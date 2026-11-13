"""Verify the released synthesis splits and serialize complete ChatML examples for training."""

import hashlib
import json
import os
import random
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_synthesis(folder, stage):
    """Validate both training and evaluation exports before selecting a training split."""
    folder = Path(folder)
    stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    hashes = stats["sha256"]
    required = ["sft.jsonl", "preference.jsonl", "evaluation.jsonl", "audit/candidates.jsonl"]
    for name in required:
        if file_hash(folder / name) != hashes[name]:
            raise ValueError(f"synthesis file does not match its published SHA-256: {name}")
    splits = {name: [json.loads(line) for line in (folder / f"{name}.jsonl").read_text(
        encoding="utf-8").splitlines()] for name in ("sft", "preference", "evaluation")}
    metadata = {name: {} for name in splits}
    families = {name: set() for name in splits}
    prompts = {name: set() for name in splits}
    with (folder / "audit/candidates.jsonl").open(encoding="utf-8") as ledger:
        for line in ledger:
            item = json.loads(line)
            if item["decision"] != "selected":
                continue
            split, number = item["split"], item["export_line"]
            if not 1 <= number <= len(splits[split]) or number in metadata[split]:
                raise ValueError(f"Invalid or repeated ledger pointer: {split}/{number}")
            row = splits[split][number - 1]
            if split == "preference":
                match = all(row[key] == item[key] for key in ("prompt", "chosen", "rejected"))
            else:
                match = row["messages"] == [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]}]
            if not match:
                raise ValueError(f"Published row and ledger disagree: {split}/{number}")
            metadata[split][number] = {key: item[key] for key in ("candidate_id", "family", "task", "level")}
            families[split].add(item["family"])
            prompts[split].add(item["prompt"])
    for name, rows in splits.items():
        if len(rows) != stats["selected"][name]["count"] or len(rows) != len(metadata[name]):
            raise ValueError(f"Export count or traceability failed for {name}")
    if families["evaluation"] != set(stats["heldout_families"]):
        raise ValueError("Unexpected evaluation families.")
    if any(families["evaluation"] & families[name] or prompts["evaluation"] & prompts[name]
           for name in ("sft", "preference")):
        raise ValueError("A held-out evaluation family or prompt occurs in training.")
    split = {"sft": "sft", "dpo": "preference"}[stage]
    return splits[split], metadata[split], dict(hashes)


def _reject_delimiters(text):
    if not text or any(marker in text for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("Empty content or reserved ChatML delimiter in example.")


def encode_prompt(tokenizer, prompt):
    """No implicit system turn, BOS, padding or answer text in the conditioning prefix."""
    _reject_delimiters(prompt)
    text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text, add_special_tokens=False)


def encode_response(tokenizer, answer):
    """Count the assistant's text and its turn terminator; no trailing newline."""
    if answer:
        _reject_delimiters(answer)
    return tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]


def select_examples(rows, metadata, tokenizer, stage, seed, limit, max_tokens):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected, skipped = [], []
    for index in order:
        row = rows[index]
        if stage == "sft":
            prompt, text = (turn["content"] for turn in row["messages"])
            answers = [text]
        else:
            prompt, answers = row["prompt"], [row["chosen"], row["rejected"]]
        if any(not answer.strip() for answer in answers):
            raise ValueError(f"Empty training completion at export line {index + 1}")
        prefix = encode_prompt(tokenizer, prompt)
        responses = [encode_response(tokenizer, ans) for ans in answers]
        longest = len(prefix) + max(map(len, responses))
        item = metadata[index + 1]
        if longest > max_tokens:
            skipped.append({"candidate_id": item["candidate_id"], "longest_sequence": longest})
            continue
        selected.append({**item, "export_line": index + 1, "prefix": prefix, "responses": responses})
        if limit and len(selected) == limit:
            break
    if not selected:
        raise ValueError("No intact examples fit --max-tokens; no training sequence was truncated.")
    return selected, skipped
