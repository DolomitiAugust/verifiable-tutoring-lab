"""Prespecified paired evaluation of Qwen base, SFT, and DPO on the synthesis held-out families.

Use --plan-only before model loading. All scoring rows are held out from SFT and DPO.
Evaluation uses training/legacy/train.py for model loading, adapter application and ChatML tokenization.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261104
MAX_NEW = 192
STAGES = ("base", "sft", "dpo")
# Fixed before generation. Pick the first exported record in each metadata-only stratum.
STRATA = (
    ("snapshot", "diagnose", "incorrect"), ("snapshot", "quiz", "none"),
    ("split_join", "coach", "correct"), ("split_join", "explain", "none"),
    ("zip_pairs", "quiz", "incorrect"), ("zip_pairs", "diagnose", "correct"),
)
MAIN_TAG = "本题输出："
TRANSFER_TAG = "变式参考输出："
ASCII_ATOM = re.compile(r"(?:[+-]?[0-9]+|[A-Za-z]+(?:[ |._+:/,-]+[A-Za-z]+)*)\Z")
MAIN_STATEMENT = re.compile(
    r"^(?:(?:原题|本题|第一题|原程序|该程序|这段程序|程序|代码)(?:的)?(?:实际|最终|最后|执行后)?"
    r"|(?:实际|最终|最后))?(?:输出|打印)(?:的(?:结果|内容))?(?:为|是|结果为|结果是)?\s*[：:]\s*(.+)$"
    r"|^(?:(?:原题|本题|第一题|原程序|该程序|这段程序|程序|代码)(?:的)?(?:实际|最终|最后|执行后)?"
    r"|(?:实际|最终|最后))?(?:输出|打印)(?:的(?:结果|内容))?(?:为|是|结果为|结果是)\s*(.+)$"
)
RUBRIC = {
    "task_fit": "0=no recognizable response to the requested teaching task; 1=partially fulfills it; 2=fulfills its specified parts with case-specific evidence.",
    "reasoning": "0=no useful account of the given code, or contradicts its execution; 1=partial correct causal account; 2=correct case-specific account of the step determining the result.",
    "followup": "0=absent, unrelated, or materially incorrect; 1=relevant but incomplete or not checkable from the response; 2=usable related exercise with a correct checkable answer. A valid alternative exercise is allowed.",
    "scope": "Qualitative review requires reading the raw answer, requested task and reference; generation metrics do not assign these scores. Mark ambiguous observations explicitly.",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def records(raw):
    return [json.loads(line) for line in raw.splitlines()]


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(dataset):
    evaluation = (dataset / "evaluation.jsonl").read_bytes()
    ledger_bytes = (dataset / "audit" / "candidates.jsonl").read_bytes()
    declared = json.loads((dataset / "stats.json").read_text(encoding="utf8"))["sha256"]
    if sha256(evaluation) != declared["evaluation.jsonl"] or sha256(ledger_bytes) != declared["audit/candidates.jsonl"]:
        raise ValueError("Evaluation inputs differ from the synthesis release manifest.")
    exports = records(evaluation)
    ledger = records(ledger_bytes)
    held = [r for r in ledger if r["split"] == "evaluation" and r["decision"] == "selected"]
    held.sort(key=lambda r: r["export_line"])
    if len(held) != 60 or len(exports) != 60 or [r["export_line"] for r in held] != list(range(1, 61)):
        raise ValueError("Expected all 60 released evaluation rows in export order.")
    held_families = {r["family"] for r in held}
    train_families = {r["family"] for r in ledger if r["split"] in ("sft", "preference") and r["decision"] == "selected"}
    if held_families != {s[0] for s in STRATA} or held_families & train_families:
        raise ValueError("Evaluation family isolation changed.")
    for row, export in zip(held, exports):
        expected = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["chosen"]}]
        if export != {"messages": expected}:
            raise ValueError(f"Export text does not match ledger row {row['candidate_id']}")
    sample = [next(r for r in held if (r["family"], r["task"], r["evidence"]) == stratum) for stratum in STRATA]
    plan = {
        "design_version": 1, "seed": SEED,
        "generation": {"sampling": False, "beams": 1, "max_new_tokens": MAX_NEW,
                       "terminal_tokens": ["<|im_end|>", "tokenizer.eos_token_id (if distinct)"],
                       "additional_string_stop_rules": [], "kv_cache": True,
                       "selection": "Minimum synthesis export line in each predeclared family/task/evidence stratum; independent of model outputs."},
        "input_hashes": {"evaluation_jsonl_sha256": sha256(evaluation), "candidate_ledger_sha256": sha256(ledger_bytes)},
        "serialization": "Same encode_prompt and encode_response as training/legacy/train.py. No prompt truncation or added system instruction. Generation begins after assistant prefix; NLL covers all response tokens including the terminal im_end.",
        "generation_cases": [{k: r[k] for k in ("candidate_id", "export_line", "family", "level", "task", "profile", "evidence", "prompt")} for r in sample],
        "reference_nll_cases": [r["candidate_id"] for r in held],
        "report_examples_before_generation": [sample[i]["candidate_id"] for i in (0, 2, 4)],
        "metrics": {
            "exact_main_tag": "One unmodified line beginning with 本题输出： and the exact reference value after it. This measures literal format/reference agreement.",
            "main_fact": "correct/wrong/unresolved: use an explicit output anchor and one unambiguous simple value; alternatively a standalone answer; no substring searching for reference numbers. Other outputs remain unresolved, not factual errors.",
            "followup": "The prompt requests a new exercise, not the supplied reference exercise. Score reference answer only when the generated response contains that exact follow-up question or its program. Otherwise mark different_or_unverified; alternate exercises are judged qualitatively.",
            "reference_nll": "Microaverage negative log probability over all reference-answer tokens (including im_end) across the 60 held-out prompts under teacher forcing. Auxiliary likelihood, not free-generation accuracy.",
            "truncation": "Record stop reason, EOS token and generated length, nonempty output, and repeated-token-4gram fraction as descriptive generation diagnostics.",
        },
        "qualitative_rubric": RUBRIC,
    }
    return plan, held, sample


def literal(fragment):
    text = fragment.strip().rstrip("。！？；;").strip()
    if text.endswith(".") and text[:-1].rstrip().lstrip("+-").isdigit():
        text = text[:-1]
    for left, right in (("**", "**"), ("\x60", "\x60"), ("「", "」"), ("“", "”"), ("'", "'"), ('"', '"')):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
    if not ASCII_ATOM.fullmatch(text):
        return None
    return str(int(text)) if re.fullmatch(r"[+-]?\d+", text) else text


def anchored_lines(text, tag):
    return [line[len(tag):].strip() for line in text.splitlines() if line.startswith(tag)]


def verdict(candidates, expected, source):
    if not candidates or None in candidates or len(set(candidates)) != 1:
        return {"status": "unresolved", "value": None, "source": source}
    value = candidates[0]
    if expected.lstrip("+-").isdigit() and not value.lstrip("+-").isdigit():
        return {"status": "unresolved", "value": None, "source": source}
    return {"status": "correct" if value == expected else "wrong", "value": value, "source": source}


def extract_main(text, expected):
    tagged = anchored_lines(text, MAIN_TAG)
    if tagged:
        return verdict([literal(v) for v in tagged], expected, "explicit_main_tag")
    # For natural phrasing only inspect the original-problem part; do not treat a
    # worked example, learner's prediction, or proposed second exercise as its answer.
    first_part = re.split(r"(?m)^\s*(?:换一道题|第二题|变式(?:题|参考|[：:])|复测(?:题|[：:]))", text, maxsplit=1)[0]
    fenced = False
    values = []
    for line in first_part.splitlines():
        stripped = line.strip()
        if stripped.startswith("\x60\x60\x60"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for sentence in re.split(r"[。！？；]", stripped):
            match = MAIN_STATEMENT.fullmatch(sentence.strip())
            if match:
                values.append(literal(match.group(1) or match.group(2)))
    if values:
        return verdict(values, expected, "explicit_output_statement")
    bare = literal(text)
    if bare and bare != expected and bare.isalpha() and text.strip() == bare:
        bare = None  # A lone English word need not be an attempted program output.
    return verdict([bare], expected, "standalone_answer")


def text_metrics(text, token_ids, row):
    main_values = anchored_lines(text, MAIN_TAG)
    follow_values = anchored_lines(text, TRANSFER_TAG)
    same_exercise = row["transfer"] in text or row["transfer_code"] in text
    follow = (verdict([literal(v) for v in follow_values], row["transfer_answer"], "same_reference_exercise")
              if same_exercise else {"status": "different_or_unverified", "value": None, "source": "exercise_not_matched"})
    grams = list(zip(token_ids, token_ids[1:], token_ids[2:], token_ids[3:]))
    return {"exact_main_tag": main_values == [row["answer"]], "has_main_tag": bool(main_values),
            "main_fact": extract_main(text, row["answer"]), "anchored_transfer_fact": follow,
            "nonempty": bool(text.strip()),
            "repeated_token_4gram_fraction": None if not grams else 1 - len(set(grams)) / len(grams)}


def eval_device(model):
    return next(model.parameters()).device


def decode_cases(torch, model, tokenizer, sample, encode_prompt, terminal_ids):
    results = []
    for row in sample:
        prefix = encode_prompt(tokenizer, row["prompt"])
        x = torch.tensor([prefix], dtype=torch.long, device=eval_device(model))
        if len(prefix) + MAX_NEW > model.config.max_position_embeddings:
            raise ValueError("Predeclared generation would exceed model context; truncation is prohibited.")
        with torch.inference_mode():
            raw = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, num_beams=1,
                                 max_new_tokens=MAX_NEW, eos_token_id=terminal_ids, pad_token_id=terminal_ids[0],
                                 use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
        new_ids = raw[0, len(prefix):].tolist()
        stopped = bool(new_ids and new_ids[-1] in terminal_ids)
        if not stopped and len(new_ids) != MAX_NEW:
            raise ValueError("Unexpected unregistered generation stop condition.")
        visible_ids = new_ids[:-1] if stopped else new_ids
        response = tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        results.append({
            **{k: row[k] for k in ("candidate_id", "export_line", "family", "level", "task", "evidence", "profile", "prompt")},
            "reference": {"main_program": row["code"], "main_answer": row["answer"], "transfer_request": row["transfer"],
                          "transfer_program": row["transfer_code"], "transfer_answer": row["transfer_answer"],
                          "full_teacher_response": row["chosen"]},
            "prompt_token_count": len(prefix), "new_tokens_including_terminal": len(new_ids),
            "visible_token_count": len(visible_ids), "generated_token_ids": new_ids,
            "terminal_id": new_ids[-1] if stopped else None, "stop_reason": "eos" if stopped else "max_new_tokens",
            "raw_response": response, "metrics": text_metrics(response, visible_ids, row),
        })
        print(f"  generation {row['candidate_id']} ({len(new_ids)} tokens, {results[-1]['stop_reason']})", flush=True)
    return results


def score_references(torch, model, tokenizer, held, encode_prompt, encode_response):
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise ValueError("Expected Qwen2ForCausalLM backbone and output head.")
    rows = []
    with torch.inference_mode():
        for row in held:
            prefix = encode_prompt(tokenizer, row["prompt"])
            target = encode_response(tokenizer, row["chosen"])
            if not prefix or not target or len(prefix) + len(target) > model.config.max_position_embeddings:
                raise ValueError("Empty or over-context reference; do not truncate this evaluation.")
            device = eval_device(model)
            x = torch.tensor([prefix + target], dtype=torch.long, device=device)
            # All predicting positions, from the last prompt token through the token preceding im_end.
            h = model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state
            hidden = h[0, len(prefix) - 1:-1]
            labels = x[0, len(prefix):]
            if len(hidden) != len(target):
                raise ValueError("Causal response alignment error.")
            # Only project answer positions, in bounded chunks, to avoid a full [T, vocab] buffer.
            nll = 0.0
            for j in range(0, len(target), 16):
                logits = model.lm_head(hidden[j:j+16]).float()
                nll += torch.nn.functional.cross_entropy(logits, labels[j:j+16], reduction="sum").item()
            if not math.isfinite(nll):
                raise ValueError(f"Nonfinite reference loss in {row['candidate_id']}")
            rows.append({"candidate_id": row["candidate_id"], "export_line": row["export_line"],
                         "family": row["family"], "answer_tokens_including_im_end": len(target),
                         "nll_sum": nll, "mean_nll": nll / len(target)})
            if len(rows) % 10 == 0:
                print(f"  NLL: {len(rows)}/{len(held)}", flush=True)
    count = sum(r["answer_tokens_including_im_end"] for r in rows)
    total = sum(r["nll_sum"] for r in rows)
    return {"sample_count": len(rows), "answer_tokens_including_im_end": count, "nll_sum": total,
            "mean_nll_per_token": total / count, "rows": rows}


def generation_summary(cases):
    def total(f):
        return sum(bool(f(r)) for r in cases)
    rep = [r["metrics"]["repeated_token_4gram_fraction"] for r in cases if r["metrics"]["repeated_token_4gram_fraction"] is not None]
    return {"cases": len(cases), "strict_main_hits": total(lambda r: r["metrics"]["exact_main_tag"]),
            "main_fact_counts": dict(Counter(r["metrics"]["main_fact"]["status"] for r in cases)),
            "reference_transfer_counts": dict(Counter(r["metrics"]["anchored_transfer_fact"]["status"] for r in cases)),
            "stop_counts": dict(Counter(r["stop_reason"] for r in cases)),
            "nonempty_count": total(lambda r: r["metrics"]["nonempty"]),
            "mean_visible_tokens": statistics.mean(r["visible_token_count"] for r in cases),
            "mean_repeated_token_4gram_fraction": statistics.mean(rep) if rep else None}


def run_stage(args, stage, held, sample, plan_hash):
    import torch
    import transformers
    from train import CHAT_FORMAT, apply_adapter, encode_prompt, encode_response, load_base

    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model, tokenizer = load_base(args.base_model, args.dtype, args.threads, args.device)
    checkpoint = None
    parent_sft_hash = None
    if stage != "base":
        checkpoint = getattr(args, stage + "_adapter")
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if metadata["stage"] != stage or not metadata["finished"] or metadata["step"] < 1 or metadata["chat_format"] != CHAT_FORMAT:
            raise ValueError("The supplied checkpoint is not a completed run of the requested stage and format.")
        parent_sft_hash = metadata.get("parent_sft_sha256")
        if stage == "dpo" and args.sft_adapter and parent_sft_hash != sha256(args.sft_adapter.read_bytes()):
            raise ValueError("DPO is not a continuation of the requested SFT checkpoint.")
        del metadata
        apply_adapter(model, checkpoint)
    model.eval()
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if encode_response(tokenizer, "") != [eos]:
        raise ValueError("Evaluation EOS differs from the training serializer.")
    terminal_ids = list(dict.fromkeys([eos] + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])))
    protocol = {"plan_sha256": plan_hash, "base_hash": model._training_base_hash, "device": str(eval_device(model)),
                "dtype": str(next(model.parameters()).dtype), "threads": torch.get_num_threads(),
                "seed": SEED, "max_new_tokens": MAX_NEW, "terminal_ids": terminal_ids,
                "reference_nll_enabled": not args.skip_nll}
    common_path = args.out / "runtime.json"
    if common_path.exists() and json.loads(common_path.read_text(encoding="utf8")) != protocol:
        raise ValueError("Runtime differs from an existing paired stage; use a separate output directory.")
    if not common_path.exists():
        write_json(common_path, protocol)
    start = time.monotonic()
    print(f"Evaluating {stage}", flush=True)
    cases = decode_cases(torch, model, tokenizer, sample, encode_prompt, terminal_ids)
    nll = None if args.skip_nll else score_references(torch, model, tokenizer, held, encode_prompt, encode_response)
    result = {
        "stage": stage, "plan_sha256": plan_hash, "seed": SEED, "base_model": str(args.base_model.resolve()),
        "adapter": str(checkpoint.resolve()) if checkpoint else None,
        "adapter_sha256": sha256(checkpoint.read_bytes()) if checkpoint else None,
        "parent_sft_sha256": parent_sft_hash, "base_hash": model._training_base_hash,
        "dtype": str(next(model.parameters()).dtype), "device": str(eval_device(model)), "threads": torch.get_num_threads(),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "terminal_ids": terminal_ids,
        "summary": generation_summary(cases), "generation": cases,
        "reference_nll": nll, "model_evaluation_elapsed_seconds": round(time.monotonic() - start, 3),
    }
    write_json(args.out / f"{stage}.json", result)
    print(f"{stage}: {result['summary']} | NLL: {nll['mean_nll_per_token'] if nll else 'disabled'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "evaluation")
    parser.add_argument("--base-model", type=Path, default=ROOT / "training" / "model" / "base")
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--dpo-adapter", type=Path)
    parser.add_argument("--stages", "--stage", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-nll", action="store_true", help="Explicitly record NLL as not evaluated.")
    parser.add_argument("--plan-only", action="store_true", help="Write/verify the preregistration without loading torch/model.")
    args = parser.parse_args(argv)
    plan, held, sample = prepare(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf8")) != plan:
            raise ValueError(f"Design or source changed; use a separate output directory from {plan_path}.")
    else:
        write_json(plan_path, plan)
    plan_hash = sha256(json_bytes(plan))
    print(f"Plan: {plan_path} ({plan_hash}); {len(sample)} generation, {len(held)} NLL cases.", flush=True)
    if args.plan_only:
        return
    for stage in args.stages:
        if stage != "base":
            checkpoint = getattr(args, stage + "_adapter")
            if checkpoint is None or not checkpoint.is_file():
                parser.error(f"--{stage}-adapter must point to the requested stage's adapter.pt.")
    for stage in args.stages:
        run_stage(args, stage, held, sample, plan_hash)


if __name__ == "__main__":
    main()
