"""Read-only model evaluation with full references, bounded generation and unseen pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import torch
import train as train


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def load_examples(path, metadata=None):
    rows = read_rows(path)
    meta = read_rows(metadata) if metadata else [{} for _ in rows]
    if len(meta) != len(rows):
        raise ValueError("Metadata and examples must have the same order and count.")
    result = []
    for i, (row, details) in enumerate(zip(rows, meta)):
        if "messages" in row:
            if [x["role"] for x in row["messages"]] != ["user", "assistant"]:
                raise ValueError("Expected complete user/assistant reference examples.")
            prompt, answer = [x["content"] for x in row["messages"]]
            answers = [answer]
        else:
            prompt, answers = row["prompt"], [row["chosen"], row["rejected"]]
        result.append({**details, **{k: v for k, v in row.items() if k not in ("messages", "chosen", "rejected")},
                       "candidate_id": row.get("candidate_id", details.get("candidate_id", f"{Path(path).stem}:{i+1:04d}")),
                       "prompt": prompt, "answers": answers, "export_line": i + 1})
    return result


def encode(rows, tokenizer, context):
    result = []
    for row in rows:
        prefix = train.encode_prompt(tokenizer, row["prompt"])
        responses = [train.encode_response(tokenizer, x) for x in row["answers"]]
        if len(prefix) + max(map(len, responses)) > context:
            raise ValueError(f"Reference exceeds context: {row['candidate_id']}; truncation prohibited.")
        result.append({**row, "prefix": prefix, "responses": responses})
    return result


def score(model, tokenizer, rows):
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            values, counts = train.response_scores(model, [row], tokenizer.pad_token_id, head_chunk=32)
            values, counts = values.cpu().tolist(), counts.cpu().tolist()
            record = {"candidate_id": row["candidate_id"], "logp_sum": values, "response_tokens": counts,
                      "mean_nll": [-v/n for v, n in zip(values, counts)]}
            if len(values) == 2:
                record.update(raw_logratio=values[0] - values[1],
                              per_token_logratio=values[0]/counts[0] - values[1]/counts[1],
                              chosen_has_higher_sequence_probability=values[0] > values[1])
            results.append(record)
            if (i + 1) % 12 == 0:
                print(f"scored {i+1}/{len(rows)}", flush=True)
    return results


def primary_answer(text, expected):
    # Conservative extraction: retain ambiguity rather than treating it as a factual error.
    values = re.findall(r"(?m)^\s*本题输出\s*[:：]\s*([^\n]+)", text)
    if not values:
        return {"status": "unresolved", "values": []}
    values = [x.strip().strip("` ").rstrip("。") for x in values]
    if len(set(values)) != 1 or expected is None:
        return {"status": "unresolved", "values": values}
    return {"status": "correct" if values[0] == str(expected).strip() else "wrong", "values": values}


def generate(model, tokenizer, rows, limit, max_new):
    results = []
    endings = sorted({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id})
    for row in rows[:limit] if limit else []:
        if len(row["prefix"]) + max_new > model.config.max_position_embeddings:
            raise ValueError("Generation would exceed model context.")
        x = torch.tensor([row["prefix"]], device=next(model.parameters()).device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                                    num_beams=1, max_new_tokens=max_new, eos_token_id=endings,
                                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                                    repetition_penalty=1.0, no_repeat_ngram_size=0)
        ids = output[0, len(row["prefix"]):].tolist()
        eos = bool(ids and ids[-1] in endings)
        visible = ids[:-1] if eos else ids
        text = tokenizer.decode(visible, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        grams = list(zip(visible, visible[1:], visible[2:], visible[3:]))
        result = {"candidate_id": row["candidate_id"], "prompt": row["prompt"],
                  "reference": row["answers"][0], "expected_main_answer": row.get("answer"),
                  "raw_response": text, "generated_token_ids": ids,
                  "generated_tokens": len(ids), "reference_tokens": len(row["responses"][0]),
                  "stop_reason": "eos" if eos else "max_new_tokens",
                  "main_fact_extraction": primary_answer(text, row.get("answer")),
                  "repeated_token_4gram_fraction": 1-len(set(grams))/len(grams) if grams else 0,
                  "seconds": time.perf_counter()-started}
        results.append(result)
        print(f"generated {row['candidate_id']} {len(ids)} {result['stop_reason']}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--preference-metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generation-limit", type=int, default=0, help="0 = NLL only; positive = first N predetermined export rows")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Evaluation output already exists; preserve it and choose a new output path.")
    if args.generation_limit < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid generation budget.")
    torch.manual_seed(20261107)
    started = time.perf_counter()
    model, tokenizer = train.load_base(args.model, args.dtype, args.threads, args.device)
    adapter_meta = None
    if args.adapter:
        adapter_meta = torch.load(args.adapter, map_location="cpu", weights_only=True)
        train.apply_adapter(model, args.adapter)
    model.eval()
    examples = encode(load_examples(args.data, args.metadata), tokenizer, model.config.max_position_embeddings)
    references = score(model, tokenizer, examples)
    generations = generate(model, tokenizer, examples, args.generation_limit, args.max_new_tokens)
    pairs = []
    if args.preferences:
        pair_examples = encode(load_examples(args.preferences, args.preference_metadata), tokenizer, model.config.max_position_embeddings)
        pairs = score(model, tokenizer, pair_examples)
    nll_sum = sum(-r["logp_sum"][0] for r in references)
    token_sum = sum(r["response_tokens"][0] for r in references)
    result = {"version": 2, "base_hash": model._training_base_hash,
              "adapter_sha256": train.file_hash(args.adapter) if args.adapter else None,
              "adapter_stage": adapter_meta["stage"] if adapter_meta else "base",
              "adapter_step": adapter_meta["step"] if adapter_meta else 0,
              "parent_sft_sha256": adapter_meta.get("parent_sft_sha256") if adapter_meta else None,
              "source_sha256": train.file_hash(Path(__file__)),
              "inputs_sha256": {str(p): train.file_hash(p) for p in [args.data, args.metadata, args.preferences, args.preference_metadata] if p},
              "protocol": {"generation_limit": args.generation_limit, "max_new_tokens": args.max_new_tokens,
                           "do_sample": False, "repetition_penalty": 1.0, "reference_truncation": False,
                           "preference_note": "Sequence likelihood preference is length-sensitive; not a human preference judgement.",
                           "automatic_fact_note": "Unresolved extractions are not factual errors or zero accuracy."},
              "summary": {"reference_nll": nll_sum/token_sum, "reference_cases": len(references), "reference_tokens": token_sum,
                          "generation_cases": len(generations), "stop_counts": dict(Counter(r["stop_reason"] for r in generations)),
                          "preference_pairs": len(pairs),
                          "raw_preference_accuracy": sum(r["chosen_has_higher_sequence_probability"] for r in pairs)/len(pairs) if pairs else None},
              "reference_nll": references, "generation": generations, "preferences": pairs,
              "runtime": {"device": args.device, "dtype": args.dtype, "torch": str(torch.__version__),
                          "seconds": time.perf_counter()-started}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train.write_json(args.out, result)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
