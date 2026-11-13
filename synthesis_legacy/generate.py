"""Generate, filter and audit Chinese educational instruction data. Python 3.10+."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import keyword
import random
import statistics
import tempfile
import tokenize
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

from curriculum import FAMILIES, LEVELS, make_case
from pedagogy import PREF_REASONS, PROFILES, TASKS, evidence_kind, reject, render

DEFAULT_SEED = 20261104
TARGETS = {"sft": 1000, "preference": 500, "evaluation": 60}
FILES = {name: name + ".jsonl" for name in TARGETS}
OUTPUT_TAG = "本题输出："
ROOT = Path(__file__).resolve().parents[1]


def digest(b):
    return hashlib.sha256(b).hexdigest()


def code_keys(code):
    """Ignore comments, whitespace and variable renaming. Keep literal values for deduplication."""
    names, full, skeleton = {}, [], []
    ignored = {tokenize.ENCODING, tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER}
    for item in tokenize.generate_tokens(io.StringIO(code).readline):
        typ, val = item.type, item.string
        if typ in ignored:
            continue
        if typ == tokenize.NAME and not keyword.iskeyword(val) and not hasattr(builtins, val):
            val = names.setdefault(val, f"_v{len(names)}")
        if typ in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}:
            val = ""
        full.append((typ, val))
        skeleton.append((typ, "" if typ in {tokenize.STRING, tokenize.NUMBER} else val))
    return tuple(digest(json.dumps(seq, ensure_ascii=False).encode()) for seq in (full, skeleton))


def execute_authored(code):
    """Only called on bounded programs rebuilt from the local, trusted curriculum module."""
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(code, "<curriculum>", "exec"), {})
    return output.getvalue().strip()


def answer_in(text, tag=OUTPUT_TAG):
    values = [line[len(tag):].strip() for line in text.splitlines() if line.startswith(tag)]
    return values[0] if len(values) == 1 else None


def inspect_candidate(case, row):
    if not all(isinstance(v, str) and v.strip() for v in (row["prompt"], row["chosen"], case["answer"], case["transfer_answer"])):
        return "empty_field"
    if execute_authored(case["code"]) != case["answer"]:
        return "incorrect_reference"
    if execute_authored(case["transfer_code"]) != case["transfer_answer"]:
        return "incorrect_transfer"
    if case["wrong"] == case["answer"] or case["code"] not in row["prompt"]:
        return "ambiguous_or_missing_problem"
    if answer_in(row["chosen"]) != case["answer"]:
        return "incorrect_chosen"
    if answer_in(row["chosen"], "变式参考输出：") != case["transfer_answer"]:
        return "incorrect_transfer_text"
    if row["task"] in ("quiz", "coach") and case["answer"] == case["transfer_answer"]:
        # A learner who repeats the first answer could pass this particular follow-up.
        return "weak_followup"
    if row["split"] == "preference":
        if not row["rejected"].strip() or row["chosen"].strip() == row["rejected"].strip():
            return "invalid_pair"
        wanted = case["wrong"] if row["reason"] == "factual_error" else case["answer"]
        if answer_in(row["rejected"]) != wanted:
            return "incorrect_negative_design"
    return "eligible"


def quota_plan():
    # Reserve one complete task family per instructional level for evaluation.
    by_level = {level: sorted(f for f, lv in FAMILIES.items() if lv == level) for level in LEVELS}
    groups = {}
    held = [by_level[level][-1] for level in LEVELS]
    for split, total in TARGETS.items():
        groups[split] = Counter()
        for i in range(total):
            level = LEVELS[i % len(LEVELS)]
            options = [by_level[level][-1]] if split == "evaluation" else by_level[level][:-1]
            groups[split][options[(i // len(LEVELS)) % len(options)]] += 1
    return groups, held


def candidate(rng, split, family, n):
    case_seed = rng.getrandbits(61)
    case = make_case(family, random.Random(case_seed))
    profile, task, variant = rng.choice(PROFILES), rng.choice(TASKS), rng.randrange(24)
    prompt, chosen = render(case, profile, task, variant)
    row = dict(candidate_id=f"C{n:05d}", split=split, family=family, level=case["level"],
               topic=case["topic"], case_seed=case_seed, profile=profile, task=task, variant=variant,
               evidence=evidence_kind(task, variant),
               prompt=prompt, chosen=chosen, code=case["code"], answer=case["answer"],
               transfer=case["transfer"], transfer_code=case["transfer_code"],
               transfer_answer=case["transfer_answer"], wrong=case["wrong"])
    if split == "preference":
        row["reason"] = rng.choice(PREF_REASONS)
        row["rejected"] = reject(case, profile, task, chosen, row["reason"], variant)
        row["chosen_shorter"] = len(chosen) < len(row["rejected"])
    row["code_key"], row["structure_key"] = code_keys(case["code"])
    row["transfer_key"] = code_keys(case["transfer_code"])[0]
    row["decision"] = inspect_candidate(case, row)
    return row


def train_row(row):
    if row["split"] == "preference":
        return {k: row[k] for k in ("prompt", "chosen", "rejected")}
    return {"messages": [{"role": "user", "content": row["prompt"]},
                         {"role": "assistant", "content": row["chosen"]}]}


def serialize(obj, pretty=False):
    return json.dumps(obj, ensure_ascii=False, sort_keys=pretty, indent=2 if pretty else None)


def encode_lines(rows):
    return "".join(serialize(row) + "\n" for row in rows).encode("utf-8")


def run(seed):
    rng = random.Random(seed)
    goals, held = quota_plan()
    plan = [(split, family) for split, quotas in goals.items()
            for family, count in quotas.items() for _ in range(count * 2)]
    rng.shuffle(plan)
    # Extra preference candidates enlarge the pool of pairs where a better response is shorter.
    extra_pairs = [("preference", family) for family, n in goals["preference"].items() for _ in range(2 * n)]
    random.Random(seed + 1).shuffle(extra_pairs)
    plan.extend(extra_pairs)
    ledger, buckets, seen_code, seen_prompt = [], defaultdict(list), set(), set()
    for i, (split, family) in enumerate(plan):
        row = candidate(rng, split, family, i + 1)
        ledger.append(row)
        if row["decision"] != "eligible":
            continue
        program_keys = {row["code_key"], row["transfer_key"]}
        if program_keys & seen_code:
            row["decision"] = "overlapping_program"
            continue
        if row["prompt"] in seen_prompt:
            row["decision"] = "duplicate_prompt"
            continue
        seen_code.update(program_keys)
        seen_prompt.add(row["prompt"])
        buckets[split, family].append(row)

    selected = {k: [] for k in TARGETS}
    for split, by_family in goals.items():
        for family, target in by_family.items():
            pool = buckets[split, family]
            if len(pool) < target:
                raise ValueError(f"{split}/{family}: only {len(pool)} eligible for {target}; inspect the curriculum or increase pool size")
            axes = ["task", "profile"] + (["reason"] if split == "preference" else [])
            counts = Counter()
            for _ in range(target):
                def priority(j):
                    row = pool[j]
                    length_count = counts["length", row["chosen_shorter"]] if split == "preference" else 0
                    return length_count, sum(counts[axis, row[axis]] for axis in axes)
                idx = min(range(len(pool)), key=priority)
                pick = pool.pop(idx)
                if split == "preference":
                    counts["length", pick["chosen_shorter"]] += 1
                for axis in axes:
                    counts[axis, pick[axis]] += 1
                pick["decision"] = "selected"
                selected[split].append(pick)
            for row in pool:
                row["decision"] = "quota_surplus"

    blobs = {}
    for split, rows in selected.items():
        rng.shuffle(rows)
        for i, row in enumerate(rows, 1):
            row["export_line"] = i
        blobs[FILES[split]] = encode_lines(train_row(row) for row in rows)

    def count(rows, field):
        return dict(sorted(Counter(r[field] for r in rows).items()))

    def describe(rows):
        return {"count": len(rows), "levels": count(rows, "level"), "topics": count(rows, "topic"),
                "families": count(rows, "family"), "tasks": count(rows, "task"),
                "profiles": count(rows, "profile"), "student_evidence": count(rows, "evidence"),
                "program_structures": len({r["structure_key"] for r in rows})}

    pairs = selected["preference"]
    pos_chars = [len(r["chosen"]) for r in pairs]
    neg_chars = [len(r["rejected"]) for r in pairs]
    stats = {"seed": seed, "candidate_count": len(ledger), "heldout_families": held,
             "decisions": count(ledger, "decision"), "generated_by_split": count(ledger, "split"),
             "decisions_by_split": {k: count([r for r in ledger if r["split"] == k], "decision") for k in TARGETS},
             "selected": {key: describe(rows) for key, rows in selected.items()},
             "preference_reasons": count(pairs, "reason"), "chosen_length_median": statistics.median(pos_chars),
             "rejected_length_median": statistics.median(neg_chars),
             "chosen_shorter": sum(a < b for a, b in zip(pos_chars, neg_chars)),
             "equal_length": sum(a == b for a, b in zip(pos_chars, neg_chars))}
    blobs["audit/candidates.jsonl"] = encode_lines(ledger)
    stats["sha256"] = {k: digest(v) for k, v in blobs.items()}
    blobs["stats.json"] = (serialize(stats, True) + "\n").encode("utf-8")
    return blobs, stats


def check_saved(folder, seed):
    blobs, stats = run(seed)
    for name, expected in blobs.items():
        if (folder / name).read_bytes() != expected:
            raise ValueError(f"Saved artifact differs from a clean regeneration: {name}")
    ledger = [json.loads(row) for row in blobs["audit/candidates.jsonl"].decode().splitlines()]
    selected = [r for r in ledger if r["decision"] == "selected"]
    if len({r["prompt"] for r in selected}) != len(selected) or len({r["code_key"] for r in selected}) != len(selected):
        raise ValueError("Selected rows overlap")
    if {r["family"] for r in selected if r["split"] == "evaluation"} & {r["family"] for r in selected if r["split"] != "evaluation"}:
        raise ValueError("Held-out family overlap")
    for split, goal in TARGETS.items():
        records = [json.loads(line) for line in (folder / FILES[split]).read_text(encoding="utf-8").splitlines()]
        matching = sorted((r for r in selected if r["split"] == split), key=lambda r: r["export_line"])
        if len(records) != goal or records != [train_row(r) for r in matching]:
            raise ValueError(f"Export invalid for {split}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--out", type=Path, default=ROOT / "data/legacy")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "build" and not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".synthesis-") as tmp:
            stage = Path(tmp) / "dataset"
            blobs, _ = run(args.seed)
            for name, data in blobs.items():
                (stage / name).parent.mkdir(parents=True, exist_ok=True)
                (stage / name).write_bytes(data)
            check_saved(stage, args.seed)
            stage.rename(out)
    stats = check_saved(out, args.seed)
    print(f"Verified {out}: {stats['candidate_count']} candidates, " +
          ", ".join(f"{k}={v['count']}" for k, v in stats["selected"].items()))


if __name__ == "__main__":
    main()

