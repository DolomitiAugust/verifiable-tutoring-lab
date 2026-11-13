"""Independently audit the saved synthesis exports. Run with Python 3.10+ (standard library)."""
import argparse
import ast
import builtins
import io
import json
import random
import statistics
from collections import Counter
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

from curriculum import FAMILIES, LEVELS, make_case
from pedagogy import reject, render


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"sft": 1000, "preference": 500, "evaluation": 60}
REASONS = ("factual_error", "incomplete", "off_target")
ANSWER_TAG = "本题输出："
TRANSFER_TAG = "变式参考输出："
# Fixed before inspecting the exports. Select the first matching line in each stratum.
SAMPLE_PLAN = (
    ("explain", "基础", "preference", "factual_error"),
    ("explain", "进阶", "sft", None),
    ("explain", "挑战", "evaluation", None),
    ("diagnose", "基础", "sft", None),
    ("diagnose", "进阶", "preference", "incomplete"),
    ("diagnose", "挑战", "preference", "off_target"),
    ("quiz", "基础", "evaluation", None),
    ("quiz", "进阶", "preference", "off_target"),
    ("quiz", "挑战", "sft", None),
    ("coach", "基础", "preference", "incomplete"),
    ("coach", "进阶", "evaluation", None),
    ("coach", "挑战", "preference", "factual_error"),
)


class Checks:
    def __init__(self):
        self.results = {}

    def add(self, name, ok, where):
        row = self.results.setdefault(name, {"passed": 0, "checked": 0, "first_failures": []})
        row["checked"] += 1
        row["passed"] += int(bool(ok))
        if not ok and len(row["first_failures"]) < 6:
            row["first_failures"].append(str(where))


def read_records(path):
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}:{number}: expected JSON object")
        records.append(record)
    return records


def execute_case_program(source):
    """The caller supplies only code rebuilt by local make_case, never ledger text."""
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, "<rebuilt curriculum case>", "exec"), {})
    return output.getvalue().strip()


def program_signature(source):
    """Ignore identifier spelling and syntax-only formatting but keep literal values."""
    tree = ast.parse(source)
    variables = {}
    for node in ast.walk(tree):
        attr = ("id" if isinstance(node, ast.Name) else
                "arg" if isinstance(node, ast.arg) else
                "name" if isinstance(node, ast.FunctionDef) else None)
        if attr is not None:
            name = getattr(node, attr)
            if not hasattr(builtins, name):
                setattr(node, attr, variables.setdefault(name, "v" + str(len(variables))))
    return ast.dump(tree, include_attributes=False)


def tagged(text, tag):
    return [line[len(tag):].strip() for line in text.splitlines() if line.startswith(tag)]


def split_fields(record, split):
    if split == "preference":
        if set(record) != {"prompt", "chosen", "rejected"}:
            raise ValueError("expected prompt/chosen/rejected")
        values = [record[key] for key in ("prompt", "chosen", "rejected")]
    else:
        if set(record) != {"messages"} or not isinstance(record["messages"], list):
            raise ValueError("expected messages")
        messages = record["messages"]
        if (len(messages) != 2 or not all(isinstance(m, dict) and set(m) == {"role", "content"}
                                          for m in messages)
                or [m["role"] for m in messages] != ["user", "assistant"]):
            raise ValueError("expected one user message and one assistant message")
        values = [m["content"] for m in messages]
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("empty/non-text field")
    return values


def negative_structure(task, reason, text, case):
    """Check verifiable text markers, not the truth of free-form educational claims."""
    output = tagged(text, ANSWER_TAG)
    if reason == "factual_error":
        return output == [case["wrong"]] and case["wrong"] != case["answer"]
    if output != [case["answer"]]:
        return False
    if reason == "incomplete":
        if task == "diagnose":
            return case["misconception"] not in text
        if task == "quiz":
            return not tagged(text, TRANSFER_TAG)
        return case["transfer"] not in text
    if reason == "off_target":
        return case["transfer"] not in text and "print(" in text and not tagged(text, TRANSFER_TAG)
    return False


def audit_row(row, split, line, fields, checks, program_sets):
    pid = row["candidate_id"]
    prompt, chosen = fields[:2]
    reason = row.get("reason")
    spot = f"{split}:{line}/{pid}"
    key = lambda name: f"{split}.{name}"
    saved = [row.get("prompt"), row.get("chosen")]
    if split == "preference":
        saved.append(row.get("rejected"))
    checks.add(key("export_text_matches_ledger"), fields == saved, spot)
    seed, family = row.get("case_seed"), row.get("family")
    trusted = family in FAMILIES and type(seed) is int and 0 <= seed < 2**61
    checks.add(key("known_family_and_seed"), trusted, spot)
    if not trusted:
        return
    case = make_case(family, random.Random(seed))
    keys = ("family", "level", "topic", "code", "answer", "wrong", "transfer", "transfer_code", "transfer_answer")
    checks.add(key("case_fields_match_seed"), all(case[k] == row.get(k) for k in keys), spot)
    for label, code_key, answer_key in (("main", "code", "answer"), ("transfer", "transfer_code", "transfer_answer")):
        code = case[code_key]
        checks.add(key("execute_" + label), execute_case_program(code) == row.get(answer_key), spot)
        program_sets[split][label].setdefault(program_signature(code), []).append(spot)
    checks.add(key("original_code_in_prompt"), case["code"] in prompt, spot)
    checks.add(key("chosen_main_answer"), tagged(chosen, ANSWER_TAG) == [case["answer"]], spot)
    checks.add(key("chosen_transfer_answer"), tagged(chosen, TRANSFER_TAG) == [case["transfer_answer"]]
               and case["transfer"] in chosen, spot)
    # Exact agreement with the trusted local text renderer verifies provenance, not pedagogy.
    try:
        expected_prompt, expected_chosen = render(case, row["profile"], row["task"], row["variant"])
        checks.add(key("prompt_chosen_match_seed_and_metadata"),
                   (prompt, chosen) == (expected_prompt, expected_chosen), spot)
        if split == "preference":
            denied = fields[2]
            checks.add(key("ordered_pair_matches_renderer"), reason in REASONS and chosen != denied
                       and denied == reject(case, row["profile"], row["task"], expected_chosen, reason, row["variant"]), spot)
            checks.add(key("negative_text_markers_match_reason"),
                       negative_structure(row["task"], reason, denied, case), spot)
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        checks.add(key("valid_template_metadata"), False, f"{spot}: {exc}")


def lengths(rows):
    groups = {}
    for reason in ("all", *REASONS):
        pairs = [(len(r["chosen"]), len(r["rejected"])) for r in rows if reason == "all" or r["reason"] == reason]
        if not pairs:
            continue
        pos, neg = zip(*pairs)
        groups[reason] = {"count": len(pairs), "chosen_mean": round(statistics.mean(pos), 2),
                          "rejected_mean": round(statistics.mean(neg), 2), "chosen_median": statistics.median(pos),
                          "rejected_median": statistics.median(neg), "chosen_shorter": sum(a < b for a, b in pairs),
                          "equal": sum(a == b for a, b in pairs), "chosen_longer": sum(a > b for a, b in pairs)}
    return groups


def run(dataset):
    checks = Checks()
    ledger = read_records(dataset / "audit" / "candidates.jsonl")
    selected = {split: [] for split in TARGETS}
    ids = set()
    for entry in ledger:
        cid = entry.get("candidate_id")
        checks.add("ledger_unique_candidate_ids", isinstance(cid, str) and cid not in ids, cid)
        ids.add(cid)
        active = entry.get("decision") == "selected"
        split = entry.get("split")
        checks.add("ledger_export_pointer_only_for_selected", ("export_line" in entry) == active, cid)
        checks.add("ledger_known_split", split in TARGETS, cid)
        if active and split in TARGETS:
            selected[split].append(entry)
    sets = {split: {"main": {}, "transfer": {}} for split in TARGETS}
    prompts = {split: set() for split in TARGETS}
    counts = {}
    inspected = []
    for split, target in TARGETS.items():
        exports = read_records(dataset / f"{split}.jsonl")
        rows = selected[split]
        counts[split] = {"exported": len(exports), "selected_in_ledger": len(rows), "target": target}
        checks.add("target_exact", len(exports) == len(rows) == target, split)
        by_line = {r.get("export_line"): r for r in rows}
        checks.add("one_to_one_ledger_pointers", len(rows) == len(by_line)
                   and set(by_line) == set(range(1, len(exports) + 1)), split)
        for line, record in enumerate(exports, 1):
            spot = f"{split}:{line}"
            try:
                fields = split_fields(record, split)
            except (ValueError, TypeError) as exc:
                checks.add(f"{split}.schema", False, f"{spot}: {exc}")
                continue
            checks.add(f"{split}.schema", True, spot)
            checks.add(f"{split}.unique_prompt_within_split", fields[0] not in prompts[split], spot)
            prompts[split].add(fields[0])
            if line not in by_line:
                continue
            row = by_line[line]
            audit_row(row, split, line, fields, checks, sets)
            inspected.append({**{k: row.get(k) for k in ("candidate_id", "task", "family", "profile", "level", "reason")},
                              "split": split, "export_line": line, "chosen": fields[1],
                              "rejected": fields[2] if split == "preference" else None})
        checks.add("unique_normalized_main_programs_within_split",
                   len(sets[split]["main"]) == len(exports), split)
    overlaps = {}
    for a, b in combinations(TARGETS, 2):
        group = f"{a}/{b}"
        prompt_count = len(prompts[a] & prompts[b])
        overlaps[group] = {"exact_prompts": prompt_count}
        checks.add("no_exact_prompt_overlap_across_splits", prompt_count == 0, group)
        for side_a in ("main", "transfer"):
            for side_b in ("main", "transfer"):
                common = sets[a][side_a].keys() & sets[b][side_b].keys()
                examples = sorted((sets[a][side_a][sig][0], sets[b][side_b][sig][0]) for sig in common)[:3]
                overlaps[group][f"{side_a}/{side_b}"] = {"distinct_shared_signatures": len(common), "examples": examples}
                checks.add("no_program_overlap_across_splits", not common, f"{group}:{side_a}/{side_b}")
    families = {split: sorted({r["family"] for r in rows}) for split, rows in selected.items()}
    withheld = set(families["evaluation"])
    checks.add("holdout_one_family_in_each_level", Counter(FAMILIES.get(f) for f in withheld) == Counter(LEVELS), families)
    for split in ("sft", "preference"):
        checks.add("no_evaluation_family_in_training", not withheld.intersection(families[split]), split)
    sample = []
    for task, level, split, reason in SAMPLE_PLAN:
        options = [r for r in inspected if (r["task"], r["level"], r["split"], r["reason"]) == (task, level, split, reason)]
        checks.add("sample_plan_strata_present", bool(options), (task, level, split, reason))
        if options:
            sample.append({k: v for k, v in min(options, key=lambda r: r["export_line"]).items() if k not in ("chosen", "rejected")})
    return {
        "all_required_checks_pass": all(v["passed"] == v["checked"] for v in checks.results.values()),
        "counts": counts, "ledger_candidates": len(ledger), "checks": checks.results, "families_by_split": families,
        "program_signature_method": "Python AST ignoring identifier spelling and formatting; literal values retained.",
        "cross_split_overlap": overlaps,
        "overlap_scope": "All four main/transfer combinations must be disjoint for each split pair. Distinct signatures count exact normalised programs, not semantic independence.",
        "preference_lengths_unicode_characters": lengths([r for r in inspected if r["split"] == "preference"]),
        "sample_plan": "First export line in each predeclared task/level/split/reason stratum: 3 SFT, 6 preference, 3 evaluation.",
        "sample_for_reading": sample,
        "scope": "The automatic checks verify counts, texts, replayed programs and observable response markers. They do not judge learning effectiveness, clarity or human preferences.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "review.json")
    args = parser.parse_args()
    result = run(args.dataset)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit {'passed' if result['all_required_checks_pass'] else 'failed'}; wrote {args.output}")
    return 0 if result["all_required_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
