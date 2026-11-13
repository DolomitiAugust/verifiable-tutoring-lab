"""Freeze then run auditable comparisons. Existing results are never overwritten."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor.structured.system import TutoringSystem, ROLE_BUDGETS, MODES, ARMS
from tutor.structured.constraints import validate_constraint

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    tmp = Path(path).with_suffix(".writing.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def execute_reference(code):
    """Run only frozen or controller-authored bounded Python after an AST allowlist."""
    tree = ast.parse(code)
    forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.ClassDef,
                 ast.Global, ast.Nonlocal, ast.Lambda, ast.Try, ast.Raise)
    calls = {"print", "sum", "range", "zip", "enumerate", "len", "list", "str", "int"}
    calls.update(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Disallowed reference construct")
        if isinstance(node, ast.Attribute) and node.attr not in {"append", "split", "join", "copy"}:
            raise ValueError("Disallowed reference attribute")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in calls:
            raise ValueError("Disallowed reference call")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True,
                            text=True, encoding="utf-8", timeout=3)
    if result.returncode:
        raise ValueError(result.stderr)
    return result.stdout.removesuffix("\n")


def selected_cases(suite, subset):
    return [c for c in suite["cases"] if subset == "all" or c["case_id"] in suite["paired_case_ids"]]


def prepare(args):
    suite = read(args.cases)
    cases = selected_cases(suite, args.subset)
    refs = []
    for case in cases:
        for turn in case["turns"]:
            output = execute_reference(turn["inputs"]["code"])
            if output != turn["reference"]["main_answer"]:
                raise ValueError("Frozen reference mismatch: " + turn["turn_id"])
            refs.append({"turn_id": turn["turn_id"], "main_answer": output})
    source = {p.name: sha(p) for p in HERE.iterdir() if p.suffix=='.py' or p.name in ('prompts.clean.json','prompts.guarded.json','experiment_design.json')}
    identity = {"kind": args.kind, "model_path": str(args.model.resolve()) if args.model else None,
                "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
                "adapter_sha256": sha(args.adapter) if args.adapter else None}
    if args.kind == "instruct" and args.mode not in ("tool_only", "tool_no_memory"):
        from tutor.structured.model_backend import verify_instruct_source
        verified = verify_instruct_source(args.model)
        identity.update(model_id=verified["model_id"], revision=verified["revision"], source_manifest_sha256=sha(args.model / "source.json"))
    design = {"source_sha256": source, "cases_sha256": sha(args.cases), "mode": args.mode,"arm":args.arm,
              "subset": args.subset, "case_ids": [c["case_id"] for c in cases], "turn_count": len(refs),
              "role_budgets": ROLE_BUDGETS, "threads": args.threads, "identity": identity,
              "references": refs, "evaluation": suite["evaluation"], "python": sys.version}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "plan.json").exists():
        plan = read(args.out / "plan.json")
        if plan["design"] != design or sha(args.out / "cases.frozen.json") != sha(args.cases):
            raise ValueError("Frozen design differs. Use a new directory; do not overwrite.")
    else:
        if any(args.out.iterdir()):
            raise ValueError("New plan requires an empty output folder")
        (args.out / "cases.frozen.json").write_bytes(args.cases.read_bytes())
        plan = {"frozen_at_utc": now(), "design": design}
        write(args.out / "plan.json", plan)
    return suite, cases, plan


def automatic_checks(case, turn, trace, prior_ids):
    main, ref = trace["tool"]["main"], turn["reference"]
    checks = []
    def add(key, passed, observed, expected):
        checks.append({"check": key, "passed": bool(passed), "observed": observed, "expected": expected})
    add("main_output", main["ok"] and main["output"] == ref["main_answer"], main["output"], ref["main_answer"])
    add("student_status", trace["student_evidence"]["status"] == ref["student_status"], trace["student_evidence"]["status"], ref["student_status"])
    add("student_submission", trace["student_evidence"]["submitted"] == (turn["inputs"]["student_answer"] is not None), trace["student_evidence"]["submitted"], turn["inputs"]["student_answer"] is not None)
    follow = trace["tool"]["followup"]
    if trace["final"]["request_policy"]["wants_followup"]:
        add("followup_delivered", bool(follow and follow["ok"]), bool(follow and follow["ok"]), True)
        if follow and follow["ok"]:
            output = execute_reference(follow["code"])
            add("followup_independent_output", output == follow["output"], follow["output"], output)
            constraint = validate_constraint(turn["inputs"]["code"], follow["code"], trace["final"]["constraint_spec"])
            add("followup_constraint", constraint["passed"], constraint, "request restriction satisfied")
    before = trace["memory"]["before"]["topics"]
    after = trace["memory"]["after"]["topics"]
    for name in ("attempts", "correct", "incorrect", "unanswered", "unverified"):
        delta = sum(r[name] for r in after) - sum(r[name] for r in before)
        expected = int(ref["student_status"] in ("correct", "incorrect")) if name == "attempts" else int(ref["student_status"] == name)
        add("memory_" + name, delta == expected, delta, expected)
    if trace["final"]["request_policy"]["wants_harder"]:
        levels = ("基础", "进阶", "挑战")
        target = levels[min(levels.index(main["difficulty"]) + 1, 2)]
        actual = follow["difficulty"] if follow and follow["ok"] else None
        add("requested_higher_difficulty", actual == target, actual, target)
        checks[-1]["counterfactual_expected_failure"] = not trace["memory"]["enabled"]
    for check in case.get("memory_checks", []):
        if check["turn_id"] != turn["turn_id"] or check["kind"] == "student_count_delta":
            continue
        kind = check["kind"]
        if kind == "retrieved_contains_turn":
            ids = [e["turn_id"] for e in trace["memory"]["retrieved"]]
            expected = prior_ids.get(check["prior_turn_id"])
            add("recall_" + check["prior_turn_id"], expected is not None and expected in ids, ids, expected)
            checks[-1]["counterfactual_expected_failure"] = not trace["memory"]["enabled"]
        elif kind == "retrieved_only_student":
            ids = [e["student_id"] for e in trace["memory"]["retrieved"]]
            add("student_isolation", all(x == check["student_id"] for x in ids), ids, check["student_id"])
        elif kind == "no_prior_student_memory":
            add("new_student_memory", not before and not trace["memory"]["retrieved"], before, [])
        elif kind == "recommendation":
            actual = trace["memory"]["recommendation"]["action"]
            add("history_based_recommendation", actual == check["action"], actual, check["action"])
            checks[-1]["counterfactual_expected_failure"] = not trace["memory"]["enabled"]
        elif kind == "before_topic_count":
            rows = [r for r in before if r["topic"] == turn["inputs"]["topic"]]
            for name, expected in check.items():
                if name in ("kind", "turn_id"):
                    continue
                observed = sum(r[name] for r in rows)
                add("prior_topic_" + name, observed == expected, observed, expected)
        else:
            raise ValueError("Unsupported frozen memory check: " + kind)
    return checks


def summarize(result):
    turns = [t for c in result["cases"] for t in c["turns"]]
    roles = [a for t in turns for a in t["trace"]["agents"].values()]
    checks = [a for t in turns for a in t["automatic_checks"]]
    teacher_rows=[t for t in turns if 'teacher' in t['trace']['agents']]
    parsed_teachers=[t for t in teacher_rows if type(t['trace']['agents']['teacher'].get('parsed')) is dict]
    false_approvals=[]
    for t in turns:
        trace=t['trace']; parsed=trace['agents'].get('reviewer',{}).get('parsed')
        current_checks={k:v for k,v in trace['final']['checks'].items() if not k.startswith('model_review_')}
        if type(parsed) is dict and parsed.get('approved') is True and not all(current_checks.values()):
            false_approvals.append(t['fixture_turn_id'])
    return {"case_count": len(result["cases"]), "turn_count": len(turns),
            "real_model_calls": sum(a.get("real_model_call", False) for a in roles),
            "generated_tokens": sum(a.get("new_tokens", 0) for a in roles),
            "role_adoption": {r: {"accepted": sum(a["accepted"] for a in roles if a["role"] == r), "calls": sum(a["role"] == r for a in roles)} for r in ROLE_BUDGETS},
            "final_llm_used": sum(t["trace"]["final"]["llm_used"] for t in turns),
            "model_freeform_prose_used":sum(t['trace']['final'].get('model_freeform_prose_used',False) for t in turns),
            "model_selected_tool_quotes":sum(t['trace']['final']['explanation_source']=='model_selected_tool_quotes' for t in turns),
            "teacher_parsed_count":len(parsed_teachers),"teacher_call_count":len(teacher_rows),
            "teacher_current_answer_matches":sum(t['trace']['agents']['teacher']['parsed'].get('answer')==t['trace']['tool']['main']['output'] for t in parsed_teachers),
            "teacher_current_status_matches":sum(t['trace']['agents']['teacher']['parsed'].get('status')==t['trace']['student_evidence']['status'] for t in parsed_teachers),
            "raw_reviewer_false_approval_count":len(false_approvals),"raw_reviewer_false_approval_turns":false_approvals,
            "final_sources": dict(Counter(t["trace"]["final"]["source"] for t in turns)),
            "automatic_passed": sum(c["passed"] for c in checks), "automatic_total": len(checks),
            "stop_reasons": dict(Counter(a.get("stop_reason", "error") for a in roles)),
            "qualitative_status": "unreviewed", "complete_case_success_claim": None,
            "interpretation": "Descriptive regression checks; generated prose still requires criterion-based review."}


def run(args):
    suite, cases, plan = prepare(args)
    if args.prepare_only:
        print("Frozen", args.out / "plan.json", "turns", plan["design"]["turn_count"], flush=True)
        return
    if (args.out / "results.json").exists() or (args.out / "memory.sqlite").exists():
        raise FileExistsError("Existing run evidence cannot be overwritten")
    started = time.perf_counter()
    backend = None
    if args.mode not in ("tool_only", "tool_no_memory"):
        from tutor.structured.model_backend import LocalQwenBackend
        backend = LocalQwenBackend(args.model, args.adapter, args.kind, args.threads)
    system = TutoringSystem(backend, args.out / "memory.sqlite", mode=args.mode,arm=args.arm)
    result = {"schema_version": 2, "started_at_utc": now(), "plan_sha256": sha(args.out / "plan.json"),
              "suite_id": suite["suite_id"], "mode": args.mode,"arm":args.arm, "subset": args.subset,
              "completed": False, "backend": getattr(backend, "metadata", {"kind": "no_model"}), "cases": []}
    prior_ids = {}
    try:
        for case in cases:
            record = {"case_id": case["case_id"], "title": case["title"], "required_qualitative_criteria": case["required_qualitative_criteria"], "turns": []}
            result["cases"].append(record)
            for turn in case["turns"]:
                if turn.get("restart_system_before"):
                    system.close()
                    system = TutoringSystem(backend, args.out / "memory.sqlite", mode=args.mode,arm=args.arm)
                trace = system.run_turn(**turn["inputs"])
                checks = automatic_checks(case, turn, trace, prior_ids)
                prior_ids[turn["turn_id"]] = trace["turn_id"]
                record["turns"].append({"fixture_turn_id": turn["turn_id"], "reference": turn["reference"], "trace": trace, "automatic_checks": checks})
                write(args.out / "results.json", result)
                print(turn["turn_id"], "checks", sum(c["passed"] for c in checks), "/", len(checks),
                      "LLM explanation", trace["final"]["llm_used"], flush=True)
        result["completed"] = True
    finally:
        system.close()
        result["finished_at_utc"] = now()
        result["elapsed_seconds_including_load"] = time.perf_counter() - started
        result["summary"] = summarize(result)
        write(args.out / "results.json", result)
    write(args.out / "metrics.json", result["summary"])
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=HERE / "cases.counterexample.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, default="multi")
    p.add_argument('--arm',choices=ARMS,default='guarded_quotes')
    p.add_argument("--subset", choices=("all", "paired"), default="all")
    p.add_argument("--kind", choices=("sft", "instruct"), default="instruct")
    p.add_argument("--model", type=Path, default=ROOT / "models/qwen2.5-0.5b-instruct")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--prepare-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
