"""Freeze and run the predeclared tutor suite. Reference fixtures never enter the system."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("cases.json")
DEFAULT_OUT = ROOT / "outputs" / "legacy_tutor"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def execute_reference(code):
    """Only the reviewed, frozen fixture programs reach CPython, never model output."""
    tree = ast.parse(code)
    forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.ClassDef,
                 ast.Global, ast.Nonlocal, ast.Lambda, ast.Try, ast.Raise)
    calls = {"print", "sum", "range", "zip", "enumerate", "len", "list", "str", "int"}
    calls.update(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError("Reference fixture contains a forbidden construct")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder references are not accepted")
        if isinstance(node, ast.Attribute) and node.attr not in {"append", "split", "join", "copy"}:
            raise ValueError("Unexpected attribute in reference fixture")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in calls:
            raise ValueError("Unexpected function in reference fixture")
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                          capture_output=True, text=True, encoding="utf8", timeout=3)
    if proc.returncode:
        raise ValueError(f"Reference execution failed: {proc.stderr}")
    return proc.stdout.strip()


def verify_suite(suite):
    assert suite["case_count"] == len(suite["cases"])
    assert suite["turn_count"] == sum(len(c["turns"]) for c in suite["cases"])
    seen, references = set(), []
    for case in suite["cases"]:
        assert case["case_id"] not in seen
        seen.add(case["case_id"])
        assert case["expected_before_run"]["completion"] in {"complete", "partial"}
        assert case["required_qualitative_criteria"]
        for turn in case["turns"]:
            assert turn["turn_id"] not in seen
            seen.add(turn["turn_id"])
            inp, ref = turn["inputs"], turn["reference"]
            assert set(inp) <= {"student_id", "topic", "code", "student_answer", "request", "background", "followup_code"}
            assert {"student_id", "topic", "code", "student_answer", "request"} <= set(inp)
            main = execute_reference(inp["code"])
            followup = execute_reference(ref["followup_code"])
            assert main == ref["main_answer"], turn["turn_id"]
            assert followup == ref["followup_answer"], turn["turn_id"]
            status = "unanswered" if inp["student_answer"] is None else (
                "correct" if inp["student_answer"] == main else "incorrect")
            assert status == ref["student_status"]
            if "followup_code" in inp:
                assert execute_reference(inp["followup_code"]) == ref["followup_answer"]
            references.append({"fixture_turn_id": turn["turn_id"], "main_answer": main,
                               "reference_followup_answer": followup, "student_status": status})
    return references


def prepare(cases_path, out, max_new_tokens, threads):
    suite = read(cases_path)
    references = verify_suite(suite)
    design = {"suite_sha256": sha(cases_path), "case_count": suite["case_count"],
              "turn_count": suite["turn_count"], "generation": {"max_new_tokens": max_new_tokens,
              "do_sample": False, "threads": threads, "all_roles_same_limit": True},
              "reference_python": sys.version, "references": references}
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        assert plan["design"] == design, "Frozen protocol differs; use a new output directory"
        assert sha(out / "cases.frozen.json") == sha(cases_path), "Frozen cases differ"
    else:
        if any(out.iterdir()):
            raise ValueError("Only an empty output directory may receive a new plan")
        (out / "cases.frozen.json").write_bytes(Path(cases_path).read_bytes())
        plan = {"frozen_at_utc": now(), "design": design}
        write(plan_path, plan)
    return suite, plan


def _snapshot(memory, side):
    value = memory.get(side, {})
    return value.get("snapshot", value) if isinstance(value, dict) else {}


def _topic(snapshot, topic):
    # The store has one bucket per topic AND structural difficulty; the fixture
    # counter checks concern the student's topic totals, not the first bucket.
    rows = [x for x in snapshot.get("topics", []) if x.get("topic") == topic]
    return {key: sum(x.get(key, 0) for x in rows)
            for key in ("turns", "attempts", "correct", "incorrect", "unanswered", "unverified")}


def assess_turn(case, turn, trace, previous_turn_ids):
    checks = []

    def add(name, passed, observed, expected):
        checks.append({"check": name, "passed": passed, "observed": observed, "expected": expected})

    tool = trace.get("tool", {})
    main = tool.get("main", {})
    ref = turn["reference"]
    add("main_execution_matches_independent_reference", main.get("ok") is True and main.get("output") == ref["main_answer"],
        {"ok": main.get("ok"), "output": main.get("output")}, ref["main_answer"])
    ev = trace.get("student_evidence", {})
    add("student_evidence_status", ev.get("status") == ref["student_status"], ev.get("status"), ref["student_status"])
    add("final_response_present", bool(trace.get("final", {}).get("text", "").strip()),
        len(trace.get("final", {}).get("text", "")), "nonempty")
    add("three_role_traces_present", all(k in trace.get("agents", {}) for k in ("planner", "teacher", "reviewer")),
        list(trace.get("agents", {})), ["planner", "teacher", "reviewer"])
    followup = tool.get("followup", {}) or {}
    add("followup_execution_verified_by_system", followup.get("ok") is True,
        {"ok": followup.get("ok"), "output": followup.get("output")}, "execution succeeds; relevance separately reviewed")
    memory = trace.get("memory", {})
    before, after = _snapshot(memory, "before"), _snapshot(memory, "after")
    topic = turn["inputs"]["topic"]
    b, a = _topic(before, topic), _topic(after, topic)
    retrieved = memory.get("retrieved", [])
    for spec in case.get("memory_checks", []):
        if spec["turn_id"] != turn["turn_id"]:
            continue
        kind = spec["kind"]
        if kind in {"student_count_delta", "before_topic_count"}:
            for key in ("attempts", "correct", "incorrect", "unanswered"):
                if key not in spec:
                    continue
                if kind == "student_count_delta":
                    value = a.get(key, 0) - b.get(key, 0)
                else:
                    value = b.get(key, 0)
                add(kind + "." + key, value == spec[key], value, spec[key])
        elif kind == "no_prior_student_memory":
            add(kind, not before.get("topics") and not retrieved, {"before": before, "retrieved": retrieved}, "no topics or retrieved episodes")
        elif kind == "retrieved_only_student":
            values = [x.get("student_id") for x in retrieved]
            add(kind, all(x == spec["student_id"] for x in values), values, spec["student_id"])
        elif kind == "retrieved_contains_turn":
            expected = previous_turn_ids.get(spec["prior_turn_id"])
            values = [x.get("turn_id") for x in retrieved]
            add(kind, expected is not None and expected in values, values, expected)
        elif kind == "recommendation":
            value = memory.get("recommendation", {}).get("action")
            add(kind, value == spec["action"], value, spec["action"])
        else:
            raise ValueError("Unknown predeclared check " + kind)
    return checks


def run(args):
    suite, plan = prepare(args.cases, args.out, args.max_new_tokens, args.threads)
    if args.prepare_only:
        print("Prepared", args.out, "cases", suite["case_count"], "turns", suite["turn_count"])
        return
    if (args.out / "results.json").exists() or (args.out / "memory.sqlite").exists():
        raise ValueError("This output already has execution state; use a new directory, never replay on changed memory")
    sys.path.insert(0, str(ROOT))
    from tutor.legacy.model_backend import LocalQwenBackend
    from tutor.legacy.system import TutoringSystem
    started = time.perf_counter()
    backend = LocalQwenBackend(model_path=args.model, adapter_path=args.adapter,
                               threads=args.threads, max_new_tokens=args.max_new_tokens)
    memory_path = args.out / "memory.sqlite"
    system = TutoringSystem(backend, memory_path)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "tutor/legacy").glob("*.py"))}
    result = {"schema_version": 1, "suite_id": suite["suite_id"], "suite_sha256": sha(args.cases),
              "plan_sha256": sha(args.out / "plan.json"), "started_at_utc": now(),
              "completed": False, "backend_requested": {"model": str(args.model) if args.model else None,
              "adapter": str(args.adapter) if args.adapter else None, "threads": args.threads,
              "max_new_tokens": args.max_new_tokens}, "source_sha256": sources, "cases": []}
    previous = {}
    try:
        for case in suite["cases"]:
            case_result = {"case_id": case["case_id"], "title": case["title"],
                           "expected_before_run": case["expected_before_run"], "turns": []}
            result["cases"].append(case_result)
            for turn in case["turns"]:
                if turn.get("restart_system_before"):
                    system.close()
                    system = TutoringSystem(backend, memory_path)
                begin = time.perf_counter()
                record = {"fixture_turn_id": turn["turn_id"], "inputs": turn["inputs"],
                          "reference": turn["reference"], "restarted_system": bool(turn.get("restart_system_before"))}
                try:
                    trace = system.run_turn(**turn["inputs"])
                    record["trace"] = trace
                    record["automatic_checks"] = assess_turn(case, turn, trace, previous)
                    previous[turn["turn_id"]] = trace["turn_id"]
                except Exception as exc:
                    record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                    record["automatic_checks"] = [{"check": "turn_completed", "passed": False, "observed": str(exc), "expected": "complete trace"}]
                record["elapsed_seconds"] = time.perf_counter() - begin
                case_result["turns"].append(record)
                write(args.out / "results.json", result)
                print(turn["turn_id"], "checks", sum(x["passed"] is True for x in record["automatic_checks"]),
                      "/", len(record["automatic_checks"]), "seconds", round(record["elapsed_seconds"], 2), flush=True)
        result["memory_episodes"] = system.memory.episodes()
        result["completed"] = True
    finally:
        system.close()
        result["finished_at_utc"] = now()
        result["elapsed_seconds_including_load"] = time.perf_counter() - started
        write(args.out / "results.json", result)
    print("Saved", args.out / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        refs = verify_suite(read(args.cases))
        print("Verified independent reference programs:", len(refs) * 2, "suite sha256:", sha(args.cases))
    else:
        run(args)


if __name__ == "__main__":
    main()
