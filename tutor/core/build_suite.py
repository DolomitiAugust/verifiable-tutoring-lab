"""Assemble a regression suite, never relabel historical unexpected outcomes."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIRED_CASES = ["tutor-C01", "tutor-C02", "tutor-C03", "tutor-C07", "tutor-C08", "tutor-S02"]


def main():
    target = HERE / "cases.v2.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a regression specification")
    cases, origins = [], {}
    for name in ("cases.json", "cases.supplement.json"):
        path = ROOT / "tutor" / name
        origins[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        original = json.loads(path.read_text(encoding="utf-8"))
        for item in original["cases"]:
            case = copy.deepcopy(item)
            case["historical_expected_before_run"] = case.pop("expected_before_run")
            case["expected_before_run"] = {"completion": "complete", "rationale": "Known-input regression target after engineering fixes, not a new surprise claim."}
            cases.append(case)
    suite = {"schema_version": 2, "suite_id": "tutor-v2-known-case-regression", "source_sha256": origins,
             "case_count": len(cases), "turn_count": sum(len(c["turns"]) for c in cases),
             "interpretation": "All inputs were known during repair. No new unexpected-success/failure claims.",
             "paired_case_ids": PAIRED_CASES,
             "evaluation": {"automatic": ["main output against independent CPython", "student submission/status", "constraint-compliant executable practice",
                 "cross-session recall and isolation", "memory counters"],
                 "qualitative_scale": {"0": "missing or incorrect", "1": "partial", "2": "complete and correct"},
                 "complete_success": "All necessary automatic checks AND every predeclared qualitative item equal 2. Unreviewed is not success.",
                 "llm_metric": "Literal model explanation adopted after gates; planning labels and tool facts do not count as LLM teaching success.",
                 "paired_comparison": "Compare multi and single on the six prespecified cases, not unequal full-suite totals.",
                 "equivalent_followup": "Output need not equal old reference if independent execution and the request's AST constraint pass.",
                 "counterfactual_memory": "tool_no_memory intentionally cannot recall prior history; do not call missing recall an implementation error.",
                 "business_claim": "No student learning benefit or general failure rate inferred."}, "cases": cases}
    target.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target, "cases", len(cases), "turns", suite["turn_count"])


if __name__ == "__main__":
    main()
