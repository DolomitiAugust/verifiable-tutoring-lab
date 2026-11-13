"""Supplement frozen cloud outputs with safe value comparison; never overwrite raw results."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def value(text):
    text = text.strip().strip("`").strip().rstrip("。")
    if len(text) > 4096:
        raise ValueError("Value too long for conservative extraction")
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        # Unquoted program string outputs are allowed. Explanatory sentences and
        # malformed containers remain unresolved rather than falsely scored wrong.
        if any(x in text for x in "[]{}\n") or re.search(r"[。！？；]|(?:因为|所以|输出|结果|应该)", text):
            raise ValueError("Unresolved output fragment")
        parsed = text
    if not isinstance(parsed, (str, int, float, bool, list, tuple, dict, set, type(None))):
        raise ValueError("Unsupported literal")
    return parsed


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def inspect(text, expected):
    if expected is None:
        return {"status": "unresolved", "reason": "no_expected_value"}
    fragments = re.findall(r"(?m)^\s*本题输出\s*[:：]\s*([^\n]+)", text)
    source = "explicit_main_tag"
    if not fragments:
        # A whole response consisting of a literal can also answer a direct-output request.
        candidate = text.strip()
        if "\n" in candidate or len(candidate) > 512:
            return {"status": "unresolved", "reason": "no_unambiguous_anchor"}
        fragments = [candidate]
        source = "standalone_value"
    try:
        target = value(str(expected))
        candidates = [value(x) for x in fragments]
    except (ValueError, TypeError, RecursionError, MemoryError):
        return {"status": "unresolved", "reason": "parse_failure", "fragments": fragments}
    if not all(same(candidates[0], x) for x in candidates):
        return {"status": "unresolved", "reason": "conflicting_values", "fragments": fragments}
    return {"status": "correct" if same(candidates[0], target) else "wrong", "source": source,
            "fragments": fragments, "expected": str(expected),
            "note": "Value comparison permits container whitespace differences; original strict extraction is preserved."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.out.resolve() == args.input.resolve():
        raise ValueError("Choose a new supplemental output file; do not replace original evaluation.")
    data = json.loads(args.input.read_text(encoding="utf8"))
    rows = [{"candidate_id": r["candidate_id"], "original_extraction": r["main_fact_extraction"],
             "safe_value_extraction": inspect(r["raw_response"], r.get("expected_main_answer"))}
            for r in data["generation"]]
    result = {"source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "protocol": "Post-run supplemental conservative value parser; no model generation is modified or repeated. Unresolved is not wrong. This is not a human teaching-quality assessment.",
              "counts": dict(Counter(r["safe_value_extraction"]["status"] for r in rows)), "rows": rows}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
