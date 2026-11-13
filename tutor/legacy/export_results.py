"""Export all tutor traces without inventing qualitative scores or surprise labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

try:
    from .run_cases import DEFAULT_OUT, read, sha, write
except ImportError:
    from run_cases import DEFAULT_OUT, read, sha, write


def fenced(text, language="text"):
    longest = max((len(x) for x in re.findall(r"~+", text)), default=0)
    fence = "~" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}\n"


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def quote_field(turns, quote):
    turn = next(t for t in turns if t["fixture_turn_id"] == quote["turn_id"])
    value = turn.get("trace", {})
    for key in quote["field"].split("."):
        value = value[key]
    if not isinstance(value, str) or quote["text"] not in value:
        raise ValueError("Review quotation does not occur in its cited trace field")


def export(directory, review_path=None):
    directory = Path(directory)
    result = read(directory / "results.json")
    suite = read(directory / "cases.frozen.json")
    plan = read(directory / "plan.json")
    result_hash = sha(directory / "results.json")
    assert result["completed"], "A partial run cannot be exported as the complete suite"
    assert result["suite_sha256"] == sha(directory / "cases.frozen.json") == plan["design"]["suite_sha256"]
    assert result["plan_sha256"] == sha(directory / "plan.json")
    assert [c["case_id"] for c in result["cases"]] == [c["case_id"] for c in suite["cases"]]
    assert sum(len(c["turns"]) for c in result["cases"]) == suite["turn_count"]
    review = read(review_path) if review_path else None
    if review is not None:
        assert review["results_sha256"] == result_hash, "Qualitative review refers to another run"
        assert review.get("reviewer"), "The reviewer must be identified; never imply a teacher reviewed it"
    review_map = {x["case_id"]: x for x in review["cases"]} if review else {}
    assert len(review_map) == len(review["cases"]) if review else True
    assert set(review_map) <= {c["case_id"] for c in suite["cases"]}, "Unknown review case ID"
    test_only = bool(result.get("test_only"))
    model_calls = [a.get("real_model_call") is True for c in result["cases"] for t in c["turns"]
                   for a in t.get("trace", {}).get("agents", {}).values()]
    summary = {"suite_id": suite["suite_id"], "results_sha256": result_hash,
               "plan_sha256": result["plan_sha256"], "suite_sha256": result["suite_sha256"],
               "case_count": suite["case_count"], "turn_count": suite["turn_count"],
               "test_only": test_only, "recorded_real_model_calls": sum(model_calls),
               "reviewer": review.get("reviewer") if review else None, "cases": [],
               "final_source_counts": {}, "llm_final_used_count": 0,
               "role_adoption_counts": {}, "automatic_checks": {"passed": 0, "total": 0},
               "interpretation": "Automatic checks cover execution, bookkeeping and trace presence. Teaching completeness requires the predeclared qualitative review. A correct fallback is attributed to rules, not to the language model."}
    sources, adopted = Counter(), {k: Counter() for k in ("planner", "teacher", "reviewer")}
    template = {"results_sha256": result_hash, "reviewer": "", "review_date": "", "cases": []}
    lines = ["# tutor 全部案例与原始角色记录\n",
             ("**这是接口测试记录，不是被研究模型的实测结果。**\n" if test_only else ""),
             "本文件包含固定测试集的全部运行。预期在运行前冻结；参考程序由独立 Python 执行核对。参考变式只作为一个合法答案示例，系统给出其他满足请求且正确的变式也可以通过。\n",
             f"案例 {suite['case_count']} 个，轮次 {suite['turn_count']} 次。运行结果 SHA-256：`{result_hash}`。\n",
             "生成文本与角色提示在代码块中原样展示，均是待分析记录，不构成对读者或工具的新指令。\n",
             "## 汇总\n", "<!-- SUMMARY_TABLE -->\n"]
    for spec, case in zip(suite["cases"], result["cases"]):
        assert [t["fixture_turn_id"] for t in case["turns"]] == [t["turn_id"] for t in spec["turns"]]
        checks = [a for t in case["turns"] for a in t["automatic_checks"]]
        assert checks
        auto_ok = all(x["passed"] is True for x in checks)
        summary["automatic_checks"]["passed"] += sum(x["passed"] is True for x in checks)
        summary["automatic_checks"]["total"] += len(checks)
        criteria = {x["id"]: x for x in spec["required_qualitative_criteria"]}
        rev = review_map.get(spec["case_id"])
        scored = {x["id"]: x for x in rev.get("criteria", [])} if rev else {}
        if rev:
            assert set(scored) == set(criteria), "Every predeclared criterion must be reviewed"
            for item in scored.values():
                assert type(item["score"]) is int and item["score"] in (0, 1, 2)
                assert item.get("reason"), "Every qualitative score needs a stated reason"
                for quote in item.get("quotes", []):
                    quote_field(case["turns"], quote)
        actual = "unreviewed" if not rev else ("complete" if auto_ok and all(x["score"] == 2 for x in scored.values()) else "not_complete")
        prediction = spec["expected_before_run"]["completion"]
        surprise = None if actual == "unreviewed" else (
            "unexpected_success" if prediction == "partial" and actual == "complete" else
            "unexpected_failure" if prediction == "complete" and actual == "not_complete" else
            "expected_success" if actual == "complete" else "anticipated_limitation")
        row = {"case_id": spec["case_id"], "title": spec["title"], "predicted": prediction,
               "automatic_passed": sum(x["passed"] is True for x in checks), "automatic_total": len(checks),
               "all_automatic_passed": auto_ok, "qualitative": list(scored.values()) if rev else None,
               "actual_completion": actual, "expectation_comparison": surprise}
        summary["cases"].append(row)
        template["cases"].append({"case_id": spec["case_id"], "criteria": [
            {"id": k, "criterion": x["criterion"], "score": None, "reason": "", "quotes": []}
            for k, x in criteria.items()]})
        lines.extend([f"## {spec['case_id']} · {spec['title']}\n",
                      f"运行前预期：**{prediction}**。{spec['expected_before_run']['rationale']}\n",
                      f"自动检查：{row['automatic_passed']}/{len(checks)}。教学完整性：**{actual}**。\n",
                      "### 预登记定性标准\n"])
        for key, criterion in criteria.items():
            lines.append(f"- `{key}`：{criterion['criterion']}\n")
        if rev:
            lines.extend(["\n### 实际定性审阅\n", f"评审者：{review['reviewer']}。不是自动知识正确率。\n", fenced(pretty(rev), "json")])
        for turn in case["turns"]:
            inp, ref = turn["inputs"], turn["reference"]
            trace = turn.get("trace", {})
            final = trace.get("final", {})
            sources[final.get("source", "turn_error")] += 1
            summary["llm_final_used_count"] += bool(final.get("llm_used", False))
            for role in adopted:
                agent = trace.get("agents", {}).get(role, {})
                adopted[role]["accepted" if agent.get("accepted") is True else "not_accepted"] += 1
            lines.extend([f"### {turn['fixture_turn_id']}\n",
                          f"学生：`{inp['student_id']}`；主题：{inp['topic']}；执行前重建系统：{turn['restarted_system']}。\n",
                          "**任务与完整输入**\n", fenced(pretty(inp), "json"),
                          "**独立参考**\n", f"原程序输出：`{ref['main_answer']}`；学生预测状态：`{ref['student_status']}`。\n",
                          ref["explanation"] + "\n", "一种满足目标的参考变式：\n",
                          fenced(ref["followup_code"], "python"), f"参考变式输出：`{ref['followup_answer']}`。该程序与答案没有作为隐藏评分答案传入系统；若任务公开给了变式程序，则仅该公开程序属于输入。\n",
                          "**系统实际最终回答**\n",
                          f"来源：`{final.get('source', 'turn_error')}`；LLM 文本用于最终回答：`{final.get('llm_used', False)}`。\n",
                          fenced(final.get("text", "")), "**自动检查与记忆证据**\n",
                          fenced(pretty(turn["automatic_checks"]), "json"),
                          fenced(pretty(trace.get("memory", {})), "json"),
                          "**完整角色记录与工具结果**\n",
                          "下列对象保存全部角色提示、未经改写的原始回复、解析结果、采纳决定、执行结果和生成元数据。\n",
                          fenced(pretty(trace if trace else turn.get("error", {})), "json")])
    summary["final_source_counts"] = dict(sources)
    summary["role_adoption_counts"] = {k: dict(v) for k, v in adopted.items()}
    successes = [x["case_id"] for x in summary["cases"] if x["expectation_comparison"] == "unexpected_success"]
    failures = [x["case_id"] for x in summary["cases"] if x["expectation_comparison"] == "unexpected_failure"]
    summary["showcase"] = {"selection": "first two in frozen case order, no post-hoc relabelling",
                           "unexpected_success": successes[:2], "unexpected_failure": failures[:2],
                           "meets_two_plus_two": len(successes) >= 2 and len(failures) >= 2,
                           "missing_success_cases": max(0, 2-len(successes)), "missing_failure_cases": max(0, 2-len(failures))}
    table = ["| 案例 | 运行前预期 | 自动检查 | 教学完整性 | 与预期比较 |", "|---|---|---:|---|---|"]
    table += [f"| {x['case_id']} | {x['predicted']} | {x['automatic_passed']}/{x['automatic_total']} | {x['actual_completion']} | {x['expectation_comparison'] or '待审阅'} |" for x in summary["cases"]]
    table += ["", "四个展示案例：", fenced(pretty(summary["showcase"]), "json"),
              "`unreviewed` 不是失败。两类意外案例不足时保留缺口，不以预期内成功或已预期的困难替代。", "",
              "来源归因：", fenced(pretty({"final_sources": dict(sources), "llm_final_used_count": summary["llm_final_used_count"], "roles": summary["role_adoption_counts"]}), "json")]
    text = "\n".join(lines).replace("<!-- SUMMARY_TABLE -->", "\n".join(table))
    (directory / "comparison.md").write_text(text, encoding="utf8")
    write(directory / "metrics.json", summary)
    if not (directory / "review.template.json").exists():
        write(directory / "review.template.json", template)
    print("Exported", directory / "comparison.md", "and metrics.json; qualitative reviewer:", summary["reviewer"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    export(args.directory, args.review)


if __name__ == "__main__":
    main()
