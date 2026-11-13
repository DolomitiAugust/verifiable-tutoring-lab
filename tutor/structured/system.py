"""Auditable revised tutoring system; no model invocation is disguised as a rule.

Models explain tool-supplied facts. This is not an unaided reasoning benchmark.
All generated explanations still require qualitative review after automatic gates.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from .constraints import request_policy, constraint_spec, constrained_followup
from .engine import FinitePython, _event_explanation
from .memory import EpisodeStore
from .contracts import parse_strict_object, planner_contract, teacher_contract, reviewer_contract

ROLE_BUDGETS = {"planner": 64, "teacher": 192, "reviewer": 96}
MODES = ("multi", "single", "tool_only", "tool_no_memory", "no_memory")
ARMS = ("clean_freeform", "guarded_quotes")


def parse_object(text):
    cleaned = text.strip()
    if cleaned.startswith("```json\n") and cleaned.endswith("```"):
        cleaned = cleaned[8:-3].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError):
        return None


def teaching_facts(main):
    """Evidence cards are explicitly tool/rule-derived, never counted as LLM prose."""
    facts, seen = [], set()
    try:
        functions={node.lineno:node for node in ast.walk(ast.parse(main.get('code',''))) if isinstance(node,ast.FunctionDef)}
    except SyntaxError:
        functions={}
    def add(topic, text):
        if (topic, text) not in seen:
            seen.add((topic, text))
            facts.append({"id": "F" + str(len(facts) + 1), "topic": topic, "text": text})
    for event in main.get("events", []):
        kind = event["action"]
        if kind == "range":
            add("range", f"range{tuple(event['arguments'])} 依次取 {event['actual_values']}，终点 {event['excluded_stop']} 不包括在内。")
        elif kind == "zip":
            add("zip", f"zip 按较短列表结束，配对为 {event['pairs']}；未配对尾项 {event['unused_tails']}。")
        elif kind == "subscript":
            add("slice", f"{event['expression']} 取下标 {event['indices']}，结果 {event['result']}。")
            if not event["indices"]:
                add("slice", "切片起点到终点之间没有沿指定步长方向可取的下标，因此结果为空；省略步长时为正向步长1。")
        elif kind == "function_definition" and event.get("defaults"):
            fn=functions.get(event.get('line'))
            params=[a.arg for a in fn.args.args[-len(event['defaults']):]] if fn else ['未知参数']*len(event['defaults'])
            for param,value in zip(params,event['defaults']):
                if type(value) in (list,dict):
                    add('default',f'参数 {param} 的默认值在定义时创建为可变容器；省略该参数时，同一个默认容器被重复传入，函数内部的后续操作仍须单独检查。')
                elif value is None:
                    guard=fn.body[0] if fn and fn.body else None
                    guarded=(isinstance(guard,ast.If) and isinstance(guard.test,ast.Compare)
                        and isinstance(guard.test.left,ast.Name) and guard.test.left.id==param
                        and len(guard.test.ops)==1 and isinstance(guard.test.ops[0],ast.Is)
                        and len(guard.test.comparators)==1 and isinstance(guard.test.comparators[0],ast.Constant)
                        and guard.test.comparators[0].value is None and len(guard.body)==1 and not guard.orelse
                        and isinstance(guard.body[0],ast.Assign) and len(guard.body[0].targets)==1
                        and isinstance(guard.body[0].targets[0],ast.Name) and guard.body[0].targets[0].id==param
                        and (isinstance(guard.body[0].value,ast.List) and not guard.body[0].value.elts
                             or isinstance(guard.body[0].value,ast.Dict) and not guard.body[0].value.keys))
                    if guarded:
                        add('default',f'参数 {param} 的默认值为 None；函数入口在该参数为 None 时创建一个新容器，因此省略该参数的各次调用不会共享一个可变默认容器。')
                    else:
                        add('default',f'参数 {param} 的默认值为 None。仅凭这个默认值不能断言容器被复用；是否创建新容器须检查函数体。')
                else:
                    add('default',f'参数 {param} 在定义时保存的默认值为 {value!r}；这个值本身不是列表或字典，不能把它当作可变默认容器。')
        if event.get("aliases"):
            add("alias", "、".join(f"{a} 与 {b} 指向同一容器" for a, b in event["aliases"]) + "，原地修改会被这些引用共同观察到。")
    if "copy(" in main.get("code", "") or "[:]" in main.get("code", ""):
        add("alias", "本题列表的浅复制创建新的外层容器；原列表追加元素不会追加到该复制列表，浅复制不等于递归复制内部对象。")
    add("execution", "实际执行输出为 " + repr(main.get("output")) + "；代码中的赋值、运算与控制流按执行顺序决定此输出。")
    return facts[:8]


class TutoringSystem:
    def __init__(self, backend, memory_path, prompts_path=None, mode="multi", arm="guarded_quotes"):
        if mode not in MODES:
            raise ValueError("Unsupported mode")
        if backend is None and mode not in ("tool_only", "tool_no_memory"):
            raise ValueError("This mode requires an explicitly supplied backend")
        if arm not in ARMS:
            raise ValueError('Unknown prompt-revision arm')
        self.arm=arm
        self.backend, self.mode = backend, mode
        self.memory_enabled = mode not in ("no_memory", "tool_no_memory")
        self.memory = EpisodeStore(memory_path)
        self.verifier = FinitePython()
        self.prompts_path = Path(prompts_path or Path(__file__).with_name("prompts.guarded.json" if arm=='guarded_quotes' else 'prompts.clean.json'))
        self.prompts = json.loads(self.prompts_path.read_text(encoding="utf-8"))

    def _agent(self, role, context):
        prompt = self.prompts["shared"] + "\n" + self.prompts[role] + "\n输入数据：\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        record = {"role": role, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "accepted": False, "parsed": None, "real_model_call": False,
                  "adopted_components": [], "max_new_tokens": ROLE_BUDGETS[role]}
        try:
            generated = self.backend.generate(prompt, role=role, max_new_tokens=ROLE_BUDGETS[role])
            if not isinstance(generated, dict) or not isinstance(generated.get("raw_response"), str):
                raise ValueError("Backend must return a recorded raw_response string")
            record.update(generated)
            parser=parse_strict_object if self.arm=='guarded_quotes' else parse_object
            record.update(parsed=parser(generated["raw_response"]), call_ok=True, adoption_reason="pending")
        except Exception as error:
            record.update(raw_response="", call_ok=False, adoption_reason="backend_error", error=type(error).__name__ + ": " + str(error))
        return record

    def run_turn(self, student_id, topic, code, student_answer, request, background=None, followup_code=None):
        for name, value, maximum in (("student_id", student_id, 100), ("topic", topic, 150), ("request", request, 3000)):
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise ValueError("Invalid " + name)
        if student_answer is not None and (not isinstance(student_answer, str) or len(student_answer) > 8192):
            raise ValueError("student_answer must be a string (including empty) or None")
        if background is not None and (not isinstance(background, str) or len(background) > 2000):
            raise ValueError("Invalid background")
        inputs = dict(student_id=student_id, topic=topic, code=code, student_answer=student_answer,
                      request=request, background=background, followup_code=followup_code)
        turn_id = uuid.uuid4().hex
        before = self.memory.snapshot(student_id)
        retrieved = self.memory.retrieve(student_id, topic, 3) if self.memory_enabled else []
        main = self.verifier.run(code)
        status = ("unverified" if not main["ok"] else "unanswered" if student_answer is None else
                  "correct" if student_answer == main["output"] else "incorrect")
        evidence = {"status": status, "student_answer": student_answer, "submitted": student_answer is not None,
                    "expected_output": main["output"], "source": "student_prediction_checked_by_interpreter", "mastery_claim": False}
        current = {"turn_id": turn_id, "student_evidence": evidence, "tool": {"main": main}}
        if self.memory_enabled:
            recommendation = self.memory.recommend(student_id, topic, main["difficulty"], current)
        else:
            recommendation = {"action": "keep", "suggested_difficulty": main["difficulty"], "evidence_episode_ids": [],
                              "source": "memory_disabled", "mastery_claim": False}
        policy = request_policy(request)
        if followup_code is not None:
            policy["wants_followup"] = True
        spec = constraint_spec(request)
        followup = None
        if main["ok"] and policy["wants_followup"]:
            followup = constrained_followup(code, main["output"], self.verifier, spec,
                        harder=policy["wants_harder"] and recommendation["action"] == "increase", provided=followup_code)
        facts = teaching_facts(main) if main["ok"] else []
        history = [{"episode_id": e["turn_id"], "student_answer": e["input"]["student_answer"],
                    "status": e["student_evidence"]["status"], "verified_answer": e["tool"]["main"]["output"],
                    "request": e["input"]["request"][:180]} for e in retrieved]
        context = {"program": code, "request": request, "background": background,
                   "student_answer": student_answer, "verified_answer": main["output"], "student_status": status,
                   "facts": facts, "topics": list(dict.fromkeys(f["topic"] for f in facts)), "history": history}
        agents, draft, adopted = {}, None, False
        checks = {}
        plan = {"focus": facts[0]["topic"], "evidence_id": facts[0]["id"]} if facts else None
        if self.mode not in ("tool_only", "tool_no_memory") and main["ok"]:
            if self.mode in ("multi", "no_memory"):
                planner = self._agent("planner", context)
                parsed = planner["parsed"]
                planner["accepted"] = bool(parsed and any(parsed.get("focus") == f["topic"] and parsed.get("evidence_id") == f["id"] for f in facts))
                if self.arm=='guarded_quotes':
                    planner['accepted']=planner_contract(parsed,facts)
                planner["adoption_reason"] = "grounded_priority_forwarded" if planner["accepted"] else "invalid_focus_or_evidence"
                planner["adopted_components"] = ["explanation_priority"] if planner["accepted"] else []
                if planner["accepted"]:
                    plan = parsed
                agents["planner"] = planner
            teacher = self._agent("teacher", {**context, "plan": plan,
                    "plan_source": "llm" if agents.get("planner", {}).get("accepted") else "rule"})
            agents["teacher"] = teacher
            draft = teacher["parsed"]
            explanation = draft.get("explanation") if draft else None
            ids = draft.get("evidence_ids") if draft else None
            checks = {"main_verified": main["ok"], "answer_matches": bool(draft and draft.get("answer") == main["output"]),
                      "status_matches": bool(draft and draft.get("status") == status),
                      "explanation_present": isinstance(explanation, str) and 12 <= len(explanation.strip()) <= 500,
                      "evidence_grounded": isinstance(ids, list) and bool(ids) and all(isinstance(x, str) and x in {f["id"] for f in facts} for x in ids),
                      "no_mastery_claim": isinstance(explanation, str) and not bool(re.search(r"已经掌握|已完全掌握|已经学会|完全理解|肯定掌握", explanation)),
                      "followup_constraint_verified": not policy["wants_followup"] or bool(followup and followup["ok"] and followup["constraint_check"]["passed"])}
            if self.arm=='guarded_quotes':
                teacher['independent_contract']=teacher_contract(draft,main['output'],status,facts,plan['evidence_id'])
                checks.update({'contract_'+k:v for k,v in teacher['independent_contract'].items()})
            review_ok = self.mode == "single"
            if self.mode in ("multi", "no_memory"):
                review_context={**context, "draft": draft, "teacher_raw_response": teacher["raw_response"], "deterministic_checks": checks}
                if self.arm=='guarded_quotes':
                    # The reviewer receives raw current facts and draft, not teacher acceptance or checker answers.
                    review_context={k:context[k] for k in ('verified_answer','student_status','student_answer','facts')}
                    review_context.update(draft=draft,required_evidence_id=plan['evidence_id'])
                reviewer = self._agent("reviewer", review_context)
                decision = reviewer["parsed"]
                reviewer["accepted"] = bool(decision and type(decision.get("approved")) is bool and
                    isinstance(decision.get("reason"), str) and 4 <= len(decision["reason"].strip()) <= 300 and
                    (not decision["approved"] or all(checks.values())))
                if self.arm=='guarded_quotes':
                    reviewer['independent_contract']=reviewer_contract(decision,draft,main['output'],status,facts,plan['evidence_id'])
                    reviewer['accepted']=reviewer['independent_contract']['valid']
                reviewer["adoption_reason"] = "review_gate_used" if reviewer["accepted"] else "invalid_or_contradictory_review"
                reviewer["adopted_components"] = ["approval_decision"] if reviewer["accepted"] else []
                review_ok = bool(reviewer["accepted"] and decision["approved"])
                agents["reviewer"] = reviewer
            adopted = bool(all(checks.values()) and review_ok and not policy["hide_answer"] and policy["wants_explanation"])
            teacher["accepted"] = adopted
            teacher["adopted_components"] = (["evidence_selection"] if self.arm=='guarded_quotes' else ["explanation"]) if adopted else []
            teacher["adoption_reason"] = "facts_status_evidence_and_review_passed" if adopted else "failed_gates_or_answer_only"
            checks["model_review_required"] = self.mode != "single"
            checks["model_review_approved"] = review_ok if self.mode != "single" else None
        text = self._final_text(main, evidence, retrieved, recommendation, policy, followup, draft if adopted else None, facts,self.arm=='guarded_quotes')
        source = "tool_rejection" if not main["ok"] else "llm_explanation_with_tool_facts" if adopted else "verified_template"
        if adopted and self.arm=='guarded_quotes': source='model_selected_tool_evidence'
        trace = {"trace_version": 2, "turn_id": turn_id, "created_at": datetime.now(timezone.utc).isoformat(),
                 "student_id": student_id, "input": inputs, "mode": self.mode, "prompt_revision_arm":self.arm,
                 "backend": getattr(self.backend, "metadata", {"kind": "no_model"}), "agents": agents,
                 "prompts_sha256": hashlib.sha256(self.prompts_path.read_bytes()).hexdigest(),
                 "tool": {"main": main, "followup": followup, "fact_cards": facts}, "student_evidence": evidence,
                 "memory": {"enabled": self.memory_enabled, "before": before, "retrieved": retrieved, "recommendation": recommendation},
                 "final": {"text": text, "source": source, "llm_used": adopted, "checks": checks,
                           "request_policy": policy, "constraint_spec": spec, "followup_source": followup["origin"] if followup else None,
                           "difficulty_target_met": followup["difficulty"] == recommendation["suggested_difficulty"] if followup and followup["ok"] and policy["wants_harder"] else None,
                           "explanation_source": ("model_selected_tool_quotes" if self.arm=='guarded_quotes' else "model_raw_explanation") if adopted else "tool_rule_explanation",
                           "model_freeform_prose_used":bool(adopted and self.arm=='clean_freeform'),
                           "llm_prose_guaranteed_correct": False, "facts_provided_to_model": True,
                           "independent_model_reasoning_claim": False, "qualitative_review_required": True}}
        self.memory.save(trace)
        trace["memory"]["after"] = self.memory.snapshot(student_id)
        return trace

    @staticmethod
    def _final_text(main, evidence, retrieved, recommendation, policy, followup, draft, facts,extractive=False):
        if not main["ok"]:
            return "当前程序超出受限核验范围，未据此判断学生正误：" + main["error"]
        if policy["hide_answer"]:
            return "先检查程序的取值范围、引用关系与控制流；本轮按要求暂不公布原题及复测答案。"
        display = repr(main["output"]) if not main["output"].strip() else main["output"]
        lines = ["本题输出：" + display]
        if not policy["wants_explanation"]:
            return lines[0]
        labels = {"correct": "你的这次预测与实际输出一致。", "incorrect": "你的这次预测与实际输出不同。",
                  "unanswered": "本轮未提交预测，不计为答对或答错。", "unverified": "未能核验本轮预测。"}
        lines.append(labels[evidence["status"]])
        if retrieved:
            old = retrieved[0]
            lines.append("同一学生、同一主题的上一条记录：预测为 " + repr(old["input"]["student_answer"]) +
                         "，实际输出为 " + repr(old["tool"]["main"]["output"]) + "；当时原话：“" + old["input"]["request"][:180] + "”。")
        if draft:
            label='模型选择的工具事实（逐字引用）：' if extractive else '模型解释（基于已提供的工具事实）：'
            lines.append(label + draft['explanation'])
        else:
            lines.append("核验依据（工具与规则）：" + "\n".join(f["text"] for f in facts))
        lines.append("执行记录：\n" + _event_explanation(main))
        if policy["wants_harder"]:
            lines.append("难度建议：" + recommendation["action"] + "；依据是已核验预测的规则统计，不代表掌握度测量。")
        if policy["wants_followup"]:
            if followup and followup["ok"]:
                lines.extend(["复测题：" + followup["description"], "```python\n" + followup["code"] + "\n```",
                              "复测参考输出：" + repr(followup["output"]),
                              "复测解释（工具与规则）：" + "\n".join(f["text"] for f in teaching_facts(followup))])
            else:
                lines.append("未交付复测：" + (followup["error"] if followup else "请求无法核验。"))
        return "\n\n".join(lines)

    def close(self):
        self.memory.close()
