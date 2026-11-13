"""Small, declared request grammar and AST-level practice constraints.

This is a bounded controller, not general natural-language understanding.
Unknown explicit restrictions cause abstention instead of unconstrained mutation.
"""
from __future__ import annotations

import ast
import copy
import re

from .engine import _related_programs, make_followup


def request_policy(request):
    # Match a sentence rather than the old arbitrary twelve-character window.
    followup = bool(re.search(r"变式|复测|练习|出题|第二题|两小题|两道|quiz|再[^。！？\n]*题", request, re.I))
    if re.search(r"(?:不要|不需(?:要)?|不用|别)(?:再)?(?:出(?:题|练习)|给[^。！？\n]*(?:练习|复测|变式)|练习|复测|变式)", request):
        followup = False
    return {"wants_followup": followup,
            "wants_explanation": not bool(re.search(r"只(?:要|需|给|输出).{0,6}(?:答案|结果)|不要解释", request)),
            "hide_answer": bool(re.search(r"(?:先|暂时)?(?:别|不要|不给).{0,4}答案", request)),
            "wants_harder": bool(re.search(r"提高难度|更难|进阶|挑战", request)),
            "source": "declared_request_grammar_v2"}


def constraint_spec(request):
    if re.search(r"默认参数|默认值", request) and re.search(r"修复|修正|改", request):
        return {"kind": "mutable_default_repair", "scope": "only_default_initialization"}
    if re.search(r"右(?:侧)?列表", request) and re.search(r"补|添加|追加|加一个", request):
        return {"kind": "append_right_list", "scope": "one_new_scalar_only"}
    if re.search(r"只[^。！？\n]*(?:改|变)[^。！？\n]*终点", request):
        return {"kind": "range_stop_only", "scope": "one_range_stop"}
    if re.search(r"只[^。！？\n]*(?:改|补|添加|调整)|不得[^。！？\n]*(?:改|变)", request):
        return {"kind": "unsupported_explicit_constraint", "scope": "abstain"}
    return {"kind": "concept_related", "scope": "heuristic_not_general_semantic_proof"}


def _dump(tree):
    return ast.dump(tree, include_attributes=False)


def _right_assignment(tree):
    zip_call = next((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "zip" and len(n.args) == 2), None)
    if zip_call is None or not isinstance(zip_call.args[1], ast.Name):
        return None
    name = zip_call.args[1].id
    return next((n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
                 and isinstance(n.value, ast.List)), None)


def repaired_defaults(tree):
    changed = copy.deepcopy(tree)
    repairs = 0
    for fn in (n for n in ast.walk(changed) if isinstance(n, ast.FunctionDef)):
        positional = fn.args.posonlyargs + fn.args.args
        offset = len(positional) - len(fn.args.defaults)
        initializers = []
        for idx, default in enumerate(fn.args.defaults):
            empty = isinstance(default, ast.List) and not default.elts or isinstance(default, ast.Dict) and not default.keys
            if not empty:
                continue
            name = positional[offset + idx].arg
            fn.args.defaults[idx] = ast.Constant(None)
            initializers.append(ast.If(ast.Compare(ast.Name(name, ast.Load()), [ast.Is()], [ast.Constant(None)]),
                [ast.Assign([ast.Name(name, ast.Store())], copy.deepcopy(default))], []))
            repairs += 1
        fn.body = initializers + fn.body
    return ast.fix_missing_locations(changed), repairs


def validate_constraint(original, candidate, spec):
    try:
        before, after = ast.parse(original), ast.parse(candidate)
    except SyntaxError:
        return {"passed": False, "kind": spec["kind"], "reason": "invalid_python"}
    kind = spec["kind"]
    passed, reason = False, "constraint_not_satisfied"
    if kind == "mutable_default_repair":
        expected, repairs = repaired_defaults(before)
        passed = repairs > 0 and _dump(expected) == _dump(after)
        reason = "canonical_None_guard_repair_only"
    elif kind == "append_right_list":
        left, right = _right_assignment(before), _right_assignment(after)
        if left is not None and right is not None and len(right.value.elts) == len(left.value.elts) + 1:
            scalar = right.value.elts[-1]
            scalar_ok = isinstance(scalar, ast.Constant) and type(scalar.value) in (int, float)
            restored = copy.deepcopy(after)
            _right_assignment(restored).value.elts.pop()
            passed = scalar_ok and _dump(before) == _dump(restored)
        reason = "exact_original_AST_plus_one_right_list_scalar"
    elif kind == "range_stop_only":
        original_calls = [n for n in ast.walk(before) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range"]
        candidate_calls = [n for n in ast.walk(after) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range"]
        changes = 0
        if len(original_calls) == len(candidate_calls):
            for a, b in zip(original_calls, candidate_calls):
                idx = 0 if len(a.args) == 1 else 1
                if len(a.args) != len(b.args) or len(a.args) not in (1, 2, 3):
                    changes = -100
                    break
                if _dump(a.args[idx]) != _dump(b.args[idx]):
                    changes += 1
                    b.args[idx] = copy.deepcopy(a.args[idx])
            passed = changes == 1 and _dump(before) == _dump(after)
        reason = "exact_original_AST_except_one_range_stop"
    elif kind == "concept_related":
        passed = _dump(before) != _dump(after) and _related_programs(original, candidate)
        reason = "same_structure_or_operation_anchors_only"
    return {"passed": bool(passed), "kind": kind, "reason": reason,
            "general_semantic_proof": False}


def constrained_followup(code, output, verifier, spec, harder=False, provided=None):
    candidates = []
    tree = ast.parse(code)
    kind = spec["kind"]
    if provided is not None:
        candidates.append((provided, "provided_program", "对照给定复测程序，预测输出并解释变化。"))
    elif kind == "mutable_default_repair":
        changed, repairs = repaired_defaults(tree)
        if repairs:
            candidates.append((ast.unparse(changed), "rule_default_repair", "把可变默认值改为 None，并在每次调用内部创建新容器，其余语句不变。"))
    elif kind == "append_right_list":
        target = _right_assignment(tree)
        if target is not None:
            target.value.elts.append(ast.Constant(1))
            candidates.append((ast.unparse(ast.fix_missing_locations(tree)), "rule_append_right", "只在 zip 的右列表末尾追加整数 1，其余程序不变。"))
    elif kind == "range_stop_only":
        for idx, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) in (1, 2, 3):
                stop_idx = 0 if len(node.args) == 1 else 1
                if isinstance(node.args[stop_idx], ast.Constant) and type(node.args[stop_idx].value) is int:
                    for delta in (1, 2, 3, -1):
                        changed = copy.deepcopy(tree)
                        list(ast.walk(changed))[idx].args[stop_idx].value += delta
                        candidates.append((ast.unparse(changed), "rule_range_stop", "只调整一个 range 的终点，其他程序结构保持不变。"))
                    break
    elif kind == "concept_related":
        proposal = make_followup(code, output, verifier, harder=harder)
        if proposal["ok"]:
            candidates.append((proposal["code"], proposal["origin"], proposal["description"]))
    attempts = []
    for candidate, origin, description in candidates:
        constraint = validate_constraint(code, candidate, spec)
        checked = verifier.run(candidate)
        attempts.append({"code": candidate, "constraint": constraint, "execution_ok": checked["ok"], "output": checked["output"]})
        if constraint["passed"] and checked["ok"] and checked["output"] != output:
            return {**checked, "origin": origin, "description": description,
                    "constraint_check": constraint, "candidate_attempts": attempts}
    return {"ok": False, "output": None, "code": None, "difficulty": None,
            "origin": "constraint_abstention", "description": None, "constraint_check": {"passed": False, "kind": kind},
            "candidate_attempts": attempts, "error": "无法在已声明约束内产生经核验的变式，未交付不合约束的替代题。"}
