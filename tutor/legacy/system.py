"""Three real role calls, bounded program facts, and evidence-only student memory.

Model-produced programs go through the same finite AST interpreter as user programs.
No eval/exec, imports, I/O, arbitrary attributes, or arbitrary Python calls are used.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
from pathlib import Path
import re
import time
import uuid

from .memory import EpisodeStore, LEVELS


class ProgramRejected(ValueError):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Function:
    def __init__(self, node, defaults, globals_):
        self.node, self.defaults, self.globals = node, defaults, globals_
        self.locals = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.locals.update(p.arg for p in node.args.posonlyargs + node.args.args)


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
            ast.Is: operator.is_, ast.IsNot: operator.is_not}
_ALLOWED = (ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.Name, ast.Constant,
            ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp,
            ast.BoolOp, ast.Compare, ast.If, ast.For, ast.While, ast.Break, ast.Continue,
            ast.FunctionDef, ast.Return, ast.Pass, ast.Call, ast.Attribute, ast.arguments,
            ast.arg, ast.keyword, ast.Load, ast.Store, ast.And, ast.Or, ast.Not, ast.USub,
            ast.UAdd, *tuple(_BINARY), *tuple(_COMPARE))
_BUILTINS = {"range", "len", "sum", "min", "max", "abs", "enumerate", "zip",
             "list", "tuple", "sorted", "str", "int", "print"}
_METHODS = {"append", "pop", "copy", "split", "join", "count", "index", "get"}


def program_difficulty(tree):
    """Declared structural label, not an empirical measure of learner difficulty."""
    compounds = (ast.For, ast.While, ast.If, ast.FunctionDef)
    def nesting(node, depth=0):
        current = depth + int(isinstance(node, compounds))
        return max([current] + [nesting(child, current) for child in ast.iter_child_nodes(node)])
    if nesting(tree) >= 2 or any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)):
        return "挑战"
    if any(isinstance(n, (ast.For, ast.While, ast.List, ast.Dict, ast.Attribute)) or
           isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in
           {"range", "zip", "enumerate", "sum", "sorted"} for n in ast.walk(tree)):
        return "进阶"
    return "基础"


class FinitePython:
    def __init__(self, max_steps=5000, timeout_seconds=0.5, max_events=160):
        self.max_steps, self.timeout_seconds, self.max_events = max_steps, timeout_seconds, max_events

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps or time.monotonic() > self.deadline:
            raise ProgramRejected("程序超过操作步数或时间上限。")

    def _check(self, value, seen=None, depth=0, budget=None):
        if seen is None:
            seen, budget = set(), [1500]
        budget[0] -= 1
        if budget[0] < 0 or depth > 12:
            raise ProgramRejected("容器嵌套或总数据量超过上限。")
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if abs(value) > 10**12 or isinstance(value, float) and not math.isfinite(value):
                raise ProgramRejected("数值超过上限。")
        elif isinstance(value, str):
            if len(value) > 4096:
                raise ProgramRejected("字符串过长。")
        elif isinstance(value, (list, tuple, dict, range)):
            if len(value) > 256 or id(value) in seen:
                raise ProgramRejected("容器过长或出现循环引用。")
            seen.add(id(value))
            if isinstance(value, dict) and any(not isinstance(k, (str, int, float, bool, type(None))) for k in value):
                raise ProgramRejected("字典键仅支持简单标量。")
            children = list(value.items()) if isinstance(value, dict) else value
            for child in children:
                self._check(child, seen, depth + 1, budget)
            seen.remove(id(value))
        else:
            raise ProgramRejected("不支持这种运行时对象。")
        return value

    def _view(self, env):
        return {key: ("<函数 " + key + ">" if isinstance(value, _Function) else repr(value))
                for key, value in env.items()}

    def _event(self, node, action, env, **extra):
        aliases = []
        names = [k for k, v in env.items() if isinstance(v, (list, dict))]
        for i, left in enumerate(names):
            aliases.extend([left, right] for right in names[i+1:] if env[left] is env[right])
        event = {"line": getattr(node, "lineno", None), "action": action,
                 "scope_depth": len(self.frames) - 1, "values": self._view(env),
                 "aliases": aliases, **extra}
        self.event_count += 1
        if len(self.events) < self.max_events:
            self.events.append(event)
        else:
            # Keep the beginning and the most recent event; disclose omitted history.
            self.events[-1] = event

    def _lookup(self, name, env):
        if name in env:
            return env[name]
        if self.local_names and name in self.local_names[-1]:
            raise ProgramRejected("局部变量在赋值前被读取：" + name)
        if name in self.globals:
            return self.globals[name]
        raise ProgramRejected("未定义变量：" + name)

    def _assign(self, target, value, env):
        self._check(value)
        if isinstance(target, ast.Name):
            if target.id.startswith("_") or target.id in _BUILTINS:
                raise ProgramRejected("不能覆盖保留名称。")
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (list, tuple, range)) or len(target.elts) != len(value):
                raise ProgramRejected("解包长度不一致。")
            for item, part in zip(target.elts, value):
                self._assign(item, part, env)
        elif isinstance(target, ast.Subscript):
            container, key = self._expr(target.value, env), self._expr(target.slice, env)
            if not isinstance(container, (list, dict)):
                raise ProgramRejected("只允许修改列表或字典元素。")
            container[key] = value
            self._check(container)
        else:
            raise ProgramRejected("不支持此赋值目标。")

    def _binary(self, kind, left, right):
        if kind is ast.Mult:
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (str, list, tuple)) and isinstance(count, int):
                    if len(seq) * max(count, 0) > (4096 if isinstance(seq, str) else 256):
                        raise ProgramRejected("重复运算会产生过大的对象。")
        if kind is ast.Pow and (not isinstance(right, int) or not 0 <= right <= 8):
            raise ProgramRejected("幂指数仅允许 0..8 的整数。")
        if kind is ast.Mod and isinstance(left, str):
            raise ProgramRejected("不支持字符串格式化运算。")
        return self._check(_BINARY[kind](left, right))

    def _call(self, node, env):
        args = [self._check(self._expr(x, env)) for x in node.args]
        kwargs = {x.arg: self._check(self._expr(x.value, env)) for x in node.keywords}
        if len(kwargs) != len(node.keywords) or None in kwargs:
            raise ProgramRejected("不支持重复关键字或展开参数。")
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "print":
                if set(kwargs) - {"sep", "end"}:
                    raise ProgramRejected("不支持 print 的该参数。")
                sep, end = kwargs.get("sep", " "), kwargs.get("end", "\n")
                if not isinstance(sep, str) or not isinstance(end, str):
                    raise ProgramRejected("print 分隔符必须为字符串。")
                text = sep.join(str(self._check(x)) for x in args) + end
                if sum(map(len, self.output)) + len(text) > 8192:
                    raise ProgramRejected("输出超过上限。")
                self.output.append(text)
                self._event(node, "print", env, printed=text)
                return None
            if name in _BUILTINS:
                functions = {"len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                             "range": range, "list": list, "tuple": tuple, "sorted": sorted,
                             "str": str, "int": int,
                             "enumerate": lambda *a, **k: list(enumerate(*a, **k)),
                             "zip": lambda *a, **k: list(zip(*a, **k))}
                if name in ("list", "tuple", "sorted", "enumerate", "zip"):
                    for value in args:
                        if isinstance(value, range) and len(value) > 256:
                            raise ProgramRejected("迭代长度过大。")
                result = self._check(functions[name](*args, **kwargs))
                if name == "range":
                    self._event(node, "range", env, arguments=args, actual_values=list(result),
                                excluded_stop=result.stop)
                elif name == "zip":
                    self._event(node, "zip", env, input_lengths=[len(a) for a in args],
                                pairs=copy.deepcopy(result),
                                unused_tails=[list(a[len(result):]) for a in args])
                return result
            fn = self._lookup(name, env)
            if not isinstance(fn, _Function):
                raise ProgramRejected("只可调用白名单函数或本程序定义的函数。")
            if len(self.frames) >= 16:
                raise ProgramRejected("函数调用层数过深。")
            params = [p.arg for p in fn.node.args.posonlyargs + fn.node.args.args]
            if len(args) > len(params) or set(kwargs) - set(params) or set(params[:len(args)]) & set(kwargs):
                raise ProgramRejected("函数参数不匹配。")
            local = dict(zip(params[-len(fn.defaults):], fn.defaults)) if fn.defaults else {}
            local.update(zip(params, args)); local.update(kwargs)
            if set(local) != set(params):
                raise ProgramRejected("缺少函数参数。")
            self.frames.append(local)
            self.local_names.append(fn.locals)
            self._event(node, "function_enter", local, function=name)
            try:
                self._block(fn.node.body, local)
                value = None
            except _Return as returned:
                value = returned.value
            finally:
                self.frames.pop()
                self.local_names.pop()
            return self._check(value)
        if isinstance(node.func, ast.Attribute):
            owner, method = self._expr(node.func.value, env), node.func.attr
            allowed = {list: {"append", "pop", "copy", "count", "index"},
                       str: {"split", "join", "count", "index"}, dict: {"get", "copy", "pop"}}
            if method not in allowed.get(type(owner), set()):
                raise ProgramRejected("该对象不支持白名单方法：" + method)
            # The receiver has a known exact built-in type; no arbitrary attributes are resolved.
            receiver_before = repr(owner)
            result = getattr(owner, method)(*args, **kwargs)
            self._check(owner)
            self._event(node, "method_" + method, env, receiver_before=receiver_before,
                        arguments=[repr(a) for a in args], result=repr(result))
            return self._check(result)
        raise ProgramRejected("不支持此调用形式。")

    def _expr(self, node, env):
        self._tick()
        if isinstance(node, ast.Constant):
            return self._check(node.value)
        if isinstance(node, ast.Name):
            return self._lookup(node.id, env)
        if isinstance(node, ast.List):
            return self._check([self._expr(x, env) for x in node.elts])
        if isinstance(node, ast.Tuple):
            return self._check(tuple(self._expr(x, env) for x in node.elts))
        if isinstance(node, ast.Dict):
            return self._check({self._expr(k, env): self._expr(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript):
            container, index = self._expr(node.value, env), self._expr(node.slice, env)
            result = self._check(container[index])
            indices = list(range(len(container)))[index] if isinstance(index, slice) and isinstance(container, (str, list, tuple)) else index
            self._event(node, "subscript", env, indices=indices, result=repr(result), expression=ast.unparse(node))
            return result
        if isinstance(node, ast.Slice):
            return slice(*(self._expr(x, env) if x is not None else None for x in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.BinOp):
            return self._binary(type(node.op), self._expr(node.left, env), self._expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self._expr(node.operand, env)
            return self._check({ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](value))
        if isinstance(node, ast.BoolOp):
            value = self._expr(node.values[0], env)
            for next_ in node.values[1:]:
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    break
                value = self._expr(next_, env)
            return value
        if isinstance(node, ast.Compare):
            left = self._expr(node.left, env)
            for op, right_node in zip(node.ops, node.comparators):
                right = self._expr(right_node, env)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            return self._call(node, env)
        raise ProgramRejected("不支持的表达式：" + type(node).__name__)

    def _block(self, statements, env):
        for node in statements:
            self._tick()
            if isinstance(node, ast.Assign):
                value = self._expr(node.value, env)
                for target in node.targets:
                    self._assign(target, value, env)
                self._event(node, "assign", env)
            elif isinstance(node, ast.AugAssign):
                old, right = self._expr(node.target, env), self._expr(node.value, env)
                # Preserve Python's in-place list += semantics and aliases.
                if isinstance(node.op, ast.Add) and isinstance(old, list):
                    if not isinstance(right, (list, tuple)) or len(old) + len(right) > 256:
                        raise ProgramRejected("列表增量过大或类型不合法。")
                    old.extend(right); self._check(old); value = old
                elif isinstance(node.op, ast.Mult) and isinstance(old, list):
                    if not isinstance(right, int) or len(old) * max(0, right) > 256:
                        raise ProgramRejected("列表重复过大或类型不合法。")
                    old[:] = old * right; self._check(old); value = old
                else:
                    value = self._binary(type(node.op), old, right)
                self._assign(node.target, value, env)
                self._event(node, "update", env)
            elif isinstance(node, ast.Expr):
                self._expr(node.value, env)
            elif isinstance(node, ast.If):
                selected = bool(self._expr(node.test, env))
                self._event(node, "condition", env, condition=selected)
                self._block(node.body if selected else node.orelse, env)
            elif isinstance(node, (ast.For, ast.While)):
                broken = False
                if isinstance(node, ast.For):
                    values = self._expr(node.iter, env)
                    if not isinstance(values, (list, tuple, str, range, dict)):
                        raise ProgramRejected("不支持的循环对象。")
                    iterator = iter(values)
                while True:
                    self._tick()
                    if isinstance(node, ast.For):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                        self._assign(node.target, item, env)
                    elif not self._expr(node.test, env):
                        break
                    self._event(node, "loop", env)
                    try:
                        self._block(node.body, env)
                    except _Break:
                        broken = True; break
                    except _Continue:
                        continue
                if not broken:
                    self._block(node.orelse, env)
            elif isinstance(node, ast.FunctionDef):
                if (node.decorator_list or node.args.vararg or node.args.kwarg or node.args.kwonlyargs
                        or node.args.posonlyargs or node.returns or any(a.annotation for a in node.args.args)):
                    raise ProgramRejected("仅支持普通位置参数和默认参数函数。")
                if node.name.startswith("_") or node.name in _BUILTINS:
                    raise ProgramRejected("保留函数名称。")
                if len(self.frames) > 1:
                    raise ProgramRejected("当前子集不支持嵌套函数。")
                defaults = [self._expr(x, env) for x in node.args.defaults]
                env[node.name] = _Function(node, defaults, self.globals)
                self._event(node, "function_definition", env, defaults=copy.deepcopy(defaults))
            elif isinstance(node, ast.Return):
                value = self._expr(node.value, env) if node.value is not None else None
                self._event(node, "return", env, returned=copy.deepcopy(value))
                raise _Return(value)
            elif isinstance(node, ast.Break):
                raise _Break()
            elif isinstance(node, ast.Continue):
                raise _Continue()
            elif not isinstance(node, ast.Pass):
                raise ProgramRejected("不支持的语句：" + type(node).__name__)

    def run(self, code):
        self.steps, self.events, self.output, self.event_count = 0, [], [], 0
        self.globals = {}; self.frames = [self.globals]
        self.local_names = []
        self.deadline = time.monotonic() + self.timeout_seconds
        difficulty = "基础"
        try:
            if not isinstance(code, str) or not code.strip() or len(code) > 5000:
                raise ProgramRejected("程序为空或超过 5000 字符。")
            tree = ast.parse(code)
            nodes = list(ast.walk(tree))
            if len(nodes) > 600:
                raise ProgramRejected("程序语法树过大。")
            difficulty = program_difficulty(tree)
            for node in nodes:
                if not isinstance(node, _ALLOWED):
                    raise ProgramRejected("不支持的语法：" + type(node).__name__)
                if isinstance(node, ast.Name) and node.id.startswith("_"):
                    raise ProgramRejected("禁止访问保留名称。")
                if isinstance(node, ast.Attribute) and node.attr not in _METHODS:
                    raise ProgramRejected("禁止访问属性：" + node.attr)
                if isinstance(node, ast.Constant):
                    self._check(node.value)
                if isinstance(node, ast.Dict) and any(k is None for k in node.keys):
                    raise ProgramRejected("禁止展开字典。")
            self._block(tree.body, self.globals)
            if not self.output:
                raise ProgramRejected("程序没有打印输出。")
            return {"ok": True, "output": "".join(self.output).strip(), "difficulty": difficulty,
                    "steps": self.steps, "events": self.events, "event_count": self.event_count,
                    "events_truncated": self.event_count > self.max_events, "error": None,
                    "source": "bounded_ast_interpreter", "code": code}
        except (Exception, RecursionError) as error:
            return {"ok": False, "output": None, "difficulty": difficulty, "steps": self.steps,
                    "events": self.events, "event_count": self.event_count,
                    "events_truncated": self.event_count > self.max_events,
                    "error": type(error).__name__ + ": " + str(error),
                    "source": "bounded_ast_interpreter", "code": code}


def _json_object(text):
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _request_policy(request):
    followup = bool(re.search(r"变式|复测|再.{0,12}(?:题|练习)|出题|练习题|第二题|两小题|两道|quiz", request, re.I))
    only_answer = bool(re.search(r"只(?:要|需|给|输出).{0,6}(?:答案|结果)|不要解释", request))
    hide_answer = bool(re.search(r"(?:先|暂时)?(?:别|不要|不给).{0,4}答案", request))
    return {"wants_followup": followup, "wants_explanation": not only_answer,
            "hide_answer": hide_answer,
            "wants_harder": bool(re.search(r"提高难度|更难|进阶|挑战", request)),
            "source": "explicit_request_keyword_rules"}


def _event_explanation(main):
    """Describe observed execution rather than inventing a student's misconception."""
    events = main["events"]
    meaningful = [e for e in events if e["action"] != "loop"]
    chosen = meaningful if len(meaningful) <= 12 else meaningful[:5] + meaningful[-7:]
    lines = []
    for e in chosen:
        where = "函数内" if e["scope_depth"] else "主程序"
        values = "，".join(f"{k}={v}" for k, v in e["values"].items() if not v.startswith("<函数 "))
        if e["action"] == "print":
            description = "打印 " + repr(e["printed"].rstrip("\n"))
        elif e["action"] == "return":
            description = "返回 " + repr(e["returned"])
        elif e["action"] == "condition":
            description = "条件为 " + str(e["condition"]) + "；" + values
        elif e["action"] == "function_definition":
            description = "定义函数；默认参数此时求值一次，保存为 " + repr(e["defaults"])
        elif e["action"] == "range":
            description = "range" + repr(tuple(e["arguments"])) + " 实际取值 " + repr(e["actual_values"]) + "；终点 " + str(e["excluded_stop"]) + " 不包含在内"
        elif e["action"] == "zip":
            description = "zip 的输入长度为 " + repr(e["input_lengths"]) + "，实际配对 " + repr(e["pairs"]) + "；未配对尾项 " + repr(e["unused_tails"])
        elif e["action"] == "subscript":
            description = e["expression"] + " 取下标 " + repr(e["indices"]) + "，得到 " + e["result"]
        elif e["action"].startswith("method_"):
            description = e["receiver_before"] + "." + e["action"][7:] + "(" + ", ".join(e["arguments"]) + ") 后，" + values + "；方法返回 " + e["result"]
        else:
            description = values or "执行此步"
        if e["aliases"]:
            description += "；" + "、".join(f"{a} 与 {b} 指向同一容器" for a, b in e["aliases"])
        lines.append(f"第 {e['line']} 行（{where}）：{description}。")
    if len(meaningful) > len(chosen) or main["events_truncated"]:
        lines.insert(5, "中间部分重复步骤略去；下列保留末尾的已执行步骤。")
    return "\n".join(lines)


def make_followup(code, original_output, verifier, harder=False):
    """Bounded local transformations. Each candidate is actually checked before use."""
    tree = ast.parse(code)
    candidates = []
    if harder:
        for index, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                changed = copy.deepcopy(tree)
                target = list(ast.walk(changed))[index]
                condition = ast.Compare(ast.BinOp(ast.Name(target.target.id, ast.Load()), ast.Mod(), ast.Constant(2)),
                                        [ast.Eq()], [ast.Constant(0)])
                target.body = [ast.If(condition, target.body, [])]
                candidates.append((ast.unparse(ast.fix_missing_locations(changed)), "ast_guard_fallback",
                                   "只在循环变量为偶数时执行原循环体，其余保持不变。"))
                break
        # A sum(range(...)) is also iteration, even when no explicit For exists.
        occupied = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        total_name, item_name = "practice_total", "practice_item"
        while total_name in occupied or item_name in occupied:
            total_name += "x"; item_name += "x"
        for statement_index, statement in enumerate(tree.body):
            if isinstance(statement, ast.FunctionDef):
                continue
            match = next((n for n in ast.walk(statement) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == "sum" and len(n.args) == 1
                          and not n.keywords and isinstance(n.args[0], ast.Call)
                          and isinstance(n.args[0].func, ast.Name) and n.args[0].func.id == "range"), None)
            if match is None:
                continue
            changed = copy.deepcopy(tree)
            counterpart = list(ast.walk(changed.body[statement_index]))[list(ast.walk(statement)).index(match)]
            class ReplaceSum(ast.NodeTransformer):
                def visit_Call(self, node):
                    return ast.Name(total_name, ast.Load()) if node is counterpart else self.generic_visit(node)
            changed = ReplaceSum().visit(changed)
            condition = ast.Compare(ast.BinOp(ast.Name(item_name, ast.Load()), ast.Mod(), ast.Constant(2)),
                                    [ast.Eq()], [ast.Constant(0)])
            prelude = [ast.Assign([ast.Name(total_name, ast.Store())], ast.Constant(0)),
                       ast.For(ast.Name(item_name, ast.Store()), copy.deepcopy(match.args[0]),
                               [ast.If(condition, [ast.AugAssign(ast.Name(total_name, ast.Store()), ast.Add(),
                                                                ast.Name(item_name, ast.Load()))], [])], [])]
            changed.body[statement_index:statement_index] = prelude
            candidates.append((ast.unparse(ast.fix_missing_locations(changed)), "ast_conditional_sum_fallback",
                               "保留同一个 range 的取值范围，将累加改为只累加其中的偶数。"))
            # A step of two can make an even filter vacuous. Offer an explicit
            # changed-range variant with a real odd/even decision as another candidate.
            if len(match.args[0].args) == 3:
                alternate = copy.deepcopy(changed)
                loop = alternate.body[statement_index + 1]
                loop.iter.args[2] = ast.Constant(1)
                loop.body[0].test.comparators[0] = ast.Constant(1)
                candidates.append((ast.unparse(ast.fix_missing_locations(alternate)), "ast_conditional_sum_fallback",
                                   "保持 range 的起点和终点，把步长改为 1，并增加条件：只累加奇数。"))
            break
    for index, node in enumerate(ast.walk(tree)):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        changes = [node.value + 1, node.value + 2, node.value - 1] if isinstance(node.value, int) else (
            [node.value + "x"] if isinstance(node.value, str) and len(node.value) > 1 else [])
        for value in changes:
            changed = copy.deepcopy(tree)
            list(ast.walk(changed))[index].value = value
            candidates.append((ast.unparse(ast.fix_missing_locations(changed)), "ast_constant_fallback",
                               f"只把原第 {node.lineno} 行的 {node.value!r} 改为 {value!r}，其余不变。"))
        if len(candidates) >= 30:
            break
    for candidate, source, description in candidates:
        checked = verifier.run(candidate)
        if checked["ok"] and checked["output"] != original_output:
            return {**checked, "origin": source, "description": description}
    return {"ok": False, "output": None, "code": None, "difficulty": None,
            "origin": "generation_failed", "description": None,
            "error": "未找到执行有效且输出不同的受限变式。"}


def _structure(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            node.value = type(node.value).__name__
        elif isinstance(node, ast.Name) and node.id not in _BUILTINS:
            node.id = "variable"
        elif isinstance(node, ast.arg):
            node.arg = "argument"
        elif isinstance(node, ast.FunctionDef):
            node.name = "function"
    return ast.dump(tree, include_attributes=False)


def _related_programs(first, second):
    if _structure(first) == _structure(second):
        return True
    def anchors(code):
        result = set()
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"range", "zip", "enumerate"}:
                result.add(node.func.id)
            elif isinstance(node, ast.Attribute) and node.attr in {"append", "split", "join"}:
                result.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                result.add("function")
        return result
    original = anchors(first)
    return bool(original) and original.issubset(anchors(second))


class TutoringSystem:
    def __init__(self, backend, memory_path, prompts_path=None):
        self.backend = backend
        self.memory = EpisodeStore(memory_path)
        self.verifier = FinitePython()
        self.prompts_path = Path(prompts_path or Path(__file__).with_name("prompts.json"))
        self.prompts = json.loads(self.prompts_path.read_text(encoding="utf-8"))

    def _agent(self, role, context):
        prompt = self.prompts["shared"] + "\n" + self.prompts[role] + "\n输入数据：\n" + json.dumps(context, ensure_ascii=False)
        try:
            generated = self.backend.generate(prompt, role=role)
            if isinstance(generated, str):
                generated = {"raw_response": generated}
            if not isinstance(generated, dict) or not isinstance(generated.get("raw_response"), str):
                raise ValueError("Backend must return raw_response text.")
            record = {**generated, "role": role, "prompt": prompt,
                      "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                      "parsed": _json_object(generated["raw_response"]),
                      "call_ok": True, "accepted": False, "adoption_reason": "pending"}
        except Exception as error:
            record = {"role": role, "prompt": prompt, "raw_response": "", "parsed": None,
                      "call_ok": False, "accepted": False, "adoption_reason": "backend_error",
                      "error": type(error).__name__ + ": " + str(error),
                      "real_model_call": False}
        record.setdefault("real_model_call", bool(getattr(self.backend, "is_real_model", False)))
        return record

    def run_turn(self, student_id, topic, code, student_answer, request, background=None, followup_code=None):
        for name, value, limit in (("student_id", student_id, 100), ("topic", topic, 150),
                                   ("request", request, 3000)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"Invalid {name}.")
        if student_answer is not None and (not isinstance(student_answer, str) or len(student_answer) > 8192):
            raise ValueError("student_answer must be text or None.")
        if background is not None and (not isinstance(background, str) or len(background) > 2000):
            raise ValueError("background must be short text or None.")
        turn_id = uuid.uuid4().hex
        inputs = {"student_id": student_id, "topic": topic, "code": code, "student_answer": student_answer,
                  "request": request, "background": background, "followup_code": followup_code}
        before = self.memory.snapshot(student_id)
        retrieved = self.memory.retrieve(student_id, topic, limit=3)
        main = self.verifier.run(code)
        supplied = student_answer is not None and bool(student_answer.strip())
        status = ("unverified" if not main["ok"] else
                  "unanswered" if not supplied else
                  "correct" if student_answer.strip() == main["output"] else "incorrect")
        evidence = {"status": status, "student_answer": student_answer, "expected_output": main["output"],
                    "source": "student_prediction_checked_by_interpreter", "mastery_claim": False}
        current = {"turn_id": turn_id, "student_evidence": evidence, "tool": {"main": main}}
        recommendation = self.memory.recommend(student_id, topic, main["difficulty"], current)
        policy = _request_policy(request)
        if followup_code is not None:
            policy["wants_followup"] = True
        history_context = [{"turn_id": r["turn_id"], "topic": r["input"]["topic"],
                            "student_answer": r["input"]["student_answer"],
                            "program": r["input"]["code"], "status": r["student_evidence"]["status"],
                            "verified_output": r["tool"]["main"]["output"]} for r in retrieved]
        facts = {"ok": main["ok"], "output": main["output"], "error": main["error"],
                 "execution": _event_explanation(main) if main["ok"] else None}
        context = {"current": inputs, "facts": facts, "student_evidence": evidence,
                   "recent_related_evidence": history_context, "difficulty_recommendation": recommendation,
                   "request_policy": policy}
        planner = self._agent("planner", context)
        parsed = planner["parsed"]
        planner["accepted"] = bool(parsed and all(isinstance(parsed.get(k), str) and 1 <= len(parsed[k]) <= 600
                                                  for k in ("focus", "strategy")))
        planner["adoption_reason"] = "valid_plan_fields_forwarded_to_teacher" if planner["accepted"] else "invalid_json_or_required_plan_fields"
        plan = parsed if planner["accepted"] else {"focus": topic, "strategy": "核对学生预测，依据实际执行过程回应请求。"}
        teacher = self._agent("teacher", {**context, "plan": plan,
                                         "plan_source": "llm" if planner["accepted"] else "rule_fallback"})
        draft = teacher["parsed"]
        followup = None
        followup_attempt = None
        if policy["wants_followup"] and main["ok"]:
            if followup_code is not None:
                followup = {**self.verifier.run(followup_code), "origin": "provided_program",
                            "description": "对照给定复测程序预测输出，并说明变化的一步。"}
            elif draft and isinstance(draft.get("followup_code"), str) and draft["followup_code"].strip():
                candidate = draft["followup_code"]
                followup_attempt = self.verifier.run(candidate)
                different = followup_attempt["ok"] and ast.dump(ast.parse(candidate)) != ast.dump(ast.parse(code))
                related = followup_attempt["ok"] and _related_programs(code, candidate)
                followup_attempt["concept_relation_check"] = {"passed": related,
                    "method": "same_normalized_structure_or_original_operation_anchors_retained",
                    "guarantees_full_pedagogical_relevance": False}
                if different and related and followup_attempt["output"] != main["output"]:
                    followup = {**followup_attempt, "origin": "llm_teacher_checked",
                                "description": "预测这道模型提出并通过执行核验的变式，并说明关键步骤。"}
            if followup is None:
                followup = make_followup(code, main["output"], self.verifier,
                                         harder=policy["wants_harder"] and recommendation["action"] == "increase")
        review_context = {**context, "teacher_raw_response": teacher["raw_response"],
                          "followup_facts": ({k: v for k, v in followup.items() if k not in ("events",)} if followup else None)}
        reviewer = self._agent("reviewer", review_context)
        decision = reviewer["parsed"]
        reviewer["accepted"] = bool(decision and isinstance(decision.get("approved"), bool)
                                     and isinstance(decision.get("reason"), str) and decision["reason"].strip())
        reviewer["adoption_reason"] = "structured_review_used_as_gate" if reviewer["accepted"] else "invalid_json_or_review_fields"
        checks = {"main_verified": main["ok"],
                  "answer_disclosure_allowed": not policy["hide_answer"],
                  "teacher_answer_matches": bool(draft and draft.get("answer") == main["output"] and main["ok"]),
                  "teacher_explanation_present": bool(draft and isinstance(draft.get("explanation"), str)
                                                       and 12 <= len(draft["explanation"]) <= 1600),
                  "teacher_feedback_present": bool(draft and isinstance(draft.get("feedback"), str)
                                                    and 4 <= len(draft["feedback"]) <= 1000),
                  "reviewer_approved": bool(reviewer["accepted"] and decision["approved"]),
                  "followup_verified_if_requested": not policy["wants_followup"] or bool(followup and followup["ok"])}
        adopted = all(checks.values())
        used_model_followup = bool(followup and followup.get("origin") == "llm_teacher_checked")
        teacher["accepted"] = adopted or used_model_followup
        teacher["adopted_components"] = (["feedback", "explanation"] if adopted else []) + (["followup_code"] if used_model_followup else [])
        teacher["adoption_reason"] = "fact_fields_and_model_review_passed_not_formal_prose_proof" if adopted else "failed_gates:" + ",".join(k for k, v in checks.items() if not v)
        text = self._final_text(main, evidence, retrieved, recommendation, policy, followup, draft if adopted else None)
        source = "tool_rejection" if not main["ok"] else "llm_with_verified_facts" if adopted or used_model_followup else "verified_template"
        trace = {"trace_version": 1, "turn_id": turn_id, "created_at": datetime.now(timezone.utc).isoformat(),
                 "student_id": student_id, "input": inputs,
                 "backend": getattr(self.backend, "metadata", {"kind": "test_backend", "real_model": False}),
                 "prompts_sha256": hashlib.sha256(self.prompts_path.read_bytes()).hexdigest(),
                 "agents": {"planner": planner, "teacher": teacher, "reviewer": reviewer},
                 "tool": {"main": main, "followup": followup, "llm_followup_attempt": followup_attempt},
                 "student_evidence": evidence,
                 "memory": {"before": before, "retrieved": retrieved, "recommendation": recommendation},
                 "final": {"text": text, "source": source, "llm_used": adopted or used_model_followup, "checks": checks,
                           "request_policy": policy,
                           "fact_source": "bounded_ast_interpreter",
                           "explanation_source": "llm_reviewed" if adopted else "execution_trace_template",
                           "followup_source": followup["origin"] if followup else None,
                           "difficulty_target_met": (followup["difficulty"] == recommendation["suggested_difficulty"]
                                                     if followup and followup["ok"] and policy["wants_harder"] else None),
                           "llm_prose_guaranteed_correct": False}}
        # Store the raw episode without recursively embedding all prior memory traces.
        episode = {k: v for k, v in trace.items() if k != "memory"}
        self.memory.save(episode)
        trace["memory"]["after"] = self.memory.snapshot(student_id)
        return trace

    def _final_text(self, main, evidence, retrieved, recommendation, policy, followup, draft):
        if not main["ok"]:
            return "这段程序未通过受限执行核验，暂不判断预测对错。原因：" + main["error"]
        lines = []
        if retrieved and policy["wants_explanation"]:
            last = retrieved[0]
            label = {"correct": "与实际输出一致", "incorrect": "与实际输出不一致",
                     "unanswered": "未提供预测", "unverified": "未能核验"}[last["student_evidence"]["status"]]
            prior = last["input"]["student_answer"]
            detail = f"你写的预测是 {prior!r}，核验输出为 {last['tool']['main']['output']!r}。" if prior is not None else ""
            stated = re.split(r"[。！？]", last["input"]["request"], maxsplit=1)[0][:160]
            lines.append(f"上次同一概念的记录中，你的预测{label}。{detail}当时你的原话是「{stated}」。这只是作答证据，不等于掌握结论。")
        if not policy["hide_answer"]:
            lines.append("本题输出：" + main["output"])
        label = {"correct": "你的这次预测与程序实际输出一致。",
                 "incorrect": "你的这次预测与程序实际输出不同。下面核对真正改变结果的步骤。",
                 "unanswered": "这次未提供预测，因此不记录为作答正确或错误。"}[evidence["status"]]
        if policy["wants_explanation"]:
            lines.append(label)
            if draft:
                lines.extend([draft["feedback"], draft["explanation"]])
            elif not policy["hide_answer"]:
                lines.extend(["按实际执行记录核对：", _event_explanation(main)])
            else:
                lines.append("先按代码顺序列出每次赋值和循环的变化，再自行预测打印内容。")
        if policy["wants_harder"]:
            lines.append(recommendation["reason"])
        if policy["wants_followup"]:
            if followup and followup["ok"]:
                lines.extend(["复测题：" + followup["description"], "```python\n" + followup["code"] + "\n```"])
                if not policy["hide_answer"]:
                    lines.append("复测参考输出：" + followup["output"])
                if policy["wants_harder"] and followup["difficulty"] != recommendation["suggested_difficulty"]:
                    lines.append("这道变式没有达到建议的结构难度，暂作为同概念复测。")
            else:
                lines.append("本轮未得到可核验的复测题，未提供未经检查的参考答案。")
        return "\n\n".join(lines)

    def close(self):
        self.memory.close()
