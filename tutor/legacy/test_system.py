"""Fast deterministic tests. Fakes are explicitly not real model evaluations."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tutor.legacy.system import FinitePython, TutoringSystem, make_followup


class InvalidBackend:
    is_real_model = False
    metadata = {"kind": "test_fake", "real_model": False}

    def __init__(self):
        self.calls = []

    def generate(self, prompt, role):
        self.calls.append((role, prompt))
        return {"raw_response": "不符合 JSON 的测试文本", "real_model_call": False}


class StructuredBackend(InvalidBackend):
    def generate(self, prompt, role):
        self.calls.append((role, prompt))
        context = json.loads(prompt.split("输入数据：\n", 1)[1])
        if role == "planner":
            answer = {"focus": "赋值顺序", "strategy": "先检查数值，再解释发生变化的变量。"}
        elif role == "teacher":
            answer = {"answer": context["facts"]["output"], "feedback": "本次预测与输出一致。",
                      "explanation": "变量先取得给定整数，print 读取该整数，所以应按赋值后保存的值核对。",
                      "followup_code": "x = 8\nprint(x)"}
        else:
            answer = {"approved": True, "reason": "答案字段与给定执行事实一致。"}
        return {"raw_response": json.dumps(answer, ensure_ascii=False), "real_model_call": False}


class InterpreterTests(unittest.TestCase):
    def test_matches_cpython_for_reviewed_programs(self):
        # These are fixed, hand-reviewed test programs, never unknown model code.
        programs = [
            "x=2+3*4\nprint(x)",
            "s=0\nfor x in range(2,9,2):\n s+=x\nprint(s)",
            "print(sum(range(3,12,3)))",
            "s='python'\nprint(s[1:5:2])",
            "a=[1,2]\nb=a\na.append(3)\nprint(b)",
            "a=[1,2]\nb=a[:]\na.append(3)\nprint(b)",
            "a=[1,2]\nb=a\na*=2\nprint(b)",
            "a=[1]\nb=a\na += [2]\nprint(b)",
            "s=0\nfor a,b in zip([2,4,8],[3,5]):\n s+=a+b\nprint(s)",
            "def put(x, bag=[]):\n bag.append(x)\n return len(bag)\na=put(2)\nb=put(3)\nprint(a+b)",
            "def f(n):\n if n<=1:\n  return 1\n return n*f(n-1)\nprint(f(4))",
            "x=0\nwhile x<8:\n x+=1\n if x==5:\n  break\nprint(x)",
            "d={'a':2}\nd['a']+=3\nprint(d.get('a'))",
            "s='ab:cd:ef'\np=s.split(':')\nprint('|'.join(p[:-1]))",
        ]
        for program in programs:
            with self.subTest(program=program):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(program, {})
                actual = FinitePython().run(program)
                self.assertTrue(actual["ok"], actual)
                self.assertEqual(actual["output"], output.getvalue().strip())
                json.dumps(actual)

    def test_rejects_dangerous_and_unbounded_programs(self):
        programs = ["import os\nprint(1)", "print(open('bad'))", "print((1).__class__)",
                    "while True:\n pass", "print('x'*1000000000)", "print(2**100000)",
                    "print(list(range(1000000000)))", "x=[]\nx.append(x)\nprint(x)",
                    "print(eval('2+2'))"]
        for program in programs:
            with self.subTest(program=program):
                self.assertFalse(FinitePython(max_steps=100).run(program)["ok"])

    def test_function_scope_and_default_reset(self):
        self.assertFalse(FinitePython().run("x=1\ndef f():\n print(x)\n x=2\nf()")["ok"])
        code = "def f(bag=[]):\n bag.append(3)\n return len(bag)\nprint(f())"
        engine = FinitePython()
        self.assertEqual(engine.run(code)["output"], "1")
        self.assertEqual(engine.run(code)["output"], "1")

    def test_harder_followup_and_execution_events(self):
        code = "s=0\nfor x in range(1,6):\n s+=x\nprint(s)"
        main = FinitePython().run(code)
        follow = make_followup(code, main["output"], FinitePython(), harder=True)
        self.assertTrue(follow["ok"])
        self.assertEqual(follow["origin"], "ast_guard_fallback")
        self.assertEqual(follow["difficulty"], "挑战")
        self.assertNotEqual(follow["output"], main["output"])
        self.assertTrue(any(e["action"] == "range" for e in main["events"]))
        short = "print(sum(range(2,8,2)))"
        follow = make_followup(short, "12", FinitePython(), harder=True)
        # All original values are even, so an unchanged output must be rejected.
        self.assertTrue(follow["ok"])
        harder = make_followup("print(sum(range(1,6)))", "15", FinitePython(), harder=True)
        self.assertEqual(harder["origin"], "ast_conditional_sum_fallback")
        self.assertEqual(harder["difficulty"], "挑战")


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test-memory-", dir=Path(__file__).parent)
        self.path = Path(self.tmp.name) / "evidence.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def turn(self, system, student="a", answer="4", request="请解释并给一道复测题"):
        return system.run_turn(student, "赋值", "x=4\nprint(x)", answer, request)

    def test_three_independent_calls_and_explicit_fallback(self):
        backend = InvalidBackend(); system = TutoringSystem(backend, self.path)
        trace = self.turn(system)
        self.assertEqual([r for r, _ in backend.calls], ["planner", "teacher", "reviewer"])
        self.assertEqual(trace["final"]["source"], "verified_template")
        self.assertFalse(trace["final"]["llm_used"])
        self.assertEqual(trace["final"]["followup_source"], "ast_constant_fallback")
        self.assertTrue(all(not a["real_model_call"] and a["raw_response"] for a in trace["agents"].values()))
        self.assertEqual(trace["student_evidence"]["status"], "correct")
        json.dumps(trace)

    def test_valid_role_handoff_is_distinguished_from_template(self):
        system = TutoringSystem(StructuredBackend(), self.path)
        trace = self.turn(system)
        self.assertEqual(trace["final"]["source"], "llm_with_verified_facts")
        self.assertEqual(trace["final"]["followup_source"], "llm_teacher_checked")
        self.assertIn("followup_code", trace["agents"]["teacher"]["adopted_components"])
        self.assertIn("plan", json.loads(trace["agents"]["teacher"]["prompt"].split("输入数据：\n")[1]))

    def test_restart_and_student_isolation(self):
        first = TutoringSystem(InvalidBackend(), self.path)
        old = self.turn(first, answer="8"); first.close()
        second = TutoringSystem(InvalidBackend(), self.path)
        new = self.turn(second, answer=None)
        self.assertEqual(new["memory"]["retrieved"][0]["turn_id"], old["turn_id"])
        counters = new["memory"]["after"]["topics"][0]
        self.assertEqual((counters["attempts"], counters["incorrect"], counters["unanswered"]), (1, 1, 1))
        other = self.turn(second, student="b", answer=None)
        self.assertEqual(other["memory"]["retrieved"], [])
        self.assertEqual(other["memory"]["before"]["topics"], [])
        self.assertNotIn(old["turn_id"], other["agents"]["planner"]["prompt"])
        self.assertEqual(other["memory"]["after"]["topics"][0]["attempts"], 0)

    def test_unanswered_does_not_erase_verified_streak(self):
        system = TutoringSystem(InvalidBackend(), self.path)
        for end in (4, 5, 6):
            system.run_turn("a", "range", f"print(sum(range(1,{end})))", str(sum(range(1, end))), "核对结果")
        result = system.run_turn("a", "range", "s=0\nfor i in range(1,6):\n s+=i\nprint(s)",
                                 None, "请提高难度，再出一道复测题和答案")
        self.assertEqual(result["memory"]["recommendation"]["action"], "increase")
        self.assertEqual(result["memory"]["after"]["topics"][0]["attempts"], 3)
        self.assertTrue(result["final"]["difficulty_target_met"])

    def test_bad_program_never_updates_correctness(self):
        system = TutoringSystem(InvalidBackend(), self.path)
        result = system.run_turn("a", "安全", "import os\nprint(1)", "1", "请核对")
        self.assertEqual(result["final"]["source"], "tool_rejection")
        self.assertEqual(result["memory"]["after"]["topics"][0]["attempts"], 0)
        self.assertEqual(result["student_evidence"]["status"], "unverified")

    def test_hide_answer_and_answer_only(self):
        system = TutoringSystem(StructuredBackend(), self.path)
        hidden = self.turn(system, answer=None, request="先不要给答案，给我提示")
        self.assertNotIn("本题输出：", hidden["final"]["text"])
        answer = self.turn(system, request="只给答案")
        self.assertEqual(answer["final"]["text"], "本题输出：4")

    def test_followup_request_paraphrases(self):
        system = TutoringSystem(InvalidBackend(), self.path)
        for request in ("请设计第二题并给答案", "再给一道同概念修复练习及答案"):
            trace = self.turn(system, request=request)
            self.assertTrue(trace["final"]["request_policy"]["wants_followup"])
            self.assertIsNotNone(trace["tool"]["followup"])


if __name__ == "__main__":
    unittest.main()
