"""Fast tests only. Scripted responses never count as actual model evaluation."""
import json
from pathlib import Path
import tempfile
import unittest

from .constraints import request_policy, constraint_spec, constrained_followup, validate_constraint
from .engine import FinitePython
from .system import TutoringSystem


class FakeBackend:
    is_real_model = False
    metadata = {"kind": "test_fake", "real_model": False}
    def __init__(self, defect=None):
        self.defect = defect
        self.calls = []
    def generate(self, prompt, role, max_new_tokens):
        self.calls.append(role)
        context = json.loads(prompt.split("输入数据：\n", 1)[1])
        if self.defect == "invalid":
            return {"raw_response": "not JSON", "real_model_call": False}
        if role == "planner":
            fact = context["facts"][0]
            obj = {"focus": fact["topic"], "evidence_id": fact["id"]}
        elif role == "teacher":
            obj = {"answer": context["verified_answer"], "status": context["student_status"],
                   "explanation": context["facts"][0]["text"], "evidence_ids": [context["facts"][0]["id"]]}
            if self.defect == "wrong_answer":
                obj["answer"] = "999"
            if self.defect == "wrong_status":
                obj["status"] = "incorrect"
            if self.defect == "invented_evidence":
                obj["evidence_ids"] = ["F999"]
            if self.defect == "mastery":
                obj["explanation"] = "你已经掌握全部概念，因此不必继续核对任何程序。"
        else:
            obj = {"approved": self.defect != "reject", "reason": "依据提供的事实逐项核对。"}
        return {"raw_response": json.dumps(obj, ensure_ascii=False), "real_model_call": False}


class ConstraintTests(unittest.TestCase):
    def test_long_request_is_recognized(self):
        self.assertTrue(request_policy("再给一道只改默认参数处理方式的修复练习及答案")["wants_followup"])
        self.assertTrue(request_policy("再给一道" + "同一个概念的" * 8 + "题")["wants_followup"])

    def test_negations(self):
        for request in ("不要再出练习", "不用复测", "不要再出题", "别给我复测题"):
            self.assertFalse(request_policy(request)["wants_followup"], request)
        self.assertTrue(request_policy("先不要答案，给一道复测")["hide_answer"])

    def test_append_constraint_not_merely_execution(self):
        code = "left=[9,24,17,32,6,18]\nright=[7,16,35,22,44]\ntotal=0\nfor a,b in zip(left,right):\n total+=a+b\nprint(total)"
        spec = constraint_spec("再只给右列表补一个数，设计第二题")
        follow = constrained_followup(code, "212", FinitePython(), spec)
        self.assertTrue(follow["ok"])
        self.assertEqual(follow["output"], "231")
        self.assertTrue(validate_constraint(code, follow["code"], spec)["passed"])
        wrong = follow["code"].replace("total = 0", "total = 1")
        self.assertFalse(validate_constraint(code, wrong, spec)["passed"])
        self.assertFalse(validate_constraint(code, code.replace("total=0", "total=1"), spec)["passed"])

    def test_default_repair_only(self):
        code = "def add(x,out=[]):\n out.append(x)\n return out\na=add(1)\nb=add(2)\nprint((a,b))"
        spec = constraint_spec("只改默认参数处理方式的修复练习")
        main = FinitePython().run(code)
        follow = constrained_followup(code, main["output"], FinitePython(), spec)
        self.assertTrue(follow["ok"])
        self.assertEqual(follow["output"], "([1], [2])")
        self.assertFalse(validate_constraint(code, follow["code"].replace("add(2)", "add(3)"), spec)["passed"])
        unsupported = code.replace("out=[]", "out=[0]")
        self.assertFalse(constrained_followup(unsupported, "x", FinitePython(), spec)["ok"])

    def test_endpoint_preserves_other_structure(self):
        code = "print(sum(range(2,8,2)))"
        spec = constraint_spec("再出一道只改终点的复测")
        self.assertTrue(validate_constraint(code, "print(sum(range(2,10,2)))", spec)["passed"])
        self.assertFalse(validate_constraint(code, "print(sum(range(3,10,2)))", spec)["passed"])

    def test_unknown_constraint_abstains(self):
        code = "x=4\nprint(x)"
        spec = constraint_spec("只改变量名的练习")
        follow = constrained_followup(code, "4", FinitePython(), spec)
        self.assertFalse(follow["ok"])
        self.assertEqual(follow["origin"], "constraint_abstention")


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tutor-v2-test-")
        self.path = Path(self.tmp.name) / "memory.sqlite"
    def tearDown(self):
        self.tmp.cleanup()
    def system(self, backend=None, mode="tool_only"):
        return TutoringSystem(backend, self.path, mode=mode)
    def turn(self, system, answer="4", request="请解释", student="a", code="x=4\nprint(x)"):
        return system.run_turn(student, "test", code, answer, request)

    def test_empty_prediction_and_blank_output_are_distinct(self):
        system = self.system()
        result = self.turn(system, answer="", code="word='python'\nprint(word[4:1])")
        self.assertEqual(result["student_evidence"]["status"], "correct")
        self.assertEqual(result["memory"]["after"]["topics"][0]["correct"], 1)
        blank = self.turn(system, answer="", code="print(' ')")
        self.assertEqual(blank["tool"]["main"]["output"], " ")
        self.assertEqual(blank["student_evidence"]["status"], "incorrect")
        self.assertEqual(self.turn(system, answer=" ", code="print(' ')")["student_evidence"]["status"], "correct")
        self.assertEqual(self.turn(system, answer=None, code="print('')")["student_evidence"]["status"], "unanswered")

    def test_memory_rejects_semantically_wrong_status(self):
        system = self.system()
        trace = self.turn(system)
        trace["turn_id"] = "bad"
        trace["student_evidence"]["status"] = "incorrect"
        with self.assertRaises(ValueError):
            system.memory.save(trace)

    def test_valid_model_text_is_attributed_exactly(self):
        backend = FakeBackend()
        trace = self.turn(self.system(backend, "multi"))
        self.assertEqual(backend.calls, ["planner", "teacher", "reviewer"])
        self.assertTrue(trace["final"]["llm_used"])
        self.assertIn(trace["agents"]["teacher"]["parsed"]["explanation"], trace["final"]["text"])
        self.assertFalse(any(a["real_model_call"] for a in trace["agents"].values()))

    def test_invalid_or_false_content_is_not_accepted(self):
        for defect in ("invalid", "wrong_answer", "wrong_status", "invented_evidence", "mastery", "reject"):
            with self.subTest(defect=defect):
                trace = self.turn(self.system(FakeBackend(defect), "multi"))
                self.assertFalse(trace["final"]["llm_used"])
                self.assertEqual(trace["final"]["source"], "verified_template")

    def test_single_and_tool_calls(self):
        backend = FakeBackend()
        trace = self.turn(self.system(backend, "single"))
        self.assertEqual(backend.calls, ["teacher"])
        self.assertTrue(trace["final"]["llm_used"])
        trace = self.turn(self.system())
        self.assertEqual(trace["agents"], {})
        self.assertFalse(trace["final"]["llm_used"])

    def test_restart_isolation_and_memory_ablation(self):
        first = self.system()
        old = self.turn(first, answer="8")
        first.close()
        second = self.system()
        current = self.turn(second)
        self.assertEqual(current["memory"]["retrieved"][0]["turn_id"], old["turn_id"])
        self.assertEqual(self.turn(second, student="b")["memory"]["retrieved"], [])
        disabled = self.system(mode="tool_no_memory")
        current = self.turn(disabled)
        self.assertEqual(current["memory"]["retrieved"], [])
        self.assertEqual(current["memory"]["recommendation"]["source"], "memory_disabled")

    def test_hidden_answer_and_no_followup(self):
        trace = self.turn(self.system(), request="先不要答案，给一道复测")
        self.assertNotIn("本题输出：", trace["final"]["text"])
        self.assertNotIn("复测参考输出：", trace["final"]["text"])
        trace = self.turn(self.system(), request="请解释，不要再出练习")
        self.assertIsNone(trace["tool"]["followup"])

    def test_program_rejection(self):
        trace = self.turn(self.system(), code="import os\nprint(4)")
        self.assertEqual(trace["student_evidence"]["status"], "unverified")
        self.assertEqual(trace["final"]["source"], "tool_rejection")


if __name__ == "__main__":
    unittest.main()
