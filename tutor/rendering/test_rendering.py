"""Controller-only regression tests. No model backend or weights are loaded."""
import tempfile
import unittest
from pathlib import Path

from .system import HistoricalV2System, PromptV2System, TutoringSystem, PromptTutoringSystem


CODE = "total = 0\nfor n in range(3, 9, 3):\n    total += n\nprint(total)"
FOLLOWUP = "total = 0\nfor n in range(3, 12, 3):\n    total += n\nprint(total)"
PAIRS = ((HistoricalV2System, TutoringSystem), (PromptV2System, PromptTutoringSystem))


class RenderingRegression(unittest.TestCase):
    def turn(self, system_class, request, provided=True, code=CODE):
        with tempfile.TemporaryDirectory(prefix="tutor-render-regression-") as folder:
            system = system_class(None, Path(folder) / "memory.sqlite", mode="tool_only")
            try:
                trace = system.run_turn("render-test", "range", code, "9", request,
                                        followup_code=FOLLOWUP if provided else None)
                self.assertFalse(trace["final"]["llm_used"])
                self.assertFalse(any(a["real_model_call"] for a in trace["agents"].values()))
                return trace
            finally:
                system.close()

    def test_hidden_answers_still_deliver_requested_exercise(self):
        for old, new in PAIRS:
            with self.subTest(system=new.__name__):
                before = self.turn(old, "先不要答案，给一道复测")
                after = self.turn(new, "先不要答案，给一道复测")
                self.assertTrue(before["tool"]["followup"]["ok"])
                self.assertNotIn("复测题：", before["final"]["text"])
                text = after["final"]["text"]
                self.assertIn("复测题：", text)
                self.assertIn(FOLLOWUP, text)
                for forbidden in ("本题输出：", "复测参考输出：", "核验依据", "执行记录"):
                    self.assertNotIn(forbidden, text)

    def test_no_explanation_does_not_cancel_exercise(self):
        for _, new in PAIRS:
            with self.subTest(system=new.__name__):
                trace = self.turn(new, "不要解释，给一道复测及答案")
                text = trace["final"]["text"]
                self.assertIn("本题输出：9", text)
                self.assertIn(FOLLOWUP, text)
                self.assertIn("复测参考输出：'18'", text)
                self.assertNotIn("核验依据", text)
                self.assertNotIn("复测解释", text)

    def test_ordinary_response_is_unchanged(self):
        for old, new in PAIRS:
            with self.subTest(system=new.__name__):
                request = "请解释并给一道复测及答案"
                self.assertEqual(self.turn(old, request)["final"]["text"],
                                 self.turn(new, request)["final"]["text"])

    def test_no_exercise_request_is_unchanged(self):
        for old, new in PAIRS:
            for request in ("请解释，不要再出练习", "先不要答案", "只要答案"):
                with self.subTest(system=new.__name__, request=request):
                    before = self.turn(old, request, provided=False)
                    after = self.turn(new, request, provided=False)
                    self.assertIsNone(after["tool"]["followup"])
                    self.assertEqual(before["final"]["text"], after["final"]["text"])

    def test_unavailable_requested_exercise_explains_failure(self):
        for _, new in PAIRS:
            with self.subTest(system=new.__name__):
                trace = self.turn(new, "先不要答案，给一道只改变量名的练习", provided=False)
                self.assertFalse(trace["tool"]["followup"]["ok"])
                self.assertIn("未交付复测：", trace["final"]["text"])
                self.assertNotIn("复测参考输出：", trace["final"]["text"])

    def test_unverified_program_rejection_is_unchanged(self):
        for old, new in PAIRS:
            with self.subTest(system=new.__name__):
                kwargs = dict(request="先不要答案，给一道复测", provided=False,
                              code="import os\nprint(9)")
                before, after = self.turn(old, **kwargs), self.turn(new, **kwargs)
                self.assertFalse(after["tool"]["main"]["ok"])
                self.assertEqual(before["final"]["text"], after["final"]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
