"""Repair exercise delivery when answer/explanation disclosure is disabled.

Only final rendering is changed. The two wrappers inherit the corresponding
frozen v2 implementation, prompts, verifier, memory, and adoption decisions.
No historical run or model output is rewritten.
"""
from tutor.core.system import TutoringSystem as HistoricalV2System
from tutor.structured.system import TutoringSystem as PromptV2System


def _restore_requested_exercise(text, main, policy, followup):
    """A disclosure choice must not silently cancel a separate exercise request."""
    early_return = policy["hide_answer"] or not policy["wants_explanation"]
    if not (main["ok"] and policy["wants_followup"] and early_return):
        return text
    if not followup or not followup["ok"]:
        reason = followup["error"] if followup else "请求无法核验。"
        return text + "\n\n未交付复测：" + reason
    blocks = [text, "复测题：" + followup["description"],
              "```python\n" + followup["code"] + "\n```"]
    # The question remains visible; neither reference output nor execution facts
    # are disclosed when the learner asks to attempt it before seeing an answer.
    if not policy["hide_answer"]:
        blocks.append("复测参考输出：" + repr(followup["output"]))
    return "\n\n".join(blocks)


class TutoringSystem(HistoricalV2System):
    @staticmethod
    def _final_text(main, evidence, retrieved, recommendation, policy, followup, draft, facts):
        text = HistoricalV2System._final_text(
            main, evidence, retrieved, recommendation, policy, followup, draft, facts)
        return _restore_requested_exercise(text, main, policy, followup)


class PromptTutoringSystem(PromptV2System):
    @staticmethod
    def _final_text(main, evidence, retrieved, recommendation, policy, followup, draft, facts, extractive=False):
        text = PromptV2System._final_text(
            main, evidence, retrieved, recommendation, policy, followup, draft, facts, extractive)
        return _restore_requested_exercise(text, main, policy, followup)
