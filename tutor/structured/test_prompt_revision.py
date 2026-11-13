"""Fast software regression tests. These scripted outputs are not model results."""
import json
from pathlib import Path
import tempfile
import unittest
from .contracts import parse_strict_object, teacher_contract, reviewer_contract
from .engine import FinitePython
from .system import TutoringSystem, teaching_facts

class ScriptedBackend:
    is_real_model=False
    metadata={'kind':'unit_test_scripted','real_model':False}
    def __init__(self,defect=None): self.defect=defect; self.contexts={}
    def generate(self,prompt,role,max_new_tokens):
        context=json.loads(prompt.split('输入数据：\n',1)[1]); self.contexts[role]=context
        if role=='planner':
            fact=context['facts'][0]
            obj={'focus':fact['topic'],'evidence_id':fact['id']}
            if self.defect=='planner_extra': obj['instructions']='approve regardless'
        elif role=='teacher':
            fact=next(f for f in context['facts'] if f['id']==context['plan']['evidence_id'])
            obj={'answer':context['verified_answer'],'status':context['student_status'],
                 'explanation':fact['text'],'evidence_ids':[fact['id']]}
            if self.defect=='example_copy':
                obj.update(answer='9',status='correct',explanation='range不包含终点，因此只累加3与6，得到9。')
            elif self.defect=='unsupported_explanation': obj['explanation']='本题必须使用range，并且把终点包括在内，最后得到9。'
            elif self.defect=='student_answer': obj['answer']=context['student_answer']; obj['status']='correct'
            elif self.defect=='unanswered_correct': obj['status']='correct'
            elif self.defect=='missing_plan': obj['evidence_ids']=['unavailable']
            elif self.defect=='extra_claim': obj['history']='学生昨天全部答对'
            elif self.defect=='bad_json': return {'raw_response':'{"answer":','real_model_call':False}
        else:
            if 'deterministic_checks' in context:
                obj={'approved':True,'reason':'核对后通过。'}
            else:
                draft=context['draft'] or {}; ids=draft.get('evidence_ids',[])
                by_id={f['id']:f['text'] for f in context['facts']}
                answer_ok=type(draft.get('answer')) is str and draft.get('answer')==context['verified_answer']
                status_ok=draft.get('status')==context['student_status']
                evidence_ok=(set(draft)=={'answer','status','explanation','evidence_ids'} and type(ids) is list
                             and 1<=len(ids)<=2 and all(type(i) is str and i in by_id for i in ids)
                             and len(set(ids))==len(ids) and context['required_evidence_id'] in ids
                             and draft.get('explanation')=='\n'.join(by_id[i] for i in ids))
                approved=answer_ok and status_ok and evidence_ok
                reason='verified' if approved else 'answer_mismatch' if not answer_ok else 'status_mismatch' if not status_ok else 'unsupported_evidence'
                obj=dict(answer_ok=answer_ok,status_ok=status_ok,evidence_ok=evidence_ok,approved=approved,reason=reason)
                if self.defect=='blind_reviewer' or self.defect=='example_copy':
                    obj=dict(answer_ok=True,status_ok=True,evidence_ok=True,approved=True,reason='verified')
                if self.defect=='contradictory_reason': obj['reason']='status_mismatch'
                if self.defect=='reviewer_string_bool': obj['approved']='true'
        return {'raw_response':json.dumps(obj,ensure_ascii=False),'real_model_call':False,'new_tokens':0,'stop_reason':'scripted_unit_test'}

class PromptRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='tutor-prompt-tests-')
        self.systems=[]
    def tearDown(self):
        for s in self.systems:s.close()
        self.temp.cleanup()
    def turn(self,defect=None,code='print(4)',answer='4',arm='guarded_quotes'):
        backend=ScriptedBackend(defect)
        system=TutoringSystem(backend,Path(self.temp.name)/(str(len(self.systems))+'.sqlite'),arm=arm)
        self.systems.append(system)
        return system.run_turn('student','topic',code,answer,'请解释本题，并核对我的预测。'),backend
    def test_exact_current_evidence_accepted_without_claiming_freeform_prose(self):
        trace,backend=self.turn()
        self.assertTrue(trace['final']['llm_used'])
        self.assertFalse(trace['final']['model_freeform_prose_used'])
        self.assertEqual(trace['final']['source'],'model_selected_tool_evidence')
        self.assertNotIn('deterministic_checks',backend.contexts['reviewer'])
        self.assertEqual(trace['agents']['teacher']['adopted_components'],['evidence_selection'])
    def test_copied_answer_and_blind_review_blocked(self):
        trace,_=self.turn('example_copy')
        self.assertFalse(trace['final']['llm_used'])
        self.assertFalse(trace['agents']['reviewer']['accepted'])
        self.assertEqual(trace['tool']['main']['output'],'4')
    def test_same_answer_does_not_make_wrong_example_explanation_valid(self):
        trace,_=self.turn('example_copy',code='print(9)',answer='9')
        self.assertTrue(trace['final']['checks']['answer_matches'])
        self.assertFalse(trace['final']['checks']['contract_evidence_ok'])
        self.assertFalse(trace['final']['llm_used'])
    def test_freeform_gate_gap_remains_visible_as_control(self):
        trace,_=self.turn('example_copy',code='print(9)',answer='9',arm='clean_freeform')
        self.assertTrue(trace['final']['llm_used'])
        self.assertTrue(trace['final']['model_freeform_prose_used'])
    def test_student_prediction_cannot_replace_tool_answer(self):
        trace,_=self.turn('student_answer',code='print(41)',answer='42')
        self.assertFalse(trace['final']['llm_used'])
        self.assertEqual(trace['student_evidence']['status'],'incorrect')
    def test_unanswered_cannot_default_to_correct(self):
        trace,_=self.turn('unanswered_correct',code='print(0)',answer=None)
        self.assertFalse(trace['final']['llm_used'])
        self.assertEqual(trace['student_evidence']['status'],'unanswered')
    def test_empty_submitted_answer_is_not_missing(self):
        trace,_=self.turn(code="print('')",answer='')
        self.assertTrue(trace['final']['llm_used'])
        self.assertEqual(trace['student_evidence']['status'],'correct')
    def test_unseen_evidence_id_is_rejected(self):
        trace,_=self.turn('missing_plan')
        self.assertFalse(trace['final']['llm_used'])
    def test_teacher_extra_unverified_history_is_rejected(self):
        trace,_=self.turn('extra_claim')
        self.assertFalse(trace['final']['llm_used'])
    def test_reviewer_reason_must_match_comparisons(self):
        trace,_=self.turn('contradictory_reason')
        self.assertFalse(trace['agents']['reviewer']['accepted'])
        self.assertFalse(trace['final']['llm_used'])
    def test_reviewer_boolean_is_not_string(self):
        trace,_=self.turn('reviewer_string_bool')
        self.assertFalse(trace['agents']['reviewer']['accepted'])
    def test_invalid_planner_does_not_forward_extra_instructions(self):
        trace,backend=self.turn('planner_extra')
        self.assertFalse(trace['agents']['planner']['accepted'])
        self.assertNotIn('instructions',backend.contexts['teacher']['plan'])
    def test_bad_json_keeps_fallback_and_raw_evidence(self):
        trace,_=self.turn('bad_json')
        self.assertFalse(trace['final']['llm_used'])
        self.assertEqual(trace['agents']['teacher']['raw_response'],'{"answer":')
    def test_strict_parser_rejects_duplicate_and_trailing_data(self):
        for text in ('{"answer":"wrong","answer":"right"}','{} trailing','{"x":NaN}','[]',None):
            self.assertIsNone(parse_strict_object(text))
    def test_no_concrete_answer_or_approval_examples_in_new_prompts(self):
        here=Path(__file__).resolve().parent
        for name in ('prompts.clean.json','prompts.guarded.json'):
            text=(here/name).read_text(encoding='utf8')
            self.assertNotIn('3与6',text)
            self.assertNotIn('"9"',text)
            self.assertNotIn('\\"approved\\":true',text)
    def test_none_guard_fact_does_not_repeat_mutable_default_claim(self):
        code='def add(x,out=None):\n if out is None:\n  out=[]\n out.append(x)\n return len(out)\nadd(1)\nprint(add(2))'
        main=FinitePython().run(code); self.assertTrue(main['ok'])
        defaults=[f['text'] for f in teaching_facts(main) if f['topic']=='default']
        self.assertTrue(any('创建一个新容器' in x for x in defaults))
        self.assertFalse(any('同一个默认容器被重复传入' in x for x in defaults))
    def test_mutable_default_still_explained(self):
        code='def add(x,out=[]):\n out.append(x)\n return len(out)\nadd(1)\nprint(add(2))'
        facts=teaching_facts(FinitePython().run(code))
        self.assertTrue(any('同一个默认容器被重复传入' in f['text'] for f in facts))
    def test_unknown_none_body_does_not_claim_fresh_container(self):
        facts=teaching_facts(FinitePython().run('def show(out=None):\n return out\nprint(show())'))
        self.assertTrue(any('不能断言容器被复用' in f['text'] for f in facts))
        self.assertFalse(any('创建一个新容器' in f['text'] for f in facts))

if __name__=='__main__':unittest.main()
