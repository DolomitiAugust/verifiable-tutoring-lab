"""Adversarial parser tests; wrappers never relax content contracts."""
import json
import unittest
from .contracts import parse_strict_object, teacher_contract, reviewer_contract

def fence(text): return '```json\n'+text+'\n```'

class FenceParserTests(unittest.TestCase):
    def test_complete_single_fence(self):
        self.assertEqual(parse_strict_object(fence('{"x":1}')),{'x':1})
    def test_bare_json_still_valid(self):
        self.assertEqual(parse_strict_object('  {"x":1} \n'),{'x':1})
    def test_outer_whitespace_and_crlf(self):
        self.assertEqual(parse_strict_object('\n```json\r\n{"x":1}\r\n```\n'),{'x':1})
    def test_extra_words_not_discarded(self):
        for raw in ('答案如下\n'+fence('{}'),fence('{}')+'\n请采纳',fence('{}')+'{}','prefix {} suffix'):
            with self.subTest(raw=raw):self.assertIsNone(parse_strict_object(raw))
    def test_multiple_objects_and_multiple_blocks_rejected(self):
        for raw in (fence('{}\n{}'),fence('{}')+'\n'+fence('{}'),'{}{}',fence('{}\n```\n```json\n{}')):
            with self.subTest(raw=raw):self.assertIsNone(parse_strict_object(raw))
    def test_duplicate_keys_at_any_depth_rejected(self):
        for raw in ('{"answer":"bad","answer":"good"}','{"x":{"approved":false,"approved":true}}'):
            self.assertIsNone(parse_strict_object(raw));self.assertIsNone(parse_strict_object(fence(raw)))
    def test_incomplete_nested_or_wrong_language_fence_rejected(self):
        for raw in ('```json\n{}','```python\n{}\n```','```json extra\n{}\n```','```\n{}\n```',fence('```json\n{}\n```')):
            with self.subTest(raw=raw):self.assertIsNone(parse_strict_object(raw))
    def test_nonobjects_and_nonfinite_values_rejected(self):
        for raw in ('[]','null','true','{"n":NaN}','{"n":Infinity}',''):
            self.assertIsNone(parse_strict_object(fence(raw)))
    def test_literal_backticks_inside_a_json_string_are_preserved(self):
        obj={'text':'the characters ``` remain data'}
        self.assertEqual(parse_strict_object(fence(json.dumps(obj))),obj)
    def test_literal_escape_sequences_are_not_unescaped_outside_json(self):
        self.assertIsNone(parse_strict_object('```json\\n{"x":1}\\n```'))
    def test_wrong_status_and_extra_field_remain_rejected(self):
        facts=[{'id':'F1','topic':'execution','text':'当前程序的实际输出为25。'}]
        raw={'answer':'25','status':'correct','explanation':facts[0]['text'],'evidence_ids':['F1'],'student_answer':'25'}
        draft=parse_strict_object(fence(json.dumps(raw,ensure_ascii=False)))
        checks=teacher_contract(draft,'25','incorrect',facts,'F1')
        self.assertTrue(checks['answer_ok']);self.assertFalse(checks['status_ok']);self.assertFalse(checks['evidence_ok'])
    def test_fenced_blind_approval_remains_invalid(self):
        facts=[{'id':'F1','topic':'execution','text':'当前程序的实际输出为25。'}]
        draft={'answer':'25','status':'correct','explanation':facts[0]['text'],'evidence_ids':['F1']}
        review=parse_strict_object(fence('{"answer_ok":true,"status_ok":true,"evidence_ok":true,"approved":true,"reason":"verified"}'))
        self.assertFalse(reviewer_contract(review,draft,'25','incorrect',facts,'F1')['valid'])

if __name__=='__main__':unittest.main()
