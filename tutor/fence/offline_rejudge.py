"""Reparse two already observed turns; no model calls or adoption claims."""
import ast
import json
from pathlib import Path
from .baseline_contracts import parse_strict_object as previous_parser
from .contracts import parse_strict_object, planner_contract, teacher_contract, reviewer_contract

HERE=Path(__file__).resolve().parent

def payload(agent):
    return json.loads(agent['prompt'].split('输入数据：\n',1)[1])

def main():
    frozen=json.loads((HERE/'observed_first_two.json').read_text(encoding='utf8'))
    rows=[]
    for turn in frozen['turns']:
        agents=turn['trace']['agents']; parsed={r:parse_strict_object(a['raw_response']) for r,a in agents.items()}
        teacher_context=payload(agents['teacher']); review_context=payload(agents['reviewer'])
        facts=teacher_context['facts']; required=teacher_context['plan']['evidence_id']
        teacher_checks=teacher_contract(parsed['teacher'],teacher_context['verified_answer'],teacher_context['student_status'],facts,required)
        # Review was actually generated against the OLD parsed draft (None), not our recovered object.
        actual_review=reviewer_contract(parsed['reviewer'],review_context['draft'],review_context['verified_answer'],review_context['student_status'],review_context['facts'],review_context['required_evidence_id'])
        hypothetical_review=reviewer_contract(parsed['reviewer'],parsed['teacher'],teacher_context['verified_answer'],teacher_context['student_status'],facts,required)
        rows.append({'turn_id':turn['fixture_turn_id'],
            'old_parse':{r:previous_parser(a['raw_response']) is not None for r,a in agents.items()},
            'new_parse':{r:p is not None for r,p in parsed.items()},
            'parsed_objects':parsed,
            'planner_content_valid':planner_contract(parsed['planner'],facts),
            'teacher_content_checks':teacher_checks,'teacher_content_valid':all(teacher_checks.values()),
            'reviewer_on_actual_recorded_context':actual_review,
            'same_review_text_against_recovered_teacher_counterfactual_only':hypothetical_review,
            'interpretation':'Reparsing cannot establish how the reviewer would respond to a changed prompt. No new model answer or final-adoption count is inferred.'})
    def functions(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text(encoding='utf8')).body if isinstance(x,ast.FunctionDef)}
    before=functions(HERE/'baseline_contracts.py'); after=functions(HERE/'contracts.py')
    unchanged={name:before[name]==after[name] for name in ('planner_contract','teacher_contract','reviewer_contract')}
    out={'model_calls':0,'study_role':'exploratory post-hoc reparse of first two observed turns',
         'content_gate_functions_ast_unchanged':unchanged,
         'old_parsed_role_objects':sum(sum(r['old_parse'].values()) for r in rows),
         'new_parsed_role_objects':sum(sum(r['new_parse'].values()) for r in rows),
         'teacher_content_valid_count':sum(r['teacher_content_valid'] for r in rows),
         'rows':rows}
    (HERE/'offline_rejudge.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False))
    if not all(unchanged.values()):raise SystemExit('Content gate changed unexpectedly')

if __name__=='__main__':main()
