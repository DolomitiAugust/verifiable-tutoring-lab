"""Independent deterministic contracts for three roles. No model or renderer imports.

Exact evidence selection is deliberately narrower than unrestricted explanation.
It proves quote fidelity to tool cards, not the general correctness of every card.
"""
import json

TEACHER_KEYS={'answer','status','explanation','evidence_ids'}
REVIEW_KEYS={'answer_ok','status_ok','evidence_ok','approved','reason'}

def parse_strict_object(text):
    if type(text) is not str: return None
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value: raise ValueError('duplicate key')
            value[key]=item
        return value
    try:
        obj=json.loads(text.strip(),object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        return obj if type(obj) is dict else None
    except (TypeError,ValueError): return None

def planner_contract(obj,facts):
    return bool(type(obj) is dict and set(obj)=={'focus','evidence_id'} and
                type(obj['focus']) is str and type(obj['evidence_id']) is str and
                any(obj['focus']==f['topic'] and obj['evidence_id']==f['id'] for f in facts))

def teacher_contract(draft,verified_answer,student_status,facts,required_evidence_id):
    is_dict=type(draft) is dict
    answer_ok=is_dict and type(draft.get('answer')) is str and draft['answer']==verified_answer
    status_ok=is_dict and type(draft.get('status')) is str and draft['status']==student_status
    ids=draft.get('evidence_ids') if is_dict else None
    text=draft.get('explanation') if is_dict else None
    by_id={f['id']:f['text'] for f in facts}
    valid_ids=(type(ids) is list and 1<=len(ids)<=2 and all(type(i) is str and i in by_id for i in ids)
               and len(set(ids))==len(ids) and required_evidence_id in ids)
    evidence_ok=bool(is_dict and set(draft)==TEACHER_KEYS and valid_ids and type(text) is str
                     and text=='\n'.join(by_id[i] for i in ids))
    return {'answer_ok':bool(answer_ok),'status_ok':bool(status_ok),'evidence_ok':evidence_ok}

def reviewer_contract(decision,draft,verified_answer,student_status,facts,required_evidence_id):
    # Recompute expected comparisons here; never trust the teacher's acceptance flag.
    expected=teacher_contract(draft,verified_answer,student_status,facts,required_evidence_id)
    schema=type(decision) is dict and set(decision)==REVIEW_KEYS
    typed=schema and all(type(decision[k]) is bool for k in ('answer_ok','status_ok','evidence_ok','approved')) and type(decision['reason']) is str
    consistent=bool(typed and all(decision[k]==v for k,v in expected.items()))
    all_good=all(expected.values())
    reasons={code for key,code in [('answer_ok','answer_mismatch'),('status_ok','status_mismatch'),('evidence_ok','unsupported_evidence')] if not expected[key]}
    reason_ok=bool(typed and decision['reason'] in ({'verified'} if all_good else reasons))
    approval_ok=bool(typed and decision['approved']==all_good)
    return {'schema_ok':bool(schema),'typed_fields':bool(typed),'independent_checks_match':consistent,
            'reason_matches_evidence':reason_ok,'approval_matches_evidence':approval_ok,
            'expected_checks':expected,'valid':bool(consistent and reason_ok and approval_ok)}
