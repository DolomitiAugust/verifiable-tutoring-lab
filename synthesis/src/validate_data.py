"""Independent export audit: stdlib only; imports no generator or renderer.

Reads frozen exports and ledger, executes programs, checks answer markers, split
separation and file hashes. Teaching quality and human preference are NOT certified.
"""
import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'dataset'
TAG='本题输出：'; TRANS='变式参考输出：'

def read(p): return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def answer(text,tag=TAG):
    found=[x[len(tag):].strip() for x in text.splitlines() if x.startswith(tag)]
    return found[0] if len(found)==1 else None
def run_program(source):
    tree=ast.parse(source)
    forbidden=(ast.Import,ast.ImportFrom,ast.With,ast.AsyncWith,ast.ClassDef)
    if any(isinstance(n,forbidden) for n in ast.walk(tree)): raise ValueError('Unexpected program capabilities')
    if any(isinstance(n,ast.Name) and n.id in ('open','eval','exec','compile','input','__import__') for n in ast.walk(tree)): raise ValueError('Unexpected dangerous name')
    out=io.StringIO()
    with redirect_stdout(out): exec(compile(tree,'<independent-export-audit>','exec'),{})
    return out.getvalue().strip()
def canonical(source,structure=False):
    # Independent AST canonicalization; retain constants for leakage checks.
    tree=ast.parse(source); names={}
    class Rename(ast.NodeTransformer):
        def visit_Name(self,n):
            if isinstance(n.ctx,(ast.Store,ast.Param)) or n.id in names:
                n.id=names.setdefault(n.id,'identifier'+str(len(names)))
            return n
        def visit_arg(self,n): n.arg=names.setdefault(n.arg,'identifier'+str(len(names))); return n
        def visit_FunctionDef(self,n):
            n.name=names.setdefault(n.name,'identifier'+str(len(names)))
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if structure: n.value=f'<{type(n.value).__name__}>'
            return n
    return ast.dump(Rename().visit(tree),include_attributes=False)
def describe(rows):
    pairs=[r for r in rows if 'rejected' in r]
    lengths=[len(r['chosen']) for r in rows]
    markers=("sep='-'","end=''","print('A#B')")
    pos=sum(len(r['chosen'])>len(r['rejected']) for r in pairs)
    ties=sum(len(r['chosen'])==len(r['rejected']) for r in pairs)
    return {'n':len(rows),'response_chars_median':statistics.median(lengths),
            'response_chars_mean':statistics.mean(lengths),
            'program_structures_same_ast_metric':len({canonical(r['code'],True) for r in rows}),
            'length_only_accuracy':(pos+.5*ties)/len(pairs) if pairs else None,
            'chosen_shorter':sum(len(r['chosen'])<len(r['rejected']) for r in pairs),
            'equal_length':ties,
            'legacy_marker_chosen':sum(any(m in r['chosen'] for m in markers) for r in rows),
            'legacy_marker_rejected':sum(any(m in r.get('rejected','') for m in markers) for r in rows)}

def validate():
    stats=json.loads((DATA/'stats.json').read_text(encoding='utf8'))
    errors=[]; checks=Counter()
    def check(name,ok,spot=''):
        checks[name]+=1
        if not ok: errors.append({'check':name,'spot':spot})
    for f,digest in stats['sha256'].items(): check('published_sha256',sha(DATA/f)==digest,f)
    ledger=read(DATA/'audit/candidates.jsonl')
    selected={s:sorted([r for r in ledger if r['split']==s and r['decision'] in ('selected','evaluation_only')],key=lambda r:r['export_line']) for s in ('sft','preference','evaluation','dev','preference_eval')}
    signatures={}; families={}; prompts={}
    for split,rows in selected.items():
        exports=read(DATA/(split+'.jsonl')); signatures[split]=set(); families[split]=set(); prompts[split]=set()
        check('count_'+split,len(rows)==len(exports)==stats['selected'][split]['count'])
        check('pointers_'+split,[r['export_line'] for r in rows]==list(range(1,len(rows)+1)))
        for row,export in zip(rows,exports):
            ident=row['candidate_id']; families[split].add(row['family']); prompts[split].add(row['prompt'])
            expected={k:row[k] for k in ('prompt','chosen','rejected')} if split.startswith('preference') else {'messages':[{'role':'user','content':row['prompt']},{'role':'assistant','content':row['chosen']}]}
            check('export_matches_ledger',export==expected,ident)
            check('main_program_exec',run_program(row['code'])==row['answer'],ident)
            check('transfer_program_exec',run_program(row['transfer_code'])==row['transfer_answer'],ident)
            check('program_in_prompt',row['code'] in row['prompt'],ident)
            check('chosen_answer',answer(row['chosen'])==row['answer'],ident)
            if split!='evaluation' and row['task']=='transfer':
                check('chosen_transfer_answer',answer(row['chosen'],TRANS)==row['transfer_answer'],ident)
                check('informative_transfer',row['answer']!=row['transfer_answer'],ident)
            if split.startswith('preference'):
                check('distinct_pair',row['chosen']!=row['rejected'],ident)
                correct_negative=row['wrong'] if row['reason']=='factual_error' else row['answer']
                check('negative_answer_matches_documented_type',answer(row['rejected'])==correct_negative,ident)
                if row['reason']=='instruction_mismatch':
                    wanted=1 if row['task']=='predict' else 2
                    check('instruction_pair_direction',len(row['chosen'].splitlines())==1 if wanted==1 else len(row['chosen'].splitlines())>1,ident)
                    check('instruction_pair_negative',len(row['rejected'].splitlines())>1 if wanted==1 else len(row['rejected'].splitlines())==1,ident)
            signatures[split].update((canonical(row['code']),canonical(row['transfer_code'])))
    overlaps={}
    splits=list(selected)
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            overlaps[a+'/'+b]=len(signatures[a]&signatures[b])
            check('no_program_overlap',not(signatures[a]&signatures[b]),a+'/'+b)
            check('no_prompt_overlap',not(prompts[a]&prompts[b]),a+'/'+b)
    for train in ('sft','preference'):
        for test in ('evaluation','dev','preference_eval'):
            check('family_separation',not(families[train]&families[test]),train+'/'+test)
    check('frozen_v1_bridge',sha(DATA/'evaluation.jsonl')==sha(ROOT.parent/'data/legacy/evaluation.jsonl'))
    natural_path=DATA/'independent_test_metadata.jsonl'
    if natural_path.exists():
        natural=read(natural_path)
        natural_exports=read(DATA/'independent_test.jsonl')
        manifest=json.loads((DATA/'independent_test_manifest.json').read_text(encoding='utf8'))
        for file,digest in manifest['sha256'].items(): check('natural_file_hash',sha(DATA/file)==digest,file)
        check('natural_count',len(natural)==len(natural_exports)==24)
        known=set().union(*signatures.values())
        for row,exp in zip(natural,natural_exports):
            check('natural_reference_execution',run_program(row['code'])==row['answer'],row['candidate_id'])
            check('natural_no_exact_program_overlap',canonical(row['code']) not in known,row['candidate_id'])
            check('natural_export_match',exp['messages']==[{'role':'user','content':row['prompt']},{'role':'assistant','content':row['chosen']}],row['candidate_id'])
            check('natural_gold_marker',answer(row['chosen'])==row['answer'],row['candidate_id'])
    old=read(ROOT.parent/'data/legacy/audit/candidates.jsonl')
    comparison={s:{'v1':describe([r for r in old if r['split']==s and r['decision']=='selected']),
                    'v2':describe(selected[s])} for s in ('sft','preference')}
    out={'passed':not errors,'checks':dict(checks),'errors':errors,
         'cross_split_program_overlap':overlaps,'comparison':comparison,
         'reason_summary':{r:describe([x for x in selected['preference'] if x['reason']==r]) for r in stats['preference_reasons']},
         'independence':'Auditor does not import curriculum/generation code. Program execution supplies fact checks; semantic teaching quality is not independently rated.',
         'limits':['Deterministic synthetic data; no real students or teacher blind ratings.',
                   'Only predefined legacy markers and length shortcut were measured; other shortcuts may remain.',
                   'Instruction-mismatch pairs intentionally test explicit concise-vs-explanation requirements.',
                   'No causal claim about model/learning improvement until downstream controlled evaluation.']}
    (ROOT.parent/'outputs').mkdir(exist_ok=True)
    (ROOT.parent/'outputs/data_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':out['passed'],'checks':sum(checks.values()),'errors':errors[:10],'comparison':comparison},ensure_ascii=False))
    return 0 if out['passed'] else 1

if __name__=='__main__': sys.exit(validate())
