"""Versioned release. No model generation; deterministic authorship is disclosed.

Run before any training. Existing v1 files are only read. The validator does not
import this module or a curriculum renderer.
"""
import ast
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import re
import statistics
import sys

sys.dont_write_bytecode = True
from curriculum_extended import make_case, TRAIN_FAMILIES, TEST_FAMILIES, DEV_FAMILIES

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset'
OLD=ROOT.parent/'data/legacy'
SEED=2026110701
TASKS=('predict','explain','diagnose','transfer')
STEMS={
 'predict':('只告诉我这段程序最后打印什么，不用解释。','我先核对答案：这段代码的输出是什么？只写结果。','先不展开过程，请给出程序输出。','请直接回答输出值，不附加讲解。','我已经推过一遍，只想确认最终输出。','先帮我对答案，用一行写出打印结果。'),
 'explain':('这段代码为什么会得到这个结果？请指出关键一步。','请算出输出，再用一两句话解释原因。','我刚学这个知识点，请结合代码讲清输出是怎么来的。','我会照着运行，但还不懂原因。请解释决定结果的那一步。','请给出结果和关键计算，不用重复整段代码。','帮我顺着程序看一下：输出是什么，依据是什么？','我想核对自己的理解，请给出输出并说明关键规则。','别只报答案，请用代码中的实际值解释结果。'),
 'diagnose':('我预测会输出{wrong}。请核对，并指出我可能在哪一步弄错。','我的答案是{wrong}，但不太确定。请告诉我实际结果，解释差别。','我把结果算成了{wrong}。能结合这段代码定位可能的误解吗？','我算出来是{wrong}。请给出正确结果，并帮我检查思路。','同学认为输出是{wrong}。请评估这个预测，并给出依据。','请帮我诊断：我预期得到{wrong}，这和程序真正做的事情一致吗？'),
 'transfer':('先解释原程序的输出，再回答这个变化：{transfer}','请用原题和变式做对照。变式是：{transfer}','我想确认是否真懂了。请给出原题结果和理由，再解答：{transfer}','请简要说明原题结果，并检查下面的改动：{transfer}','原程序会输出什么？为什么？随后再看：{transfer}','请先给出原结果，再用同样规则处理这个变化：{transfer}')}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def lines(path,rows): path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')

def signature(code,skeleton=False):
    tree=ast.parse(code); names={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and node.id not in {'print','range','sum','len','list','dict','set','sorted','reversed','enumerate','zip','any','all'}:
            node.id=names.setdefault(node.id,'v'+str(len(names)))
        if isinstance(node,ast.arg): node.arg=names.setdefault(node.arg,'v'+str(len(names)))
        if isinstance(node,ast.FunctionDef): node.name=names.setdefault(node.name,'v'+str(len(names)))
        if skeleton and isinstance(node,ast.Constant): node.value=f'<{type(node.value).__name__}>'
    return sha(ast.dump(tree).encode())

def execute(code):
    out=io.StringIO()
    with redirect_stdout(out): exec(compile(code,'<bounded-authored-program>','exec'),{})
    return out.getvalue().strip()

def prompt(case,task,variant):
    request=STEMS[task][variant%len(STEMS[task])].format(**case)
    # Instruction order is independent of response label.
    block=f"```python\n{case['code']}\n```"
    return request+'\n'+block if variant%2 else block+'\n'+request

def response(case,task,variant,answer=None):
    answer=case['answer'] if answer is None else str(answer)
    output='本题输出：'+answer
    if task=='predict': return output
    trace=case['reasoning']
    if answer!=case['answer']:
        # Same surface construction on both sides; no negative-only vocabulary.
        # A false output remains an error even when the shared trace exposes it.
        trace=trace.replace(case['answer'],answer)
    if task=='diagnose':
        trace=case['misconception']+' '+trace
    if task=='transfer':
        trace+='\n变式参考输出：'+case['transfer_answer']
    return output+'\n'+trace if variant%2 else trace+'\n'+output

def make_row(rng,split,family,number):
    case_seed=rng.getrandbits(61); case=make_case(family,random.Random(case_seed))
    variant=rng.randrange(48)
    task=rng.choice(TASKS)
    reason=None
    if split in ('preference','preference_eval'):
        reason=rng.choice(('factual_error','instruction_mismatch'))
        if reason=='instruction_mismatch': task=rng.choice(('predict','explain'))
    p=prompt(case,task,variant); chosen=response(case,task,variant)
    row={**case,'candidate_id':f'V2-{split}-{number:05d}','split':split,'task':task,
         'profile':'unspecified','evidence':'incorrect' if task=='diagnose' else 'none',
         'case_seed':case_seed,'variant':variant,'prompt':p,'chosen':chosen,'reason':reason}
    if reason:
        row['rejected']=response(case,task,variant,case['wrong']) if reason=='factual_error' else response(case,'explain' if task=='predict' else 'predict',variant)
    row['decision']='eligible'
    # Program facts have a second source: actual Python execution.
    if execute(case['code'])!=case['answer']: row['decision']='incorrect_reference'
    elif execute(case['transfer_code'])!=case['transfer_answer']: row['decision']='incorrect_transfer'
    elif case['answer']==case['wrong']: row['decision']='ambiguous_negative'
    elif task=='transfer' and case['answer']==case['transfer_answer']: row['decision']='same_output_transfer'
    elif reason and row['chosen']==row['rejected']: row['decision']='identical_pair'
    row['program_signature']=signature(case['code'])
    row['transfer_signature']=signature(case['transfer_code'])
    row['structure_key']=signature(case['code'],True)
    return row

def export(row,preference=False):
    if preference: return {k:row[k] for k in ('prompt','chosen','rejected')}
    return {'messages':[{'role':'user','content':row['prompt']},{'role':'assistant','content':row['chosen']}]}

def build():
    DATA.mkdir(parents=True,exist_ok=True); (DATA/'audit').mkdir(exist_ok=True)
    rng=random.Random(SEED); ledger=[]; selected={}; used=set(); prompts=set()
    old_ledger=[json.loads(x) for x in (OLD/'audit/candidates.jsonl').read_text(encoding='utf8').splitlines()]
    bridge=sorted([x for x in old_ledger if x['split']=='evaluation' and x['decision']=='selected'],key=lambda x:x['export_line'])
    for row in bridge:
        row=dict(row,source_version='v1_frozen_bridge')
        row['program_signature']=signature(row['code']); row['transfer_signature']=signature(row['transfer_code'])
        row['structure_key']=signature(row['code'],True)
        ledger.append(row); used.update((row['program_signature'],row['transfer_signature'])); prompts.add(row['prompt'])
    selected['evaluation']=bridge
    # Fixed family counts and selection before model outputs. No rewriting after training.
    specs=(('sft',1000,TRAIN_FAMILIES),('preference',500,TRAIN_FAMILIES),
           ('dev',36,DEV_FAMILIES),('preference_eval',100,TEST_FAMILIES))
    for split,target,families in specs:
        bank=[]
        for i in range(target*5):
            row=make_row(rng,split,families[i%len(families)],i+1)
            if row['decision']=='eligible':
                keys={row['program_signature'],row['transfer_signature']}
                if keys & used or row['prompt'] in prompts: row['decision']='overlapping_program'
                else: used.update(keys); prompts.add(row['prompt'])
            bank.append(row)
        pool=[x for x in bank if x['decision']=='eligible']; counts=Counter(); picks=[]
        while len(picks)<target:
            if not pool: raise ValueError((split,'insufficient candidates',len(picks)))
            def priority(x):
                fact=counts['reason',x['reason']] if split.startswith('preference') else 0
                length=counts['direction',(len(x['chosen'])>len(x.get('rejected','')))-(len(x['chosen'])<len(x.get('rejected','')))] if split.startswith('preference') else 0
                return (fact,counts['family',x['family']],counts['task',x['task']],length)
            row=min(pool,key=priority); pool.remove(row)
            row['decision']='selected' if split in ('sft','preference') else 'evaluation_only'
            row['export_line']=len(picks)+1; picks.append(row)
            for k in ('reason','family','task'): counts[k,row[k]]+=1
            counts['direction',(len(row['chosen'])>len(row.get('rejected','')))-(len(row['chosen'])<len(row.get('rejected','')))]+=1
        for row in pool: row['decision']='quota_surplus'
        selected[split]=picks; ledger.extend(bank)
        lines(DATA/(split+'.jsonl'),[export(x,split.startswith('preference')) for x in picks])
        if split not in ('sft','preference'): lines(DATA/(split+'_metadata.jsonl'),picks)
    # Preserve bridge bytes, including original prompts and outputs.
    (DATA/'evaluation.jsonl').write_bytes((OLD/'evaluation.jsonl').read_bytes())
    lines(DATA/'audit/candidates.jsonl',ledger)
    stats={'version':'synthesis-v2','seed':SEED,'heldout_families':list(TEST_FAMILIES),
           'development_families':list(DEV_FAMILIES),'training_families':list(TRAIN_FAMILIES),
           'authorship':'Codex-authored deterministic programs and templates; no external model API, teacher ratings or student observations.',
           'candidate_count':len(ledger),'decisions':dict(Counter(x['decision'] for x in ledger)),
           'selected':{s:{'count':len(v),'families':dict(Counter(x['family'] for x in v)),
                         'tasks':dict(Counter(x['task'] for x in v)),
                         'program_structures':len({x['structure_key'] for x in v}),
                         'response_chars_median':statistics.median(len(x['chosen']) for x in v)} for s,v in selected.items()},
           'preference_reasons':dict(Counter(x['reason'] for x in selected['preference'])),
           'sha256':{name:sha((DATA/name).read_bytes()) for name in (
               'dev.jsonl','dev_metadata.jsonl','evaluation.jsonl','preference.jsonl',
               'preference_eval.jsonl','preference_eval_metadata.jsonl','sft.jsonl','audit/candidates.jsonl')}}
    dump(DATA/'stats.json',stats)
    dump(DATA/'split_manifest.json',{'seed':SEED,'frozen_before_training':True,
         'policy':{'sft':'training only','preference':'training only',
                   'dev':'epoch selection by NLL only; 3 new families absent from training',
                   'evaluation':'unchanged v1 bridge test; never train or tune',
                   'preference_eval':'100 new pairs from frozen test families; final evaluation only'},
         'sha256':{**stats['sha256'],'stats.json':sha((DATA/'stats.json').read_bytes())}})
    print(json.dumps({'counts':{s:len(v) for s,v in selected.items()},'decisions':stats['decisions']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=DATA)
    args=parser.parse_args()
    DATA=args.output_dir.resolve()
    if (DATA/'stats.json').exists():
        raise SystemExit('Release already exists. Use --output-dir with a new directory; frozen data must not be overwritten.')
    build()
