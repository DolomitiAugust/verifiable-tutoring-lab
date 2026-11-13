"""Bounded post-hoc deployment calibration, never a new training run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
training=HERE.parent
PROJECT=HERE.parents[3]
sys.path.insert(0,str(training))
import train as train
import evaluate as evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class MixedProjection(nn.Module):
    """Mix effective LoRA updates B@A, rather than multiply interpolated A/B."""
    def __init__(self,original,sft_a,sft_b,dpo_a,dpo_b,scale):
        super().__init__()
        self.original=original
        for name,value in [('sft_a',sft_a),('sft_b',sft_b),('dpo_a',dpo_a),('dpo_b',dpo_b)]:
            self.register_buffer(name,value.detach().clone())
        self.scale=scale
        self.fraction=0.0

    def forward(self,x):
        base=self.original(x)
        if self.fraction==0:
            delta=F.linear(F.linear(x.float(),self.sft_a),self.sft_b)
        elif self.fraction==1:
            delta=F.linear(F.linear(x.float(),self.dpo_a),self.dpo_b)
        else:
            sft=F.linear(F.linear(x.float(),self.sft_a),self.sft_b)
            dpo=F.linear(F.linear(x.float(),self.dpo_a),self.dpo_b)
            delta=(1-self.fraction)*sft+self.fraction*dpo
        return base+(self.scale*delta).to(base.dtype)


def generate(model,tokenizer,rows,penalty):
    result=[]
    endings=sorted({tokenizer.convert_tokens_to_ids('<|im_end|>'),tokenizer.eos_token_id})
    for row in rows:
        x=torch.tensor([row['prefix']])
        start=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=192,
                do_sample=False,num_beams=1,use_cache=True,eos_token_id=endings,
                pad_token_id=tokenizer.pad_token_id,repetition_penalty=penalty,no_repeat_ngram_size=0)
        ids=out[0,len(row['prefix']):].tolist();eos=bool(ids and ids[-1] in endings)
        text=tokenizer.decode(ids[:-1] if eos else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        result.append({'candidate_id':row['candidate_id'],'family':row['family'],'task':row['task'],
                       'prompt':row['prompt'],'reference':row['answers'][0],'expected_main_answer':row['answer'],
                       'raw_response':text,'generated_token_ids':ids,'new_tokens':len(ids),
                       'stop_reason':'eos' if eos else 'max_new_tokens','seconds':time.perf_counter()-start,
                       'primary_value_proxy':evaluate.primary_answer(text,row['answer'])})
        print('pilot',penalty,row['candidate_id'],len(ids),'eos',eos,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'run_001');args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh output folder; previous evidence is preserved.')
    source=training/'retry_imports/retry_048f9a92c0ef_numeric_review/retry_results'
    sft_path=source/'checkpoints/selected_sft.pt';dpo_path=source/'runs/dpo/adapter.pt'
    dev=PROJECT/'synthesis/data/legacy/dev.jsonl';meta=dev.with_name('dev_metadata.jsonl')
    fractions=[0.0,0.05,0.1,0.25,1.0]
    raw=evaluate.load_examples(dev,meta)
    pilot=[]
    for family in sorted({r['family'] for r in raw}):
        for task in ('predict','explain'):
            pilot.append(next(r for r in raw if r['family']==family and r['task']==task))
    plan={'kind':'Post-hoc effective-LoRA-update interpolation, no optimizer/training.',
          'equation':'DeltaW(lambda) = (1-lambda)*DeltaW_SFT + lambda*DeltaW_DPO at each Q/V projection',
          'fractions':fractions,'selection':'minimum full-reference micro NLL on all 36 dev rows; tie within 1e-6 favors smaller fraction',
          'pilot_ids':[r['candidate_id'] for r in pilot],'pilot_penalties':[1.0,1.08],'max_new_tokens':192,
          'pilot_rule':'Inspect full text, explanation facts and unresolved outputs; fewer repetitions/EOS alone never counts as a fix. Stop if pilot has clear factual/teaching failures; no test-set tuning.',
          'scope':'Exploratory after both prior tests were seen; dev has only 4 AST structures. Not blind and not independent confirmation.',
          'inputs':{str(p):sha(p) for p in [sft_path,dpo_path,dev,meta,Path(__file__),training/'train.py',training/'evaluate.py',training/'data_io.py',training/'base_source.json']},
          'device':'cpu','threads':4,'seed':20261108}
    args.out.mkdir(parents=True);save(args.out/'plan.json',plan)
    torch.manual_seed(20261108)
    model,tokenizer=train.load_base(PROJECT/'training/legacy/model/base','float32',4,'cpu')
    sft=torch.load(sft_path,map_location='cpu',weights_only=True);dpo=torch.load(dpo_path,map_location='cpu',weights_only=True)
    assert dpo['parent_sft_sha256']==sha(sft_path) and sft['settings']==dpo['settings']
    assert sft['settings']=={'rank':16,'alpha':32,'last_layers':24}
    assert sft['base_hash']==dpo['base_hash']==model._training_base_hash
    assert sft['stage']=='sft' and dpo['stage']=='dpo' and sft['finished'] and dpo['finished']
    modules=[]
    for i,block in enumerate(model.model.layers):
        for proj in ('q_proj','v_proj'):
            prefix=f'model.layers.{i}.self_attn.{proj}.'
            mix=MixedProjection(getattr(block.self_attn,proj),sft['weights'][prefix+'lora_A'],sft['weights'][prefix+'lora_B'],
                                dpo['weights'][prefix+'lora_A'],dpo['weights'][prefix+'lora_B'],sft['settings']['alpha']/sft['settings']['rank'])
            setattr(block.self_attn,proj,mix);modules.append(mix)
    model.requires_grad_(False);model.eval()
    encoded=evaluate.encode(raw,tokenizer,model.config.max_position_embeddings)
    candidates=[]
    for fraction in fractions:
        for m in modules:m.fraction=fraction
        start=time.perf_counter();scores=evaluate.score(model,tokenizer,encoded)
        nll=-sum(r['logp_sum'][0] for r in scores)/sum(r['response_tokens'][0] for r in scores)
        item={'fraction':fraction,'dev_nll':nll,'seconds':time.perf_counter()-start,'scores':scores}
        candidates.append(item);save(args.out/'dev_candidates.json',candidates)
        print('candidate',fraction,'dev NLL',nll,flush=True)
    best=candidates[0]
    for item in candidates[1:]:
        if item['dev_nll']<best['dev_nll']-1e-6:best=item
    save(args.out/'selection.json',{'fraction':best['fraction'],'dev_nll':best['dev_nll'],'rule':plan['selection']})
    for m in modules:m.fraction=best['fraction']
    pilot_encoded=evaluate.encode(pilot,tokenizer,model.config.max_position_embeddings)
    for penalty in plan['pilot_penalties']:
        rows=generate(model,tokenizer,pilot_encoded,penalty)
        save(args.out/f'pilot_penalty_{penalty}.json',{'fraction':best['fraction'],'penalty':penalty,'max_new_tokens':192,
              'source_plan_sha256':sha(args.out/'plan.json'),'rows':rows,
              'proxy_counts':dict(Counter(r['primary_value_proxy']['status'] for r in rows))})
    save(args.out/'completed.json',{'completed':True,'training_run':False,'test_set_read':False,
                                  'next':'Full-text review before any deployment or test evaluation; no automatic improvement claim.'})


if __name__=='__main__':main()
