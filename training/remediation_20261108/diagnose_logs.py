"""Reproduce diagnostic summaries without changing any measured run."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
training=HERE.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output={'scope':'Descriptive diagnostics; no single-factor causal attribution.', 'runs':{}}
    for name,directory in [('first',training/'imports/run_5bf9f4d325f1/results'),
                           ('retry',training/'retry_imports/retry_048f9a92c0ef_numeric_review/retry_results')]:
        source=directory/'runs/dpo/metrics.jsonl'
        rows=[json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
        tail=rows[-100:]
        chosen=[r['policy_chosen_logp']-r['reference_chosen_logp'] for r in tail]
        rejected=[r['policy_rejected_logp']-r['reference_rejected_logp'] for r in tail]
        output['runs'][name]={'dpo_log_sha256':sha(source),'steps':len(rows),
           'last_100_both_logps_below_frozen_reference':sum(a<0 and b<0 for a,b in zip(chosen,rejected)),
           'last_100_mean_chosen_logp_change':sum(chosen)/len(chosen),
           'last_100_mean_rejected_logp_change':sum(rejected)/len(rejected),
           'final_logged_step':rows[-1],
           'dev_epoch_nll':[json.loads((directory/f'evaluation/sft_epoch_{i}_dev.json').read_text(encoding='utf8'))['summary']['reference_nll'] for i in (1,2,3)]}
    output['implementation_checks']={
        'response_labels':'prompt labels=-100, response labels active with causal shift',
        'eos':'encode_response appends atomic im_end',
        'dpo':'negative logsigmoid(beta*((chosen-rejected)-(reference_chosen-reference_rejected)))',
        'sft_selection':'minimum dev NLL among epochs 1,2,3; no intermediate sub-epoch SFT selection',
        'dpo_selection':'fixed 500 steps; no intermediate DPO dev checkpoints or early stopping in the frozen original run',
        'code_sha256':{p.name:sha(p) for p in [training/'train.py',training/'data_io.py',training/'run_retry.py']}}
    output['sources']=[{'url':'https://arxiv.org/abs/2305.18290','kind':'original DPO paper; relative likelihood objective'},
                       {'url':'https://huggingface.co/docs/trl/dpo_trainer','kind':'official trainer documentation; loss and score diagnostics; accessed 2026-11-08'}]
    (HERE/'training_diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{key:value for key,value in v.items() if key not in ('final_logged_step',)} for k,v in output['runs'].items()},ensure_ascii=False))


if __name__=='__main__':main()
