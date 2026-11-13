"""Recompute read-only data diagnostics from frozen inputs; no training."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]
DATA = PROJECT / 'synthesis/dataset'
training = PROJECT / 'revisions/v2/training'
TASKS = ('predict', 'explain', 'diagnose', 'transfer')
REASONS = ('factual_error', 'instruction_mismatch')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_key(code):
    """Published v2 metric: name-normalized AST, constants replaced by type."""
    tree = ast.parse(code)
    names = {}
    excluded = {'print', 'range', 'sum', 'len', 'list', 'dict', 'set', 'sorted',
                'reversed', 'enumerate', 'zip', 'any', 'all'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in excluded:
            node.id = names.setdefault(node.id, 'v' + str(len(names)))
        if isinstance(node, ast.arg):
            node.arg = names.setdefault(node.arg, 'v' + str(len(names)))
        if isinstance(node, ast.FunctionDef):
            node.name = names.setdefault(node.name, 'v' + str(len(names)))
        if isinstance(node, ast.Constant):
            node.value = f'<{type(node.value).__name__}>'
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def table(rows):
    count = Counter((row['reason'], row['task']) for row in rows)
    return {reason: {task: count[reason, task] for task in TASKS} for reason in REASONS}


def main():
    source_names = ['sft.jsonl', 'preference.jsonl', 'dev.jsonl',
                    'independent_test.jsonl', 'independent_test_metadata.jsonl',
                    'preference_eval_metadata.jsonl', 'audit/candidates.jsonl', 'stats.json']
    paths = [DATA/name for name in source_names] + [training/'data_io.py', PROJECT/'synthesis/synthesis_legacy/build_data.py']
    hashes_before = {str(path): sha(path) for path in paths}
    ledger = read_rows(DATA/'audit/candidates.jsonl')
    selected = {split: [r for r in ledger if r.get('split') == split and
                       r.get('decision') == ('evaluation_only' if split == 'dev' else 'selected')]
                for split in ('sft', 'preference', 'dev')}
    sft, pref, dev, natural = [read_rows(DATA/name) for name in
                              ('sft.jsonl', 'preference.jsonl', 'dev.jsonl', 'independent_test.jsonl')]
    texts = [r['messages'][1]['content'] for r in sft]
    assert len(selected['dev']) == len(dev) == 36
    for row in selected['dev']:
        exported = dev[row['export_line'] - 1]['messages']
        assert exported == [{'role': 'user', 'content': row['prompt']},
                            {'role': 'assistant', 'content': row['chosen']}]
    assert len(set(structural_key(r['code']) for r in selected['dev'])) == 4
    recomputed = [structural_key(r['code']) for r in selected['sft']]
    mismatches = [r['candidate_id'] for r, key in zip(selected['sft'], recomputed) if key != r['structure_key']]
    assert not mismatches
    assert len(set(recomputed)) == 145
    source = (training/'data_io.py').read_text(encoding='utf8')
    response_function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'encode_response')
    response_source = '\n'.join(source.splitlines()[response_function.lineno-1:response_function.end_lineno])
    suspicious = ['OMATIC', 'UIAlert', 'MemoryWarning', 'DonaldTrump', 'DUCTON']
    model_repetition = {}
    eval_paths = []
    for run, folder in [('first', training/'imports/run_5bf9f4d325f1/results'),
                        ('retry', training/'retry_imports/retry_048f9a92c0ef_numeric_review/retry_results')]:
        model_repetition[run] = {}
        for stage in ('base', 'sft', 'dpo', 'continued_sft'):
            path = folder/'evaluation'/f'{stage}_independent.json'
            eval_paths.append(path)
            obj = json.loads(path.read_text(encoding='utf8'))
            counts = [r['raw_response'].count('本题输出') for r in obj['generation']]
            model_repetition[run][stage] = {
                'cases': len(counts), 'label_occurrences_at_least_2': sum(n >= 2 for n in counts),
                'occurrence_count_distribution': dict(sorted(Counter(counts).items())),
                'source': str(path), 'sha256': sha(path)}
    report = {
        'schema': 'training-data-diagnosis-20261108-v1',
        'scope': 'Read-only descriptive diagnosis, not causal identification; no new training or model generation.',
        'reviewer': 'Codex separate data-diagnosis agent',
        'source_sha256': hashes_before,
        'sft': {
            'examples': len(sft), 'unique_program_signatures': len(set(r['program_signature'] for r in selected['sft'])),
            'ast_metric': 'Same v2 published structure_key: normalize non-excluded names in ast.walk order, replace ast.Constant.value by its Python type marker, then SHA256(ast.dump).',
            'recomputed_ast_structures': len(set(recomputed)),
            'ledger_ast_structures': len(set(r['structure_key'] for r in selected['sft'])),
            'structure_key_mismatches': mismatches,
            'maximum_examples_per_ast_structure': max(Counter(recomputed).values()),
            'unique_exact_completions': len(set(texts)),
            'exact_completion_duplicate_excess': len(texts)-len(set(texts)),
            'completion_label_occurrence_counts': dict(sorted(Counter(t.count('本题输出：') for t in texts).items())),
            'completions_with_adjacent_equal_lines': sum(any(a == b for a, b in zip(t.splitlines(), t.splitlines()[1:])) for t in texts),
            'completions_containing_observed_garbage_fragments': {term: sum(term in t for t in texts) for term in suspicious},
            'prompts_without_python_code_fence': sum('```python' not in r['messages'][0]['content'] for r in sft),
            'reserved_chat_markers_in_completions': sum(any(term in t for term in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')) for t in texts),
            'numeric_normalization_only_note': 'Optional regular-expression diagnostic below is not an AST count and must not replace the 145-structure metric.',
            'unique_completions_after_numeric_regex_normalization': len(set(re.sub(r'[-+]?\d+(?:\.\d+)?', '<N>', t) for t in texts)),
        },
        'preference_training_reason_by_task': table(selected['preference']),
        'preference_test_reason_by_task': table(read_rows(DATA/'preference_eval_metadata.jsonl')),
        'preference_interpretation': '243/250 (97.2%) factual-error pairs occur in diagnose/transfer; all instruction-mismatch pairs occur in predict/explain. The selector balances task counts, while the generator limits instruction-mismatch to predict/explain. This confounds task and negative type; it does not prove the cause of DPO degradation.',
        'distribution': {
            'prompt_char_median': {name: statistics.median(len(r['messages'][0]['content']) for r in rows) for name, rows in [('sft', sft), ('dev', dev), ('natural', natural)]},
            'response_char_median': {name: statistics.median(len(r['messages'][1]['content']) for r in rows) for name, rows in [('sft', sft), ('dev', dev), ('natural', natural)]},
            'dev_examples': len(dev), 'dev_ast_structures': len(set(r['structure_key'] for r in selected['dev'])),
            'dev_families': dict(Counter(r['family'] for r in selected['dev'])),
            'natural_authorship': 'Same Codex project, individually authored natural requests; not external human data.',
        },
        'eos_serialization_evidence': {
            'source': str(training/'data_io.py'), 'sha256': sha(training/'data_io.py'),
            'function': 'encode_response', 'line_start': response_function.lineno, 'line_end': response_function.end_lineno,
            'verbatim_source': response_source,
            'finding': 'Source explicitly appends tokenizer.convert_tokens_to_ids("<|im_end|>") to every encoded assistant response. This is code evidence, not a fresh tokenization or training run.',
        },
        'generated_label_repetition': model_repetition,
        'calculation_notes': [
            'Counts use frozen JSONL rows; training ledger rows are decision=selected. Published dev rows are marked decision=evaluation_only; all 36 dev export pointers/content are matched independently.',
            'The reason-by-task table includes explicit zero cells for both reasons and all four task labels.',
            'Exact duplicate excess is total examples minus distinct full response strings; it is not duplicated-program count.',
            'Adjacent equal lines are literal string equality, not an overall semantic repetition measure.',
            'The generated-label statistic counts literal 本题输出 occurrences in each complete raw_response; it is not the audited four-token repetition metric.',
            'Absence of listed garbage fragments in source does not exclude every possible data-quality issue.',
        ],
        'bounded_remedies': [
            'Build a separately versioned reason-by-task cross-balanced preference set; support instruction mismatches for diagnose/transfer and factual-error alternatives for predict/explain.',
            'Diversify request/response forms and preserve a small general-instruction mixture; verify references and EOS while keeping the old frozen release.',
            'Use a prespecified development suite spanning training-domain novel instances, held-out families, factual explanation and termination; do not tune on the already observed 24 natural tests.',
            'These data remedies require a separately evaluated new training run; decoding stops alone cannot correct a wrong first value or explanation.',
        ],
    }
    assert hashes_before == {str(path): sha(path) for path in paths}
    report['input_files_unchanged_after_recomputation'] = True
    (OUT/'data_diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (OUT/'data_diagnosis.md').write_text('''# 数据诊断：有限证据与可做的修正

冻结数据中没有模型输出里常见的乱码尾巴：1000条SFT均恰好含一次“本题输出”，没有相邻重复行，序列化代码也明确追加结束符。因此，不能把生成重复直接归因于数据复制了坏尾巴或漏加EOS。

数据仍较窄：1000个程序只有145个同口径AST结构，所有题都使用代码与固定输出标签。偏好对还有明显耦合：250个事实负例中243个属于诊断或迁移；250个指令负例则全部属于预测或解释。开发集36条也只有4种程序结构。这些现象可能限制泛化，但尚不能证明它们就是追加训练变差的原因。

可做的修正是另建版本，交叉均衡任务与负例原因，增加自然表达、答案样式及少量通用指令，并预先登记更全面的开发检查。修正是否有效仍需新实验验证；本诊断未修改原数据，也没有运行新训练或生成。
''', encoding='utf8')
    print(json.dumps({'sft': len(sft), 'ast': len(set(recomputed)), 'reason_by_task': table(selected['preference']), 'unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
