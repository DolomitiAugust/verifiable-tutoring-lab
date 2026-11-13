"""Redraw paired training figures from saved evaluation and full-text review records.

No weights are loaded and no metric is inferred from an answer-extraction proxy.
Outputs are regenerated variants; committed report assets are not overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIRST = ROOT/'training/imports/run_5bf9f4d325f1'
RETRY = ROOT/'training/retry_imports/retry_048f9a92c0ef_numeric_review/retry_results'
REVIEW = ROOT/'training/retry_imports/retry_048f9a92c0ef/new_full_text_fact_review.json'
STAGES = ['base', 'sft', 'dpo', 'continued_sft']
COLORS = ['#555555', '#0072B2', '#E69F00', '#009E73']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def nll(report):
    return np.array([row['mean_nll'][0] for row in report['reference_nll']])


def pair(left, right, count):
    a, b = left['reference_nll'], right['reference_nll']
    assert len(a) == len(b) == count
    assert [x['candidate_id'] for x in a] == [x['candidate_id'] for x in b]
    assert [x['response_tokens'] for x in a] == [x['response_tokens'] for x in b]


def build(language):
    tr = lambda en, zh: en if language == 'en' else zh
    labels = [tr('Base', '基础'), 'SFT', 'DPO', tr('Continued SFT', '继续SFT')]
    first = {s: {kind: read(FIRST/'results/evaluation'/f'{s}_{kind}.json')
                 for kind in ['bridge_nll', 'independent']} for s in STAGES}
    facts = read(FIRST/'full_text_fact_review.json')
    verdict = {(row['stage'], row['candidate_id']): row['judgement'] for row in facts['rows']}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.2), gridspec_kw={'width_ratios': [1.05, .95]})
    base = first['base']['bridge_nll']
    markers = ['o', '^', 's']
    for stage, color, marker in zip(STAGES[1:], COLORS[1:], markers):
        pair(base, first[stage]['bridge_nll'], 60)
        axes[0].scatter(nll(base), nll(first[stage]['bridge_nll']), s=13, alpha=.7,
                        color=color, marker=marker, label=labels[STAGES.index(stage)])
    upper = max(float(nll(first[s]['bridge_nll']).max()) for s in STAGES) + .2
    axes[0].plot([0, upper], [0, upper], '--', color='#777777', lw=.8)
    axes[0].set(xlim=(0, upper), ylim=(0, upper), xlabel=tr('Base reference NLL', '基础模型参考NLL'),
                ylabel=tr('Trained reference NLL', '训练后参考NLL'))
    axes[0].set_title(tr('(a) 60 paired bridge references', '(a) 60条配对桥接参考'), fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlim=(-.5, 3.5), ylim=(23.5, -.5), xticks=range(4), yticks=range(24))
    axes[1].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[1].set_yticklabels([str(i+1).zfill(2) for i in range(24)], fontsize=7)
    for x, stage in enumerate(STAGES):
        for y, row in enumerate(first[stage]['independent']['generation']):
            status = verdict[stage, row['candidate_id']]
            stop = 'E' if row['stop_reason'] == 'eos' else 'L'
            value = {'correct': 'C', 'wrong': 'W', 'unresolved': '?'}[status]
            axes[1].add_patch(plt.Rectangle((x-.48, y-.47), .96, .94, color='#edf0f2'))
            axes[1].text(x, y, stop+'/'+value, ha='center', va='center', fontsize=6.5)
    axes[1].set_title(tr('(b) 24 natural-question records', '(b) 24条自然题记录'), fontsize=10)
    fig.text(.1, .025, tr('E: EOS; L: token cap. C/W/?: full-text primary-output review, not teaching quality.',
                         'E为结束标记，L为长度上限；C/W/?为全文主值复核，不等于教学质量。'), fontsize=7)
    fig.subplots_adjust(left=.1, right=.97, bottom=.18, top=.9, wspace=.4)

    retry = {s: {kind: read(RETRY/'evaluation'/f'{s}_{kind}.json')
                 for kind in ['bridge_nll', 'independent']} for s in STAGES}
    facts_retry = read(REVIEW)
    second, ax = plt.subplots(1, 2, figsize=(7.0, 4.0), gridspec_kw={'width_ratios': [1, 1.15]})
    for x, kind, count in [(0, 'independent', 24), (1, 'bridge_nll', 60)]:
        pair(retry['sft'][kind], retry['dpo'][kind], count)
        delta = nll(retry['dpo'][kind])-nll(retry['sft'][kind])
        ax[0].scatter(x+np.linspace(-.15, .15, count), delta, s=12, alpha=.7, color=COLORS[1])
        ax[0].boxplot([delta], positions=[x], widths=.45, showfliers=False,
                      medianprops={'color': COLORS[2]})
    ax[0].axhline(0, color='#777777', ls='--', lw=.8)
    ax[0].set_xticks([0, 1], [tr('Natural\nn=24', '自然题\nn=24'), tr('Bridge\nn=60', '桥接题\nn=60')])
    ax[0].set_ylabel(tr('Reference NLL: retry DPO minus SFT', '参考NLL：追加DPO减SFT'))
    ax[0].set_title(tr('(a) Paired reference-fit change', '(a) 配对参考拟合变化'), fontsize=10)
    start = np.zeros(4)
    for status, color, hatch, label in [('correct', COLORS[3], '', tr('Correct', '对')),
                                       ('wrong', COLORS[2], '//', tr('Wrong', '错')),
                                       ('unresolved', '#d2d7dc', '..', tr('Unresolved', '未判定'))]:
        values = np.array([sum(x['stage'] == s and x['judgement'] == status for x in facts_retry['rows']) for s in STAGES])
        ax[1].barh(range(4), values, left=start, color=color, hatch=hatch, label=label, edgecolor='white')
        for y, (value, left) in enumerate(zip(values, start)):
            if value: ax[1].text(left+value/2, y, str(value), ha='center', va='center', fontsize=8)
        start += values
    assert list(start) == [24]*4
    ax[1].set_yticks(range(4), labels); ax[1].invert_yaxis()
    ax[1].set(xlim=(0, 24), xlabel=tr('Full-text primary-output counts', '全文主值复核题数'))
    ax[1].set_title(tr('(b) Recorded factual status', '(b) 实录事实状态'), fontsize=10)
    ax[1].legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=3, frameon=False, fontsize=7)
    second.subplots_adjust(left=.1, right=.97, bottom=.28, top=.88, wspace=.6)
    second.text(.1, .025, tr('One exploratory run; item distributions are not uncertainty across training seeds.',
                           '单次探索性运行；逐题分布不代表不同训练种子的统计不确定性。'), fontsize=7)
    return [('training-cloud-evidence', fig), ('training-retry-comparison', second)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    parser.add_argument('--outdir', type=Path, default=ROOT/'reports/generated')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans' if args.language == 'en' else 'Microsoft YaHei',
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    output = args.outdir/args.language; output.mkdir(parents=True, exist_ok=True)
    for name, fig in build(args.language):
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(output/(name+'.'+extension), dpi=300, facecolor='white')
        plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
