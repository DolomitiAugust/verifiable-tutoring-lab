"""Redraw cost, cash and review-stress figures from recorded scenario assumptions.

These calculations are not customer, workload, revenue or learning observations.
The source report assets are preserved; generated variants use a separate folder.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GREY = '#0072B2', '#E69F00', '#8A9198'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def build(language):
    tr = lambda en, zh: en if language == 'en' else zh
    assumptions = read(ROOT/'business/business_assumptions.json')
    results = read(ROOT/'business/results/business_outputs.json')
    stress = read(ROOT/'business/validation/results/quality_cost_stress.json')
    reference = next(x for x in results['units'] if x['id'] == 'reference')
    names = [tr('Model API', '模型API'), tr('Dedicated cloud', '专属云资源'),
             tr('Human review', '人工复核'), tr('Initial onboarding', '首次接入'), tr('Incremental acquisition', '增量获客')]
    values = [reference['api'], reference['cloud'], reference['review'], reference['onboarding'],
              reference['assumptions']['cac_per_new_contract']]
    assert abs(sum(values)-23307.68) < 1e-7
    costs, ax = plt.subplots(figsize=(6.2, 3.0))
    bars = ax.barh(names, values, color=[GREY, GREY, ORANGE, ORANGE, GREY], height=.56)
    bars[3].set_hatch('///'); ax.invert_yaxis(); ax.set_xlim(0, 9800)
    for bar, value in zip(bars, values):
        ax.text(value+110, bar.get_y()+bar.get_height()/2, f'{value:,.2f}'.rstrip('0').rstrip('.'), va='center', fontsize=8)
    ax.set_xlabel(tr('Cost per full-year new contract (CNY; assumed)', '每份全年新合同费用（元；假设）'))
    ax.set_title(tr('Baseline delivery and acquisition costs', '基准交付与获客成本'), loc='left', fontsize=10)
    costs.subplots_adjust(left=.27, right=.96, bottom=.21, top=.85)

    cash, ax = plt.subplots(figsize=(6.2, 3.1))
    for scenario, color, marker, style in zip(results['cash_scenarios'], [GREY, BLUE, ORANGE], ['o', 's', '^'], [':', '-', '--']):
        rows = scenario['months']; lag = scenario['collection_lag_months']
        x = [0]+[r['month'] for r in rows]
        y = [assumptions['cash_plan']['fundraise_cny']/10000]+[r['closing_cash']/10000 for r in rows]
        ax.plot(x, y, color=color, marker=marker, linestyle=style, markersize=3,
                label=tr(f'{lag}-month collection lag', f'回款延迟{lag}个月'))
        ax.text(12.2, y[-1], f'{y[-1]:.2f}', va='center', fontsize=8)
    ax.set(xlim=(0, 13.5), ylim=(0, 105), xticks=range(0, 13, 2),
           xlabel=tr('Months after assumed funding', '假设融资后的月份'),
           ylabel=tr('Closing cash (10,000 CNY)', '月末现金（万元）'))
    ax.set_title(tr('Scenario: annual contracts start in months 4, 7 and 10', '情景：第4、7、10月启用年合同'), loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8);cash.subplots_adjust(left=.14, right=.95, bottom=.22, top=.85)

    review, (left, right) = plt.subplots(1, 2, figsize=(6.2, 3.2))
    rows = stress['rows']; x = [r['review_fraction']*100 for r in rows]
    y = [r['contribution_after_incremental_cac_cny']/10000 for r in rows]
    zero = -y[0]*100/(y[-1]-y[0]); assert abs(zero-17.028493827160494) < 1e-8
    for ax in [left, right]:
        ax.plot(x, y, color=BLUE, marker='o', markersize=3, lw=1.4)
        ax.axhline(0, color='#777777', ls='--', lw=.8)
    left.set(xlim=(-2, 102), ylim=(-30, 7), xticks=[0, 25, 50, 75, 100],
             ylabel=tr('Contribution (10,000 CNY)', '获客后贡献（万元）'))
    left.set_title(tr('A  Full stress range', 'A  完整压力范围'), loc='left', fontsize=10)
    right.set(xlim=(-.7, 25.7), ylim=(-3.5, 7), xticks=[0, 5, 10, 15, 20, 25])
    right.set_title(tr('B  Critical range enlarged', 'B  关键区间放大'), loc='left', fontsize=10)
    right.axvline(zero, color=ORANGE, ls=':', lw=1.1);right.scatter([zero], [0], color=ORANGE, marker='D', s=22)
    right.annotate(tr(f'{zero:.2f}%: zero contribution', f'{zero:.2f}%：贡献为零'), xy=(zero, 0), xytext=(3, -2.5),
                   fontsize=7, arrowprops={'arrowstyle': '-', 'lw': .7})
    review.text(.5, .13, tr('Human-review turn share (assumed, %)', '人工复核轮次比例（假设，%）'), ha='center', fontsize=8)
    review.text(.1, .025, tr('Cost sensitivity only; model fallback is not a measured human-review rate.',
                            '仅为成本敏感性；模型回退率不是实测人工复核率。'), fontsize=7)
    review.subplots_adjust(left=.14, right=.97, bottom=.29, top=.84, wspace=.35)
    return [('business-costs', costs), ('business-cash', cash), ('business-review-stress', review)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    parser.add_argument('--outdir', type=Path, default=ROOT/'reports/generated')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans' if args.language == 'en' else 'Microsoft YaHei',
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    output = args.outdir/args.language;output.mkdir(parents=True, exist_ok=True)
    for name, fig in build(args.language):
        for extension in ['pdf', 'svg', 'png']:fig.savefig(output/(name+'.'+extension), dpi=300, facecolor='white')
        plt.close(fig)
    print(output)


if __name__ == '__main__':main()
