"""Render the data synthesis selection experiment from its paired permutation results."""

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports'
SOURCE = ROOT / 'synthesis_legacy/selection_experiment.json'
FIGURE = REPORT / 'assets/zh' / "selection-ablation.png"
LEDGER = ROOT / 'data/legacy' / "audit" / "candidates.jsonl"
SUPPLY_FIGURE = REPORT / 'assets/zh' / "family-supply.png"
POLICIES = {
    "A_random": ("A", "随机", "#0072B2", "o"),
    "B_labels": ("B", "标签均衡", "#E69F00", "s"),
    "C_length_labels": ("C", "长度与标签均衡", "#009E73", "^"),
}
REASONS = {
    "factual_error": "事实错误",
    "incomplete": "教学缺项",
    "off_target": "目标偏移",
}
INK = "#263746"


def load_experiment():
    experiment = json.loads(SOURCE.read_text(encoding="utf-8"))
    trials = experiment["trials"]
    if not trials or len(trials) != experiment["design"]["repeats"]:
        raise ValueError("Trial count differs from the declared number of permutations.")
    observations = {}
    for policy in POLICIES:
        rows = [t["policies"][policy]["metrics"] for t in trials]
        metrics = ["long_heuristic", *("short_rate_" + reason for reason in REASONS)]
        observations[policy] = {}
        for metric in metrics:
            values = [row[metric] for row in rows]
            expected = experiment["policies_summary"][policy][metric]["mean"]
            if not all(0 <= value <= 1 for value in values) or not math.isclose(fmean(values), expected):
                raise ValueError(f"Invalid or inconsistent metric: {policy}/{metric}")
            observations[policy][metric] = values
    return observations


def plot_style():
    available = {face.name for face in font_manager.fontManager.ttflist}
    families = ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                "PingFang SC", "SimHei", "WenQuanYi Zen Hei")
    font = next((name for name in families if name in available), None)
    if not font:
        raise RuntimeError("Install a Chinese sans-serif font such as Noto Sans CJK SC.")
    plt.rcParams.update({
        "font.family": font, "font.size": 10, "text.color": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#8C959C", "axes.linewidth": 0.7,
        "axes.spines.right": False, "axes.spines.top": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "grid.color": "#E6E9EC", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "savefig.bbox": None,
    })


def save_figure_pair(fig, output: Path, title: str, source: str):
    """Export the same artists as vector PDF and 300 dpi PNG, without cropping.

    Explicit canvas dimensions keep embedded paper text at its intended size.
    PDF Type 42 fonts embed searchable text; no graph artists are rasterized.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        destination = output.with_suffix(suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        metadata = ({"Title": title, "Subject": source, "Creator": "Matplotlib; Verifiable Tutoring Lab",
                     "CreationDate": None, "ModDate": None} if suffix == ".pdf"
                    else {"Title": title, "Source": source})
        try:
            fig.savefig(temporary, format=suffix[1:], dpi=300, facecolor="white", metadata=metadata)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def draw_long_rule(ax, observations):
    ax.grid(axis="y")
    ax.axhline(50, color="#66737D", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    ax.text(2.25, 50.9, "50% 参考", ha="right", va="bottom", fontsize=8.2, color="#596773")
    for index, (policy, (_, _, color, marker)) in enumerate(POLICIES.items()):
        values = [100 * v for v in observations[policy]["long_heuristic"]]
        # Jitter only horizontally. Every plotted ordinate is the observed value.
        offsets = [0.44 * ((i + 0.5) / len(values) - 0.5) for i in range(len(values))]
        random.Random(35 + index).shuffle(offsets)
        ax.scatter([index + dx for dx in offsets], values, marker=marker, s=13,
                   linewidths=0.65, facecolors="none", edgecolors=color, alpha=0.76, zorder=3)
        mean = fmean(values)
        ax.hlines(mean, index - 0.29, index + 0.29, color=INK, linewidth=1.4, zorder=4)
        ax.text(index, max(values) + 1.5, f"{mean:.2f}%", ha="center", va="bottom",
                color=INK, fontsize=9.3, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    ax.set_title("(a)  选长命中率", loc="left", pad=10, weight="bold")
    ax.set_xlim(-0.44, 2.44)
    ax.set_ylim(46.5, 92)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_xticks(range(3), [style[0] for style in POLICIES.values()])
    ax.set_ylabel("选长规则命中率（%）", labelpad=4)
    ax.set_xlabel("筛选策略", labelpad=4)


def draw_short_by_reason(ax, observations):
    ax.grid(axis="x")
    for category, reason in enumerate(REASONS):
        for row, (policy, (_, _, color, marker)) in enumerate(POLICIES.items()):
            mean = 100 * fmean(observations[policy]["short_rate_" + reason])
            y = 2 - category + (1 - row) * 0.22
            ax.scatter(mean, y, s=29, marker=marker, linewidths=0.7,
                       edgecolors=color, facecolors=color, zorder=3)
            ax.annotate(f"{mean:.2f}", (mean, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9.0, color=INK)
    for boundary in (0.5, 1.5):
        ax.axhline(boundary, color="#E3E8EB", linewidth=0.65, zorder=1)
    ax.set_title("(b)  按缺陷类别分组", loc="left", pad=10, weight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.49, 2.49)
    ax.set_yticks([2, 1, 0], list(REASONS.values()))
    ax.set_xlabel("该类中首选较短的比例（%）", labelpad=4)
    ax.tick_params(axis="y", length=0, pad=4)


def constrained_families():
    expected = json.loads(SOURCE.read_text(encoding="utf-8"))["family_supply_and_quota"]
    counts, topics = defaultdict(Counter), {}
    with LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "preference" or row["decision"] not in ("selected", "quota_surplus"):
                continue
            family = row["family"]
            topics[family] = row["topic"]
            counter = counts[family]
            difference = len(row["chosen"]) - len(row["rejected"])
            counter["eligible"] += 1
            counter["short" if difference < 0 else "long" if difference > 0 else "equal"] += 1
            if row["decision"] == "selected":
                counter["quota"] += 1
                counter["selected_short"] += difference < 0
    if counts.keys() != expected.keys():
        raise ValueError("Experiment and candidate ledger differ in eligible families.")
    limited = []
    for family, summary in expected.items():
        counter = counts[family]
        if any(counter[key] != value for key, value in summary.items()):
            raise ValueError(f"Supply or release quota differs from the experiment: {family}")
        supply, target = counter["short"], counter["quota"] // 2
        if supply < target:
            if counter["selected_short"] != supply:
                raise ValueError(f"Available and released short pairs require separate markers: {family}")
            limited.append((topics[family], supply, target))
    limited.sort(key=lambda row: row[2] - row[1], reverse=True)
    return limited, len(expected)


def draw_family_supply():
    rows, family_count = constrained_families()
    fig, ax = plt.subplots(figsize=(6.55, 2.85))
    fig.subplots_adjust(left=0.195, right=0.98, top=0.78, bottom=0.29)
    blue = POLICIES["A_random"][2]
    ax.grid(axis="x")
    for y, (_, supply, target) in enumerate(rows):
        ax.hlines(y, supply, target, color="#8C959C", linewidth=1.3, zorder=2)
        ax.scatter(supply, y, marker="o", s=40, color=blue, zorder=3)
        ax.scatter(target, y, marker="s", s=42, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=3)
        ax.annotate(f"{supply} / {target}   缺口 {target - supply}", ((supply + target) / 2, y),
                    xytext=(0, 7), textcoords="offset points", va="bottom", ha="center", fontsize=10)
    upper = max(row[2] for row in rows)
    ax.set_xlim(0, 4 * math.ceil(upper / 4))
    ax.set_xticks(range(0, 4 * math.ceil(upper / 4) + 1, 4))
    ax.set_ylim(len(rows) - 0.65, -0.65)
    labels = {"列表别名与复制": "列表别名", "break 与 continue": "循环控制",
              "递归的停止条件": "递归停止条件"}
    ax.set_yticks(range(len(rows)), [labels.get(topic, topic) for topic, _, _ in rows])
    ax.set_xlabel("回答对数", labelpad=3)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=10)
    ax.tick_params(axis="x", length=3, width=0.7, labelsize=9)
    fig.text(0.195, 0.925, "首选较短回答对的供给限制", ha="left", fontsize=11, weight="bold")
    fig.text(0.195, 0.855, f"仅显示受限的 {len(rows)} / {family_count} 个题族", ha="left", fontsize=9.5, color="#53616C")
    handles = [
        Line2D([], [], color=blue, marker="o", linestyle="none", markersize=5,
               label="可用供给 = 实际入选"),
        Line2D([], [], markeredgecolor=INK, markerfacecolor="white", marker="s", linestyle="none",
               markersize=5, label="近半目标：floor(配额 / 2)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, frameon=False, fontsize=9.4, columnspacing=1.7, handlelength=0.9)
    save_figure_pair(fig, SUPPLY_FIGURE, "题族内较短首选的供给约束", f"{SOURCE.name}; data/legacy/audit/{LEDGER.name}")
    plt.close(fig)


def main():
    observations = load_experiment()
    plot_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.55, 3.50),
                                     gridspec_kw={"width_ratios": [0.90, 1.10]})
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.30, top=0.85, wspace=0.48)
    for ax in (left, right):
        ax.tick_params(width=0.7, length=3, labelsize=9.2)
    draw_long_rule(left, observations)
    draw_short_by_reason(right, observations)
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=5.7,
                      color=color, label=f"{letter}  {label}")
               for letter, label, color, marker in POLICIES.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3,
               frameon=False, fontsize=9.3, columnspacing=1.55, handlelength=0.9, handletextpad=0.45)
    fig.text(0.5, 0.025, "50 次配对置换：左图显示全部观测与均值，右图显示分组均值。",
             ha="center", fontsize=8.8, color="#53616C")
    FIGURE.parent.mkdir(exist_ok=True)
    save_figure_pair(fig, FIGURE, "data synthesis 筛选策略对照", SOURCE.name)
    plt.close(fig)
    draw_family_supply()


if __name__ == "__main__":
    main()
