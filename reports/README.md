# Reports and figure sources

[English](README.md) | [简体中文](#中文说明)

The two PDFs describe the same frozen datasets and experiments. English is the concise research version; Chinese adds short conceptual notes. Both retain the 14 scientific figures, primary comparison tables, failed responses, and uncertainty boundaries. Personal conversations and submission material are excluded.

## Online figure previews

[English gallery](figures.md) | [中文图表总览](figures.zh-CN.md) | [Viewing/download help](../docs/viewing-files.md). Use the embedded PNGs if GitHub cannot render an individual SVG.

## Build the PDFs

From the repository root, with an existing Tectonic installation:

```bash
python -B reports/build_reports.py --engine /path/to/tectonic
```

This one build command uses `source/en/` and `source/zh/` plus shared assets. It does not run models. The supplied typography uses Times New Roman, Segoe UI, Consolas, SimSun, SimHei and Microsoft YaHei. Install these fonts where licensed and available, or edit the `fontspec`/`xeCJK` declarations in both language masters to installed equivalents. A different font can change pagination.

## Redraw figures from saved records

```bash
python -m pip install -r reports/requirements-figures.txt
python -B reports/plotting/build_diagrams.py
python -B reports/plotting/plot_training_comparisons.py --language en
python -B reports/plotting/plot_training_comparisons.py --language zh
python -B reports/plotting/plot_business.py --language en
python -B reports/plotting/plot_business.py --language zh
```

- `build_diagrams.py` redraws four bilingual schematics in `assets/`, binding their counts and cost inset to saved source records. Icons are conceptual metaphors.
- `plot_training_comparisons.py` reads both cloud evaluations and the corresponding full-text factual reviews. It checks paired case IDs/reference-token counts and draws the first-run evidence grid plus retry NLL differences and factual-status counts.
- `plot_business.py` reads the saved assumptions/results and redraws baseline costs, cash under three collection lags, and review-cost sensitivity. These are deterministic scenarios, not business observations.
- `plot_selection.py` and `plot_agents.py` preserve the earlier selection and memory/adoption plotting logic. Their source records remain part of the relevant comparisons.

The two comparison scripts write regenerated PDF/SVG/PNG variants to `reports/generated/{en,zh}/`, leaving the report's committed assets unchanged. Pass `--outdir` to choose another destination. Figure fonts use DejaVu Sans for English and Microsoft YaHei for Chinese; on other systems, change the Chinese `font.family` setting to an installed CJK font such as Noto Sans CJK SC. No training or model inference occurs in these commands. Generated variants can differ in layout from the committed figures while using the same observations. `design-prompts/` preserves the conceptual figure briefs; the PNG cover illustration is explanatory artwork.

## 中文说明

两份最终PDF共享同一套代码、冻结数据与实验记录。中文只增加少量原理解释，不增加实验或改变结论。`source/`是报告译文源稿，`assets/`是报告实际使用的图，`plotting/`保留可执行绘图源码，`design-prompts/`是科研示意图的设计说明。

先安装绘图依赖，再运行上面的命令。后训练绘图直接读取已有逐题NLL、终止信息和全文事实审阅，检查配对题目及参考token数量；“未判定”仍单列，不能直接算错。商业图按既定价格、用量、人工和回款假设计算，不是实际客户或收入。概念图中的机器人只说明职责。

`build_diagrams.py`会重画并替换四组概念图；训练、商业绘图默认写入`generated/`，便于与保留的报告图比较。修改图形外观不应修改冻结科学结果。重建PDF前需准备对应字体；Linux可用已安装的CJK字体替换源稿及绘图设置，但应重新检查分页。`validation.json`记录本次最终PDF、引用、图形路径和排版检查，全文案例的英译证据在`case-study-evidence.en.json`。
