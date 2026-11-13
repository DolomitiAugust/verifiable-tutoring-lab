# Evidence to pilot — Academic Figure Prompt, palette A

## Layer 1 — Global description

Create a 162 × 116 mm editable academic evidence-and-validation diagram titled
**From prototype evidence to a testable pilot**. The upper part has three
columns: observed project evidence, current limits, and proposed validation
gates. Below, embed a small quantitative cost-scenario curve and a carefully
qualified assumption note. Visually distinguish experiments from commercial
calculations; the cost threshold is not model accuracy or permitted error.

## Layer 2 — Section-by-section description

=== OBSERVED ===

Use a white blue-outlined module with **Frozen teaching data: 1,000 SFT / 500
pairs**, **Two cloud runs: SFT → DPO + control**, and **209/209 tool-only
checks; Not an LLM teaching score**. These are completed project artifacts,
not customer or learner outcomes. A thin solid arrow leads to current limits.

=== CURRENT LIMITS ===

Use a white orange-outlined module. Show **3 added linked groups: 54 calls / 18
turns; 0/18 model-prose finals**, followed by **Learning gains and paid demand
unobserved**. The adoption count must stay limited to these three groups.

=== PROPOSED GATES ===

Use a dashed orange frame, with a small green heading. List **Teacher blind
review: Facts + requested constraints**, **Voluntary course trial: Unaided
transfer + total work**, and **Paid pilot + renewal: Costs + actual receipts**.
Use a dashed arrow into this column to mean a proposed validation step. No
check mark or completed-state badge may imply these stages have happened.

=== COST INSET ===

Separate the inset with a thin grey rule and the explicit title **COST INSET —
DETERMINISTIC BUSINESS ASSUMPTIONS**. Plot contribution after incremental CAC
for one full-year new contract against assumed human-review turn share, from
0 to 25%, in units of 10,000 CNY. Use the saved scenario's linear cost
calculation. The blue curve declines through zero. Mark the assumed 2% share
with a green square and the computed zero-contribution point, approximately
17.03%, with an orange diamond and vertical dotted guide. The dashed horizontal
line is zero contribution, not a performance baseline.

In a neighboring dashed white note module retain **2% review is assumed**,
**17.03%: zero contribution after CAC in this scenario**, and **Not an allowed
error rate or a learning-quality threshold**. Do not invent confidence
intervals, measured customer observations, or a probability of commercial
success.

## Layer 3 — Global annotations and data binding

Bind data counts to frozen data synthesis files, the two cloud-run step summaries to saved
records, the regression to `tool_regression_v2/metrics.json`, and added model
adoption to the three specified linkage groups. Bind the numerical inset to
`quality_cost_stress.json` and `business_assumptions.json`. Compute the exact
threshold as −intercept/slope; save both unrounded value and displayed 17.03%.
The review fraction was not measured; fallback is not equivalent to human
review. Pair the same numbers with translated axis text in Chinese. Arrow
semantics are solid evidence dependency and dashed proposed validation.

## Layer 4 — Style specifications

Use white for at least 70% of the canvas, with white module fills. Restrained
palette A: blue `#0072B2`, orange `#E69F00`, green `#009E73`; neutral ink
`#263746`, secondary text `#62727C`, and rules `#CCD5DA`. Use 7.4–11.2-pt
searchable sans-serif text and thin outlines. Retain positive and negative
values and a visible zero. No gauges, pie charts, gradients, shadows, pseudo
learning curves or causal claims about student benefit. Export PDF/SVG/PNG,
inspect fonts and canvas bounds, and review both color and greyscale at actual
report width.
