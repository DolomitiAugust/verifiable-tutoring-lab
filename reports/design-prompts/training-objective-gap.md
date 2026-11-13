# DPO objective gap — source-preserving Chinese localization

## Layer 1 — Global description

Localize the existing 162 × 78.74 mm English diagnostic diagram into Chinese.
Retain its optimizer robot, measured-score modules, arrow paths and qualified
interpretation. This is a translation of saved diagnostics, not a new model run.

## Layer 2 — Section-by-section description

=== MEASURED SCORE CHANGES ===

Retain chosen mean sequence-log-probability change −37.761 nat and rejected mean
change −59.936 nat, relative to the frozen SFT reference during the last 100
logged retry-DPO updates. The small robot is labelled as an optimizer metaphor.

=== JOINT DECLINE COUNT ===

Retain 96/100 final logged updates with both scores below their frozen-reference
values. Do not reinterpret this as an output-accuracy or student-learning rate.

=== RELATIVE OBJECTIVE ===

The difference of the two mean changes is +22.175 nat. Explain that an expanded
relative preference margin can lower DPO loss while both absolute scores fall.
The figure explicitly says that this does not establish factual correctness.

## Layer 3 — Global annotations and binding

Read `training_diagnosis.json`, verify exact stored values and the computed
difference before formatting. Preserve the English PDF/SVG/PNG bytes. The
Chinese wording must identify sequence-level log probability, the last-100
window, the frozen reference and the distinction from generation accuracy.

## Layer 4 — Style specifications

Use the original palette-A geometry and white background, with Chinese
sans-serif text and embedded fonts. Preserve line art, modules and connectors;
export editable SVG, PDF and 300-dpi PNG. Check each label inside its original
module, inspect Chinese glyphs, and review the result at report width.
