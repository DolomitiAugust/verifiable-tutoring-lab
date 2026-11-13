# Synthesis atlas — Academic Figure Prompt, palette A

## Layer 1 — Global description

Create a publication-ready, information-rich, editable vector research diagram,
162 mm wide by 125 mm high, titled **From research ideas to frozen teaching
data**. Arrange four clearly separated panels: the two distinct research
frameworks across the top, the local executable factory in the middle, and
frozen training/development/evaluation purposes along the bottom. A small
factory and a friendly inspector are explanatory metaphors. They must remain
subordinate to the method and must not resemble measured experimental outputs.

## Layer 2 — Section-by-section description

=== A. WHAT TO SELECT ===

Use a white panel, a small steel-blue section label and a thin grey rule.
Place three outlined trace sheets at left and the exact paper name
**NaturalThoughts** at right. Explain the sequence **Annotate reasoning traces
→ select → distill**, and retain **Strategies, verbosity, difficulty proxies**.
This summarizes content selection and supervised distillation in the paper,
not a framework implementation by this project.

=== B. HOW ITEMS ADVANCE ===

Use a neighboring white panel with a warm-orange section label and thin grey
rule. Draw a miniature queue of three nonempty record boxes with short grey
lines inside each, joined by small arrows. Label **Matrix**, **Stateful items +
role queues**, and **Shared model / tool services**. Add **Scheduling is not a
truth criterion**. The distinction from NaturalThoughts must be visible.

=== C. LOCAL EXECUTABLE FACTORY ===

Join the research panels with a thin dashed transfer route labelled **Adapt
ideas only**. The local pipeline is left to right: an outlined factory labelled
**Rules + templates**; a nonempty candidate card labelled **8,240 candidates**
containing **Code, requests, response pairs**; a small inspector holding a
magnifier labelled **Execute + verify**; a **Filter** module containing **Facts,
task, length, program overlap**; and **Freeze splits**, containing **Hash, seed,
export ledger**. Keep arrows in open gaps, outside all text and node borders.

=== D. FROZEN PURPOSES ===

Place three separate white modules: blue **Train only: 1,000 SFT / 500 pairs**;
orange **Select epoch: 36 dev**; green **Evaluate only: 60 bridge / 100
preferences / 24 natural**. Do not connect evaluation back into training.
Retain the note **No DeepSeek-teacher distillation or distributed cluster was
run here. Icons are metaphors.**

## Layer 3 — Global annotations and evidence binding

Solid arrows mean local data processing; dashed transfer means conceptual
adaptation, not experimental reproduction. Freeze counts and identities from
`synthesis/dataset/stats.json`, the six saved JSONL files and the passing
`validation.json`; retain their hashes in provenance. Literature descriptions
are bound to the supplied NaturalThoughts and Matrix papers and the verified
background section. Do not import either paper's benchmark gains or GPU
throughput as a result of this project. English and Chinese versions have the
same nodes, numbers, arrow semantics and panel arrangement. Exact rendered
language labels are recorded in the companion provenance JSON.

## Layer 4 — Style specifications

At least 70% white space by pixel area; white module fills; no gradient, shadow,
banner or decorative colored panel. Use steel blue `#0072B2`, warm orange
`#E69F00`, bluish green `#009E73`, charcoal `#263746`, grey `#62727C`, and thin
light-grey rules `#CCD5DA`. Use consistent sans-serif typography, approximately
7.4–11.2 pt at the supplied figure size. Friendly vector line-art faces may have
two eyes and a restrained smile. Embed fonts in PDF, retain searchable text in
SVG, and export a 300-dpi PNG from the same drawing. Review both languages in
color and greyscale for missing glyphs, node-label collisions, arrow-label
collisions, clipping and density. This prompt was implemented with native
vector code rather than a generative data-image model.
