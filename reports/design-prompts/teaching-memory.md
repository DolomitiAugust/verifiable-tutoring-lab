# Teaching and memory — Academic Figure Prompt, palette A

## Layer 1 — Global description

Create an editable 162 × 125 mm, white-background academic mechanism figure
titled **Teaching calls, evidence gates, and memory**. Combine a role and
memory architecture in the upper panel with the historical empty-prediction
failure and its correction below. Three friendly small role figures and a
memory-drawer icon help explain the structure, while all empirical counts
remain literal, source-bound annotations. The title's distinction is essential:
a model draft, a verified answer, and a learner record are different objects.

## Layer 2 — Section-by-section description

=== A. CALLS, EVIDENCE AND PERSISTENCE ===

Use a small blue section heading **SHARED WEIGHTS WITHIN EACH CONDITION** and a
thin grey rule. Draw a blue shared-weight bus feeding three separate role
calls: blue **Planner** with a clipboard, orange **Teacher** with a small open
book, and green **Reviewer** with a magnifier. These are separate calls sharing
one checkpoint within a condition, not three independently trained models.
Sequential candidate arrows point toward a white **Content gate** containing
**Facts / constraints / review**. Its green branch leads to **Adopt**; its
orange dashed branch leads to **Fallback**. Keep every arrow clear of words.

Below the roles, a nonempty wide white module reads **Shared context: verified
facts + retrieved learner history** and supplies each role. A **Restricted
execution** module contains **Code + student prediction**. A **SQLite: learner
+ topic** module contains **Raw events + counters** and a tiny drawer icon in a
dedicated left gutter. Drawer lines must never cross text. A directed store
arrow connects observed tool/student evidence to memory; a retrieval arrow
supplies the shared context. Mark content provenance separately as **model text
or tool/template**.

Include the precisely scoped empirical line **Three added linked groups: 54
calls / 18 turns; 0/18 final model prose adopted.** This refers only to the
three added cloud-SFT linkage groups. Do not extend zero adoption to the
historical official-Instruct diagnostic.

=== B. HISTORICAL S02 CAUSAL CHAIN ===

Use an orange section label **PRESENCE IS NOT STRING TRUTHINESS — HISTORICAL
S02**. Three white boxes show: **Input semantics: None = no submission; "" =
submitted empty string**; **Old check: truthiness; bool("".strip()) = False →
marked unanswered**; **Feedback → memory error; attempts +0 / correct +0;
unanswered +1**. Orange arrows show propagation of the same defect.

A distinct green correction strip states **Revised, on this S02 case:
student_answer is not None → compare empty strings → attempts +1; correct +1**. Historical and revised
behavior must not be merged or relabeled. This lower panel is a causal example,
not a complete runtime chronology or an estimate of learning effectiveness.

## Layer 3 — Global annotations and evidence binding

Bind architecture to the implemented system and memory source; bind S02 to
its unchanged input, executed empty output, status and counter increments in
the supplementary result. Bind 54 calls, 18 turns and zero adopted finals to
`original_retry`, `clean_first`, and `clean_retry` in
`tutor/core/reviews/retry_link_summary.json`. Validate the totals rather
than entering new results. Chinese and English drawings have identical module
and connector structure. Log source hashes and all exact visible labels.

## Layer 4 — Style specifications

Use at least 70% white background. White modules have 0.8-pt outlines; top
labels and thin grey rules organize the figure. Palette: `#0072B2`, `#E69F00`,
`#009E73`, charcoal `#263746`, grey `#62727C`, light grey `#CCD5DA`. Do not use
colored section fills, shadows, gradients or oversized cartoon art. Preserve
7.4–11.2-pt searchable sans-serif labels. Export PDF with embedded fonts,
editable SVG and 300-dpi PNG. Review text against both other text and small
decorative line art, then inspect greyscale. The figures are an architecture
and failure explanation, never a claim that the model improved teaching.
