# AI assistance and evaluation scope

OpenAI Codex assisted with research synthesis, deterministic data-generation code, training/evaluation implementation, technical review, translation, diagrams, and report preparation. Known recent assistance identifiers include GPT-6-astra, GPT-6.1-terra, and GPT-6.1-sol; exact historical per-call variants are not fully recoverable. Model-assisted checks from this project are not independent or blinded teacher evaluation.

The evaluated training model is the official **Qwen2.5-0.5B base**, followed by the recorded SFT and DPO adapters. The separately sourced **Qwen2.5-0.5B-Instruct** is diagnostic and is not represented as the result of this project's training. Planner, teacher, and reviewer are role-specific calls sharing a model, not independently trained agents.

Dataset construction uses deterministic programs and templates; it did not call a teacher model to generate every row. Recorded role prompts and model inputs are retained with the experiment records where selected. Original Chinese inputs remain Chinese; English report excerpts are translations, not a new English-language model evaluation.

Scientific plots are code-rendered from saved data. The unlabelled role-and-memory illustration is AI-generated explanatory artwork and is not experimental evidence; the image tool did not expose its exact model identifier. Figure design briefs and source information accompany the selected plotting assets.

Personal chat transcripts and opaque internal delegation records are excluded from this release. A complete historical authoring-prompt archive cannot be certified. This qualification does not imply that missing text was reconstructed or that the released records form an independently reproduced model experiment.
