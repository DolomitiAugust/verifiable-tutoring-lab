# Publication scope

This repository is a de-identified research release assembled from the most complete inspectable artifacts, rather than a copy of a submission directory or an automatic selection of the newest files.

## Selection decisions

- `synthesis/` contains the extended dataset used in both cloud runs. `synthesis_legacy/` and `data/legacy/` preserve the earlier selection experiment and bridge references required to interpret the data comparison.
- `training/` contains the training/evaluation implementation, the first cloud run, the exploratory all-layer run, and the continued-SFT controls. These are different experimental conditions. The original failed numerical audit and its separate post-hoc numerical review remain distinguishable.
- `tutor/legacy/` preserves the implementation underlying the historical four case studies. `tutor/core/`, `tutor/structured/`, and `tutor/fence/` retain the controls and prompt-contract experiments used in the analysis. `tutor/rendering/` contains the later isolated output-control repair. Repair tests are not new language-model successes.
- `business/` preserves calculation code, explicit assumptions, source indices, and scenario outputs. Prices and product descriptions are dated source observations; demand, usage, review burden, and collections are assumptions.
- `reports/` contains only the two final report PDFs, their bilingual sources, shared figures, and selected plotting/design sources. The English and Chinese editions use the same research evidence.

Large model/checkpoint binaries, environments, recovery/upload scripts, notebooks with transient outputs, slide decks, obsolete report PDFs, duplicate export snapshots, personal chat/delegation archives, and third-party full-text papers/pages are excluded.

## Transformation and evidence

Identifiers, local paths, course labels, filenames, and imports were normalized for publication. Scientific measurements and case outcomes were retained. The PDFs were rebuilt from edited LaTeX sources; identity fields were removed from content and metadata, rather than covered with visual overlays.

Historical SHA-256 values refer to original pre-publication artifacts. Path normalization and de-identification change some serialized files, so those old hashes must not be treated as hashes of the released copies. `release-manifest.json` records current release-file hashes; `scripts/verify_release.py` verifies that manifest and basic artifact consistency. Neither hash verification nor controller tests establish teaching quality.

Original local artifacts and historical versions are outside the new repository and were not overwritten. This publication starts a fresh Git history with a neutral repository-level author identity. Repository visibility is initially private; no scheduled visibility change is configured.

## Reproduction boundary

Saved data, answers, reviews, and descriptive statistics are inspectable without model weights. A model rerun requires the pinned official base model, compatible dependencies, new adapters, and a new output directory. Original adapter/optimizer binaries are not distributed, so the repository does not supply a ready-to-load copy of every historical checkpoint. No model rerun was performed for publication.

Third-party source material is referenced by official links and bibliographic citations instead of being republished in full. No open-source license grant is assigned by this release.
