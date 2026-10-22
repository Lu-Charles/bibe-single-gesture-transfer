# BIBE 2026 paper 54 analysis package

Reproduction materials for **Cross Gesture Transfer After Single Gesture Calibration in High Density Electromyography**, Charles Lu.

This package is specific to the BIBE CSL-HDEMG study. It is separate from the later GRABMyo/SeNic manuscript and the `emg-gesture-recognition` repository. It preserves the original BIBE results and adds explicitly post-hoc review analyses and a limited revision follow-up.

## Private review copy

This repository is private for author review. It is not a public code release. Public release remains a separate decision. The revised clean and highlighted manuscripts, supplement, and reviewer response are in [`review_documents/`](review_documents/). Their code-availability wording still records public release as pending and should be updated only after a public release is verified.

## Reproduce the paper tables

Python 3 and NumPy are required. The analysis-only commands read the included numerical records:

```sh
python -m pip install numpy
python reproduce_analysis.py
python revision_analysis.py
python reproduce_followup.py
```

The first command checks all four original BIBE tables, participant contrasts, descriptive bootstrap intervals and implementation controls against the archived summaries. The second checks original means from 3,840 method-case trial-prediction records and writes error decompositions, calibration-specific confusion matrices, exact sign-flip results and search-loss saturation counts. The third recomputes the new follow-up means from 288 saved method-case records. Each reports `passed: true` and checks numerical equality to 1e-12. These checks do not retrain models or replay raw signals.

Optional figure regeneration (requires Matplotlib):

```sh
python -m pip install matplotlib
python plot_figures.py
```

This writes the calibrated/other-class plot, revised control plot, calibration-specific confusion matrices, and pipeline schematic into `figures/`.

## Contents and scientific scope

- `primary_cases.csv`, `control_cases.csv`, `participant_summary.csv`, `analysis.json`, and `controls_analysis.json`: exact numerical inputs for the original BIBE tables.
- `trial_predictions.json`: original saved trial predictions, ground-truth labels, trial IDs, and spatial-search diagnostics; contains no raw biosignals or checkpoint weights.
- `revision_analysis.json`: post-hoc results. Confusion rows are normalized within cases, averaged across 20 pairs within each participant, then equally across four participants, separately for each calibration gesture.
- `followup_protocol.json`, `followup_results.json`, `followup_summary.json`: new exploratory experiments on session pair 1 to 2, all four original evaluation participants and all eight calibration gestures. Identity regularization, activated baseline subtraction and literal upstream mapping are retained regardless of outcome. No new participant cohort is claimed.
- `original_verification/`: original verification records, clearly distinct from this release's analysis reproduction checks.
- `original_pipeline/`: preserved study scripts for methodological inspection; the original workspace and data setup are required to execute them. `run_followup.py` accepts that workspace explicitly and writes to a separate output directory.
- `upstream_source_sha256.json`: pinned third-party SAL file hashes. The upstream code is credited and not redistributed here.

The independent sample size is four participants. Repeated frames, trials, calibration choices and session pairs do not increase it. Sign-flip tests require a symmetric sign-exchangeable null and have a minimum two-sided p-value of 0.125. Bootstrap intervals are descriptive. The follow-up was planned after the original results were known, and is not an independent confirmation or external preregistration.

## Raw data and full pipeline reproduction

Obtain CSL-HDEMG from its provider under the provider's registration and usage terms. The study uses provider gesture IDs 8, 9, 12, 13, 16, 21, 23 and 24. Preserve whole-trial calibration/scoring separation and the two missing repetitions noted in the paper. The original upstream SAL revision is `7c5a075a58dff06566b82965f8235ba744377fb8` at https://github.com/joao-binenbojm/spatial-adaptation-layer . Check its source hashes against the included ledger.

The original pipeline snapshots use the historical workspace layout under `research/runs/`, raw files under `data/public/csl-hdemg/`, and a verified upstream source snapshot. They are not an automatic raw-data bootstrap. Raw recordings, processed frame caches, original/follow-up checkpoints and full frame-prediction archives are retained locally and are not hosted in this analysis-only repository. Public access to those model banks is not claimed. The complete numerical inputs for the manuscript tables and error matrices are included here.

`environment.json` records the original experiment environment. `release_validation.json` records the actual analysis validation performed for this release. Historical metadata is preserved as evidence, not represented as independent timestamped preregistration. Personal absolute workspace prefixes in protocol/verification documentation are replaced by `$WORKSPACE`; numerical result records are unchanged.

## License and citation

Author-authored scripts are distributed under the included MIT license. Dataset and third-party software rights remain with their respective owners. Cite the paper title and the exact repository revision used. This package does not claim acceptance, publication, or endorsement by the dataset or SAL authors.
