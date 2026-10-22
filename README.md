# Cross-gesture transfer after single-gesture calibration

Code and numerical results for **Cross Gesture Transfer After Single Gesture Calibration in High Density Electromyography**, Charles Lu.

This study evaluates gain correction, spatial adaptation, and classifier fine-tuning on CSL-HDEMG. In four evaluation participants, high accuracy on the calibrated gesture did not consistently preserve accuracy on the other seven gestures. The repository contains the records and scripts used to reproduce the reported tables, error analyses, and figures.

The [manuscript](paper/manuscript.pdf) and [supplement](paper/supplement.pdf) describe the methods and results. Editable Word versions are included in the same directory.

## Installation

The numerical analysis uses Python and NumPy. It was validated with Python 3.12 and NumPy 2.3.5.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the results

Run these commands from the repository root:

```sh
python scripts/reproduce_tables.py
python scripts/analyze_predictions.py
python scripts/check_followup.py
```

- `reproduce_tables.py` verifies the four original manuscript tables, participant contrasts, bootstrap intervals, and implementation controls against the archived summaries.
- `analyze_predictions.py` computes error destinations, confusion matrices by calibration gesture, exact sign-flip tests, and search-loss saturation counts from saved trial predictions.
- `check_followup.py` verifies the safeguard and preprocessing follow-up summaries from saved trial predictions.

Each command reports `"passed": true` on successful completion. Numerical comparisons use an absolute tolerance of 1e-12. The first two commands also write their analysis outputs to `results/`.

To regenerate the figures:

```sh
python -m pip install -r requirements-figures.txt
python scripts/plot_figures.py
```

Figures are written to `figures/`. These commands reproduce analyses from the included records; they do not retrain models or replay raw recordings.

## Repository structure

```text
data/                   Per-case metrics and saved trial predictions
results/                Aggregated results and analysis checks
protocols/              Original and follow-up experiment configurations
scripts/                Analysis, plotting, and follow-up scripts
src/original_pipeline/  Archived training and verification source
provenance/             Environment, upstream hashes, and verification records
figures/                Manuscript and supplementary figures
paper/                  Manuscript and supplement in PDF and Word formats
```

## Data and experimental design

The primary analysis includes 3,840 method-case records: six methods, four participants, 20 ordered session pairs per participant, and eight calibration gestures. Participant 1 was used for development; participants 2–5 were used for evaluation. Calibration and scoring use disjoint trials. All eight classes appear in source training; “other” refers to classes absent from target calibration. Participant means receive equal weight.

`data/primary_cases.csv` and `data/control_cases.csv` contain per-case metrics; `data/participant_summary.csv` contains participant aggregates. `data/trial_predictions.json` contains trial labels, predictions, identities, and spatial-search diagnostics. The follow-up trial predictions and summaries are in `data/followup_results.json` and `results/followup_summary.json`.

The safeguard and preprocessing experiments use session pair 1→2, with 32 calibration cases per variant across the same four participants. These exploratory experiments were added after the original results were known. They are not an independent replication. The independent sample size remains four; bootstrap intervals are descriptive, and exact two-sided sign-flip tests have a minimum attainable p-value of 0.125 under a sign-exchangeable null.

Raw CSL-HDEMG recordings must be obtained from the dataset provider under its registration and usage terms. Provider gesture IDs are 8, 9, 12, 13, 16, 21, 23, and 24. Raw recordings, frame caches, and model checkpoints are not included here.

## Training source and provenance

The archived training code in `src/original_pipeline/scripts/` uses the original experiment workspace layout, including `scripts/`, `data/public/csl-hdemg/`, and `research/runs/`. Running that code requires the raw dataset, prepared caches, and upstream source snapshot. It is retained for methodological reference and is not a standalone installation of the full training pipeline.

`scripts/run_followup.py` uses the same workspace through its `--workspace` argument and writes new experiment outputs to `--output`. The exact settings and verification procedures are recorded in `protocols/followup_protocol.json`.

The upstream [Spatial Adaptation Layer implementation](https://github.com/joao-binenbojm/spatial-adaptation-layer) is pinned to revision `7c5a075a58dff06566b82965f8235ba744377fb8`. File hashes are recorded in `provenance/upstream_source_sha256.json`. Original experiment environments and verification records are retained in `provenance/`; workspace prefixes in historical records are represented by `$WORKSPACE`.

`SHA256SUMS.json` records the files distributed in this repository. Regenerating figures may change embedded file timestamps without changing the plotted results.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Cite the manuscript title and the repository revision used for an analysis.

Author-written code is distributed under the [MIT license](LICENSE). Dataset and third-party software rights remain with their respective owners.
