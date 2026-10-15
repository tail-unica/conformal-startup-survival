# Conformal Risk Control for Startup Success Prediction through Machine Learning

This repository contains the code and the reproducible pipeline for applying
**Conformal Risk Control (CRC)** to the machine-learning prediction of startup
success. It builds on *"The Future Is Not a Feature: A Look-Ahead Bias-Free
Evaluation Framework for Startup Success Prediction through Machine Learning"*
([tail-unica/startup-survival](https://github.com/tail-unica/startup-survival)),
from which the dataset and the feature and target engineering are inherited.

## Overview

A classifier outputs a score, and turning that score into a decision means picking
a threshold. A plain `p >= 0.5` gives **no guarantee** on the errors the model
makes, and in this task the costly one is the **false negative**: a startup that
will succeed, flagged as a failure, and missed.

**Conformal Risk Control** (Angelopoulos et al., 2024) picks the threshold from a
held-out **calibration** set so that the **False Negative Rate** of the success
class is controlled at a target level `alpha`:

> E[FNR_test] <= alpha

It returns the largest threshold `t*` whose finite-sample FNR bound stays at or
below `alpha`; predicting *positive* whenever `p >= t*` then keeps the expected
test FNR under `alpha`. The guarantee is distribution-free and holds for any
classifier, but it is **marginal**: it holds on average over the population, not
within each subgroup of it.

Three research questions follow from that:

- **RQ1 — Global-level CRC.** One threshold, calibrated on the whole calibration
  set. Does the realized test FNR stay at `alpha` across many calibration/test
  re-splits, and what does the other side of the confusion matrix pay for it?
- **RQ2 — Subgroups under global-level CRC.** With that single threshold, how is
  the FNR distributed across **industry sectors** and **founding-year cohorts**
  (`YearFounded`)?
- **RQ3 — Group-level CRC.** A separate threshold per subgroup (Mondrian CRC).
  How close does each group's FNR get to `alpha`, and what does it cost on the
  pooled metrics?

Seven model families are implemented: Random Forest (`rf`), LightGBM (`lgb`), a
Multi-Layer Perceptron (`mlp`), Decision Tree (`dt`), Logistic Regression (`lr`),
an RBF-kernel Support Vector Machine (`svm`), and TabPFN v2 (`tabpfn`). The default
configuration runs `rf`, `lgb`, `mlp` and `tabpfn`.

## Data

The dataset comes from
[tail-unica/startup-survival](https://github.com/tail-unica/startup-survival),
whose panel is derived from **PitchBook** data. Features and target are computed
under a strict temporal constraint (a fixed look-back **time window**), so every
feature only uses information available at the decision point. That repository
and its paper document the full construction.

| File | Content | Status |
|------|---------|--------|
| `data/dataset_controlled.csv` | **The bias-controlled dataset** of that project (`data/processed/dataset_controlled.csv` there): one row per firm, `CompanyID`, the binary `Target`, and the engineered features. The exact input of every experiment here. | Released |
| PitchBook extraction and panels | The raw data the dataset is built from. | **Not released** |

`HQCountry` and `PrimaryIndustrySector` are kept raw in the CSV and
frequency-encoded after the split, on the training rows only. The raw sector
names are also the industry-sector subgroups of RQ2 and RQ3.

## Repository structure

```
.
├── config/
│   └── config.yaml              # Splits, seeds, conformal grid, model families and hyperparameters
├── data/
│   └── dataset_controlled.csv   # Final bias-controlled dataset (released)
├── src/
│   ├── utils.py                 # CRC threshold, conformal experiments, subgroup tables, plots
│   ├── encoding.py              # Frequency encoding, fitted on the training split
│   └── models/                  # One class per model family, keyed by model_type
├── notebook.ipynb               # The whole pipeline, from the dataset to RQ1–RQ3
├── pyproject.toml               # Dependencies and ruff configuration
├── uv.lock                      # Exact resolution, committed
├── .env                         # W&B entity and project (only needed with USE_WANDB)
└── README.md
```

## Getting started

**Python 3.12**, and [uv](https://docs.astral.sh/uv/) to manage the environment. If
you do not have uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/tail-unica/conformal-startup-survival
cd conformal-startup-survival
uv sync --group dev
```

`uv sync` creates `.venv/` with the exact versions recorded in `uv.lock` and
installs the project itself, so `import src...` works from the notebook. The `dev`
group adds ruff and the Jupyter kernel (`ipykernel`).

Key dependencies (`uv.lock` holds the exact resolution): `pandas`, `numpy`,
`scikit-learn`, `lightgbm`, `torch`, `tabpfn`, `scipy`, `matplotlib`, `seaborn`,
`wandb`, `PyYAML`, `python-dotenv`.

**TabPFN needs a CUDA GPU** in practice: `device="auto"` falls back to CPU, where it
is impractical past a few thousand rows. The checkpoint is downloaded to the TabPFN
cache on first use (override the location with `TABPFN_MODEL_CACHE_DIR`). Remove
`tabpfn` from `model_type` in `config.yaml` to run on a CPU-only machine.

### Weights & Biases, only if you want it

Tracking is **optional**: with `USE_WANDB = False` (the default, in the first cell
of the notebook) the runs happen locally, nothing is logged remotely, and the
results are kept in memory, which is all the analysis cells need. To use it,
create an account at <https://wandb.ai/site>, create an entity and a project, fill
them in `.env`, and log in once:

```bash
wandb login
```

Then set `USE_WANDB = True`: a sweep agent drives the same runs and logs the
training metrics, the conformal tables and the FNR plots of each model.

## Running the pipeline

Open `notebook.ipynb` in an editor (VS Code, PyCharm) and pick the interpreter in
`.venv/bin/python` as the kernel; `uv sync --group dev` has already installed it.
For JupyterLab, `uv run --with jupyterlab jupyter lab`.

Run it top to bottom. Each section opens with what it does and how
to read its output:

1. **Splits, encoding, imputation and scaling.** One stratified split into
   training, validation and the **conformal pool**.
2. **Sweep** (only with `USE_WANDB`) and **training**. One run per model family:
   fit, score the pool once, run the conformal experiment on those scores.
3. **RQ1.** Pre- and post-calibration tables, the target `alpha` vs realized FNR
   plot, the per-metric boxplots over seeds.
4. **RQ2.** Realized FNR per industry sector and per cohort under the global
   threshold: tables, grouped boxplots, per-model line plots.
5. **RQ3.** The same views with per-group thresholds, and the Δ table
   (Mondrian − standard CRC) on the pooled metrics.

Training is the only slow step; everything after it reuses the stored pool
probabilities.

## Configuration

`config/config.yaml` holds every setting, so that a choice is made in one place:

| key | what it decides |
|---|---|
| `paths.dataset` | the dataset the notebook reads |
| `time_window`, `last_year` | the look-back window and the last year the data covers (inherited from the dataset construction) |
| `random_seed` | the seed of the train/validation/pool split and of the models |
| `train_frac`, `validation_frac`, `calibration_frac`, `test_frac` | the split geometry, as shares of the whole dataset: 50 / 10 / 20 / 20 |
| `frequency_encoding` | which columns are encoded, and the share below which a category is pooled into `Others` |
| `conformal.alpha` | the grid of target FNR levels |
| `conformal.seeds` | the calibration/test re-splits of the pool: a list, an int, or `{min, max}` (inclusive) |
| `sweep_settings` | the model families (`model_type`) and the hyperparameters of each, under a `<family>_` prefix |

## The experiments in detail

### Split and the conformal pool

The dataset is split once, stratified on the target: **50%** training, **10%**
validation (early stopping of the MLP), and the remaining **40%** as the conformal
pool. The pool is never seen during training. Each model scores it once, and the
conformal experiments then re-split it **50/50** into a calibration half and a test
half, once per seed in `conformal.seeds`: the 20% / 20% of the config. Re-splitting
the scores, not the data, is what makes many seeds cheap.

The frequency encoding, the KNN imputer and the `RobustScaler` are fitted on the
training rows alone. Each model family picks its own matrix (`wants_scaled` in
`src/models/`): the tree models and TabPFN read the imputed features, the others
the imputed and scaled ones.

### The CRC threshold

`conformal_fnr_threshold` in `src/utils.py` scans the calibration positives' scores
and returns the largest `t` with

> (n · FNR_cal(t) + 1) / (n + 1) <= alpha

where `n` is the number of calibration positives. If `alpha < 1 / (n + 1)` the
target is unreachable at that calibration size; the threshold falls to 0 and the
row is flagged `infeasible`.

Every seed also carries a **base row** at `p >= 0.5`, as an uncalibrated reference.
It depends on how each family scales its probabilities (class weighting, Platt
scaling), so it is a reference, not a comparison between models.

### Subgroups and group-level calibration

`fnr_by_group_table` runs the subgroup analysis twice on the same seeds:

- **marginal**: the global threshold, its FNR measured within each group
  (`FNR_marg`);
- **conditional** (Mondrian): one threshold per group, fitted on that group's
  calibration rows and applied to the same group in the test half (`FNR_cond`).

A group with fewer than `min_group` positives in its calibration half (20 in the
notebook) is flagged **`infeasible`** and falls back to the global threshold: a
per-group threshold fitted on a handful of positives carries no guarantee. A group
with no positives in a test half has an undefined FNR, reported as `NaN` and left
out of the mean over seeds.

`run_conformal_fnr_group_calibrated` applies the per-group thresholds and measures
the metrics on the whole test half, so the RQ3 global tables read the same way as
the RQ1 ones.

### Hyperparameters

The active `sweep_settings` block is a **grid** with one value per hyperparameter
(the best ones found), so the only axis it crosses is `model_type`. To search for
hyperparameters instead, comment it out and enable the Bayesian block above it,
with `USE_WANDB = True`.

### Adding a model family

Write a subclass of `Model` in `src/models/` that sets `wants_scaled` and
implements `from_sweep` (and overrides `fit`, `predict_proba` or `score` if the
defaults do not fit), register it in `MODELS` in `src/models/__init__.py`, and add
its hyperparameters to `sweep_settings` under a `<family>_` prefix. The training
cell does not change.

## Development

```bash
uv run ruff check src    # lint
uv run ruff format src   # format
```

Lint and format rules live in `pyproject.toml` (ruff, line length 100).

## Reproducing the results

The dataset is released, so the results can be reproduced without PitchBook
access:

```bash
uv sync --group dev
```

then run `notebook.ipynb` top to bottom on the `.venv` kernel. The default `config.yaml` is the one the
reported results come from.

## Citation

If you use this code, please cite the accompanying work and the project it builds
on:

```bibtex
@article{startup_survival,
  title   = {The Future Is Not a Feature: A Look-Ahead Bias-Free Evaluation
             Framework for Startup Success Prediction through Machine Learning},
  author  = {Casti, Giulio and Mandas, Marco and Piras, Luca and Marras, Mirko and Fenu, Gianni},
  year    = {2026}
}
```

## License and data rights

The source code is released for academic and research use. The dataset is
inherited from
[tail-unica/startup-survival](https://github.com/tail-unica/startup-survival); the
original PitchBook-derived panel is **not** included and remains subject to
PitchBook's licensing terms.
