# Conformal Risk Control for Startup Success Prediction through Machine Learning

This repository contains the code and the reproducible pipeline for applying
**Conformal Risk Control (CRC)** to the machine-learning prediction of startup
success. It builds on a previous project — *"The Future Is Not a
Feature: A Look-Ahead Bias-Free Evaluation Framework for Startup Success
Prediction through Machine Learning"* ([tail-unica/startup-survival](https://github.com/tail-unica/startup-survival))
— from which both the dataset and the feature/target-engineering methodology are
inherited.

## Overview

Predicting startup success from early-stage signals is a popular machine-learning
task. A good classifier, however, only outputs a score: turning that score into a
decision means picking a threshold, and a plain 0.5 cut-off gives **no guarantee**
on the kind of errors the model makes. In this setting the most costly mistake is
a **false negative**, flagging a startup that will actually succeed as a failure,
and therefore missing it.

This project uses **Conformal Risk Control** (Angelopoulos et al., 2022) to
select the decision threshold so that the **False Negative Rate (FNR)** of the
positive (success) class is provably controlled at a user-chosen target level
`alpha`:

> E[FNR_test] <= alpha

Given a held-out **calibration** set, CRC returns the largest threshold whose
finite-sample FNR upper bound stays below `alpha`; predicting *positive* whenever
`prob >= t*` then keeps the expected test FNR under `alpha`. The guarantee is
distribution-free and holds for any underlying classifier.

The pipeline trains several model families on the same startup dataset and
studies the CRC guarantee empirically through three research questions:

- **RQ1 — Global-level CRC.** A single conformal threshold is calibrated on the
  whole calibration set. Across many random calibration/test re-splits we check
  that the realized test FNR lands on or below the target `alpha` for every model.
- **RQ2 — Subgroups under global-level CRC.** With a single *marginal* threshold,
  the realized FNR is broken down by **industry sector** and by **founding-year cohort** (`YearFounded`), to reveal subgroups that remain under-covered even
  when the marginal risk is controlled.
- **RQ3 — Group-level CRC.** A separate threshold is calibrated per
  subgroup, pulling each group's realized FNR back toward the target `alpha`, and
  the same global and per-subgroup views of RQ1/RQ2 are reproduced for comparison.

Model families implemented: Random Forest (`rf`), LightGBM (`lgb`), a
Multi-Layer Perceptron (`mlp`), Decision Tree (`dt`), Logistic Regression
(`lr`), an RBF Support Vector Machine (`svm`), and TabPFN v2 (`tabpfn`); the
default configuration runs all of them. Hyperparameters are tuned with a
**Weights & Biases** sweep. TabPFN needs a GPU in practice.

## Data

The dataset and the underlying feature/target engineering come from the previous
project [tail-unica/startup-survival](https://github.com/tail-unica/startup-survival),
whose original panel is derived from **PitchBook** data. Features and the
prediction target are computed under a strict temporal constraint (a fixed
look-back **time window**), so that every feature is computed only from
information available up to the decision point (i.e. free of look-ahead bias).
Refer to that repository and its accompanying paper for the full construction of
the dataset.

This repository ships `data/dataset_controlled.csv`, the bias-controlled dataset
of that project (`data/processed/dataset_controlled.csv` there): one row per
firm, with `CompanyID`, the binary `Target`, and the engineered features. The
categorical columns `HQCountry` and `PrimaryIndustrySector` are kept raw and
frequency-encoded after the split, on the training rows only; the raw sector
names label the industry-sector subgroups in RQ2/RQ3.

Due to PitchBook's licensing and intellectual-property restrictions, the original
raw panel is **not** redistributed here.

## Repository structure

```
.
├── config/
│   └── config.yaml              # Paths, time window, splits, seed, conformal grid, sweep
├── data/
│   └── dataset_controlled.csv   # Final bias-controlled dataset
├── src/
│   ├── utils.py                 # Conformal Risk Control, metrics, and plotting helpers
│   ├── encoding.py              # Frequency encoding fitted on the training split
│   └── models/                  # One wrapper per model family, keyed by model_type
├── notebook.ipynb               # Main reproducible pipeline (end to end)
├── pyproject.toml               # Dependencies and ruff configuration
├── uv.lock                      # Exact resolution of the dependencies
├── .env                         # Template for the environment variables
└── README.md
```

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) and requires
**Python 3.12**.

```bash
git clone https://github.com/tail-unica/conformal-startup-survival
cd conformal-startup-survival
uv sync          # creates .venv with the exact versions in uv.lock
```

Open `notebook.ipynb` with the `.venv` kernel. Lint and format with ruff:

```bash
uv run ruff check src
uv run ruff format src
```

## Configuration

Experiment tracking and hyperparameter sweeps use **Weights & Biases**.

If you don't have an account, create one at https://wandb.ai/site

Then follow the instructions on the website to create an **entity** and a **project**.

Fill the `.env` with your own entity and project.

Then log in once:

```bash
wandb login
```

All paths, the look-back window (`time_window`), the final year (`last_year`),
the random seed (`random_seed`), the train/calibration/validation/test fractions,
the hyperparameter grids, and the **conformal settings** are defined in
`config/config.yaml`. The `conformal` block holds the target-risk grid
(`alpha`) for the FNR and the range of `seeds` used for the calibration/test
re-splits in the robustness check.

## Step-by-step usage

The entire workflow is driven by **`notebook.ipynb`**. Proceed through the
sections in order.

### 1. Split, imputation and scaling

Loads `dataset_controlled.csv`, separates `CompanyID` and `Target`, and performs a
stratified split into train / validation / (calibration+test) using the seed and
fractions from `config.yaml`. The categorical columns are frequency-encoded on
the training split (`frequency_encoding` in `config.yaml`). Missing values are imputed with a KNN imputer and
features are scaled with a robust scaler (both fit on the training split only).
The calibration+test rows are kept together as a single **pool** that the
conformal step re-splits later.

### 2. Sweep creation and training

Initialize the W&B sweep, then start the agent. The training function is called
once per hyperparameter configuration: it trains the model, logs
validation/test metrics and plots to W&B, and stores the model's predicted
probabilities on the calibration+test pool (in `conformal_store` /
`conformal_pool_store`) for the conformal analysis.

`config.yaml` ships the **best hyperparameters** already found for each model, so
the default grid sweep reproduces the reported models directly. The commented
Bayesian sweep block can be enabled instead to search for new hyperparameters.

### 3. RQ1 — Global-level CRC

For each model, a single conformal threshold is calibrated per target `alpha`
and evaluated over many random calibration/test re-splits of the pool. The cells
produce the mean-over-seeds summary tables (`crc_summary_tables`), the target
`alpha` vs realized test FNR plot (`plot_conformal_fnr`), and per-metric
seed-to-seed boxplots (`plot_crc_metric_boxplots`). Points on or below the
`y = x` diagonal mean the risk is controlled.

### 4. RQ2 — Subgroups under global-level CRC

Using the single *marginal* threshold, the realized FNR is reported per industry
sector and per founding-year cohort (`fnr_by_group_table`). Where a group's
`FNR_marg` sits above the target `alpha`, that subgroup is under-covered by
marginal-only control. Results are shown as tables, grouped boxplots
(`plot_fnr_by_group_boxplots`), and per-model line plots
(`plot_fnr_by_group_lines`).

### 5. RQ3 — Group-level (Mondrian) CRC

A separate conformal threshold is calibrated per subgroup
(`run_conformal_fnr_group_calibrated`, and the `conditional` mode of
`fnr_by_group_table`), for both the industry-sector and the founding-year-cohort
groupings. The cells reproduce the RQ1 global tables/boxplots (now with
per-group thresholds applied and metrics pooled over the whole test half) and the
RQ2 per-subgroup views, so the conditional FNR (`FNR_cond`) can be compared
directly against the marginal one (`FNR_marg`).

## Reproducing the results

1. Install the environment and configure W&B (see above).
2. Run the split/imputation/scaling cell.
3. Launch the sweep with the best hyperparameters in `config.yaml` to train the
   models and populate the conformal stores.
4. Execute the RQ1, RQ2, and RQ3 cells to obtain the CRC control tables, the
   FNR-vs-`alpha` plots, and the per-subgroup marginal/conditional comparisons.

## Citation

If you use this code, please cite the accompanying work and the previous project
it builds on:

```bibtex
@article{startup_survival,
  title   = {The Future Is Not a Feature: A Look-Ahead Bias-Free Evaluation
             Framework for Startup Success Prediction through Machine Learning},
  authors  = {Casti Giulio and Mandas Marco and Piras Luca and Marras Mirko and Fenu Gianni},
  year    = {2026}
}
```

## License and data rights

The source code is released for academic and research use. The dataset is
inherited from the previous project
[tail-unica/startup-survival](https://github.com/tail-unica/startup-survival); the
original PitchBook-derived panel is **not** included here and remains subject to
PitchBook's licensing terms.
