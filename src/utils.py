import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def set_seed(seed):
    """
    Fixes all relevant random sources so that experiments (MLP training,
    DataLoader shuffling, SHAP sampling) are fully reproducible.
    Sklearn / LightGBM models also rely on this through numpy.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensors(X, y):
    """
    Converts input features and target labels to PyTorch tensors.

    :param X: Input features as a NumPy array (or tensor-compatible structure).
    :param y: Target labels, can be a pandas Series or a NumPy array.
    :return: A tuple containing the input features and target labels as PyTorch tensors.
    """

    y_np = y.values if hasattr(y, "values") else y
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32).unsqueeze(1),
    )


def get_probs(loader, model, device):
    """
    Computes the predicted probabilities and true labels for a given data loader, model, and device.

    :param loader: A PyTorch DataLoader that provides batches of input features and target labels.
    :param model: A PyTorch model that will be used to make predictions on the input features.
    :param device: The device (e.g., 'cpu' or 'cuda') on which the model and data will be processed.
    :return: A tuple containing two NumPy arrays: the predicted probabilities and the true labels.
    """

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    return np.array(all_probs).flatten(), np.array(all_labels).flatten().astype(int)


def _order_metrics(keys):
    """
    Reorder metric keys to the paper's column order:
    AUC, F1 (test), Prec (test), Rec (test), Acc (test), Acc (train).
    Matching is case-insensitive on substrings; unmatched keys are appended.
    """

    def _slot(k):
        s = k.lower().replace("_", " ").replace("-", " ")
        is_train = "train" in s
        if "auc" in s:
            return 0
        if "f1" in s or "f_1" in s or s.strip() == "f1":
            return 1
        if "prec" in s:
            return 2
        if "rec" in s:  # recall (not 'prec')
            return 3
        if "acc" in s:
            return 5 if is_train else 4
        return 99

    return sorted(keys, key=lambda k: (_slot(k), k.lower()))


def conformal_fnr_threshold(probs_cal, labels_cal, alpha, pos_label=1):
    """
    Conformal Risk Control (Angelopoulos et al., 2022) threshold that controls
    the False Negative Rate of the positive class at level ``alpha``.

    Positive class = ``pos_label``. Decision rule downstream: predict positive
    when ``prob >= t``. The FNR is monotone non-decreasing in the threshold, so
    we return the LARGEST threshold ``t*`` whose CRC upper bound stays <= alpha:

        (n * R_hat(t) + 1) / (n + 1) <= alpha

    where ``n`` is the number of positive calibration examples and ``R_hat(t)``
    the empirical FNR on those positives. Guarantees ``E[FNR_test(t*)] <= alpha``.

    If ``alpha < 1/(n+1)`` the target is infeasible for this calibration size and
    ``0.0`` is returned (predict everything positive, best effort).

    :param probs_cal:  calibration predicted probabilities of the positive class.
    :param labels_cal: calibration true labels.
    :param alpha:      target risk level for the FNR.
    :param pos_label:  label of the positive class (default 1 = success).
    :return:           the conformal threshold ``t*`` (float).
    """
    probs = np.asarray(probs_cal, dtype=float)
    labels = np.asarray(labels_cal)
    pos_probs = probs[labels == pos_label]
    n = len(pos_probs)
    if n == 0:
        return 0.0

    # Candidate thresholds where R_hat steps, plus 0.0 (predict all positive).
    candidates = np.concatenate([[0.0], np.sort(np.unique(pos_probs))])

    best_t = 0.0
    for t in candidates:
        r_hat = np.mean(pos_probs < t)
        bound = (n * r_hat + 1.0) / (n + 1.0)
        if bound <= alpha:
            best_t = t  # keep the largest feasible threshold
        else:
            break  # bound is non-decreasing in t -> stop
    return float(best_t)


def resolve_seeds(spec):
    """
    Normalize a ``conformal.seeds`` config value into a concrete list of ints.

    ``conformal.seeds`` is consumed directly by :func:`run_conformal_fnr_experiment`
    (it is NOT a W&B sweep parameter), so it must resolve to actual integers.
    Accepted forms:
    - list/tuple of ints           -> used as-is,
    - single int ``n``             -> ``[n]``,
    - dict ``{"min": a, "max": b}``-> ``list(range(a, b + 1))`` (inclusive),
      any ``distribution`` key is ignored; a ``values`` list is used directly.

    :param spec: the raw config value.
    :return:     list of int seeds.
    :raises ValueError: if the spec cannot be interpreted.
    """
    if isinstance(spec, dict):
        if "values" in spec:
            return [int(s) for s in spec["values"]]
        if "min" in spec and "max" in spec:
            return list(range(int(spec["min"]), int(spec["max"]) + 1))
        raise ValueError(
            "conformal.seeds dict must contain 'values' or both 'min' and 'max'; "
            f"got keys {list(spec.keys())}"
        )
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [int(s) for s in spec]
    raise ValueError(f"Unsupported conformal.seeds spec: {spec!r}")


def _classification_metrics(pred_pos, pos_mask, n_pos, n_total):
    """Classification metrics for a boolean prediction vector."""
    tp = int(np.sum(pred_pos & pos_mask))
    fp = int(np.sum(pred_pos & ~pos_mask))
    fn = n_pos - tp
    tn = n_total - tp - fp - fn
    n_neg = n_total - n_pos
    n_pred_pos = tp + fp
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / n_pred_pos if n_pred_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, FNR=fnr, FPR=fpr)


def run_conformal_fnr_experiment(probs_pool, labels_pool, alphas, seeds, pos_label=1):
    """
    Verify conformal FNR control across several random calibration/test splits.

    The pool (calibration + test rows) is fixed; the model probabilities are
    computed once. For every ``seed`` the pool is re-split 50/50 (stratified) into
    a calibration and a test half; for every ``alpha`` a conformal threshold is
    fit on the calibration half (:func:`conformal_fnr_threshold`) and the realized
    FNR (and recall/precision/positive-rate) is measured on the test half.

    :param probs_pool:  predicted positive-class probabilities on the pool.
    :param labels_pool: true labels on the pool.
    :param alphas:      iterable of target FNR levels.
    :param seeds:       iterable of random seeds for the cal/test re-split.
    :param pos_label:   positive class label (default 1 = success).
    :return: tidy long-format DataFrame with ``k + 1`` rows per seed (``k`` =
             number of alphas): one *base* row (no calibration, default 0.5
             threshold) plus one *conformal* row per target alpha (threshold
             ``t*``). Columns:
             - ``seed``,
             - ``calibrated`` (bool): ``False`` on the base row, ``True`` on the
               conformal rows,
             - ``alpha`` (NaN on the base row),
             - ``threshold`` (0.5 on the base row, ``t*`` on conformal rows),
             - ``infeasible`` (only meaningful on conformal rows),
             - ``AUC`` (threshold-free, on this seed's test half; reported only
               on the base row, NaN on conformal rows),
             - ``accuracy``, ``precision``, ``recall``, ``f1``, ``FNR``, ``FPR``
               at the row's operating point.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    seeds = resolve_seeds(seeds)
    probs = np.asarray(probs_pool, dtype=float)
    labels = np.asarray(labels_pool)
    idx = np.arange(len(labels))

    rows = []
    for seed in seeds:
        cal_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=labels, random_state=seed)
        p_cal, y_cal = probs[cal_idx], labels[cal_idx]
        p_test, y_test = probs[test_idx], labels[test_idx]

        n_pos_cal = int(np.sum(y_cal == pos_label))
        pos_mask = y_test == pos_label
        n_pos_test = int(np.sum(pos_mask))
        n_total = len(y_test)

        # AUC is threshold-free -> computed once on this seed's test half and
        # reported on the base row only (NaN on the conformal rows).
        try:
            auc_val = roc_auc_score(y_test, p_test)
        except ValueError:
            auc_val = float("nan")

        # Base row: no calibration, default 0.5 threshold.
        base = _classification_metrics(p_test >= 0.5, pos_mask, n_pos_test, n_total)
        base_row = {
            "seed": seed,
            "calibrated": False,
            "alpha": float("nan"),
            "threshold": 0.5,
            "infeasible": False,
            "AUC": auc_val,
        }
        base_row.update(base)
        rows.append(base_row)

        # One conformal row per target alpha, at the CRC threshold t*.
        for alpha in alphas:
            t = conformal_fnr_threshold(p_cal, y_cal, alpha, pos_label)
            conf = _classification_metrics(p_test >= t, pos_mask, n_pos_test, n_total)
            conf_row = {
                "seed": seed,
                "calibrated": True,
                "alpha": alpha,
                "threshold": t,
                "infeasible": alpha < 1.0 / (n_pos_cal + 1.0),
                "AUC": float("nan"),
            }
            conf_row.update(conf)
            rows.append(conf_row)

    cols = [
        "seed",
        "calibrated",
        "alpha",
        "threshold",
        "infeasible",
        "AUC",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "FNR",
        "FPR",
    ]
    return pd.DataFrame(rows)[cols]


def run_conformal_fnr_by_group(
    probs_pool, labels_pool, groups_pool, alphas, seeds, mode="marginal", pos_label=1, min_group=30
):
    """
    Group-aware conformal FNR evaluation across random cal/test re-splits.

    Like :func:`run_conformal_fnr_experiment`, the pool is re-split 50/50
    (stratified on the label) for every ``seed``. For every ``alpha`` a
    threshold is fit on the calibration half and the realized FNR is measured
    on the test half, but here **per group**:

    - ``mode="marginal"``: ONE threshold is calibrated on the whole calibration
      half (:func:`conformal_fnr_threshold`) and its realized FNR is reported
      within each group of the test half. Shows how marginal control degrades
      across subgroups.
    - ``mode="conditional"`` (Mondrian): a SEPARATE threshold is calibrated on
      each group's calibration rows and applied within that group on the test
      half, controlling the FNR within every group.

    A group is flagged ``infeasible`` when it has fewer than ``min_group``
    positive calibration examples, or when ``alpha < 1/(n_pos_cal + 1)`` so the
    target is unreachable for that calibration size. In ``"conditional"`` mode an
    infeasible group falls back to the marginal threshold (a per-group threshold
    fitted on a handful of positives is noise); the rows are still returned and
    flagged, so callers can exclude them from control checks.

    :param probs_pool:  predicted positive-class probabilities on the pool.
    :param labels_pool: true labels on the pool.
    :param groups_pool: group label per pool row (e.g. cohort or sector).
    :param alphas:      iterable of target FNR levels.
    :param seeds:       seeds (or a spec understood by :func:`resolve_seeds`).
    :param mode:        ``"marginal"`` or ``"conditional"``.
    :param pos_label:   positive class label (default 1 = success).
    :param min_group:   min positive calibration examples for a feasible group.
    :return: tidy DataFrame, one row per ``(seed, alpha, group)`` with columns
             ``seed, alpha, group, threshold, infeasible, n_pos, n_total,
             FNR_conf, recall_conf, pos_rate_conf``. ``recall_conf == 1 -
             FNR_conf``; if a group has no test positives in a seed both are NaN
             (undefined, not 0), so mean-over-seeds skips that seed instead of
             being pulled toward 0.
    """
    from sklearn.model_selection import train_test_split

    if mode not in ("marginal", "conditional"):
        raise ValueError(f"mode must be 'marginal' or 'conditional', got {mode!r}")

    seeds = resolve_seeds(seeds)
    probs = np.asarray(probs_pool, dtype=float)
    labels = np.asarray(labels_pool)
    groups = np.asarray(groups_pool)
    idx = np.arange(len(labels))
    unique_groups = np.unique(groups)

    rows = []
    for seed in seeds:
        cal_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=labels, random_state=seed)
        p_cal, y_cal, g_cal = probs[cal_idx], labels[cal_idx], groups[cal_idx]
        p_test, y_test, g_test = probs[test_idx], labels[test_idx], groups[test_idx]

        for alpha in alphas:
            t_marg = conformal_fnr_threshold(p_cal, y_cal, alpha, pos_label)

            for g in unique_groups:
                cal_g = g_cal == g
                n_pos_cal_g = int(np.sum(y_cal[cal_g] == pos_label))
                infeasible = n_pos_cal_g < min_group or alpha < 1.0 / (n_pos_cal_g + 1.0)

                if mode == "conditional" and not infeasible:
                    t = conformal_fnr_threshold(p_cal[cal_g], y_cal[cal_g], alpha, pos_label)
                else:
                    t = t_marg

                test_g = g_test == g
                yt = y_test[test_g]
                pt = p_test[test_g]
                pos_mask = yt == pos_label
                n_pos = int(np.sum(pos_mask))
                n_total = int(len(yt))
                pred_pos = pt >= t
                tp = int(np.sum(pred_pos & pos_mask))
                n_pred_pos = int(np.sum(pred_pos))

                fnr = (n_pos - tp) / n_pos if n_pos > 0 else float("nan")
                recall = tp / n_pos if n_pos > 0 else float("nan")
                pos_rate = n_pred_pos / n_total if n_total > 0 else float("nan")

                rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "group": g,
                        "threshold": float(t),
                        "infeasible": bool(infeasible),
                        "n_pos": n_pos,
                        "n_total": n_total,
                        "FNR_conf": fnr,
                        "recall_conf": recall,
                        "pos_rate_conf": pos_rate,
                    }
                )

    return pd.DataFrame(rows)


def plot_conformal_fnr(df, model_name=None):
    """
    Plot target alpha vs realized test FNR for the conformal FNR experiment.

    Shows the mean realized FNR over seeds with a +/- std band, the per-seed
    points, and the ``y = x`` reference: points on or below the diagonal mean the
    risk is controlled (``E[FNR_test] <= alpha``).

    :param df:         DataFrame from :func:`run_conformal_fnr_experiment`.
    :param model_name: optional model label for the title.
    :return:           matplotlib Figure.
    """
    conf = df[df["calibrated"]]
    grp = conf.groupby("alpha")["FNR"]
    alphas = np.array(sorted(conf["alpha"].unique()), dtype=float)
    mean = grp.mean().loc[alphas].values
    std = grp.std(ddof=0).loc[alphas].values

    fig, ax = plt.subplots(figsize=(7, 6))

    lo = min(alphas.min(), conf["FNR"].min())
    hi = max(alphas.max(), conf["FNR"].max())
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", label="target (y = x)")

    ax.scatter(conf["alpha"], conf["FNR"], color="#1f77b4", alpha=0.35, s=20, label="per-seed FNR")
    ax.plot(alphas, mean, color="#c62828", lw=2, marker="o", label="mean realized FNR")
    ax.fill_between(
        alphas, mean - std, mean + std, color="#c62828", alpha=0.15, label="+/- std over seeds"
    )

    ax.set_xlabel("Target risk alpha")
    ax.set_ylabel("Realized test FNR")
    title = "Conformal FNR control"
    if model_name is not None:
        title += f" - {str(model_name).upper()}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


CRC_CATEGORICAL = [
    "#2a78d6",
    "#008300",
    "#e87ba4",
    "#eda100",
    "#1baf7a",
    "#eb6834",
    "#4a3aa7",
    "#e34948",
]


def sorted_groups(values):
    """Unique non-null group labels, sorted numerically when possible."""
    uniq = [g for g in pd.unique(values) if pd.notna(g)]
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=str)


def assign_model_colors(models, palette=CRC_CATEGORICAL):
    """Fixed categorical hue per model, first-seen order (color follows the
    model, not its rank), so a model keeps its color across every figure."""
    order = []
    for m in models:
        if m not in order:
            order.append(m)
    return {m: palette[i % len(palette)] for i, m in enumerate(order)}


def fnr_by_group_table(probs_pool, labels_pool, groups, alphas, seeds, min_group=30):
    """Mean-over-seeds per-(group, alpha) FNR table plus the raw per-seed
    marginal and conditional frames.

    Returns ``(tab, raw_marg, raw_cond)``. ``tab`` holds the marginal and
    conditional realized FNR (mean over seeds), the coverage gaps above alpha
    (blank where non-positive) and the pool group sizes. ``raw_marg`` and
    ``raw_cond`` are the tidy per-``(seed, alpha, group)`` frames (both carry a
    ``FNR_conf`` column) for the per-subgroup boxplots and line plots."""
    raw_marg = run_conformal_fnr_by_group(
        probs_pool, labels_pool, groups, alphas, seeds, mode="marginal", min_group=min_group
    )
    raw_cond = run_conformal_fnr_by_group(
        probs_pool, labels_pool, groups, alphas, seeds, mode="conditional", min_group=min_group
    )

    def _agg(raw, col):
        fnr = raw.groupby(["group", "alpha"])["FNR_conf"].mean().round(3).rename(col).reset_index()
        infeas = (
            raw.groupby(["group", "alpha"])["infeasible"].max().rename("infeasible").reset_index()
        )
        return fnr.merge(infeas, on=["group", "alpha"])

    marg = _agg(raw_marg, "FNR_marg").drop(columns="infeasible")
    cond = _agg(raw_cond, "FNR_cond")
    tab = marg.merge(cond, on=["group", "alpha"])

    tab["FNR_marg - alpha"] = (tab["FNR_marg"] - tab["alpha"]).where(lambda s: s > 0).round(3)
    tab["FNR_cond - alpha"] = (tab["FNR_cond"] - tab["alpha"]).where(lambda s: s > 0).round(3)

    pool_counts = pd.DataFrame({"group": np.asarray(groups), "label": np.asarray(labels_pool)})
    sizes = pd.concat(
        [
            pool_counts.groupby("group").size().rename("n_pool"),
            pool_counts.groupby("group")["label"].sum().astype(int).rename("n_pos_pool"),
        ],
        axis=1,
    ).reset_index()
    tab = tab.merge(sizes, on="group")
    tab = tab.sort_values(["alpha", "group"]).reset_index(drop=True)
    return tab, raw_marg, raw_cond


def run_conformal_fnr_group_calibrated(
    probs_pool, labels_pool, groups_pool, alphas, seeds, pos_label=1, min_group=30
):
    """Group-level (Mondrian) CRC evaluated with GLOBAL pooled metrics.

    Mirrors :func:`run_conformal_fnr_experiment` (same 50/50 stratified re-split
    per seed, same base 0.5 row and schema) but, for every conformal row, a
    SEPARATE threshold is calibrated on each group's calibration rows and applied
    within that group on the test half; the six metrics are then measured on the
    whole (pooled) test half. Use it for the RQ3 "as-RQ1" global tables/boxplots.

    ``threshold`` is NaN on the group-calibrated rows (no single scalar);
    ``infeasible`` is True if any group was infeasible for that alpha. Infeasible
    groups fall back to the marginal threshold, as in
    :func:`run_conformal_fnr_by_group`.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    seeds = resolve_seeds(seeds)
    probs = np.asarray(probs_pool, dtype=float)
    labels = np.asarray(labels_pool)
    groups = np.asarray(groups_pool)
    idx = np.arange(len(labels))
    unique_groups = np.unique(groups)

    rows = []
    for seed in seeds:
        cal_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=labels, random_state=seed)
        p_cal, y_cal, g_cal = probs[cal_idx], labels[cal_idx], groups[cal_idx]
        p_test, y_test, g_test = probs[test_idx], labels[test_idx], groups[test_idx]

        pos_mask = y_test == pos_label
        n_pos_test = int(np.sum(pos_mask))
        n_total = len(y_test)
        try:
            auc_val = roc_auc_score(y_test, p_test)
        except ValueError:
            auc_val = float("nan")

        base = _classification_metrics(p_test >= 0.5, pos_mask, n_pos_test, n_total)
        base_row = {
            "seed": seed,
            "calibrated": False,
            "alpha": float("nan"),
            "threshold": 0.5,
            "infeasible": False,
            "AUC": auc_val,
        }
        base_row.update(base)
        rows.append(base_row)

        for alpha in alphas:
            pred_pos = np.zeros(n_total, dtype=bool)
            any_infeasible = False
            t_marg = conformal_fnr_threshold(p_cal, y_cal, alpha, pos_label)
            for g in unique_groups:
                cal_g = g_cal == g
                test_g = g_test == g
                n_pos_cal_g = int(np.sum(y_cal[cal_g] == pos_label))
                if n_pos_cal_g < min_group or alpha < 1.0 / (n_pos_cal_g + 1.0):
                    any_infeasible = True
                    t = t_marg
                else:
                    t = conformal_fnr_threshold(p_cal[cal_g], y_cal[cal_g], alpha, pos_label)
                pred_pos[test_g] = p_test[test_g] >= t
            conf = _classification_metrics(pred_pos, pos_mask, n_pos_test, n_total)
            conf_row = {
                "seed": seed,
                "calibrated": True,
                "alpha": alpha,
                "threshold": float("nan"),
                "infeasible": any_infeasible,
                "AUC": float("nan"),
            }
            conf_row.update(conf)
            rows.append(conf_row)

    cols = [
        "seed",
        "calibrated",
        "alpha",
        "threshold",
        "infeasible",
        "AUC",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "FNR",
        "FPR",
    ]
    return pd.DataFrame(rows)[cols]


def crc_summary_tables(df_conf):
    """Two mean-over-seeds summary tables from a CRC experiment frame: the
    pre-calibration base row (0.5 threshold; AUC is threshold-free and lives
    here) and one post-calibration row per target alpha."""
    base_cols = ["AUC", "accuracy", "precision", "recall", "f1", "FNR", "FPR"]
    base = df_conf[~df_conf["calibrated"]][base_cols].mean().round(3).to_frame("mean").T
    base.index = ["threshold 0.5 (pre)"]
    conf = (
        df_conf[df_conf["calibrated"]]
        .groupby("alpha")[["threshold", "accuracy", "precision", "recall", "f1", "FNR", "FPR"]]
        .mean()
        .round(3)
    )
    return base, conf


def plot_crc_metric_boxplots(
    model_df,
    metrics=("accuracy", "precision", "recall", "f1", "FNR", "FPR"),
    title_prefix="Global-level CRC",
):
    """One figure per metric: seed-to-seed distribution grouped by model; within
    each model group the first (gray) box is the uncalibrated baseline (0.5),
    then one box per target alpha at the conformal threshold. ``model_df`` maps
    model name -> its CRC experiment frame."""

    models = list(model_df)
    alphas = sorted(
        {a for df in model_df.values() for a in df.loc[df["calibrated"], "alpha"].unique()}
    )
    base_color = "#c9c8c2"
    ramp = plt.cm.Blues(np.linspace(0.45, 0.9, len(alphas)))
    box_colors = [base_color] + [tuple(c) for c in ramp]
    box_labels = ["no conformal"] + [f"$\\alpha$ = {a:g}" for a in alphas]
    n_box = len(box_colors)
    gap = 1.0
    group_span = n_box + gap
    INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

    def _series(df, metric, alpha):
        if alpha is None:
            return df.loc[~df["calibrated"], metric].to_numpy()
        return df.loc[df["calibrated"] & (df["alpha"] == alpha), metric].to_numpy()

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(0.55 * len(models) * n_box + 3, 4.2))
        positions, data, colors = [], [], []
        for gi, model_type in enumerate(models):
            df = model_df[model_type]
            for j, alpha in enumerate([None, *alphas]):
                positions.append(gi * group_span + j)
                data.append(_series(df, metric, alpha))
                colors.append(box_colors[j])
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=INK, linewidth=1.4),
            whiskerprops=dict(color=MUTED, linewidth=1.0),
            capprops=dict(color=MUTED, linewidth=1.0),
            boxprops=dict(linewidth=0.0),
        )
        for patch, c in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(c)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.8)
        centers = [gi * group_span + (n_box - 1) / 2 for gi in range(len(models))]
        ax.set_xticks(centers)
        ax.set_xticklabels([m.upper() for m in models], fontsize=11, color=INK)
        ax.tick_params(axis="x", length=0)
        for gi in range(1, len(models)):
            ax.axvline(gi * group_span - (gap + 1) / 2, color=GRID, linewidth=0.8)
        ax.set_ylabel(metric, fontsize=11, color=INK)
        ax.set_title(
            f"{title_prefix} — {metric} (distribution over seeds)", fontsize=12, color=INK, pad=10
        )
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="y", colors=MUTED, labelcolor=INK)
        ax.legend(
            handles=[
                Patch(facecolor=c, edgecolor="white", label=lab)
                for c, lab in zip(box_colors, box_labels, strict=True)
            ],
            frameon=False,
            fontsize=9,
            ncol=n_box,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
        )
        fig.tight_layout()
        plt.show()


def plot_fnr_by_group_boxplots(
    raw_by_model, dim_name, model_color, ylabel="realized FNR (marginal)"
):
    """One boxplot figure per target alpha, models side by side. X-axis is the
    subgroups; within each cluster one box per model (seed-to-seed FNR), colored
    by the fixed per-model hue. Dashed line at the target alpha."""
    from matplotlib.lines import Line2D

    INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    items = list(raw_by_model.items())
    n_box = len(items)
    gap = 1.0
    span = n_box + gap
    all_groups = sorted_groups(pd.concat([r["group"] for _, r in items]))
    plot_alphas = sorted({a for _, r in items for a in r["alpha"].unique()})

    for alpha in plot_alphas:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(all_groups) * n_box + 3), 5.6))
        positions, data, colors = [], [], []
        for gi, g in enumerate(all_groups):
            for j, (model_type, raw) in enumerate(items):
                sub = raw[(raw["alpha"] == alpha) & (raw["group"] == g)]
                positions.append(gi * span + j)
                data.append(sub["FNR_conf"].dropna().to_numpy())
                colors.append(model_color[model_type])
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.8,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=INK, linewidth=1.2),
            whiskerprops=dict(color=MUTED, linewidth=1.0),
            capprops=dict(color=MUTED, linewidth=1.0),
            boxprops=dict(linewidth=0.0),
        )
        for patch, c in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(c)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.8)
        ax.axhline(alpha, color="#c1121f", lw=1.3, ls="--", zorder=5)
        centers = [gi * span + (n_box - 1) / 2 for gi in range(len(all_groups))]
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [str(g) for g in all_groups], rotation=45, ha="right", fontsize=9, color=INK
        )
        ax.tick_params(axis="x", length=0)
        for gi in range(1, len(all_groups)):
            ax.axvline(gi * span - (gap + 1) / 2, color=GRID, linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11, color=INK)
        ax.set_title(f"{dim_name} — $\\alpha$ = {alpha:g}\n", fontsize=12, color=INK, pad=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        handles = [
            Patch(facecolor=model_color[mt], edgecolor="white", label=mt.upper()) for mt, _ in items
        ]
        handles.append(
            Line2D(
                [0], [0], color="#c1121f", lw=1.3, ls="--", label=f"target $\\alpha$ = {alpha:g}"
            )
        )
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=9,
            ncol=min(len(handles), 4),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.5),
        )
        fig.tight_layout()
        plt.show()


def plot_fnr_by_group_lines(raw_by_model, dim_name, model_color, ylabel="realized FNR (marginal)"):
    """One line-plot figure per target alpha: one line per model (saturated mean
    over seeds inside a lighter +/-1 std band). Dashed line at the target alpha.
    Numeric groups (cohorts) use their value on x; non-numeric fall back to
    ordinal positions."""
    INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    all_groups = sorted_groups(pd.concat([r["group"] for r in raw_by_model.values()]))
    try:
        x = np.array(all_groups, dtype=float)
    except (TypeError, ValueError):
        x = np.arange(len(all_groups), dtype=float)
    plot_alphas = sorted({a for r in raw_by_model.values() for a in r["alpha"].unique()})

    for alpha in plot_alphas:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(all_groups) + 2), 4.6))
        for model_type, raw in raw_by_model.items():
            g = raw[raw["alpha"] == alpha].groupby("group")["FNR_conf"]
            mean = g.mean().reindex(all_groups)
            std = g.std(ddof=0).reindex(all_groups).fillna(0.0)
            color = model_color[model_type]
            lo = (mean - std).to_numpy(dtype=float)
            hi = (mean + std).to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, lo, color=color, lw=0.8, alpha=0.55)
            ax.plot(x, hi, color=color, lw=0.8, alpha=0.55)
            ax.plot(x, mean.to_numpy(dtype=float), color=color, lw=2.2, label=model_type.upper())
        ax.axhline(alpha, color="#c1121f", lw=1.3, ls="--", label=f"target $\\alpha$ = {alpha:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(g) for g in all_groups], rotation=45, ha="right", fontsize=9, color=INK
        )
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel, fontsize=11, color=INK)
        ax.set_title(f"{dim_name} — $\\alpha$ = {alpha:g}\n", fontsize=12, color=INK, pad=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.legend(
            frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18)
        )
        fig.tight_layout()
        plt.show()
