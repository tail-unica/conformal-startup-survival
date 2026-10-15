"""Frequency encoding of categorical features, fitted on a training split alone."""

import pandas as pd

DEFAULT_MIN_FREQUENCY = 0.05
DEFAULT_OTHER_LABEL = "Others"


def fit_frequency_encoding(
    train_series, min_frequency=DEFAULT_MIN_FREQUENCY, other_label=DEFAULT_OTHER_LABEL
):
    """
    Learns the training share of every category of one categorical column.

    Categories holding less than ``min_frequency`` of the training rows are
    pooled into a single ``other_label`` bucket, whose share is the sum of
    theirs. Missing values are part of the sample and go into the same bucket:
    they are rows the model will see, not rows to drop.

    Shares are relative (count / n_training_rows) rather than raw counts, so the
    value of a category does not depend on how many rows the training split has.
    Within one split the choice is immaterial: RobustScaler divides by the IQR
    and absorbs any constant factor.

    :param train_series:  categorical column of the training split.
    :param min_frequency: share below which a category is pooled into
                          ``other_label``. A category exactly at the threshold
                          is kept.
    :param other_label:   key holding the pooled share.
    :return:              dict {category: share}, always including
                          ``other_label`` (0.0 when nothing was pooled), whose
                          values sum to 1.
    """
    if not 0.0 <= min_frequency <= 1.0:
        raise ValueError(f"min_frequency must be in [0, 1], got {min_frequency}")

    series = pd.Series(train_series)
    n_rows = len(series)
    if n_rows == 0:
        raise ValueError("cannot fit a frequency encoding on an empty training split")

    shares = series.value_counts(dropna=True) / n_rows

    kept = shares[shares >= min_frequency]
    encoding = {category: float(share) for category, share in kept.items()}
    # everything not kept (rare categories and the missing values excluded by
    # value_counts) is what is left of the sample
    encoding[other_label] = float(1.0 - kept.sum())
    return encoding


def apply_frequency_encoding(series, encoding, other_label=DEFAULT_OTHER_LABEL):
    """
    Replaces each category with its training share, as learned by
    :func:`fit_frequency_encoding`.

    A category that is missing, unseen in training, or pooled at fit time all
    resolve to the same ``other_label``.

    :param series:      categorical column of any split.
    :param encoding:    mapping returned by :func:`fit_frequency_encoding`.
    :param other_label: key of the pooled share inside ``encoding``.
    :return:            float Series of shares, on the input's index.
    """
    other_share = encoding.get(other_label, 0.0)
    lookup = {k: v for k, v in encoding.items() if k != other_label}

    series = pd.Series(series)
    encoded = series.map(lookup)
    return encoded.astype(float).fillna(other_share)
