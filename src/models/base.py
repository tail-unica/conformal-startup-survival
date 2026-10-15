"""Common interface for the model families the sweep trains.

A family is defined by two answers:

- which matrix it is fitted on: imputed-only, or imputed and scaled;
- how it turns probabilities into labels.

One subclass gives both, so the training loop never branches on the model
type and a new family is a new file plus one line in the registry.
"""

from __future__ import annotations


class Model:
    """A model family: how to build it, fit it and score it.

    Subclasses set ``wants_scaled`` and implement ``from_sweep``.
    The default ``fit``/``predict``/``predict_proba`` cover any estimator with
    the scikit-learn API; the families that deviate override them.
    """

    #: True when the family is fitted on RobustScaler output rather than on the
    #: imputed-but-unscaled matrix. Distance- and gradient-based learners (LR,
    #: SVM, MLP) need it; trees and TabPFN do not.
    wants_scaled: bool = False

    def __init__(self, model, seed: int):
        """Wrap a built estimator.

        :param model: The estimator, already constructed with its hyperparameters.
        :param seed: Seed of this run, kept for the families that draw random numbers.
        """
        self.model = model
        self.seed = seed

    @classmethod
    def from_sweep(cls, cfg, seed: int) -> Model:
        """Build the model from one sweep run's ``wandb.config``.

        Each subclass reads only the keys carrying its own family prefix
        (``rf_*``, ``lgb_*``, ``svm_*``, …), which is why config.yaml's
        sweep_settings block maps onto these classes unchanged.

        :param cfg: One sweep run's ``wandb.config``.
        :param seed: Seed of this run.
        :return: The built family.
        """
        raise NotImplementedError

    def matrices(self, split: dict) -> tuple:
        """Pick the matrices this family is fitted and scored on.

        :param split: The split dict built in the notebook
            (``X_{train,val,test}_{imp,scaled}``).
        :return: ``(train, val, test)``, scaled or merely imputed.
        """
        if self.wants_scaled:
            return split["X_train_scaled"], split["X_val_scaled"], split["X_test_scaled"]
        return split["X_train_imp"], split["X_val_imp"], split["X_test_imp"]

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        """Fit the model.

        :param X_train: Training design matrix.
        :param y_train: Training labels.
        :param X_val: Validation matrix; only the MLP uses it, for early stopping.
        :param y_val: Validation labels, same.
        """
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Score the positive class.

        :param X: Design matrix.
        :return: Probability of the positive class, as a 1-D array.
        """
        return self.model.predict_proba(X)[:, 1]

    def score(self, X):
        """Probabilities and labels for one matrix, as ``(probs, preds)``.

        Both come from one method so that the families whose inference is
        expensive pay for a single pass. The default keeps the estimator's own
        decision rule; the families where that rule disagrees with thresholding
        their own probabilities override this.

        :param X: Design matrix.
        :return: ``(probs, preds)``.
        """
        return self.predict_proba(X), self.model.predict(X)

    def score_by_threshold(self, X):
        """What ``score`` returns for the families that decide at p >= 0.5.

        Shared rather than repeated: they override ``score`` to call it.

        :param X: Design matrix.
        :return: ``(probs, preds)``.
        """
        probs = self.predict_proba(X)
        return probs, threshold_at_half(probs)


def threshold_at_half(probs):
    """Label rows by ``p >= 0.5``.

    Used by the families whose ``predict`` would otherwise disagree with their
    own probabilities: SVC decides on the sign of the decision function while
    ``predict_proba`` goes through Platt scaling, and for the MLP and TabPFN
    calling ``predict`` would pay for a second forward pass over the same rows.

    :param probs: Probabilities of the positive class.
    :return: Integer labels.
    """
    return (probs >= 0.5).astype(int)
