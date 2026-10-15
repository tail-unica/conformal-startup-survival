"""The five families that are thin wrappers over a scikit-learn estimator."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.base import Model


class RandomForestModel(Model):
    """Random forest, built from the ``rf_*`` keys of the sweep."""

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the forest from the ``rf_*`` keys."""
        return cls(
            RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                max_features="log2",
                class_weight="balanced",
                random_state=seed,
            ),
            seed,
        )


class LightGBMModel(Model):
    """Gradient-boosted trees, built from the ``lgb_*`` keys and the shared ones."""

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the booster; ``scale_pos_weight`` is left to :meth:`fit`."""
        # scale_pos_weight is set from the training split, so it is filled in by
        # fit() rather than here.
        return cls(
            lgb.LGBMClassifier(
                n_estimators=cfg.lgb_n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.lgb_max_depth,
                min_child_weight=cfg.min_child_weight,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                reg_alpha=cfg.reg_alpha,
                reg_lambda=cfg.reg_lambda,
                random_state=seed,
                verbose=-1,
            ),
            seed,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit, weighting the positive class by the training split's imbalance."""
        # Class imbalance is corrected by the negative-to-positive ratio of the
        # training split (80/20 -> 4), the LightGBM equivalent of the
        # class_weight="balanced" the other estimators take at construction.
        counts = y_train.value_counts()
        self.model.set_params(scale_pos_weight=float(counts[0] / counts[1]))
        self.model.fit(X_train, y_train)


class DecisionTreeModel(Model):
    """A single tree, built from the ``dt_*`` keys of the sweep."""

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the tree from the ``dt_*`` keys."""
        return cls(
            DecisionTreeClassifier(
                max_depth=cfg.dt_max_depth,
                min_samples_leaf=cfg.dt_min_samples_leaf,
                min_samples_split=cfg.dt_min_samples_split,
                criterion=cfg.dt_criterion,
                class_weight="balanced",
                random_state=seed,
            ),
            seed,
        )


class LogisticRegressionModel(Model):
    """Logistic regression on the scaled matrix, from the ``lr_*`` keys."""

    wants_scaled = True

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the regression from the ``lr_*`` keys."""
        return cls(
            LogisticRegression(
                C=cfg.lr_C,
                penalty=cfg.lr_penalty,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
            seed,
        )


class SVMModel(Model):
    """RBF-kernel support vector machine, from the ``svm_*`` keys."""

    # An RBF kernel compares distances, so unscaled features would let the
    # widest-range column dominate.
    wants_scaled = True

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the machine from the ``svm_*`` keys, with probabilities on."""
        return cls(
            SVC(
                C=cfg.svm_C,
                gamma=cfg.svm_gamma,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                cache_size=1000,
                random_state=seed,
            ),
            seed,
        )

    def score(self, X):
        """Decide at p >= 0.5, like every other family."""
        # SVC.predict() takes the sign of the decision function while
        # predict_proba() goes through Platt scaling, and the two disagree.
        # Every other model here decides at p >= 0.5, and the comparison table
        # is only meaningful if the SVM decides the same way.
        return self.score_by_threshold(X)
