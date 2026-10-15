"""TabPFN v2, run locally."""

from __future__ import annotations

from tabpfn import TabPFNClassifier

from src.models.base import Model

#: The same checkpoint the Prior Labs API served as "v2_default". Locally this
#: is a file name, downloaded to the TabPFN cache on first use (override the
#: location with TABPFN_MODEL_CACHE_DIR).
CHECKPOINT = "tabpfn-v2-classifier-v2_default.ckpt"


class TabPFNModel(Model):
    """TabPFN v2, built from the ``tabpfn_*`` keys of the sweep."""

    # Imputed but unscaled, like the trees: TabPFN normalises its inputs
    # internally and does not want RobustScaler output.
    wants_scaled = False

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the classifier from the ``tabpfn_*`` keys."""
        return cls(
            TabPFNClassifier(
                model_path=CHECKPOINT,
                n_estimators=cfg.tabpfn_n_estimators,
                balance_probabilities=True,
                # The v2 checkpoint declares a 10k-row pretraining limit and the
                # training split of the full dataset is larger, so without this
                # fit() refuses. The API applied a 50k limit to the same model,
                # so lifting it is what keeps the local run equivalent, and it
                # keeps the train set identical to the one the other models see
                # instead of subsampling it.
                ignore_pretraining_limits=True,
                # CUDA when the machine has it, CPU otherwise. TabPFN on CPU
                # past a thousand rows is impractical, so this model is meant
                # for the GPU server.
                device="auto",
                random_state=seed,
            ),
            seed,
        )

    def score(self, X):
        """Decide at p >= 0.5, reusing the single forward pass."""
        # For a binary task predict() is argmax(predict_proba), i.e. the same
        # p >= 0.5 rule, so calling it would pay for a second forward pass over
        # the same rows.
        return self.score_by_threshold(X)
