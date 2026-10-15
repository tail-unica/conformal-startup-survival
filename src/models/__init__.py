"""The model families the sweep can train, keyed by the sweep's ``model_type``.

Adding a family means writing its class and adding one line here; the training
loop does not change.
"""

from __future__ import annotations

from src.models.base import Model
from src.models.mlp import MLP, MLPModel
from src.models.sklearn_models import (
    DecisionTreeModel,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
)
from src.models.tabpfn import TabPFNModel

#: Keys are exactly the values of sweep_settings.parameters.model_type in
#: config.yaml.
MODELS: dict[str, type[Model]] = {
    "rf": RandomForestModel,
    "lgb": LightGBMModel,
    "dt": DecisionTreeModel,
    "lr": LogisticRegressionModel,
    "svm": SVMModel,
    "mlp": MLPModel,
    "tabpfn": TabPFNModel,
}


def build(model_type: str, cfg, seed: int) -> Model:
    """Build the family named by ``model_type`` from one sweep run's config.

    :param model_type: Key of :data:`MODELS`, as config.yaml's sweep declares it.
    :param cfg: One sweep run's ``wandb.config``.
    :param seed: Seed of this run.
    :return: The built family.
    :raises ValueError: If ``model_type`` names no family.
    """
    try:
        family = MODELS[model_type]
    except KeyError:
        raise ValueError(
            f"unknown model_type {model_type!r}; expected one of {sorted(MODELS)}"
        ) from None
    return family.from_sweep(cfg, seed)


__all__ = [
    "MLP",
    "MODELS",
    "DecisionTreeModel",
    "LightGBMModel",
    "LogisticRegressionModel",
    "MLPModel",
    "Model",
    "RandomForestModel",
    "SVMModel",
    "TabPFNModel",
    "build",
]
