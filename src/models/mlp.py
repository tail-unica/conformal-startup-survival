"""The Multi-Layer Perceptron: the network and the family that trains it."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import Model
from src.utils import get_probs, to_tensors

EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


class MLP(nn.Module):
    """The network itself: tanh layers with optional batch norm and dropout."""

    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3, batch_norm=True):
        """Stack the layers.

        :param input_size: Number of features.
        :param hidden_sizes: Width of each hidden layer, or one int for a single one.
        :param dropout_rate: Dropout applied after every hidden layer.
        :param batch_norm: Whether to insert a batch-norm layer after each linear one.
        """
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        """Run the batch through the stack.

        :param x: Input batch.
        :return: Logits, one per row.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLPModel(Model):
    """The family that trains :class:`MLP`, from the ``mlp_*`` keys of the sweep.

    The network is built in :meth:`fit`, because its input size is known only
    once the design matrix is in hand.
    """

    wants_scaled = True

    def __init__(self, hidden_sizes, learning_rate, dropout_rate, weight_decay, batch_size, seed):
        """Keep the hyperparameters until :meth:`fit` can build the network.

        :param hidden_sizes: Width of each hidden layer.
        :param learning_rate: Adam learning rate.
        :param dropout_rate: Dropout applied after every hidden layer.
        :param weight_decay: Adam weight decay.
        :param batch_size: Rows per batch.
        :param seed: Seed of this run, used for the weights and the shuffle order.
        """
        super().__init__(model=None, seed=seed)
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_sweep(cls, cfg, seed):
        """Build the family from the ``mlp_*`` keys and the shared ones."""
        return cls(
            hidden_sizes=[int(x) for x in cfg.hidden_sizes.split(",")],
            learning_rate=cfg.mlp_learning_rate,
            dropout_rate=cfg.dropout_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            seed=seed,
        )

    def _loader(self, X, y, shuffle=False):
        """Wrap a matrix and its labels in a DataLoader.

        :param X: Design matrix.
        :param y: Labels.
        :param shuffle: Whether to shuffle, with a seeded generator.
        :return: The loader.
        """
        X_t, y_t = to_tensors(X, y)
        if not shuffle:
            return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size)
        # Seeded generator so that shuffle order is reproducible across runs.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the network, stopping early on the validation loss.

        :param X_train: Training matrix, scaled.
        :param y_train: Training labels.
        :param X_val: Validation matrix, scaled; required here, unlike the other
            families, because early stopping reads it.
        :param y_val: Validation labels.
        """
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val)

        self.model = MLP(
            input_size=X_train.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
            batch_norm=True,
        ).to(self.device)

        # Class weight: pos_weight = n_neg / n_pos (loss balancing)
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_auc = 0.0
        patience_counter = 0

        # Temporary path for best model checkpoint
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        best_model_path = os.path.join(tmp_dir, "best_model.pt")

        for epoch in range(EPOCHS):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(X_batch)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_auc = roc_auc_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

            scheduler.step(val_auc)

            print(
                f"Epoch {epoch + 1:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # Early stopping on validation AUC, keeping the best checkpoint.
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}. Best AUC: {best_auc:.4f}")
                    break

        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        self.model.eval()

    def predict_proba(self, X):
        """Score the positive class in input order."""
        # Labels are needed to build the loader but not used: get_probs returns
        # them in input order, so they match the y the caller already holds.
        loader = self._loader(X, np.zeros(len(X)))
        probs, _ = get_probs(loader, self.model, self.device)
        return probs

    def score(self, X):
        """Decide at p >= 0.5, reusing the single forward pass."""
        # One forward pass: predict() would run the network a second time over
        # the same rows.
        return self.score_by_threshold(X)
