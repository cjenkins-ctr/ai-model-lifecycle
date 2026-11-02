"""Scikit-learn baseline models — the default training algorithm."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from app.database.models import AlgorithmType
from app.logging_config import get_logger
from app.models.registry import EvaluatorRegistry, TrainerRegistry

logger = get_logger(__name__)


# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_dataset(dataset_name: str):
    """Load a dataset by name. Supports sklearn built-in datasets."""
    from sklearn import datasets

    if dataset_name == "iris":
        data = datasets.load_iris()
    elif dataset_name == "wine":
        data = datasets.load_wine()
    elif dataset_name == "digits":
        data = datasets.load_digits()
    elif dataset_name == "breast_cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "moon":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
        data = type('Bunch', (), {'data': X, 'target': y, 'target_names': ['class_0', 'class_1']})()
    elif dataset_name == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=1000, noise=0.1, random_state=42)
        data = type('Bunch', (), {'data': X, 'target': y, 'target_names': ['class_0', 'class_1']})()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use: iris, wine, digits, breast_cancer, moon, circles")

    return data


# ─── Trainer Classes ─────────────────────────────────────────────────────────


class SklearnTrainer:
    """Base trainer for scikit-learn models."""

    model_class = LogisticRegression
    default_params = {"max_iter": 1000}

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def prepare_data(self, dataset_name: str, test_size: float = 0.2, random_state: int = 42):
        """Split dataset into train/validation sets."""
        data = load_dataset(dataset_name)
        X_train, X_val, y_train, y_val = train_test_split(
            data.data, data.target, test_size=test_size, random_state=random_state
        )
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        return self

    def train(self, ctx):
        """Execute training loop."""
        self.prepare_data(ctx.config.get("dataset", "iris"))

        config = ctx.config
        dataset = config.get("dataset", "iris")
        epochs = config.get("epochs", 1)  # sklearn doesn't use epochs, but we track for consistency
        model_type = config.get("model_type", "logistic_regression")

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(kernel="rbf"),
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(models.keys())}")

        model = models[model_type]
        model.fit(self.X_train, self.y_train)
        self.model = model

        # Simulate epoch-by-epoch logging (sklearn trains in one pass)
        for epoch in range(1, epochs + 1):
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            train_acc = accuracy_score(self.y_train, train_pred)
            val_acc = accuracy_score(self.y_val, val_pred)

            ctx.log_metric("train/accuracy", train_acc, step=epoch)
            ctx.log_metric("val/accuracy", val_acc, step=epoch)
            ctx.log_metric("train/loss", 1.0 - train_acc, step=epoch)
            ctx.log_metric("val/loss", 1.0 - val_acc, step=epoch)

            logger.info(
                "epoch",
                epoch=epoch,
                train_acc=round(train_acc, 4),
                val_acc=round(val_acc, 4),
            )

        # Save checkpoint
        checkpoint_dir = ctx.registry_path / ctx.experiment_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)

        ctx.checkpoint_path = checkpoint_path
        ctx.save_checkpoint(str(checkpoint_path), "model")

        logger.info("model_trained", model_type=model_type, checkpoint=str(checkpoint_path))


class RandomSklearnTrainer(SklearnTrainer):
    """Trainer that automatically selects a model type."""
    model_class = None  # Discovered at runtime
    default_params = {}

    def train(self, ctx):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        config = ctx.config
        dataset = config.get("dataset", "iris")
        epochs = config.get("epochs", 1)
        self.prepare_data(dataset)

        model.fit(self.X_train, self.y_train)
        self.model = model

        for epoch in range(1, epochs + 1):
            train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
            val_acc = accuracy_score(self.y_val, model.predict(self.X_val))

            ctx.log_metric("train/accuracy", train_acc, step=epoch)
            ctx.log_metric("val/accuracy", val_acc, step=epoch)
            ctx.log_metric("train/loss", 1.0 - train_acc, step=epoch)
            ctx.log_metric("val/loss", 1.0 - val_acc, step=epoch)

        checkpoint_dir = ctx.registry_path / ctx.experiment_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)

        ctx.checkpoint_path = checkpoint_path
        ctx.save_checkpoint(str(checkpoint_path), "model")


# ─── Evaluator Classes ────────────────────────────────────────────────────────


class SklearnEvaluator:
    """Evaluate a trained scikit-learn model."""

    def evaluate(self, ctx):
        if not ctx.checkpoint_path or not Path(ctx.checkpoint_path).exists():
            raise RuntimeError("No checkpoint found. Run training first.")

        with open(ctx.checkpoint_path, "rb") as f:
            model = pickle.load(f)

        # Re-load the validation data for evaluation
        config = ctx.config
        dataset = config.get("dataset", "iris")
        from sklearn.model_selection import train_test_split
        data = load_dataset(dataset)
        _, X_val, _, y_val = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            "precision_weighted": float(precision_score(y_val, y_pred, average="weighted")),
            "recall_weighted": float(recall_score(y_val, y_pred, average="weighted")),
        }

        ctx.final_metrics = metrics

        logger.info("evaluation_complete", metrics=metrics)
        return metrics


# ─── Registration ─────────────────────────────────────────────────────────────


@TrainerRegistry.register(AlgorithmType.SKLEARN)
class SklearnTrainerEntry(SklearnTrainer):
    """Entry-point trainer for sklearn."""
    pass


@EvaluatorRegistry.register(AlgorithmType.SKLEARN)
class SklearnEvaluatorEntry(SklearnEvaluator):
    """Entry-point evaluator for sklearn."""
    pass


__all__ = [
    "SklearnTrainer",
    "SklearnEvaluator",
    "load_dataset",
    "TrainerRegistry",
    "EvaluatorRegistry",
]
