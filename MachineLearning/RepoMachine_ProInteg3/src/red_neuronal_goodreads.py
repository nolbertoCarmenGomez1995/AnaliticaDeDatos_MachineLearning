"""Red neuronal para clasificar libros destacados de Goodreads."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_COLUMNS = [
    "paginas",
    "cantidad_calificaciones",
    "cantidad_resenas",
    "cantidad_ediciones",
]
TARGET_COLUMN = "alta_calificacion"


def prepare_neural_data(raw: pd.DataFrame, minimum_rating: float = 3.5) -> pd.DataFrame:
    """Crea una clase binaria de calidad didáctica y valida sus predictoras."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw debe ser un pandas.DataFrame.")

    required = {"title", "pages", "average_rating", "ratings_count", "reviews_count", "num_editions"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"Faltan columnas necesarias: {sorted(missing)}.")

    frame = pd.DataFrame(
        {
            "paginas": pd.to_numeric(raw["pages"], errors="coerce"),
            "calificacion": pd.to_numeric(raw["average_rating"], errors="coerce"),
            "cantidad_calificaciones": pd.to_numeric(raw["ratings_count"], errors="coerce"),
            "cantidad_resenas": pd.to_numeric(raw["reviews_count"], errors="coerce"),
            "cantidad_ediciones": pd.to_numeric(raw["num_editions"], errors="coerce"),
        }
    )
    frame = frame[
        frame["paginas"].ge(20)
        & frame["calificacion"].between(minimum_rating, 5.0)
        & frame[FEATURE_COLUMNS].notna().any(axis=1)
    ].copy()
    frame[TARGET_COLUMN] = frame["calificacion"].ge(4.2).astype(int)
    frame = frame.drop(columns="calificacion")
    frame = frame.reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError("Se necesitan al menos 100 libros para entrenar la red neuronal.")
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("La variable objetivo no contiene las dos clases.")
    return frame


def fit_neural_model(
    books: pd.DataFrame,
    *,
    sample_size: int = 20_000,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Entrena MLPClassifier y evalúa solo sobre datos reservados."""
    if sample_size < 100:
        raise ValueError("sample_size debe ser al menos 100.")
    sample = books.sample(n=min(sample_size, len(books)), random_state=SEED)
    x = sample[FEATURE_COLUMNS]
    y = sample[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )
    pipeline = Pipeline(
        [
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
            (
                "red",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "confusion_matrix": cm.tolist(),
        "positive_rate": float(y_test.mean()),
    }
    return {
        "model": pipeline,
        "metrics": metrics,
        "probabilities": probabilities,
        "test_labels": y_test.to_numpy(),
        "test_predictions": predictions,
    }
