"""Clasificador textual ligero con atención tipo transformer sobre títulos Goodreads."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
PAD, UNK = 0, 1
TARGET_COLUMN = "alta_calificacion"


def prepare_text_data(
    raw: pd.DataFrame,
    *,
    minimum_rating: float = 3.5,
    sample_size: int = 12_000,
) -> pd.DataFrame:
    """Deriva una clase didáctica y conserva únicamente el título como texto predictivo."""
    required = {"title", "average_rating"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"Faltan columnas necesarias: {sorted(missing)}.")
    frame = pd.DataFrame(
        {
            "texto": raw["title"].fillna("").astype(str).str.lower(),
            "calificacion": pd.to_numeric(raw["average_rating"], errors="coerce"),
        }
    )
    frame = frame[
        frame["texto"].str.strip().ne("")
        & frame["calificacion"].between(minimum_rating, 5.0)
    ].copy()
    frame[TARGET_COLUMN] = frame["calificacion"].ge(4.2).astype(int)
    frame = frame[["texto", TARGET_COLUMN]].reset_index(drop=True)
    frame = frame.sample(n=min(sample_size, len(frame)), random_state=SEED).reset_index(drop=True)
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("La variable objetivo no contiene las dos clases.")
    return frame


class AttentionClassifier(nn.Module):
    """Mini-transformer encoder: embeddings, atención multi-cabeza y promedio ponderado."""

    def __init__(self, vocab_size: int, hidden_size: int = 32, heads: int = 4) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size debe ser divisible entre heads.")
        self.hidden_size = hidden_size
        self.heads = heads
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = tokens.shape
        x = self.embedding(tokens)
        head_size = self.hidden_size // self.heads

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch_size, length, self.heads, head_size).transpose(1, 2)

        query = split_heads(self.query(x))
        key = split_heads(self.key(x))
        value = split_heads(self.value(x))
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_size)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = attention @ value
        context = context.transpose(1, 2).reshape(batch_size, length, self.hidden_size)
        pooled = (context * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.output(pooled), attention


def fit_transformer(
    texts: pd.Series,
    labels: pd.Series,
    *,
    epochs: int = 3,
    max_length: int = 24,
    max_vocab: int = 5000,
) -> dict[str, Any]:
    """Entrena y evalúa el modelo sin filtrar ejemplos con `train_test_split`."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    counter: dict[str, int] = {}
    for text in train_texts:
        for token in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", str(text).lower()):
            counter[token] = counter.get(token, 0) + 1
    vocab = ["<PAD>", "<UNK>"] + [
        token
        for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[: max_vocab - 2]
    ]
    token_ids = {token: index for index, token in enumerate(vocab)}

    def encode(series: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in series:
            tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", str(text).lower())[:max_length]
            ids = [token_ids.get(token, UNK) for token in tokens]
            ids = ids + [PAD] * (max_length - len(ids))
            mask = [token != PAD for token in ids]
            rows.append(ids)
            masks.append(mask)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    x_train, train_mask = encode(train_texts)
    x_test, test_mask = encode(test_texts)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    model = AttentionClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_counts = y_train_tensor.bincount(minlength=2).clamp_min(1).float()
    class_weights = len(y_train_tensor) / (2 * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_train, train_mask)
        loss = loss_function(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, attention = model(x_test, test_mask)
        predictions = logits.argmax(dim=1)
    cm = confusion_matrix(y_test_tensor.numpy(), predictions.numpy(), labels=[0, 1])
    return {
        "model": model,
        "vocab": vocab,
        "metrics": {
            "accuracy": float(accuracy_score(y_test_tensor.numpy(), predictions.numpy())),
            "f1": float(f1_score(y_test_tensor.numpy(), predictions.numpy(), zero_division=0)),
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "confusion_matrix": cm.tolist(),
        },
        "attention": attention,
        "test_texts": test_texts.to_list(),
        "test_labels": y_test_tensor,
        "test_predictions": predictions,
    }
