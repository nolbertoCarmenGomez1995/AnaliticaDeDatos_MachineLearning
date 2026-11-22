"""Ejecuta las prácticas adaptadas de redes neuronales y transformers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .hf_loader import HFDatasetLoader
from .red_neuronal_goodreads import fit_neural_model, prepare_neural_data
from .transformer_goodreads import fit_transformer, prepare_text_data

DATASET_ID = "pfaha/goodreads-books"
TARGET_RULE = "alta_calificacion = 1 cuando average_rating >= 4.2"


def run_advanced_models(
    raw: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Entrena ambos modelos prácticos y escribe un resumen automático."""
    neural_frame = prepare_neural_data(raw)
    neural = fit_neural_model(neural_frame)
    transformer_frame = prepare_text_data(raw)
    transformer = fit_transformer(
        transformer_frame["texto"],
        transformer_frame["alta_calificacion"],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "modelos_avanzados_goodreads.txt"
    lines = [
        "RESUMEN — RED NEURONAL Y TRANSFORMER GOODREADS",
        "=" * 62,
        f"Fecha de ejecución: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Dataset: {DATASET_ID}, split train.",
        f"Variable objetivo didáctica: {TARGET_RULE}.",
        "La calificación original no se incluye como predictora.",
        "",
        "SEMANA 6 — RED NEURONAL MLP",
        "Tipo: clasificación supervisada.",
        "Modelo único de esta práctica: MLPClassifier.",
        "Variables predictoras: paginas, cantidad_calificaciones, "
        "cantidad_resenas, cantidad_ediciones.",
        "Preprocesamiento: imputación mediana y escalado ajustados con entrenamiento.",
        f"Arquitectura: (32, 16); activación=relu; semilla=42.",
        f"Métricas: {neural['metrics']}",
        "",
        "SEMANA 7 — MINI-TRANSFORMER",
        "Tipo: clasificación supervisada de texto.",
        "Modelo: embeddings + atención multi-cabeza + clasificación.",
        "Predictor: título normalizado; no se utilizan calificación ni conteos.",
        f"Configuración: 4 cabezas, 3 épocas, longitud máxima 24, vocabulario máximo 5000.",
        f"Métricas: {transformer['metrics']}",
        "",
        "Interpretación: comparar siempre con el baseline de predecir la clase "
        "mayoritaria. Una exactitud alta puede ocultar bajo recall si las clases "
        "están desbalanceadas.",
        "",
        "Limitación: son prácticas de clase adaptadas a Goodreads; no reemplazan "
        "la validación comercial ni demuestran causalidad.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "neural": neural,
        "transformer": transformer,
        "summary_path": summary_path,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw = HFDatasetLoader(dataset_name=DATASET_ID, split="train").load()
    result = run_advanced_models(raw, output_dir=project_root / "reportes")
    print(f"Resumen: {result['summary_path']}")
    print("Red neuronal:", result["neural"]["metrics"])
    print("Transformer:", result["transformer"]["metrics"])


if __name__ == "__main__":
    main()
