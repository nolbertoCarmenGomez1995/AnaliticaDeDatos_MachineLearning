import unittest

import pandas as pd

from src.red_neuronal_goodreads import fit_neural_model, prepare_neural_data
from src.transformer_goodreads import fit_transformer, prepare_text_data


def make_raw(rows=600):
    return pd.DataFrame(
        {
            "title": [f"Libro destacado {i}" if i % 2 else f"Libro breve {i}" for i in range(rows)],
            "pages": [80 + i % 700 for i in range(rows)],
            "average_rating": [4.3 if i % 3 == 0 else 3.8 for i in range(rows)],
            "ratings_count": [1000 if i % 3 == 0 else 20 for i in range(rows)],
            "reviews_count": [100 if i % 3 == 0 else 2 for i in range(rows)],
            "num_editions": [i % 12 for i in range(rows)],
        }
    )


class NeuralGoodreadsTests(unittest.TestCase):
    def test_prepares_binary_target_without_original_rating(self):
        frame = prepare_neural_data(make_raw())

        self.assertIn("alta_calificacion", frame.columns)
        self.assertNotIn("calificacion", frame.columns)
        self.assertEqual(set(frame["alta_calificacion"]), {0, 1})

    def test_trains_and_evaluates_mlp(self):
        frame = prepare_neural_data(make_raw())
        result = fit_neural_model(frame, sample_size=600)

        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(len(result["test_labels"]), result["metrics"]["test_rows"])
        self.assertEqual(len(result["test_predictions"]), result["metrics"]["test_rows"])


class TransformerGoodreadsTests(unittest.TestCase):
    def test_prepares_title_only_text(self):
        frame = prepare_text_data(make_raw(), sample_size=600)

        self.assertEqual(list(frame.columns), ["texto", "alta_calificacion"])
        self.assertTrue(frame["texto"].str.len().gt(0).all())

    def test_trains_tiny_attention_classifier(self):
        frame = prepare_text_data(make_raw(), sample_size=600)
        result = fit_transformer(frame["texto"], frame["alta_calificacion"], epochs=2)

        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(len(result["test_predictions"]), result["metrics"]["test_rows"])
        self.assertEqual(result["model"].heads, 4)


if __name__ == "__main__":
    unittest.main()
