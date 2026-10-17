import pandas as pd
from datasets import load_dataset

class HFDatasetLoader:
    """Clase encargada de obtener un dataset de Hugging Face y retornarlo como DataFrame de pandas."""

    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> pd.DataFrame:
        """Carga el dataset desde Hugging Face y devuelve un DataFrame de pandas."""
        try:
            print(f"Cargando dataset '{self.dataset_name}' (split: '{self.split}')...")
            ds = load_dataset(self.dataset_name, split=self.split)
            df = pd.DataFrame(ds)
            print(f"Dataset cargado con éxito. Filas: {len(df)}, Columnas: {len(df.columns)}")
            return df
        except Exception as e:
            print(f"Error al cargar el dataset desde Hugging Face: {e}")
            raise e