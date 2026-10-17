import os
import pandas as pd

class DatasetSummarizer:
    """Clase para analizar un DataFrame y exportar su resumen descriptivo a un archivo .txt."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def export_summary(self, filepath: str):
        """Genera y guarda un resumen estructurado del DataFrame en un archivo .txt."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RESUMEN DEL DATAFRAME TRANSFORMADO\n")
            f.write("=" * 60 + "\n\n")

            # 1. Dimensiones
            f.write("1. ESTRUCTURA Y DIMENSIONES:\n")
            f.write(f"   - Cantidad de Filas: {self.df.shape[0]}\n")
            f.write(f"   - Cantidad de Columnas: {self.df.shape[1]}\n\n")

            # 2. Columnas, tipos y nulos
            f.write("2. COLUMNAS, TIPOS DE DATOS Y VALORES NULOS:\n")
            f.write(f"{'Columna':<30} {'Tipo de Dato':<15} {'Valores Nulos':<15}\n")
            f.write("-" * 60 + "\n")
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                nulls = self.df[col].isnull().sum()
                f.write(f"{col:<30} {dtype:<15} {nulls:<15}\n")

            # 3. Estadísticas descriptivas
            f.write("\n3. ESTADÍSTICAS DESCRIPTIVAS:\n")
            f.write("-" * 60 + "\n")
            f.write(self.df.describe(include="all").to_string())
            f.write("\n")

        print(f"Resumen exportado exitosamente a: {filepath}")