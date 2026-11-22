import unittest
from pathlib import Path

import pandas as pd

from dashboard import (
    AREA_AGIL,
    AREA_EXTENSA,
    SIN_IDIOMA,
    candidatos_exhibicion,
    cargar_libros,
    filtrar_libros,
    resumir_extension,
    resumir_participacion,
    resumir_composicion_idioma,
    resumir_estadisticas_calificacion,
    resumir_reparto_extension,
    ranking_categoria,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "AE3" / "resultados"


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.libros = cargar_libros(RESULTS_DIR / "libros_preparados_enriquecidos.csv")

    def test_extension_summary_matches_ae3_export(self):
        actual = resumir_extension(self.libros).set_index("categoria_extension")
        expected = pd.read_csv(
            RESULTS_DIR / "t2_reparto_por_extension.csv"
        ).set_index("categoria_extension")

        for area in actual.index:
            self.assertEqual(
                actual.loc[area, "registros_edicion"],
                expected.loc[area, "registros_edicion"],
            )
            self.assertEqual(
                actual.loc[area, "obras_asignadas"],
                expected.loc[area, "obras_asignadas"],
            )

    def test_area_repartition_matches_ae3_export(self):
        actual = resumir_reparto_extension(self.libros).set_index("categoria_extension")
        expected = pd.read_csv(
            RESULTS_DIR / "t2_reparto_por_extension.csv"
        ).set_index("categoria_extension")
        for area in actual.index:
            self.assertEqual(
                actual.loc[area, "registros_edicion"],
                expected.loc[area, "registros_edicion"],
            )
            self.assertAlmostEqual(
                actual.loc[area, "porcentaje_registros"],
                expected.loc[area, "porcentaje_registros"],
                places=2,
            )
            self.assertEqual(
                actual.loc[area, "obras_asignadas"],
                expected.loc[area, "obras_asignadas"],
            )
            self.assertAlmostEqual(
                actual.loc[area, "porcentaje_obras"],
                expected.loc[area, "porcentaje_obras"],
                places=2,
            )

    def test_rating_statistics_include_quartiles_and_iqr(self):
        actual = resumir_estadisticas_calificacion(self.libros).set_index(
            "categoria_extension"
        )
        for area in actual.index:
            ratings = self.libros.loc[
                self.libros["categoria_extension"].eq(area), "average_rating"
            ]
            self.assertAlmostEqual(actual.loc[area, "calificacion_media"], ratings.mean())
            self.assertAlmostEqual(actual.loc[area, "cuartil_1"], ratings.quantile(0.25))
            self.assertAlmostEqual(actual.loc[area, "mediana"], ratings.median())
            self.assertAlmostEqual(actual.loc[area, "cuartil_3"], ratings.quantile(0.75))
            self.assertAlmostEqual(
                actual.loc[area, "rango_intercuartil"],
                ratings.quantile(0.75) - ratings.quantile(0.25),
            )

    def test_population_and_language_tables_match_prepared_data(self):
        population = pd.read_csv(RESULTS_DIR / "t1_control_poblacion.csv")
        final_population = population.loc[
            population["criterio"].str.startswith("Población final:")
        ].iloc[0]
        self.assertEqual(final_population["registros"], len(self.libros))

        composition = pd.read_csv(RESULTS_DIR / "t6_composicion_idioma.csv")
        self.assertEqual(
            composition[f"registros_{AREA_AGIL}"].sum(),
            len(self.libros.loc[self.libros["categoria_extension"].eq(AREA_AGIL)]),
        )
        self.assertEqual(
            composition[f"registros_{AREA_EXTENSA}"].sum(),
            len(self.libros.loc[self.libros["categoria_extension"].eq(AREA_EXTENSA)]),
        )
        self.assertAlmostEqual(composition[f"porcentaje_{AREA_AGIL}"].sum(), 100.0)
        self.assertAlmostEqual(composition[f"porcentaje_{AREA_EXTENSA}"].sum(), 100.0)

    def test_reader_metrics_match_ae3_without_double_counting(self):
        actual = resumir_participacion(self.libros).set_index("categoria_extension")
        expected = pd.read_csv(
            RESULTS_DIR / "t7_participacion_series.csv"
        ).set_index("categoria_extension")

        for area in actual.index:
            self.assertEqual(
                actual.loc[area, "calificaciones_totales"],
                expected.loc[area, "calificaciones_totales"],
            )
            self.assertEqual(
                actual.loc[area, "resenas_totales"],
                expected.loc[area, "resenas_totales"],
            )
            self.assertEqual(
                actual.loc[area, "obras_asignadas"],
                expected.loc[area, "obras_asignadas"],
            )

    def test_publisher_and_language_rankings_match_ae3_exports(self):
        for column, filename in (
            ("publisher", "t3_top_editoriales.csv"),
            ("idioma_normalizado", "t4_top_idiomas.csv"),
        ):
            actual = ranking_categoria(self.libros, column)
            expected = pd.read_csv(RESULTS_DIR / filename)
            keys = ["categoria_extension", column]
            actual_keys = actual[keys].reset_index(drop=True)
            expected_keys = expected[keys].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_keys, expected_keys)
            for _, group in actual.groupby("categoria_extension"):
                self.assertTrue(group["obras_unicas"].is_monotonic_decreasing)

    def test_filters_and_publisher_ranking_use_selected_records(self):
        filtrados = filtrar_libros(
            self.libros,
            areas=[AREA_EXTENSA],
            idiomas=["Spanish"],
            rango_calificacion=(4.0, 5.0),
        )

        self.assertFalse(filtrados.empty)
        self.assertTrue(filtrados["categoria_extension"].eq(AREA_EXTENSA).all())
        self.assertTrue(filtrados["idioma_normalizado"].eq("Spanish").all())
        self.assertTrue(filtrados["average_rating"].between(4.0, 5.0).all())

        ranking = ranking_categoria(filtrados, "publisher")
        self.assertTrue(ranking["categoria_extension"].eq(AREA_EXTENSA).all())
        self.assertLessEqual(len(ranking), 10)

    def test_clearing_area_filter_means_all_areas(self):
        sin_restriccion = filtrar_libros(self.libros)
        areas_vacias = filtrar_libros(self.libros, areas=[])
        self.assertEqual(set(areas_vacias["book_id"]), set(sin_restriccion["book_id"]))

    def test_language_composition_uses_filtered_records_and_includes_missing(self):
        spanish = filtrar_libros(self.libros, idiomas=["Spanish"])
        resumen_espanol = resumir_composicion_idioma(spanish)
        self.assertTrue(resumen_espanol["idioma"].eq("Spanish").all())
        for _, grupo in resumen_espanol.groupby("categoria_extension"):
            self.assertAlmostEqual(grupo["porcentaje_registros"].sum(), 100.0)

        sin_idioma = filtrar_libros(self.libros, idiomas=[SIN_IDIOMA])
        resumen_sin_idioma = resumir_composicion_idioma(sin_idioma)
        self.assertTrue(resumen_sin_idioma["idioma"].eq(SIN_IDIOMA).all())
        self.assertEqual(resumen_sin_idioma["registros_edicion"].sum(), len(sin_idioma))
        for _, grupo in resumen_sin_idioma.groupby("categoria_extension"):
            self.assertAlmostEqual(grupo["porcentaje_registros"].sum(), 100.0)

    def test_publisher_and_format_filters_match_real_values(self):
        por_editorial = filtrar_libros(self.libros, busqueda_editorial="penguin")
        self.assertFalse(por_editorial.empty)
        self.assertTrue(
            por_editorial["publisher"].str.contains("penguin", case=False).all()
        )

        por_formato = filtrar_libros(self.libros, formatos=["Tapa blanda"])
        self.assertFalse(por_formato.empty)
        self.assertTrue(por_formato["formato_grupo"].eq("Tapa blanda").all())

    def test_candidate_list_respects_combined_filters_and_deduplicates_works(self):
        seleccion = filtrar_libros(
            self.libros,
            idiomas=["Spanish"],
            formatos=["Tapa blanda"],
        )
        self.assertEqual(len(seleccion), 5714)

        candidatos = candidatos_exhibicion(seleccion)
        self.assertEqual(len(candidatos), 20)
        self.assertEqual(candidatos["work_id"].nunique(), len(candidatos))
        self.assertTrue(candidatos["idioma_normalizado"].eq("Spanish").all())
        self.assertTrue(candidatos["formato_grupo"].eq("Tapa blanda").all())
        self.assertTrue(candidatos["average_rating"].is_monotonic_decreasing)

    def test_filtered_work_uses_a_matching_edition_for_work_metrics(self):
        libros = pd.DataFrame([
            {
                "book_id": 1,
                "work_id": 10,
                "title": "Edición global representativa",
                "first_author": "Autor original",
                "pages": 320,
                "categoria_extension": AREA_AGIL,
                "average_rating": 4.2,
                "rango_calificacion": "4,0 a 4,49",
                "ratings_count": 900,
                "reviews_count": 90,
                "publisher": "Editorial A",
                "idioma_normalizado": "English",
                "is_part_of_series": False,
                "formato_grupo": "Tapa dura",
                "registro_representativo_obra": True,
            },
            {
                "book_id": 2,
                "work_id": 10,
                "title": "Edición española de bolsillo",
                "first_author": "Autor original",
                "pages": 300,
                "categoria_extension": AREA_AGIL,
                "average_rating": 4.5,
                "rango_calificacion": "4,5 a 5,0",
                "ratings_count": 120,
                "reviews_count": 15,
                "publisher": "Editorial B",
                "idioma_normalizado": "Spanish",
                "is_part_of_series": True,
                "formato_grupo": "Tapa blanda",
                "registro_representativo_obra": False,
            },
        ])
        seleccion = filtrar_libros(
            libros,
            idiomas=["Spanish"],
            formatos=["Tapa blanda"],
        )

        extension = resumir_extension(seleccion).set_index("categoria_extension")
        actividad = resumir_participacion(seleccion).set_index("categoria_extension")
        candidatos = candidatos_exhibicion(seleccion)

        self.assertEqual(extension.loc[AREA_AGIL, "obras_asignadas"], 1)
        self.assertEqual(extension.loc[AREA_AGIL, "porcentaje_obras"], 100.0)
        self.assertEqual(actividad.loc[AREA_AGIL, "calificaciones_totales"], 120)
        self.assertEqual(actividad.loc[AREA_AGIL, "resenas_totales"], 15)
        self.assertEqual(candidatos.iloc[0]["book_id"], 2)


if __name__ == "__main__":
    unittest.main()
