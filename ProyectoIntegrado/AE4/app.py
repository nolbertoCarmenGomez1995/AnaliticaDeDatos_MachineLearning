from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
RESULTS_DIR = PROJECT_DIR / "AE3" / "resultados"
DATA_PATH = RESULTS_DIR / "libros_preparados_enriquecidos.csv"
SENSITIVITY_PATH = RESULTS_DIR / "t2b_sensibilidad_umbral.csv"
POPULATION_PATH = RESULTS_DIR / "t1_control_poblacion.csv"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard import (  # noqa: E402
    AREA_AGIL,
    AREA_EXTENSA,
    ORDEN_AREAS,
    SIN_IDIOMA,
    candidatos_exhibicion,
    cargar_libros,
    filtrar_libros,
    ranking_categoria,
    resumir_calificaciones,
    resumir_composicion_idioma,
    resumir_estadisticas_calificacion,
    resumir_reparto_extension,
)

COLORES_AREA = {AREA_AGIL: "#2a78d6", AREA_EXTENSA: "#eb6834"}

st.set_page_config(
    page_title="Goodreads | Perfil para exhibición",
    page_icon="📚",
    layout="wide",
)


@st.cache_data(show_spinner="Cargando el dataset preparado de AE3…")
def cargar_datos(ruta: str) -> pd.DataFrame:
    return cargar_libros(ruta)


@st.cache_data(show_spinner=False)
def cargar_sensibilidad(ruta: str) -> pd.DataFrame:
    archivo = Path(ruta)
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró la tabla de sensibilidad de AE3: {archivo}")
    return pd.read_csv(archivo, encoding="utf-8")


@st.cache_data(show_spinner=False)
def cargar_tabla_resultados(ruta: str) -> pd.DataFrame:
    archivo = Path(ruta)
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró una tabla requerida de AE3: {archivo}")
    return pd.read_csv(archivo, encoding="utf-8")


st.title("Goodreads | Perfil de catálogo para exhibición")
st.markdown(
    "Compara **Lecturas Ágiles** y **Grandes Clásicos y Sagas** para orientar la "
    "distribución del espacio y priorizar editoriales e idiomas."
)
st.info(
    "Análisis descriptivo de título, primer autor, páginas, calificación, editorial e idioma. "
    "Las conclusiones orientan una conversación con la librería; Goodreads no mide sus ventas."
)

if not DATA_PATH.is_file():
    st.error(
        "No se encontró el CSV preparado de AE3. Ejecuta primero "
        "`AE3/notebooks/AE3_transformacion_enriquecimiento.ipynb`."
    )
    st.stop()

libros = cargar_datos(str(DATA_PATH))

with st.expander("Necesidad, preguntas de negocio y dataset", expanded=False):
    st.markdown(
        "**Necesidad.** En el escenario hipotético de una librería en Nueva York, el espacio "
        "limitado debe repartirse entre lecturas ágiles y obras extensas, y se quieren "
        "identificar editoriales e idiomas para revisar compras importadas. La diversidad de "
        "la ciudad hace pertinente explorar idiomas, pero no prueba la demanda de la tienda."
    )
    st.markdown(
        "**Pregunta principal.** Entre los libros destacados de Goodreads "
        "(calificación promedio ≥ 3,5), ¿cómo se distribuyen por extensión, qué editoriales "
        "e idiomas concentran más títulos en cada área y cómo varían las calificaciones?"
    )
    st.markdown(
        """
**Preguntas de negocio**

1. **P1:** ¿Cuál es la población de estudio y cuántos registros se excluyen por cada criterio?
2. **P2:** ¿Qué número y porcentaje de títulos reúne cada área con el umbral de páginas definido?
3. **P3:** ¿Qué editoriales concentran más títulos destacados en cada área y con qué calificación promedio?
4. **P4:** ¿Qué idiomas concentran más títulos destacados en cada área y con qué calificación promedio?
5. **P5:** ¿Cómo se distribuye la calificación promedio de las ediciones destacadas en cada área?
6. **P6:** ¿Cuál es la composición por idioma de cada área?
"""
    )
    st.markdown(
        "**Dataset.** `pfaha/goodreads-books`, Hugging Face, configuración `default`, "
        "split `train`, revisión `43b3198cd7ec9c142d56181f1a09ed9e8c112610`. "
        "La fuente tiene 500.000 registros y 24 columnas; AE3 deja 235.104 ediciones "
        "y 20 columnas preparadas, incluyendo el primer autor."
    )
    st.markdown(
        "**Periodo.** No aplica: la fuente no contiene una fecha de venta, lectura o reseña "
        "que permita analizar tendencias temporales."
    )

with st.sidebar:
    st.header("Filtros")
    areas_seleccionadas = st.multiselect(
        "Área de exhibición",
        options=ORDEN_AREAS,
        default=ORDEN_AREAS,
        help="Puedes seleccionar una o ambas. Si dejas el campo vacío, se incluyen ambas áreas.",
    )
    idiomas_disponibles = sorted(
        libros["idioma_normalizado"].dropna().astype(str).unique().tolist()
    )
    idiomas_seleccionados = st.multiselect(
        "Idioma",
        options=[SIN_IDIOMA, *idiomas_disponibles],
        default=[],
        help="Selecciona uno o varios idiomas. Vacío significa todos, incluidos los no informados.",
    )
    formatos_disponibles = sorted(libros["formato_grupo"].dropna().unique().tolist())
    formatos_seleccionados = st.multiselect(
        "Formato (solo lista de candidatos)",
        options=formatos_disponibles,
        default=[],
        help=(
            "Filtro operativo para encontrar ediciones de un formato concreto. "
            "No cambia los gráficos ni las respuestas P1–P6."
        ),
    )
    editorial = st.text_input(
        "Buscar editorial",
        help="Busca por coincidencia parcial, sin distinguir mayúsculas y minúsculas.",
    )
    minimo_rating = float(libros["average_rating"].min())
    rango_rating = st.slider(
        "Calificación promedio",
        min_value=minimo_rating,
        max_value=5.0,
        value=(minimo_rating, 5.0),
        step=0.1,
        help="La base ya excluye libros con calificación inferior a 3,5.",
    )
    st.caption(
        "La población ya viene filtrada en AE3: calificación ≥ 3,5 y páginas válidas "
        "entre 1 y 5.000."
    )
    st.caption(
        "Los criterios distintos se combinan con **Y**; dentro de idioma, formato o área "
        "se combina con **O**. Formato solo limita la lista de candidatos; si lo dejas vacío, "
        "incluye todos los formatos."
    )

libros_analisis = filtrar_libros(
    libros,
    areas=areas_seleccionadas,
    idiomas=idiomas_seleccionados,
    busqueda_editorial=editorial,
    rango_calificacion=rango_rating,
)

libros_para_candidatos = filtrar_libros(
    libros,
    areas=areas_seleccionadas,
    idiomas=idiomas_seleccionados,
    formatos=formatos_seleccionados,
    busqueda_editorial=editorial,
    rango_calificacion=rango_rating,
)

if libros_analisis.empty:
    st.warning("No hay registros con esos filtros. Cambia o limpia alguna selección.")
    st.stop()

titulos_unicos = int(libros_analisis["work_id"].nunique())
candidatos = candidatos_exhibicion(libros_para_candidatos)
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Registros analizados", f"{len(libros_analisis):,}")
kpi_2.metric("Títulos únicos", f"{titulos_unicos:,}")
kpi_3.metric("Calificación media", f"{libros_analisis['average_rating'].mean():.2f} / 5")
kpi_4.metric("Mediana de páginas", f"{libros_analisis['pages'].median():,.0f}")
st.caption(
    "Los resultados principales usan las seis variables definidas para el análisis. "
    "El filtro de formato solo se aplica a la lista auxiliar de candidatos."
)

tabs = st.tabs([
    "Resumen para decidir",
    "Candidatos para exhibición",
    "Extensión (P2)",
    "Editoriales (P3)",
    "Idiomas (P4 y P6)",
    "Calificaciones (P5)",
    "Población y método (P1)",
    "Conclusiones",
    "Datos",
])

with tabs[0]:
    st.subheader("Resumen de la selección")
    st.markdown(
        "Compara el número de obras únicas por área y la distribución de páginas en el catálogo "
        "que coincide con los filtros de idioma, editorial, calificación y área."
    )
    resumen_decision = resumir_reparto_extension(libros_analisis)
    resumen_decision["etiqueta_libros"] = resumen_decision.apply(
        lambda fila: (
            f"{int(fila['obras_asignadas']):,} títulos "
            f"({fila['porcentaje_obras']:.1f} %)"
        ),
        axis=1,
    )
    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(
        px.bar(
            resumen_decision,
            x="categoria_extension",
            y="obras_asignadas",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            text="etiqueta_libros",
            labels={
                "categoria_extension": "Área de exhibición",
                "obras_asignadas": "Títulos únicos",
            },
            title="Libros por área en la selección actual",
        ),
        width="stretch",
    )
    derecha.plotly_chart(
        px.box(
            libros_analisis,
            x="categoria_extension",
            y="pages",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            points=False,
            labels={
                "categoria_extension": "Área",
                "pages": "Páginas",
            },
            title="Distribución de páginas y umbral de 350",
        ),
        width="stretch",
    )
    area_prioritaria = resumen_decision.sort_values(
        ["obras_asignadas", "categoria_extension"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    if not candidatos.empty:
        primero = candidatos.iloc[0]
        st.info(
            f"**Hallazgo de la selección:** hay {len(libros_analisis):,} registros "
            f"({titulos_unicos:,} títulos únicos). El área con más libros es "
            f"**{area_prioritaria['categoria_extension']}** "
            f"({int(area_prioritaria['obras_asignadas']):,}; "
            f"{area_prioritaria['porcentaje_obras']:.1f} %). El candidato con mayor "
            f"calificación dentro de los filtros es **{primero['title']}** "
            f"({primero['average_rating']:.2f}/5)."
        )
    st.caption(
        "El área de Grandes Clásicos y Sagas se aproxima con libros de más de 350 páginas; "
        "no implica que cada título sea un clásico o pertenezca a una saga."
    )

with tabs[1]:
    st.subheader("Títulos candidatos para revisar en exhibición")
    st.markdown(
        f"Hay **{len(libros_para_candidatos):,}** ediciones que cumplen los criterios "
        "seleccionados, incluyendo el filtro opcional de formato. La lista muestra hasta "
        "20 obras únicas, cada una con su título y primer autor; se elige una edición "
        "coincidente por obra."
    )
    st.info(
        "Orden de la lista: mayor calificación promedio y, en caso de empate, título y autor. "
        "El formato solo se usa para limitar esta lista; no altera los análisis P1–P6."
    )
    if candidatos.empty:
        st.info("No hay títulos con información suficiente para formar la lista de candidatos.")
    else:
        candidatos_grafico = candidatos.head(15).sort_values(
            "average_rating", ascending=True, kind="stable"
        )
        st.plotly_chart(
            px.bar(
                candidatos_grafico,
                x="average_rating",
                y="title",
                color="categoria_extension",
                color_discrete_map=COLORES_AREA,
                orientation="h",
                hover_data=[
                    "first_author",
                    "pages",
                    "idioma_normalizado",
                    "publisher",
                    "formato_grupo",
                ],
                labels={
                    "average_rating": "Calificación promedio",
                    "title": "Título",
                    "first_author": "Primer autor",
                    "pages": "Páginas",
                    "categoria_extension": "Área de exhibición",
                    "idioma_normalizado": "Idioma",
                    "formato_grupo": "Formato (filtro auxiliar)",
                    "publisher": "Editorial",
                },
                title="Quince títulos con mayor calificación promedio",
            ),
            width="stretch",
        )
        candidatos_visibles = candidatos.assign(
            first_author=candidatos["first_author"].fillna("Sin autor informado")
        ).rename(columns={
            "title": "Título",
            "first_author": "Primer autor",
            "categoria_extension": "Área",
            "pages": "Páginas",
            "formato_grupo": "Formato (auxiliar)",
            "idioma_normalizado": "Idioma",
            "publisher": "Editorial",
            "average_rating": "Calificación (1–5)",
        })
        st.dataframe(
            candidatos_visibles[
                [
                    "Título",
                    "Primer autor",
                    "Área",
                    "Páginas",
                    "Formato (auxiliar)",
                    "Idioma",
                    "Editorial",
                    "Calificación (1–5)",
                ]
            ].style.format({
                "Calificación (1–5)": "{:.2f}",
            }),
            hide_index=True,
            width="stretch",
            height=600,
        )
        st.download_button(
            "Descargar lista de candidatos (CSV)",
            data=candidatos.to_csv(index=False).encode("utf-8"),
            file_name="goodreads_candidatos_exhibicion.csv",
            mime="text/csv",
        )

with tabs[6]:
    st.subheader("P1. Población de estudio y método")
    st.markdown(
        "**Pregunta:** ¿Cuál es la población de estudio y cuántos registros se excluyen "
        "por cada criterio?"
    )
    st.caption(
        "Métrica: cantidad de registros de edición y porcentaje sobre los 500.000 registros "
        "originales. Unidad: ediciones; no es una cantidad de ventas."
    )
    poblacion = cargar_tabla_resultados(str(POPULATION_PATH))
    kpi_a, kpi_b, kpi_c = st.columns(3)
    kpi_a.metric("Dataset original", f"{int(poblacion.iloc[0]['registros']):,} ediciones")
    poblacion_final = poblacion.loc[
        poblacion["criterio"].str.startswith("Población final:")
    ].iloc[0]
    kpi_b.metric("Población final", f"{int(poblacion_final['registros']):,} ediciones")
    kpi_c.metric("Retenido del dataset", f"{poblacion_final['porcentaje_del_dataset']:.2f}%")

    exclusiones = poblacion.loc[
        poblacion["criterio"].isin([
            "Calificación faltante",
            "Calificación < 3.5",
            "Destacados con páginas faltantes",
            "Destacados con páginas < 1",
            "Destacados con páginas > 5.000",
        ])
    ]
    izquierda, derecha = st.columns([1.2, 1])
    izquierda.plotly_chart(
        px.bar(
            exclusiones,
            x="registros",
            y="criterio",
            orientation="h",
            color="criterio",
            labels={"registros": "Ediciones excluidas", "criterio": "Criterio"},
            title="Exclusiones aplicadas en AE3",
        ),
        width="stretch",
    )
    poblacion_visible = poblacion.rename(columns={
        "criterio": "Criterio",
        "registros": "Ediciones",
        "porcentaje_del_dataset": "% del dataset original",
    })
    derecha.dataframe(
        poblacion_visible.style.format({"% del dataset original": "{:.2f}%"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "**Respuesta e interpretación.** De 500.000 registros, quedan 235.104 "
        "(47,02 %) después de seleccionar calificación ≥ 3,5 y páginas válidas. "
        "La tabla permite identificar cuánto aporta cada regla al recorte."
    )
    st.markdown(
        "**Decisión que podría apoyar:** delimitar qué población respalda los análisis "
        "posteriores. **Limitación:** los resultados describen solo esta selección de "
        "Goodreads, no todo su catálogo ni el de una librería."
    )

with tabs[2]:
    st.subheader("P2. Reparto por extensión")
    st.markdown(
        "**Pregunta:** ¿Qué número y porcentaje de títulos reúne cada área con el umbral "
        "de páginas definido?"
    )
    st.caption(
        "Métricas: títulos únicos y su porcentaje por área; se informa también el número de "
        "ediciones coincidentes y la mediana de páginas. El umbral seleccionado es 350 páginas."
    )
    st.caption(
        "La categoría **Obras extensas** es el proxy disponible para el área de Grandes Clásicos "
        "y Sagas: se asigna por superar 350 páginas. El dataset no identifica géneros ni sagas."
    )
    extension = resumir_reparto_extension(libros_analisis)
    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(
        px.bar(
            extension,
            x="categoria_extension",
            y="obras_asignadas",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            labels={
                "categoria_extension": "Área",
                "obras_asignadas": "Títulos únicos",
            },
            title="Títulos únicos por área",
            text_auto=True,
        ),
        width="stretch",
    )
    derecha.plotly_chart(
        px.bar(
            extension,
            x="categoria_extension",
            y="porcentaje_obras",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            labels={
                "categoria_extension": "Área",
                "porcentaje_obras": "% de títulos únicos",
            },
            title="Participación del total de títulos",
            text_auto=True,
        ),
        width="stretch",
    )
    extension_visible = extension[
        [
            "categoria_extension",
            "obras_asignadas",
            "porcentaje_obras",
            "registros_edicion",
            "paginas_mediana",
        ]
    ].rename(columns={
        "categoria_extension": "Área",
        "obras_asignadas": "Títulos únicos",
        "porcentaje_obras": "% de títulos",
        "registros_edicion": "Ediciones",
        "paginas_mediana": "Mediana de páginas",
    })
    st.dataframe(
        extension_visible.style.format({
            "% de títulos": "{:.2f}%",
            "Mediana de páginas": "{:.0f}",
        }),
        hide_index=True,
        width="stretch",
    )
    resumen_actual = extension.set_index("categoria_extension")
    st.info(
        "**Respuesta en la selección actual:** "
        f"Lecturas Ágiles reúne {resumen_actual.loc[AREA_AGIL, 'obras_asignadas']:,} "
        f"títulos únicos ({resumen_actual.loc[AREA_AGIL, 'porcentaje_obras']:.2f} %); "
        f"Obras extensas reúne "
        f"{resumen_actual.loc[AREA_EXTENSA, 'obras_asignadas']:,} títulos únicos "
        f"({resumen_actual.loc[AREA_EXTENSA, 'porcentaje_obras']:.2f} %)."
    )
    st.markdown(
        "**Decisión que podría apoyar:** estimar qué proporción de títulos corresponde a cada "
        "área. **Limitación:** el porcentaje de libros no equivale al porcentaje de estante; "
        "no se midieron dimensiones de libros ni de mobiliario."
    )
    st.caption(
        "La clasificación se basa en páginas. El umbral de 350 se eligió por la distribución "
        "observada y se contrasta con distintos cortes en el gráfico de sensibilidad."
    )

    st.subheader("Sensibilidad al umbral de páginas")
    sensibilidad = cargar_sensibilidad(str(SENSITIVITY_PATH))
    sensibilidad = sensibilidad.loc[
        sensibilidad["categoria_extension"].isin(ORDEN_AREAS)
    ]
    st.plotly_chart(
        px.line(
            sensibilidad,
            x="umbral_paginas",
            y="porcentaje_registros",
            color="categoria_extension",
            markers=True,
            color_discrete_map=COLORES_AREA,
            labels={
                "umbral_paginas": "Umbral de páginas",
                "porcentaje_registros": "% de ediciones",
                "categoria_extension": "Área",
            },
            title="Cómo cambia el reparto al mover el corte",
        ),
        width="stretch",
    )
    st.caption("La sensibilidad es el cálculo global de AE3; no cambia con los filtros de la página.")

with tabs[3]:
    st.subheader("P3. Editoriales con más títulos destacados")
    st.markdown(
        "**Pregunta:** ¿Qué editoriales concentran más títulos destacados en cada área "
        "y con qué calificación promedio?"
    )
    st.caption(
        "El top 10 se ordena por obras únicas (work_id); también se muestran sus ediciones. "
        "La calificación media y mediana se calculan sobre las ediciones de cada editorial."
    )
    editoriales = ranking_categoria(libros_analisis, "publisher")
    if editoriales.empty:
        st.info("No hay editoriales informadas para los filtros seleccionados.")
    else:
        izquierda, derecha = st.columns([1.2, 1])
        izquierda.plotly_chart(
            px.bar(
                editoriales,
                x="obras_unicas",
                y="publisher",
                color="categoria_extension",
                color_discrete_map=COLORES_AREA,
                orientation="h",
                barmode="group",
                labels={
                    "obras_unicas": "Títulos únicos",
                    "publisher": "Editorial",
                    "categoria_extension": "Área",
                },
                title="Top 10 por área",
            ),
            width="stretch",
        )
        editoriales_visible = editoriales[[
            "categoria_extension",
            "publisher",
            "registros_edicion",
            "obras_unicas",
            "calificacion_promedio",
            "calificacion_mediana",
        ]].rename(columns={
            "categoria_extension": "Área",
            "publisher": "Editorial",
            "registros_edicion": "Ediciones",
            "obras_unicas": "Títulos únicos",
            "calificacion_promedio": "Calificación media (1–5)",
            "calificacion_mediana": "Calificación mediana (1–5)",
        })
        derecha.dataframe(
            editoriales_visible.style.format({
                "Calificación media (1–5)": "{:.2f}",
                "Calificación mediana (1–5)": "{:.2f}",
            }),
            hide_index=True,
            width="stretch",
        )
        lideres = editoriales.groupby("categoria_extension", sort=False).head(1)
        texto_lideres = "; ".join(
            f"{fila['categoria_extension']}: {fila['publisher']} "
            f"({fila['obras_unicas']:,} títulos únicos; "
            f"{fila['calificacion_promedio']:.2f}/5)"
            for _, fila in lideres.iterrows()
        )
        st.info(f"**Respuesta en la selección actual:** {texto_lideres}.")
    st.caption(
        "Una obra puede aparecer en más de una editorial si tiene varias ediciones; "
        "la calificación media resume las ediciones de cada sello."
    )
    st.markdown(
        "**Decisión que podría apoyar:** preparar una lista de editoriales para investigar "
        "en el mercado local. **Limitación:** número de títulos y nota de Goodreads no "
        "demuestran ventas, rentabilidad ni condiciones comerciales; los sellos de un mismo "
        "grupo editorial no están unificados."
    )
    st.caption(
        "No se suman calificaciones o reseñas por editorial: esas métricas pertenecen a la "
        "obra y pueden repetirse en varias ediciones. La media de P3 es descriptiva por "
        "registro de edición; la calificación de la obra puede repetirse."
    )

with tabs[4]:
    st.subheader("P4. Idiomas con más títulos destacados")
    st.markdown(
        "**Pregunta:** ¿Qué idiomas concentran más títulos destacados en cada área "
        "y con qué calificación promedio?"
    )
    st.caption(
        "El top 10 se ordena por obras únicas (work_id); también se muestran sus ediciones. "
        "La calificación media y mediana se calculan sobre las ediciones de cada idioma."
    )
    idiomas = ranking_categoria(libros_analisis, "idioma_normalizado")
    if idiomas.empty:
        sin_idioma = int(libros_analisis["idioma_normalizado"].isna().sum())
        if sin_idioma:
            st.info(
                f"La selección contiene {sin_idioma:,} ediciones sin idioma informado. "
                "Consulta su cantidad y participación en la composición por idioma de P6."
            )
        else:
            st.info("No hay idiomas informados para los filtros seleccionados.")
    else:
        izquierda, derecha = st.columns([1.2, 1])
        izquierda.plotly_chart(
            px.bar(
                idiomas,
                x="obras_unicas",
                y="idioma_normalizado",
                color="categoria_extension",
                color_discrete_map=COLORES_AREA,
                orientation="h",
                barmode="group",
                labels={
                    "obras_unicas": "Títulos únicos",
                    "idioma_normalizado": "Idioma",
                    "categoria_extension": "Área",
                },
                title="Top 10 por área",
            ),
            width="stretch",
        )
        idiomas_visible = idiomas[[
            "categoria_extension",
            "idioma_normalizado",
            "registros_edicion",
            "obras_unicas",
            "calificacion_promedio",
            "calificacion_mediana",
        ]].rename(columns={
            "categoria_extension": "Área",
            "idioma_normalizado": "Idioma",
            "registros_edicion": "Ediciones",
            "obras_unicas": "Títulos únicos",
            "calificacion_promedio": "Calificación media (1–5)",
            "calificacion_mediana": "Calificación mediana (1–5)",
        })
        derecha.dataframe(
            idiomas_visible.style.format({
                "Calificación media (1–5)": "{:.2f}",
                "Calificación mediana (1–5)": "{:.2f}",
            }),
            hide_index=True,
            width="stretch",
        )
        lideres = idiomas.groupby("categoria_extension", sort=False).head(1)
        texto_lideres = "; ".join(
            f"{fila['categoria_extension']}: {fila['idioma_normalizado']} "
            f"({fila['obras_unicas']:,} títulos únicos; "
            f"{fila['calificacion_promedio']:.2f}/5)"
            for _, fila in lideres.iterrows()
        )
        st.info(f"**Respuesta en la selección actual:** {texto_lideres}.")
    st.caption(
        "Una obra puede aparecer en varios idiomas si tiene ediciones traducidas. El inglés "
        "predomina en Goodreads, pero la fuente no identifica la ciudad de sus usuarios ni "
        "permite atribuir esa composición al mercado de Nueva York."
    )
    st.markdown(
        "**Decisión que podría apoyar:** explorar idiomas para futuras compras importadas. "
        "**Limitación:** la presencia en Goodreads no representa la demanda local ni "
        "específicamente la de Nueva York. La media de P4 es descriptiva por edición; "
        "la calificación puede repetirse para una misma obra."
    )

    st.divider()
    st.subheader("P6. Composición por idioma de cada área")
    st.markdown(
        "**Pregunta:** ¿Cuál es la composición por idioma de cada área? "
        "Los resultados se recalculan con la selección actual."
    )
    composicion = resumir_composicion_idioma(libros_analisis)
    composicion_grafico = composicion.groupby(
        "categoria_extension", sort=False
    ).head(10)
    izquierda, derecha = st.columns([1.2, 1])
    izquierda.plotly_chart(
        px.bar(
            composicion_grafico,
            x="porcentaje_registros",
            y="idioma",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            orientation="h",
            barmode="group",
            labels={
                "porcentaje_registros": "% del área filtrada",
                "idioma": "Idioma",
                "categoria_extension": "Área",
            },
            title="Diez idiomas con mayor participación por área",
        ),
        width="stretch",
    )
    composicion_visible = composicion.rename(columns={
        "categoria_extension": "Área",
        "idioma": "Idioma",
        "registros_edicion": "Ediciones",
        "porcentaje_registros": "% del área filtrada",
    })
    derecha.dataframe(
        composicion_visible.style.format({"% del área filtrada": "{:.2f}%"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "La participación se calcula sobre las ediciones filtradas de cada área e incluye "
        "“Sin idioma informado”. El gráfico muestra los diez idiomas principales por área; "
        "la tabla contiene todos los idiomas de la selección."
    )
    lideres_idioma = composicion.groupby("categoria_extension", sort=False).head(1)
    lectura_idioma = "; ".join(
        f"{fila['categoria_extension']}: {fila['idioma']} "
        f"({fila['porcentaje_registros']:.2f} %)"
        for _, fila in lideres_idioma.iterrows()
    )
    st.markdown(
        f"**Respuesta en la selección actual:** idioma con mayor participación: {lectura_idioma}. "
        "**Decisión posible:** investigar oferta multilingüe para el escenario hipotético de "
        "Nueva York. **Limitación:** estas proporciones describen Goodreads, no la composición "
        "lingüística ni las preferencias de compra de la ciudad."
    )

with tabs[5]:
    st.subheader("P5. Distribución de las calificaciones")
    st.markdown(
        "**Pregunta:** ¿Cómo se distribuye la calificación promedio de las ediciones "
        "destacadas en cada área?"
    )
    st.caption(
        "Métricas: porcentaje de ediciones en rangos de calificación; media y mediana "
        "en escala de 1 a 5; desviación estándar, cuartiles e intervalo intercuartil."
    )
    distribucion = resumir_calificaciones(libros_analisis)
    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(
        px.bar(
            distribucion,
            x="categoria_extension",
            y="porcentaje_registros",
            color="rango_calificacion",
            barmode="stack",
            labels={
                "categoria_extension": "Área",
                "porcentaje_registros": "% de ediciones",
                "rango_calificacion": "Rango de calificación",
            },
            title="Rangos de calificación por área",
        ),
        width="stretch",
    )
    resumen_calificaciones = (
        resumir_estadisticas_calificacion(libros_analisis)
    )
    derecha.plotly_chart(
        px.box(
            libros_analisis,
            x="categoria_extension",
            y="average_rating",
            color="categoria_extension",
            color_discrete_map=COLORES_AREA,
            points=False,
            range_y=[3.5, 5],
            labels={
                "categoria_extension": "Área",
                "average_rating": "Calificación promedio",
            },
            title="Distribución, mediana y cuartiles por área",
        ),
        width="stretch",
    )
    st.dataframe(
        resumen_calificaciones.rename(columns={
            "categoria_extension": "Área",
            "registros_edicion": "Ediciones",
            "calificacion_media": "Media",
            "desviacion_estandar": "Desviación estándar",
            "minimo": "Mínimo",
            "cuartil_1": "Q1",
            "mediana": "Mediana",
            "cuartil_3": "Q3",
            "maximo": "Máximo",
            "rango_intercuartil": "Rango intercuartil",
        }).style.format({
            "Media": "{:.2f}",
            "Desviación estándar": "{:.2f}",
            "Mínimo": "{:.2f}",
            "Q1": "{:.2f}",
            "Mediana": "{:.2f}",
            "Q3": "{:.2f}",
            "Máximo": "{:.2f}",
            "Rango intercuartil": "{:.2f}",
        }, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        distribucion.style.format({"porcentaje_registros": "{:.2f}%"}),
        hide_index=True,
        width="stretch",
    )
    if not resumen_calificaciones.empty:
        lectura = "; ".join(
            f"{fila['categoria_extension']}: media {fila['calificacion_media']:.2f}/5, "
            f"mediana {fila['mediana']:.2f}, Q1 {fila['cuartil_1']:.2f} y "
            f"Q3 {fila['cuartil_3']:.2f}"
            for _, fila in resumen_calificaciones.iterrows()
        )
        st.info(f"**Respuesta en la selección actual:** {lectura}.")
    st.markdown(
        "**Decisión que podría apoyar:** comparar la recepción promedio entre áreas; "
        "si las notas son similares, no usarlas por sí solas para asignar más espacio. "
        "**Limitación:** solo se analizan registros con nota ≥ 3,5; la calificación es "
        "promedio de Goodreads y puede repetirse en varias ediciones."
    )
with tabs[7]:
    st.subheader("Hallazgos, recomendación y limitaciones")
    reparto_final = resumir_reparto_extension(libros_analisis)
    ratings_finales = resumir_estadisticas_calificacion(libros_analisis)
    editoriales_finales = ranking_categoria(libros_analisis, "publisher")
    idiomas_finales = ranking_categoria(libros_analisis, "idioma_normalizado")
    resumen_reparto = "; ".join(
        f"{fila['categoria_extension']}: {int(fila['obras_asignadas']):,} títulos únicos "
        f"({fila['porcentaje_obras']:.2f} %)"
        for _, fila in reparto_final.iterrows()
    )
    resumen_rating = "; ".join(
        f"{fila['categoria_extension']}: media {fila['calificacion_media']:.2f}, "
        f"mediana {fila['mediana']:.2f}, IQR {fila['rango_intercuartil']:.2f}"
        for _, fila in ratings_finales.iterrows()
    )
    lideres_editoriales = editoriales_finales.groupby(
        "categoria_extension", sort=False
    ).head(1)
    lideres_idiomas = idiomas_finales.groupby(
        "categoria_extension", sort=False
    ).head(1)
    resumen_editoriales = "; ".join(
        f"{fila['categoria_extension']}: {fila['publisher']}"
        for _, fila in lideres_editoriales.iterrows()
    ) or "Sin editorial informada en la selección"
    resumen_idiomas = "; ".join(
        f"{fila['categoria_extension']}: {fila['idioma_normalizado']}"
        for _, fila in lideres_idiomas.iterrows()
    ) or "Sin idioma informado en la selección"
    st.markdown(
        f"""
**Hallazgos en la selección actual**

- **Reparto:** {resumen_reparto}.
- **Calificaciones:** {resumen_rating}.
- **Editorial con más títulos por área:** {resumen_editoriales}.
- **Idioma con más títulos por área:** {resumen_idiomas}.
"""
    )
    st.markdown(
        "**Recomendación para el cliente:** revisar la lista de títulos mejor calificados "
        "y contrastar el reparto por área, las editoriales líderes y los idiomas con ventas, "
        "inventario y preferencias locales antes de decidir compras o espacio. No convertir "
        "automáticamente el porcentaje de libros en porcentaje de estante."
    )
    st.markdown(
        """
**Limitaciones**

- No hay ventas, costos, inventario, disponibilidad ni medidas físicas de estantería.
- Por ello, no se calculan ingresos ni rentabilidad; las calificaciones no demuestran demanda local.
- La ubicación de la tienda en Nueva York es hipotética. Goodreads no registra la ubicación
  de sus usuarios, así que la composición por idioma no estima el mercado de la ciudad.
- La población se filtra a calificaciones ≥ 3,5; no compara libros destacados con todo el catálogo.
- El umbral de 350 páginas es una regla analítica. Los datos no identifican géneros ni sagas.
- La lista de candidatos se ordena por la calificación promedio, sin considerar el número de votos.
- La fuente no permite observar cambios a través del tiempo.
"""
    )

with tabs[8]:
    st.subheader("Datos analíticos por edición")
    columnas_analisis = [
        "title",
        "first_author",
        "pages",
        "average_rating",
        "publisher",
        "idioma_normalizado",
        "categoria_extension",
    ]
    datos_analisis = libros_analisis[columnas_analisis].rename(columns={
        "title": "Título",
        "first_author": "Primer autor",
        "pages": "Páginas",
        "average_rating": "Calificación promedio",
        "publisher": "Editorial",
        "idioma_normalizado": "Idioma",
        "categoria_extension": "Área",
    })
    st.write(
        f"Se muestran hasta 1.000 de **{len(datos_analisis):,}** registros que cumplen "
        "los filtros analíticos. Incluyen las seis variables y la categoría de extensión derivada. "
        "El botón descarga todas las filas que cumplen los filtros."
    )
    st.dataframe(
        datos_analisis.head(1000),
        hide_index=True,
        width="stretch",
        height=500,
    )
    st.download_button(
        "Descargar datos filtrados (CSV)",
        data=datos_analisis.to_csv(index=False).encode("utf-8"),
        file_name="goodreads_libros_filtrados.csv",
        mime="text/csv",
    )
    st.caption(
        "Fuente: pfaha/goodreads-books, revisión 43b3198cd7ec9c142d56181f1a09ed9e8c112610. "
        "Goodreads mide actividad de sus usuarios, no ventas de una librería."
    )
