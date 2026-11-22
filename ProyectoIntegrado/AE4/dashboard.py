from __future__ import annotations

from pathlib import Path

import pandas as pd

AREA_AGIL = "Lecturas Ágiles"
AREA_EXTENSA = "Obras extensas"
ORDEN_AREAS = [AREA_AGIL, AREA_EXTENSA]
ORDEN_RANGOS = ["3,5 a 3,99", "4,0 a 4,49", "4,5 a 5,0"]
SIN_IDIOMA = "Sin idioma informado"

COLUMNAS_REQUERIDAS = {
    "book_id",
    "work_id",
    "title",
    "first_author",
    "pages",
    "categoria_extension",
    "average_rating",
    "ratings_count",
    "reviews_count",
    "publisher",
    "idioma_normalizado",
    "is_part_of_series",
    "formato_grupo",
    "rango_calificacion",
    "registro_representativo_obra",
}


def _convertir_booleano(serie: pd.Series, nombre: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(serie):
        resultado = serie
    else:
        valores = serie.astype("string").str.strip().str.lower()
        resultado = valores.map({"true": True, "false": False, "1": True, "0": False})
    if resultado.isna().any():
        raise ValueError(f"La columna {nombre!r} contiene valores booleanos inválidos o nulos.")
    return resultado.astype(bool)


def cargar_libros(ruta_csv: str | Path) -> pd.DataFrame:
    ruta = Path(ruta_csv)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el dataset preparado de AE3: {ruta}")

    libros = pd.read_csv(ruta, encoding="utf-8")
    faltantes = sorted(COLUMNAS_REQUERIDAS.difference(libros.columns))
    if faltantes:
        raise ValueError(f"El CSV preparado no contiene las columnas requeridas: {faltantes}")
    if libros.empty:
        raise ValueError(f"El CSV preparado está vacío: {ruta}")

    for columna in ("pages", "average_rating", "ratings_count", "reviews_count", "work_id"):
        libros[columna] = pd.to_numeric(libros[columna], errors="raise")
    for columna in ("is_part_of_series", "registro_representativo_obra"):
        libros[columna] = _convertir_booleano(libros[columna], columna)

    if libros["book_id"].isna().any() or libros["book_id"].duplicated().any():
        raise ValueError("book_id debe identificar una sola vez cada edición.")
    if libros["work_id"].isna().any():
        raise ValueError("work_id no puede contener valores nulos.")
    if libros["registro_representativo_obra"].sum() != libros["work_id"].nunique():
        raise ValueError("Debe existir exactamente una edición representativa por obra.")
    if not libros["categoria_extension"].isin(ORDEN_AREAS).all():
        raise ValueError("El CSV contiene categorías de extensión desconocidas.")

    return libros


def filtrar_libros(
    libros: pd.DataFrame,
    areas: list[str] | None = None,
    idiomas: list[str] | None = None,
    formatos: list[str] | None = None,
    busqueda_editorial: str = "",
    rango_calificacion: tuple[float, float] = (3.5, 5.0),
) -> pd.DataFrame:
    mascara = pd.Series(True, index=libros.index)

    if areas:
        mascara &= libros["categoria_extension"].isin(areas)

    if idiomas:
        idiomas_seleccionados = [idioma for idioma in idiomas if idioma != SIN_IDIOMA]
        mascara_idioma = libros["idioma_normalizado"].isin(idiomas_seleccionados)
        if SIN_IDIOMA in idiomas:
            mascara_idioma |= libros["idioma_normalizado"].isna()
        mascara &= mascara_idioma

    if formatos:
        mascara &= libros["formato_grupo"].isin(formatos)

    consulta = busqueda_editorial.strip()
    if consulta:
        mascara &= libros["publisher"].astype("string").str.contains(
            consulta, case=False, regex=False, na=False
        )

    minimo, maximo = rango_calificacion
    mascara &= libros["average_rating"].between(minimo, maximo, inclusive="both")
    return libros.loc[mascara].copy()


def _ediciones_representativas(libros: pd.DataFrame) -> pd.DataFrame:
    orden = libros.assign(
        _book_id_num=pd.to_numeric(libros["book_id"], errors="coerce")
    ).sort_values(
        ["work_id", "ratings_count", "_book_id_num"],
        ascending=[True, False, True],
        kind="stable",
    )
    return orden.drop_duplicates(subset="work_id", keep="first").drop(
        columns="_book_id_num"
    )


def candidatos_exhibicion(libros: pd.DataFrame, limite: int = 20) -> pd.DataFrame:
    if limite < 1:
        raise ValueError("El límite de candidatos debe ser mayor que cero.")
    candidatos = (
        libros.assign(_book_id_num=pd.to_numeric(libros["book_id"], errors="coerce"))
        .sort_values(
            ["work_id", "average_rating", "_book_id_num"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(subset="work_id", keep="first")
        .drop(columns="_book_id_num")
    )
    return candidatos.sort_values(
        ["average_rating", "title", "first_author"],
        ascending=[False, True, True],
        kind="stable",
    ).head(limite).copy()


def resumir_reparto_extension(libros: pd.DataFrame) -> pd.DataFrame:
    resumen = resumir_extension(libros)
    return resumen[[
        "categoria_extension",
        "registros_edicion",
        "porcentaje_registros",
        "obras_asignadas",
        "porcentaje_obras",
        "paginas_mediana",
    ]]


def resumir_extension(libros: pd.DataFrame) -> pd.DataFrame:
    filas = []
    total_registros = len(libros)
    representativas = _ediciones_representativas(libros)
    total_obras = len(representativas)

    for area in ORDEN_AREAS:
        grupo = libros.loc[libros["categoria_extension"].eq(area)]
        obras_area = representativas.loc[
            representativas["categoria_extension"].eq(area)
        ]
        filas.append({
            "categoria_extension": area,
            "registros_edicion": len(grupo),
            "porcentaje_registros": len(grupo) / total_registros * 100 if total_registros else 0,
            "obras_con_ediciones_en_area": grupo["work_id"].nunique(),
            "obras_asignadas": obras_area["work_id"].nunique(),
            "porcentaje_obras": (
                obras_area["work_id"].nunique() / total_obras * 100 if total_obras else 0
            ),
            "paginas_mediana": grupo["pages"].median(),
        })
    return pd.DataFrame(filas)


def ranking_categoria(
    libros: pd.DataFrame,
    columna: str,
    limite: int = 10,
) -> pd.DataFrame:
    filas = []
    for area in ORDEN_AREAS:
        grupo = libros.loc[
            libros["categoria_extension"].eq(area) & libros[columna].notna()
        ]
        if grupo.empty:
            continue
        resumen = (
            grupo.groupby(columna, observed=True)
            .agg(
                registros_edicion=("book_id", "size"),
                obras_unicas=("work_id", "nunique"),
                calificacion_promedio=("average_rating", "mean"),
                calificacion_mediana=("average_rating", "median"),
            )
            .reset_index()
            .sort_values(
                ["obras_unicas", "registros_edicion", columna],
                ascending=[False, False, True],
                kind="stable",
            )
            .head(limite)
        )
        resumen.insert(0, "categoria_extension", area)
        resumen["porcentaje_registros_con_valor"] = (
            resumen["registros_edicion"] / len(grupo) * 100
        )
        total_obras_con_valor = grupo["work_id"].nunique()
        resumen["porcentaje_obras_con_valor"] = (
            resumen["obras_unicas"] / total_obras_con_valor * 100
            if total_obras_con_valor
            else 0
        )
        filas.append(resumen)

    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def resumir_calificaciones(libros: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for area in ORDEN_AREAS:
        grupo = libros.loc[libros["categoria_extension"].eq(area)]
        if grupo.empty:
            continue
        for rango in ORDEN_RANGOS:
            cantidad = int(grupo["rango_calificacion"].eq(rango).sum())
            filas.append({
                "categoria_extension": area,
                "rango_calificacion": rango,
                "registros_edicion": cantidad,
                "porcentaje_registros": cantidad / len(grupo) * 100,
            })
    return pd.DataFrame(filas)


def resumir_estadisticas_calificacion(libros: pd.DataFrame) -> pd.DataFrame:
    estadisticas = (
        libros.groupby("categoria_extension", observed=True)
        .agg(
            registros_edicion=("average_rating", "count"),
            calificacion_media=("average_rating", "mean"),
            desviacion_estandar=("average_rating", "std"),
            minimo=("average_rating", "min"),
            cuartil_1=("average_rating", lambda valores: valores.quantile(0.25)),
            mediana=("average_rating", "median"),
            cuartil_3=("average_rating", lambda valores: valores.quantile(0.75)),
            maximo=("average_rating", "max"),
        )
        .reset_index()
    )
    estadisticas["rango_intercuartil"] = (
        estadisticas["cuartil_3"] - estadisticas["cuartil_1"]
    )
    return estadisticas


def resumir_participacion(libros: pd.DataFrame) -> pd.DataFrame:
    representativas = _ediciones_representativas(libros)
    filas = []
    for area in ORDEN_AREAS:
        grupo = representativas.loc[representativas["categoria_extension"].eq(area)]
        filas.append({
            "categoria_extension": area,
            "obras_asignadas": grupo["work_id"].nunique(),
            "calificaciones_totales": int(grupo["ratings_count"].sum()),
            "resenas_totales": int(grupo["reviews_count"].sum()),
            "calificaciones_mediana_por_obra": grupo["ratings_count"].median(),
            "resenas_mediana_por_obra": grupo["reviews_count"].median(),
            "obras_en_series": int(grupo["is_part_of_series"].sum()),
            "porcentaje_obras_en_series": (
                grupo["is_part_of_series"].mean() * 100 if len(grupo) else 0
            ),
        })
    resultado = pd.DataFrame(filas)
    total_calificaciones = resultado["calificaciones_totales"].sum()
    total_resenas = resultado["resenas_totales"].sum()
    resultado["porcentaje_calificaciones"] = (
        resultado["calificaciones_totales"] / total_calificaciones * 100
        if total_calificaciones else 0
    )
    resultado["porcentaje_resenas"] = (
        resultado["resenas_totales"] / total_resenas * 100 if total_resenas else 0
    )
    return resultado


def resumir_formato(libros: pd.DataFrame) -> pd.DataFrame:
    filas = []
    formatos = sorted(libros["formato_grupo"].dropna().unique())
    for area in ORDEN_AREAS:
        grupo = libros.loc[libros["categoria_extension"].eq(area)]
        for formato in formatos:
            cantidad = int(grupo["formato_grupo"].eq(formato).sum())
            filas.append({
                "categoria_extension": area,
                "formato_grupo": formato,
                "registros_edicion": cantidad,
                "porcentaje_registros": cantidad / len(grupo) * 100 if len(grupo) else 0,
            })
    return pd.DataFrame(filas)


def resumir_composicion_idioma(libros: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for area in ORDEN_AREAS:
        grupo = libros.loc[libros["categoria_extension"].eq(area)]
        if grupo.empty:
            continue
        conteos = (
            grupo["idioma_normalizado"]
            .fillna(SIN_IDIOMA)
            .value_counts()
            .rename_axis("idioma")
            .rename("registros_edicion")
            .reset_index()
        )
        conteos.insert(0, "categoria_extension", area)
        conteos["porcentaje_registros"] = conteos["registros_edicion"] / len(grupo) * 100
        filas.append(conteos)

    if not filas:
        return pd.DataFrame(
            columns=[
                "categoria_extension",
                "idioma",
                "registros_edicion",
                "porcentaje_registros",
            ]
        )
    return pd.concat(filas, ignore_index=True).sort_values(
        ["categoria_extension", "registros_edicion", "idioma"],
        ascending=[True, False, True],
        kind="stable",
        ignore_index=True,
    )
