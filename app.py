"""Interfaz Streamlit: búsqueda y respuesta sobre UNA sola colección multimodal.

Convención de la casa: nunca se usan ternarios que contengan llamadas `st.*`.
Un ternario evalúa ambas ramas y el DeltaGenerator de la rama no usada se filtra
a la salida. Todos los renderizados van con `if/elif` explícitos.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src import config, describe, embedder, ingest, rag, store

st.set_page_config(page_title="Motor DATABiQ", page_icon="🔎", layout="wide")

MODALITIES = {
    "Todas": None,
    "Texto": "text",
    "Imagen": "image",
    "Audio": "audio",
    "PDF": "pdf",
    "Video": "video",
}

# Cuántos fragmentos mira el modo recuperación. El del `.env` siempre está entre las
# opciones y el resto sirven para mirar más ancho: sin capa de generación, nadie paga
# por esos fragmentos de más.
TOP_K_CHOICES = sorted({5, 8, 12, 20, config.TOP_K})
TOP_K_DEFAULT = TOP_K_CHOICES.index(config.TOP_K)

_UPLOAD_EXT = ingest.IMAGE_EXT | ingest.AUDIO_EXT | ingest.PDF_EXT | ingest.TEXT_EXT
if config.video_enabled():
    # Sin PyAV no se pueden muestrear fotogramas: ofrecer video solo llevaría a un
    # error por archivo, así que no se ofrece.
    _UPLOAD_EXT = _UPLOAD_EXT | ingest.VIDEO_EXT
UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in _UPLOAD_EXT)


@st.cache_resource(show_spinner="Cargando EmbeddingGemma 2 (744M parámetros)…")
def _warm_up():
    """Fuerza la carga del modelo una vez y muestra el spinner. Es un singleton."""
    return embedder.get_model()


def _formulario(
    clave: str, verbo: str, con_top_k: bool = False
) -> tuple[bool, str, str | None, int]:
    """Consulta + filtro de modalidad. `clave` aísla los widgets entre pestañas.

    Con `con_top_k` añade cuántos fragmentos recuperar: en modo recuperación no hay capa
    de generación que pague por ellos, así que se puede mirar más ancho.
    """
    with st.form(f"form_{clave}"):
        # NO convertir esto a ternario: es la convención de la casa (ver el docstring del
        # módulo). Un ternario con llamadas `st.*` evalúa ambas ramas y el DeltaGenerator
        # de la no usada se filtra a la salida. Ruff sugiere SIM108 aquí; se ignora a
        # conciencia y el `noqa` documenta por qué.
        if con_top_k:  # noqa: SIM108
            columnas = st.columns([4, 1, 1])
        else:
            columnas = st.columns([4, 1])

        pregunta = columnas[0].text_input(
            "Pregunta",
            placeholder="p. ej. ¿qué propone el paper sobre la atención multi-cabeza?",
            key=f"pregunta_{clave}",
            label_visibility="collapsed",
        )
        elegida = columnas[1].selectbox(
            "Modalidad", list(MODALITIES), key=f"modalidad_{clave}", label_visibility="collapsed"
        )
        top_k = config.TOP_K
        if con_top_k:
            top_k = int(
                columnas[2].selectbox(
                    "Top-K",
                    TOP_K_CHOICES,
                    index=TOP_K_DEFAULT,
                    key=f"topk_{clave}",
                    label_visibility="collapsed",
                    help="Cuántos fragmentos recuperar",
                )
            )
        enviado = st.form_submit_button(verbo, type="primary")

    return enviado, pregunta, MODALITIES[elegida], top_k


def _filas_ranking(hits: list[dict]) -> list[dict]:
    """El ranking como tabla: puesto, modalidad, archivo, dónde y similitud.

    `donde` es la página del PDF o el tramo del video, que es lo que vuelve comprobable
    una cita.
    """
    filas = []
    for puesto, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        if meta.get("page"):
            donde = f"página {meta['page']}"
        elif meta.get("start") is not None:
            donde = f"{meta['start']}–{meta['end']} s"
        else:
            donde = ""
        filas.append(
            {
                "puesto": puesto,
                "modalidad": meta.get("modality", ""),
                "archivo": meta.get("name", ""),
                "donde": donde,
                "similitud": round(float(hit["similarity"]), 4),
            }
        )
    return filas


# Tope de vectores que se piden al medir las bandas: con un `k` alto el ranking es
# completo, pero traerlo entero por consulta en un índice grande sería caro.
TOPE_MEDICION = 50


def _sin_texto_propio(nombre: str, documento: str) -> bool:
    """¿A ese vector el LLM solo le llega el nombre del archivo o su tramo?

    Son las tres formas de quedarse sin texto: la imagen o el audio sin descripción (el
    documento ES el nombre), la página de un PDF sin capa de texto, y el fragmento de
    video sin transcripción. Se encuentran, pero no se pueden responder.
    """
    return (
        documento.strip() in ("", nombre)
        or "sin texto extraíble" in documento
        or " · fragmento " in documento
    )


def _bandas(hits: list[dict]) -> list[dict]:
    """Mínimo, máximo y amplitud de la similitud dentro de cada modalidad.

    Es la medición que explica por qué un acierto puede depender de milésimas: si dos
    bandas se solapan, el ranking entre modalidades discrimina poco.
    """
    valores: dict[str, list[float]] = {}
    for hit in hits:
        clave = hit["metadata"].get("modality", "?")
        valores.setdefault(clave, []).append(float(hit["similarity"]))
    return [
        {
            "modalidad": modalidad,
            "n": len(numeros),
            "mín": round(min(numeros), 4),
            "máx": round(max(numeros), 4),
            "amplitud": round(max(numeros) - min(numeros), 4),
        }
        for modalidad, numeros in sorted(valores.items())
    ]


def _busca_y_cronometra(pregunta: str, modalidad: str | None, top_k: int) -> list[dict]:
    """Busca y anota lo que tardó: la latencia es una métrica viva, no un dato de memoria."""
    arranque = time.perf_counter()
    encontrados = rag.retrieve(pregunta, top_k=top_k, modality=modalidad)
    st.session_state.setdefault("latencias", []).append(time.perf_counter() - arranque)
    return encontrados


def _panel_metricas() -> None:
    """Lo medible del índice y del uso de la sesión, con una acción detrás de cada cifra.

    **Nada de aquí llama al modelo**: todo sale del metadata de los vectores, de la
    carpeta de caché y de las latencias que la propia sesión va midiendo. Es la medición
    que antes se hacía a mano en la terminal, ya dentro de la aplicación.
    """
    items = store.inventory()
    if not items:
        st.info("Índice vacío: añade archivos para poder medirlo.")
        return

    por_archivo: dict[str, dict] = {}
    sin_texto = 0
    vivos: dict[str, bool] = {}
    for item in items:
        meta = item["metadata"]
        nombre = meta.get("name", "?")
        ruta = str(meta.get("path", ""))
        if _sin_texto_propio(nombre, item["document"]):
            sin_texto += 1
        if ruta not in vivos:
            vivos[ruta] = Path(ruta).exists()
        entrada = por_archivo.setdefault(
            nombre,
            {
                "archivo": nombre,
                "modalidad": meta.get("modality", "?"),
                "vectores": 0,
                "mb": 0.0,
                "rutas": set(),
            },
        )
        entrada["vectores"] += 1
        entrada["rutas"].add(ruta)
        if vivos[ruta]:
            entrada["mb"] = round(Path(ruta).stat().st_size / 1024 / 1024, 2)

    huerfanos = [ruta for ruta, existe in vivos.items() if not existe]

    st.markdown("#### El índice")
    columnas = st.columns(4)
    columnas[0].metric("Vectores", len(items))
    columnas[1].metric("Archivos", len(por_archivo))
    columnas[2].metric("Sin texto propio", sin_texto)
    columnas[3].metric("Sin archivo en disco", len(huerfanos))
    st.caption(
        "«Sin texto propio»: al LLM solo le llega el nombre del archivo, la página sin capa "
        "de texto o el tramo del video. Se encuentran, pero no se responden — se arreglan "
        "en la sección **Textos para revisar**, aquí abajo."
    )

    # Los modelos que producen los textos van aquí y no en la sección de textos: esta es
    # la ficha del sistema, y aquella el editor donde se corrigen.
    productores = []
    if config.VISION_MODEL:
        productores.append(f"visión `{config.VISION_MODEL}`")
    if config.WHISPER_MODEL:
        productores.append(f"transcripción `{config.WHISPER_MODEL}`")
    if productores:
        st.caption("El texto lo producen " + " y ".join(productores) + ".")

    st.dataframe(
        [
            {
                "archivo": entrada["archivo"],
                "modalidad": entrada["modalidad"],
                "vectores": entrada["vectores"],
                "MB": entrada["mb"],
                "rutas": len(entrada["rutas"]),
            }
            for entrada in sorted(por_archivo.values(), key=lambda e: e["vectores"], reverse=True)
        ],
        hide_index=True,
    )
    st.caption(
        "`vectores` es el fan-out —10 imágenes dan 10 vectores y un video de 129 MB da 3, uno "
        "por fragmento— y `rutas` mayor que 1 significa **el mismo archivo indexado desde dos "
        "sitios**, que crea un vector por copia."
    )

    if huerfanos:
        st.warning(f"{len(huerfanos)} vector(es) apuntan a archivos que ya no están en disco.")
        if st.button(f"Borrar los {len(huerfanos)} vectores huérfanos"):
            sobrantes = [
                item["id"] for item in items if str(item["metadata"].get("path", "")) in huerfanos
            ]
            store.delete_ids(sobrantes)
            st.rerun()

    st.divider()
    st.markdown("#### Bandas de similitud por modalidad")
    st.caption(
        "Es lo que menos discrimina del sistema: si dos bandas se solapan, cuál gana puede "
        "depender de milésimas. Dentro de cada modalidad el orden sí funciona."
    )
    with st.form("form_bandas"):
        consulta = st.text_input(
            "Consulta para medir",
            placeholder="p. ej. ¿de qué trata el corpus?",
            label_visibility="collapsed",
        )
        medir = st.form_submit_button("Medir", type="primary")
    if medir and consulta.strip():
        encontrados = _busca_y_cronometra(consulta, None, min(store.total(), TOPE_MEDICION))
        st.session_state["bandas"] = _bandas(encontrados)
        st.session_state["bandas_consulta"] = consulta
    if st.session_state.get("bandas_consulta"):
        st.caption(
            f"«{st.session_state['bandas_consulta']}» · ranking completo hasta {TOPE_MEDICION}"
        )
        st.dataframe(st.session_state["bandas"], hide_index=True)

    st.divider()
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("#### Caché de copias")
        copias = [c for c in config.CACHE_DIR.glob("*") if c.is_file()]
        megas = sum(c.stat().st_size for c in copias) / 1024 / 1024
        st.metric("Archivos", len(copias))
        st.caption(f"{megas:.1f} MB · se poda sola al reindexar")
    with derecha:
        st.markdown("#### Latencia de recuperación")
        latencias = sorted(st.session_state.get("latencias") or [])
        if latencias:
            st.metric("Mediana de la sesión", f"{latencias[len(latencias) // 2]:.3f} s")
            st.caption(f"{len(latencias)} consulta(s) · mediana histórica medida: 0,115 s")
        else:
            st.caption("Sin consultas en esta sesión todavía.")

    if config.generation_enabled():
        st.divider()
        st.markdown("#### Consumo de la sesión")
        gasto = st.session_state.get("gasto") or {}
        de_entrada = int(gasto.get("prompt", 0))
        de_salida = int(gasto.get("completion", 0))
        columnas = st.columns(3)
        columnas[0].metric("Tokens de entrada", de_entrada)
        columnas[1].metric("Tokens de salida", de_salida)
        columnas[2].metric("Preguntas", int(gasto.get("preguntas", 0)))
        coste = (
            de_entrada / 1_000_000 * config.PRICE_INPUT
            + de_salida / 1_000_000 * config.PRICE_OUTPUT
        )
        st.caption(
            f"≈ **${coste:.6f}** en hora valle; en hora peak cuesta el doble. Tarifas de "
            f"entrada ${config.PRICE_INPUT}/M y salida ${config.PRICE_OUTPUT}/M, verificadas "
            "el 2026-10-07 y ajustables en el `.env`."
        )


def _panel_descripciones() -> None:
    """Genera y revisa el texto de imágenes, audio y video antes de indexarlos.

    El paso de revisión no es decorativo: un texto inventado se embebe de forma
    permanente y contamina todas las búsquedas futuras en silencio. El modelo propone,
    la persona aprueba — y hace falta, porque la transcripción oye «Datavik» donde dice
    «DATABiQ».

    En el video hay **una fila por fragmento**, que es la unidad que se embebe.
    """
    st.divider()
    st.subheader("Textos para revisar")
    st.caption(
        "Solo lo que produce un modelo. **El PDF no aparece**: su texto se extrae del "
        "documento, no lo genera la visión ni Whisper."
    )

    carpeta = config.DATA_DIR / "uploads"
    medios = (
        [
            p
            for p in sorted(carpeta.iterdir())
            if ingest.modality_of(p) in ("image", "audio", "video")
        ]
        if carpeta.exists()
        else []
    )
    guardadas = describe.load_descriptions()
    pendientes = [p for p in medios if describe.needs_text(p, guardadas)]

    if pendientes and st.button(f"Generar texto de {len(pendientes)} archivo(s)"):
        with st.spinner("Describiendo imágenes, transcribiendo audio y video…"):
            resultado = describe.generate_missing(medios)
        if resultado["errors"]:
            st.error("Fallos: " + "; ".join(resultado["errors"]))
        st.success(f"{resultado['generated']} textos generados. Revísalos abajo.")
        st.rerun()

    if not guardadas:
        st.caption("Sin textos: los archivos se encontrarán, pero no se podrán describir.")
        return

    st.caption("Revísalos y corrígelos antes de indexar — se embeben de forma permanente.")
    # La clave de un fragmento es `archivo#v2`; se muestra legible y se traduce al guardar.
    visibles = {clave: describe.label(clave) for clave in guardadas}
    invertido = {texto: clave for clave, texto in visibles.items()}
    filas = st.data_editor(
        [{"archivo": visibles[clave], "descripcion": texto} for clave, texto in guardadas.items()],
        hide_index=True,
        key="editor_descripciones",
    )

    if st.button("Guardar y reindexar", type="primary"):
        editadas = {
            invertido.get(fila["archivo"], fila["archivo"]): fila["descripcion"] for fila in filas
        }
        describe.save_descriptions(editadas)
        if not medios:
            st.warning("No hay imágenes, audios ni videos en `data/uploads/` que reindexar.")
            return
        with st.spinner(f"Reindexando {len(medios)} archivo(s)…"):
            resumen = ingest.ingest_paths(medios, labels=editadas)
        # Decir cuántos se saltaron: si no, corregir una palabra y ver «0 actualizados»
        # parecería que el botón no hizo nada.
        st.success(
            f"{resumen['ingested']} vectores actualizados · "
            f"{resumen.get('intact', 0)} ya estaban al día."
        )
        if resumen["errors"]:
            st.error("Errores: " + "; ".join(resumen["errors"]))


def _render_hit(position: int, hit: dict) -> None:
    metadata = hit["metadata"]
    modality = metadata.get("modality", "?")
    name = metadata.get("name", "?")
    path = Path(metadata.get("path", ""))

    st.markdown(f"**{position}. {name}**")
    st.caption(f"`{modality}` · similitud `{hit['similarity']:.4f}`")
    if metadata.get("page"):
        st.caption(f"página {metadata['page']}")
    if modality == "video":
        st.caption(f"fragmento {metadata.get('start')}–{metadata.get('end')} s")

    if modality == "image":
        if path.exists():
            st.image(str(path))
        else:
            st.warning("La imagen ya no está en disco.")

    elif modality == "pdf":
        # `pdf_page_images` lee de la caché: no vuelve a renderizar el PDF.
        pages = ingest.pdf_page_images(path)
        index = int(metadata.get("page", 1)) - 1
        if 0 <= index < len(pages):
            st.image(str(pages[index]))
        else:
            st.warning("No se pudo recuperar esa página.")

    elif modality == "audio":
        if path.exists():
            st.audio(str(path))
        else:
            st.warning("El audio ya no está en disco.")

    elif modality == "video":
        if path.exists():
            # Abre el reproductor en el fragmento que coincidió, no en el minuto 0.
            st.video(str(path), start_time=int(metadata.get("start") or 0))
        else:
            st.warning("El video ya no está en disco.")

    elif modality == "text":
        st.markdown(hit["document"])

    else:
        st.write(hit["document"])


# ---------------------------------------------------------------- barra lateral
# Sin conteos del índice: viven en la pestaña Métricas, con su detalle por archivo.
# Aquí queda lo que es de la aplicación —qué modelo corre y en qué dispositivo— y lo que
# se acciona: añadir archivos, revisar sus textos y vaciar el índice.
with st.sidebar:
    st.caption(f"**Modelo** · {config.MODEL_ID}")
    st.caption(f"**Dimensión** · {config.EMBED_DIM}d (matryoshka)")
    st.caption(f"**Dispositivo** · {embedder.describe()}")

    st.divider()
    st.subheader("Añadir archivos")
    uploads = st.file_uploader(
        "Imágenes, audio, PDF, texto o video",
        accept_multiple_files=True,
        type=UPLOAD_TYPES,
        label_visibility="collapsed",
    )
    if st.button("Indexar", type="primary") and uploads:
        destination = config.DATA_DIR / "uploads"
        destination.mkdir(parents=True, exist_ok=True)
        copiados = []
        for upload in uploads:
            target = destination / upload.name
            target.write_bytes(upload.getbuffer())
            copiados.append(target)
        with st.spinner(f"Indexando {len(copiados)} archivo(s)…"):
            # Las descripciones ya revisadas entran como `labels`: es lo que vuelve una
            # imagen RESPONDIBLE (forma interleaved) en vez de solo ENCONTRABLE.
            summary = ingest.ingest_paths(copiados, labels=describe.load_descriptions())
        st.success(f"{summary['ingested']} vectores añadidos.")
        if summary["skipped"]:
            st.warning("Omitidos: " + ", ".join(summary["skipped"]))
        if summary["errors"]:
            st.error("Errores: " + "; ".join(summary["errors"]))

    st.divider()
    if config.generation_partial():
        st.warning(
            "Hay una `DEEPSEEK_API_KEY` en el entorno (heredada de otro proyecto) "
            "pero faltan `AI_MODEL` y `AI_MODEL_BASE_URL`. Defínelas en `.env` "
            "para activar la generación."
        )
    elif not config.generation_enabled():
        st.caption("Sin capa de generación: modo solo-recuperación.")
    if st.button("Borrar índice"):
        st.session_state["confirm_reset"] = True
    if st.session_state.get("confirm_reset"):
        st.warning("Esto borra todos los vectores.")
        if st.button("Confirmar borrado", type="primary"):
            with st.spinner("Borrando…"):
                store.reset()
            st.session_state["confirm_reset"] = False
            st.rerun()
        if st.button("Cancelar"):
            st.session_state["confirm_reset"] = False
            st.rerun()


# ------------------------------------------------------------------- principal
st.title("Motor DATABiQ")
st.caption(
    "EmbeddingGemma 2 proyecta las cuatro modalidades en el mismo espacio de 768 "
    "dimensiones; el modelo de lenguaje lee los fragmentos recuperados."
)

_warm_up()

# Tres trabajos distintos, tres pestañas: **responder** (necesita el LLM y paga por los
# tokens), **recuperar** (el ranking, que es la parte multimodal y no cuesta nada) y
# **medir** (lo que se puede contar del índice y de la sesión, sin llamar al modelo).
tab_preguntar, tab_recuperar, tab_metricas = st.tabs(["Preguntar", "Recuperar", "Métricas"])

with tab_preguntar:
    st.markdown(
        "**Pregunta en lenguaje natural sobre tus archivos.** Recupera de texto, imágenes, "
        "audio, PDF y video a la vez, y responde citando lo que encontró."
    )
    enviado, pregunta, modalidad, _ = _formulario("preguntar", "Preguntar")

    if enviado and pregunta.strip():
        if store.total() == 0:
            st.warning("El índice está vacío. Añade archivos en la barra lateral.")
        else:
            with st.spinner("Recuperando fragmentos en todas las modalidades…"):
                encontrados = _busca_y_cronometra(pregunta, modalidad, config.TOP_K)
            respuesta = None
            if config.generation_enabled() and encontrados:
                with st.spinner("Leyendo los fragmentos y redactando la respuesta…"):
                    uso: dict = {}
                    respuesta = rag.answer(pregunta, encontrados, uso=uso)
                # Se acumula en la sesión para que la pestaña de métricas pueda poner
                # precio a lo consumido: el gasto por pregunta no se ve en ningún otro sitio.
                gasto = st.session_state.setdefault(
                    "gasto", {"prompt": 0, "completion": 0, "preguntas": 0}
                )
                gasto["prompt"] += uso.get("prompt", 0)
                gasto["completion"] += uso.get("completion", 0)
                gasto["preguntas"] += 1
            # Claves propias de esta pestaña: compartirlas con la otra haría que un
            # resultado apareciera en las dos.
            st.session_state["ask_pregunta"] = pregunta
            st.session_state["ask_hits"] = encontrados
            st.session_state["ask_respuesta"] = respuesta

    if st.session_state.get("ask_pregunta"):
        st.divider()
        st.markdown(f"### {st.session_state['ask_pregunta']}")

        respuesta = st.session_state.get("ask_respuesta")
        hits = st.session_state.get("ask_hits") or []

        if respuesta:
            st.markdown(respuesta)
            st.caption(
                "Generado a partir de las fuentes de abajo. Los números entre corchetes "
                "las referencian."
            )
        elif not config.generation_enabled():
            st.info(
                "Sin capa de generación configurada: abajo están los fragmentos recuperados, "
                "pero nadie los redacta. Para eso está la pestaña **Recuperar**, o rellena "
                "`DEEPSEEK_API_KEY` en `.env`."
            )
        else:
            st.info("No se recuperó ningún fragmento para esa pregunta.")

        st.markdown(f"#### Fuentes · {len(hits)} fragmento(s)")
        for posicion, hit in enumerate(hits, start=1):
            with st.container(border=True):
                _render_hit(posicion, hit)

with tab_recuperar:
    st.markdown(
        "**Mira el ranking, sin capa de generación.** La lista ordenada por similitud, con "
        "la página del PDF o el tramo del video donde coincidió."
    )
    enviado, pregunta, modalidad, top_k = _formulario("recuperar", "Recuperar", con_top_k=True)

    if enviado and pregunta.strip():
        if store.total() == 0:
            st.warning("El índice está vacío. Añade archivos en la barra lateral.")
        else:
            with st.spinner(f"Buscando los {top_k} fragmentos más parecidos…"):
                encontrados = _busca_y_cronometra(pregunta, modalidad, top_k)
            st.session_state["find_pregunta"] = pregunta
            st.session_state["find_hits"] = encontrados
            st.session_state["find_top_k"] = top_k

    if st.session_state.get("find_pregunta"):
        hits = st.session_state.get("find_hits") or []
        st.divider()
        st.markdown(f"### {st.session_state['find_pregunta']}")

        if not hits:
            st.info("No se recuperó ningún fragmento para esa pregunta.")
        else:
            st.caption(
                f"Top-{st.session_state.get('find_top_k')} · {len(hits)} fragmento(s) · "
                "la similitud **no** es comparable entre modalidades: cada una ocupa su banda."
            )
            st.dataframe(_filas_ranking(hits), hide_index=True)

            elegido = st.selectbox(
                "Ver el fragmento",
                range(len(hits)),
                format_func=lambda posicion: (
                    f"{posicion + 1}. {hits[posicion]['metadata']['name']}"
                ),
                key="find_elegido",
            )
            with st.container(border=True):
                _render_hit(int(elegido) + 1, hits[int(elegido)])

with tab_metricas:
    st.markdown(
        "**Revisar los textos y medir el índice.** Primero lo que hay que aprobar antes de "
        "indexarlo; después las cifras del corpus y de la sesión. Nada de aquí llama al "
        "modelo de lenguaje: el texto lo producen la visión local y Whisper cuando lo pides."
    )
    if config.vision_enabled() or config.transcribe_enabled():
        _panel_descripciones()
    _panel_metricas()
