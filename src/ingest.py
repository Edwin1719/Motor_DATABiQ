"""Ingesta: archivos del disco -> vectores en Chroma.

Cada modalidad se prepara antes de embeder:
  - audio  -> WAV mono 16 kHz en caché (es lo que el modelo espera)
  - pdf    -> imagen PNG por página para RECUPERAR + texto de la página para RESPONDER
  - texto  -> fragmentos con solape, codificados con el prefijo `Document`
  - imagen -> PNG RGB de un solo fotograma (el encoder no acepta GIF animado ni paleta)
  - video  -> fragmentos de VIDEO_CHUNK_SECONDS, con los fotogramas ya muestreados a
              1 fps por PyAV: el modelo no decodifica, solo mira fotogramas

Representación dual: el vector sale siempre de la modalidad (una página de PDF se
embede como imagen, una foto como foto), pero el campo `document` guarda el TEXTO
que verá el LLM. Sin ese texto el RAG recupera bien y responde mal, porque el
modelo de lenguaje solo recibiría un nombre de archivo.

Para las modalidades sin texto extraíble (imagen, audio) se acepta un mapa
`labels` {nombre_de_archivo: descripción} que se usa como `document`. Es la vía
para captions de un dataset o etiquetas de clase; sin él, el LLM solo puede citar
que el archivo coincidió, no hablar de su contenido.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config, embedder, store, video

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
PDF_EXT = {".pdf"}
TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".csv", ".html", ".css", ".sql", ".java",
}

TEXT_CHUNK = 1200
TEXT_OVERLAP = 200
PDF_DPI = 110
# Muestreo que exige el encoder de audio: mono a 16 kHz.
SAMPLE_RATE = 16000

# Clave interna: el texto que verá el LLM. No se guarda en el payload de Chroma.
DOCUMENT = "_document"


def modality_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in PDF_EXT:
        return "pdf"
    if ext in TEXT_EXT:
        return "text"
    return None


def _stable_id(path: Path, suffix: str = "") -> str:
    """Hash estable del origen: reingerir el mismo archivo actualiza, no duplica."""
    digest = hashlib.sha1(f"{path.resolve()}{suffix}".encode("utf-8")).hexdigest()
    return digest[:32]


def _version(path: Path) -> str:
    """Versión del archivo para nombrar su caché: tamaño + fecha de modificación.

    **No** es el hash de la ruta —ese era el fallo: al reemplazar un PDF, sus páginas PNG
    seguían siendo las viejas y el vector salía de la imagen anterior mientras el texto
    ya era el nuevo— y tampoco un hash del contenido, porque `_cache_path()` se llama
    **una vez por página** y hashear obligaría a releer el documento entero N veces.
    Cualquier edición real cambia el tamaño o la fecha.
    """
    stat = path.stat()
    return hashlib.sha1(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:8]


def _cache_prefix(path: Path) -> str:
    """Prefijo de las copias de caché de un archivo: hash de la ruta + versión."""
    return f"{_stable_id(path)[:16]}-{_version(path)}"


def _cache_path(path: Path, suffix: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{_cache_prefix(path)}{suffix}"


def _huella(version: str, texto: str) -> str:
    """Huella de lo que determina el vector: la versión del archivo y su texto.

    El modelo es determinista, y sin cambiar el archivo lo único que puede cambiar el
    vector es el texto. Así que una huella igual significa vector igual, y reindexar
    deja de costar 40 s cuando solo se corrigió una pieza.
    """
    return hashlib.sha1(f"{version}|{texto}".encode("utf-8")).hexdigest()[:16]


def _intacto(guardadas: Mapping[str, dict], item_id: str, huella: str) -> bool:
    """¿Ese id ya está en el índice con la misma huella?"""
    return guardadas.get(item_id, {}).get("fingerprint") == huella


def _prune_cache(path: Path) -> None:
    """Borra las copias de caché de esa ruta que no sean ya de la versión actual.

    Sin esto, cada edición de un archivo dejaría atrás sus PNG y WAV anteriores. Se
    limpian también las de los dos esquemas viejos —`{prefijo}.png` (imagen y audio) y
    `{prefijo}_p001.png` (páginas de PDF), ambos del hash de ruta sin versión—, que de
    otro modo quedarían invisibles para siempre.
    """
    prefijo = _stable_id(path)[:16]
    actual = _cache_prefix(path)
    sobrantes = [c for c in config.CACHE_DIR.glob(f"{prefijo}-*") if not c.name.startswith(actual)]
    sobrantes += [c for c in config.CACHE_DIR.glob(f"{prefijo}.*")]
    sobrantes += list(config.CACHE_DIR.glob(f"{prefijo}_*"))
    for viejo in sobrantes:
        viejo.unlink()


def audio_to_16k_mono(src: Path) -> Path:
    """El modelo pide mono a 16 kHz: se decodifica con soundfile y se remuestrea.

    **Por qué no `librosa.load`:** en este entorno (librosa 1.0.0 sobre Python 3.14)
    `librosa.load` revienta con `RuntimeError: cannot cache function '__o_fold': no
    locator available for file .../librosa/core/notation.py`. El import perezoso de
    `librosa.core.notation` no resuelve su localizador de fichero, así que el módulo no
    carga y **toda** ingesta de audio falla. Medido el 2026-10-09.

    `soundfile` + `scipy.signal.resample_poly` hacen lo mismo con dos dependencias que el
    entorno ya tiene (scipy llega con scikit-learn, que librosa ya exigía), y
    `resample_poly` aplica su filtro antialias por defecto. `librosa` queda solo como
    respaldo, por si algún formato que soundfile no lea sí lo lea él.
    """
    dst = _cache_path(src, ".wav")
    if dst.exists():
        return dst

    try:
        _resample_con_soundfile(src, dst)
    except Exception:
        _resample_con_librosa(src, dst)
    return dst


def _resample_con_soundfile(src: Path, dst: Path) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    # `always_2d=True` normaliza la forma: un WAV mono llega como (n, 1) y no como (n,).
    muestras, nativo = sf.read(str(src), dtype="float32", always_2d=True)
    mono = muestras.mean(axis=1)  # estéreo -> mono, como lo escucha el modelo

    if not len(mono) or float(abs(mono).max()) == 0.0:
        # Silencio digital: remuestrear solo añadiría artefactos de borde, y el modelo
        # necesita el archivo igual para que el clip exista y sea encontrable.
        sf.write(str(dst), mono, SAMPLE_RATE)
        return

    if nativo != SAMPLE_RATE:
        from math import gcd

        divisor = gcd(int(nativo), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, int(nativo) // divisor)
    sf.write(str(dst), np.asarray(mono, dtype="float32"), SAMPLE_RATE)


def _resample_con_librosa(src: Path, dst: Path) -> None:
    """Respaldo para formatos que soundfile no sepa leer. Import perezoso a propósito."""
    import librosa
    import soundfile as sf

    samples, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    sf.write(str(dst), samples, SAMPLE_RATE)


def image_to_rgb(src: Path) -> Path:
    """Normaliza la imagen a un PNG RGB de un solo fotograma, sobre fondo blanco.

    El encoder de visión espera UNA imagen `(alto, ancho, 3)` en RGB, y hay tres
    formatos cotidianos que no lo cumplen:

      - **GIF animado**: llega como pila de fotogramas `(N, alto, ancho, 3)` y el
        procesador revienta con "too many values to unpack (expected 3)".
      - **Paleta ('P') o escala de grises ('L')**: el array no tiene 3 ejes.
      - **PNG con alfa ('RGBA')**: el cuarto canal no es RGB, y hay que **componerlo
        sobre blanco** — ver el comentario del cuerpo.

    Se conserva el primer fotograma y se aplica la rotación EXIF, que es la que
    traen las fotos de móvil y que sin aplicarla dejaría la imagen girada.
    Se cachea en disco: reindexar no vuelve a convertir.
    """
    from PIL import Image, ImageOps

    dst = _cache_path(src, ".png")
    if not dst.exists():
        with Image.open(src) as image:
            # Abrir ya sitúa el primer fotograma: no hay que recorrer la animación.
            source = ImageOps.exif_transpose(image).convert("RGBA")

        # Se compone sobre BLANCO a propósito, no se descarta el alfa. `convert("RGB")`
        # a secas tira el canal alfa sin componer nada, y los píxeles transparentes
        # llevan negro por dentro: un logo con fondo transparente se convertía en un
        # cuadrado NEGRO con el logo encima, invisible para el encoder y para cualquier
        # modelo de visión que lo describiera después.
        canvas = Image.new("RGB", source.size, (255, 255, 255))
        canvas.paste(source, mask=source.getchannel("A"))
        canvas.save(str(dst), "PNG")
    return dst


def pdf_pages(src: Path) -> list[tuple[Path, str]]:
    """Cada página como (imagen PNG, texto extraído).

    La imagen es lo que se embede: el encoder de visión la codifica en un vector
    (no extrae texto, la convierte en números — por eso no hace falta OCR). El
    texto es lo que recibe el LLM para poder responder. Son dos representaciones
    del mismo objeto: el vector sirve para recuperar, el texto para responder.
    """
    import pymupdf

    pages: list[tuple[Path, str]] = []
    with pymupdf.open(str(src)) as document:
        for number, page in enumerate(document, start=1):
            out = _cache_path(src, f"_p{number:03d}.png")
            if not out.exists():
                page.get_pixmap(dpi=PDF_DPI).save(str(out))
            pages.append((out, page.get_text().strip()))
    return pages


def pdf_page_images(src: Path) -> list[Path]:
    """Solo las rutas de las imágenes de página (las usa la UI para previsualizar)."""
    return [image for image, _ in pdf_pages(src)]


def text_chunks(text: str, size: int = TEXT_CHUNK, overlap: int = TEXT_OVERLAP) -> list[str]:
    collapsed = " ".join(text.split())
    if len(collapsed) <= size:
        return [collapsed] if collapsed else []
    step = size - overlap
    return [
        collapsed[start : start + size]
        for start in range(0, len(collapsed), step)
        if collapsed[start : start + size].strip()
    ]


def _label_for(path: Path, labels: Mapping[str, str] | None) -> str | None:
    if not labels:
        return None
    return labels.get(path.name) or labels.get(str(path))


def _media(
    items: list[tuple[str, Any, dict]],
    meta: dict,
    clave: str,
    payload: Any,
    texto: str | None = None,
) -> None:
    """Añade un ítem de media: `clave` es `image`, `audio` o `video`.

    Si hay texto —una descripción o una transcripción— va **dentro del mismo vector**,
    en la forma interleaved de la documentación del modelo y con el marcador
    `<|clave|>` que su procesador expande (para el video, a sus tokens de fotogramas).

    No es un detalle de estilo: un segundo vector de texto duplicaría el archivo en el
    ranking. Y no repetirlo en cada rama es lo que evita que un cambio de esta forma se
    aplique en tres sitios y se olvide en uno.
    """
    if texto:
        meta[DOCUMENT] = texto
        items.append(("media", {"text": f"{texto} <|{clave}|>", clave: payload}, meta))
    else:
        items.append(("media", {clave: payload}, meta))


def _collect(
    path: Path,
    modality: str,
    items: list[tuple[str, Any, dict]],
    labels: Mapping[str, str] | None,
    guardadas: Mapping[str, dict] | None = None,
) -> None:
    """Añade a `items` tuplas (kind, payload, metadata): kind es 'text', 'media' o 'intacto'.

    `'intacto'` significa que ese id ya está en el índice con la misma huella: aporta su
    id —para que el reindexado no lo tome por huérfano— pero **no** un payload, así que
    ni se normaliza el archivo ni se vuelve a embeber. Con `guardadas` vacío no se salta
    nada, que es lo correcto para una ingesta nueva. Vale para las cuatro modalidades.
    """
    guardadas = guardadas or {}
    base = {
        "modality": modality,
        "name": path.name,
        "path": str(path.resolve()),
        "source": str(path.resolve().parent),
    }
    label = _label_for(path, labels)

    if modality == "text":
        version = _version(path)
        body = path.read_text(encoding="utf-8", errors="replace")
        for index, chunk in enumerate(text_chunks(body)):
            item_id = _stable_id(path, f"#{index}")
            huella = _huella(version, chunk)
            meta = {**base, "chunk": index, "id": item_id, "fingerprint": huella}
            if _intacto(guardadas, item_id, huella):
                items.append(("intacto", None, meta))
                continue
            items.append(("text", chunk, meta))

    elif modality == "pdf":
        version = _version(path)
        for number, (page_image, page_text) in enumerate(pdf_pages(path), start=1):
            item_id = _stable_id(path, f"#p{number}")
            huella = _huella(version, page_text)
            meta = {**base, "page": number, "id": item_id, "fingerprint": huella}
            # `pdf_pages()` ya trajo la página —y su PNG sale de la caché por versión, así
            # que es barato—, pero embeberla no lo es: ~0,25 s por página, que en un PDF
            # de 200 páginas serían ~50 s en cada reindexado sin cambios.
            if _intacto(guardadas, item_id, huella):
                items.append(("intacto", None, meta))
                continue
            # El texto de la página es lo que permite responder sobre el contenido.
            meta[DOCUMENT] = page_text or f"{path.name} · página {number} (sin texto extraíble)"
            _media(items, meta, "image", str(page_image))

    elif modality == "audio":
        huella = _huella(_version(path), label or "")
        meta = {**base, "id": _stable_id(path), "fingerprint": huella}
        if _intacto(guardadas, meta["id"], huella):
            items.append(("intacto", None, meta))
            return
        _media(items, meta, "audio", str(audio_to_16k_mono(path)), label)

    elif modality == "video":
        # El modelo no decodifica video: se le entregan los fotogramas ya
        # muestreados a 1 fps. Un video más largo que VIDEO_CHUNK_SECONDS se
        # fragmenta, porque el procesador solo admite VIDEO_MAX_FRAMES por vector
        # y cada fragmento es un vector con su tramo (inicio, fin).
        version = _version(path)
        duracion = video.probe(path)["duration"]
        for numero, (inicio, fin) in enumerate(
            video.chunks(duracion, config.VIDEO_CHUNK_SECONDS), start=1
        ):
            # El texto del FRAGMENTO manda sobre el del archivo entero: `video.mp4#v2`
            # es más específico que `video.mp4`, y es lo que lo vuelve respondible.
            propia = (labels or {}).get(f"{path.name}#v{numero}") or label
            huella = _huella(version, propia or "")
            meta = {
                **base,
                "chunk": numero,
                "start": round(inicio, 2),
                "end": round(fin, 2),
                "id": _stable_id(path, f"#v{numero}"),
                "fingerprint": huella,
            }
            # La huella se mira ANTES de decodificar: un fragmento intacto se ahorra sus
            # 2,79 s de decodificación y sus 3,41 s de embedding (medidos).
            if _intacto(guardadas, meta["id"], huella):
                items.append(("intacto", None, meta))
                continue
            fotogramas = video.frames(
                path, inicio, fin, fps=config.VIDEO_FPS, max_frames=config.VIDEO_MAX_FRAMES
            )
            if not len(fotogramas):
                continue
            _media(items, meta, "video", fotogramas, propia)

    else:  # imagen
        # La imagen se normaliza (primer fotograma, RGB sobre blanco): el encoder no
        # acepta GIF animado, paleta ni transparencia. El `path` del metadata conserva
        # el archivo original, que es el que muestra la interfaz.
        huella = _huella(_version(path), label or "")
        meta = {**base, "id": _stable_id(path), "fingerprint": huella}
        if _intacto(guardadas, meta["id"], huella):
            items.append(("intacto", None, meta))
            return
        _media(items, meta, "image", str(image_to_rgb(path)), label)


def _mmss(seconds: float | None) -> str:
    """Segundos -> `MM:SS`, para que la cita de un fragmento se lea de un vistazo."""
    if seconds is None:
        return "?"
    total = int(round(float(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _display_text(metadata: dict, payload: Any) -> str:
    """Texto que verá el LLM. Es el punto donde un RAG gana o pierde."""
    if metadata.get(DOCUMENT):
        return str(metadata[DOCUMENT])
    if metadata["modality"] == "text":
        return str(payload)
    if metadata["modality"] == "pdf":
        return f"{metadata['name']} · página {metadata['page']}"
    if metadata["modality"] == "video":
        # Sin una descripción propia, al menos que la cita diga QUÉ tramo del
        # video coincidió: "video.mp4 · fragmento 00:32–01:04".
        return f"{metadata['name']} · fragmento {_mmss(metadata.get('start'))}–{_mmss(metadata.get('end'))}"
    return metadata["name"]


def ingest_paths(
    paths: Iterable[str | Path],
    *,
    labels: Mapping[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Embede y guarda. Devuelve un resumen con lo ingestado, lo omitido y los errores."""
    items: list[tuple[str, Any, dict]] = []
    skipped: list[str] = []
    errors: list[str] = []
    files = [Path(p) for p in paths]

    for position, path in enumerate(files, start=1):
        if progress:
            progress(position, len(files))
        modality = modality_of(path)
        if modality is None:
            skipped.append(f"{path.name} (extensión no soportada)")
            continue
        try:
            # Lo que ya está con la misma huella no se vuelve a preparar: así reindexar
            # deja de costar 40 s cuando solo se corrigió una pieza.
            _collect(path, modality, items, labels, guardadas=store.metadatos_por_ruta(path))
            # Ya está escrita la copia de la versión actual: las anteriores sobran.
            _prune_cache(path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    # Los ítems intactos no se re-embeben, pero cuentan como producidos: si no, la poda
    # los tomaría por huérfanos y borraría vectores que siguen siendo válidos.
    intactos = [meta["id"] for kind, _, meta in items if kind == "intacto"]

    if not items:
        return {
            "ingested": 0,
            "intact": 0,
            "skipped": skipped,
            "errors": errors,
            "by_modality": {},
        }

    # Texto y media se codifican en dos llamadas (el texto lleva prefijo, la media no),
    # pero se reensamblan en el orden original para no perder la correspondencia.
    vectors: list[np.ndarray | None] = [None] * len(items)
    text_slots = [i for i, (kind, _, _) in enumerate(items) if kind == "text"]
    media_slots = [i for i, (kind, _, _) in enumerate(items) if kind == "media"]

    if text_slots:
        encoded = embedder.encode_text(
            [items[i][1] for i in text_slots], prompt_name="Document", progress=True
        )
        for slot, index in enumerate(text_slots):
            vectors[index] = encoded[slot]

    if media_slots:
        try:
            encoded = embedder.encode_media([items[i][1] for i in media_slots], progress=True)
            for slot, index in enumerate(media_slots):
                vectors[index] = encoded[slot]
        except Exception:
            # Un archivo que el encoder no sepa leer no debe tumbar la ingesta
            # completa: se reintenta uno por uno y se descartan los que fallen,
            # reportándolos en `errors`.
            for index in media_slots:
                try:
                    vectors[index] = embedder.encode_media([items[index][1]])[0]
                except Exception as exc:
                    errors.append(f"{items[index][2]['name']}: {type(exc).__name__}: {exc}")

    # Solo sobreviven los ítems que se lograron codificar. ids, metadatos y
    # documentos se alinean con esa misma lista, o se desincronizarían.
    keep = [i for i, vector in enumerate(vectors) if vector is not None]
    if not keep:
        return {
            "ingested": 0,
            "intact": len(intactos),
            "skipped": skipped,
            "errors": errors,
            "by_modality": {},
        }

    matrix = np.vstack([vectors[i] for i in keep]).astype("float32")
    ids = [items[i][2]["id"] for i in keep]
    # `id` y la clave interna DOCUMENT no se guardan en el payload de Chroma.
    metadatas = [
        {k: v for k, v in items[i][2].items() if k != "id" and not k.startswith("_")}
        for i in keep
    ]
    documents = [_display_text(items[i][2], items[i][1]) for i in keep]

    store.add_items(ids=ids, embeddings=matrix, metadatas=metadatas, documents=documents)

    # Reingerir no debe dejar vectores de una versión anterior del archivo: si el PDF
    # perdió páginas o el video se acortó, esos ids ya no se producen y seguirían
    # apareciendo en las búsquedas. Solo se podan los archivos que SÍ se ingirieron
    # (uno que falló conserva lo que tenía) y solo lo que ese archivo ya no produce.
    producidos: dict[str, set[str]] = {}
    for index, (kind, _, meta) in enumerate(items):
        # Los intactos entran aquí: se saltaron justamente porque siguen válidos.
        if kind == "intacto" or vectors[index] is not None:
            producidos.setdefault(meta["path"], set()).add(meta["id"])
    for ruta, nuevos in producidos.items():
        sobrantes = set(store.ids_for_path(ruta)) - nuevos
        if sobrantes:
            store.delete_ids(sorted(sobrantes))

    by_modality: dict[str, int] = {}
    for meta in metadatas:
        by_modality[meta["modality"]] = by_modality.get(meta["modality"], 0) + 1

    return {
        "ingested": len(keep),
        "intact": len(intactos),
        "skipped": skipped,
        "errors": errors,
        "by_modality": by_modality,
    }
