"""Convierte media en texto indexable: describe imágenes, transcribe audio y video.

Por qué local: mantiene la promesa del proyecto —los datos no salen del equipo— y no
tiene costo por archivo. Y por qué un modelo pequeño basta: no pedimos prosa, pedimos
una o dos frases factuales para que la imagen sea **encontrable por lo que dice**.

**Para el audio NO se usa Ollama**, aunque el modelo declare torre de audio: su API solo
acepta `audio` en `/api/embed`, así que `/api/generate` lo descarta en silencio y el
modelo acaba pidiendo el audio. Se transcribe con `faster-whisper`, local también.

**Para el video, el texto sale de su audio, no de sus fotogramas.** Medido el 2026-10-08
con el video promocional del usuario (75 s): una pasada de Whisper cuesta 3,5 s y
devuelve la narración entera, mientras que describir tres fotogramas por fragmento con
el modelo de visión costaba ~4 s cada uno y aportaba generalidades («una oficina con
pantallas»). Los segmentos de Whisper traen marcas de tiempo, así que se reparten entre
los fragmentos por su punto medio. Un video **mudo** se queda sin texto propio.

Dos cosas aprendidas midiendo, no supuestas:

  - **`think=False` no es opcional.** Los modelos con razonamiento activo tardan ~100 s
    por imagen contra ~5 s con él apagado (medido con `gemma4:e2b`: 101 s → 5,2 s).
  - **La imagen se normaliza antes de enviarla.** `ingest.image_to_rgb()` la deja en RGB
    sobre fondo blanco; un PNG con fondo transparente aplanado a negro hacía que el
    modelo describiera un cuadrado negro, con razón.

Las descripciones se guardan en `data/descripciones.json` para **revisarlas antes de
indexarlas**: una descripción inventada se embebe *para siempre* y contamina todas las
búsquedas futuras, así que el modelo propone y la persona aprueba. En el video hay una
entrada por fragmento, con la clave `archivo#v1`, `archivo#v2`…
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config, ingest, video

PROMPT = (
    "Describe esta imagen en UNA o DOS frases factuales, en español. Di qué es y, si "
    "contiene texto, transcríbelo. No inventes nada que no veas: si algo no lo "
    "distingues, no lo menciones."
)


def describe_image(src: str | Path) -> str:
    """Una o dos frases factuales sobre la imagen. Cadena vacía si no se pudo."""
    if not config.vision_enabled():
        return ""

    prepared = ingest.image_to_rgb(Path(src))
    payload = {
        "model": config.VISION_MODEL,
        "prompt": PROMPT,
        "images": [base64.b64encode(prepared.read_bytes()).decode()],
        "stream": False,
        "think": False,  # medido: ~20x más rápido, y para esto sobra
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{config.VISION_BASE_URL.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return str(json.loads(response.read()).get("response", "")).strip()


_whisper_model = None


def _whisper():
    """Carga el modelo de Whisper una sola vez: hacerlo por archivo sería carísimo."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            # int8 en CPU (rápido y de sobra para esto); float16 solo con CUDA probada.
            compute_type="int8" if config.WHISPER_DEVICE == "cpu" else "float16",
        )
    return _whisper_model


def _raw_segments(samples: np.ndarray) -> list:
    """Segmentos crudos de Whisper, con sus marcas de tiempo.

    El idioma va **fijo** (`WHISPER_LANGUAGE`) y no se autodetecta: con audios cortos la
    detección falla — es la lección que costó el enfoque bilingüe del proyecto ORION.
    `beam_size` se deja en su default (5): allí se bajó a 3 por latencia de tiempo real,
    pero aquí la transcripción corre una sola vez por archivo.
    """
    if not config.transcribe_enabled():
        return []

    segments, _ = _whisper().transcribe(
        samples,
        language=config.WHISPER_LANGUAGE,
        vad_filter=True,  # descarta silencios: un pitch tiene pausas
    )
    return list(segments)


def transcribe_audio(src: str | Path) -> str:
    """Transcribe un audio a texto con Whisper local. Cadena vacía si no se pudo."""
    if not config.transcribe_enabled():
        return ""

    # Se le pasa el ARRAY, no la ruta. Si recibe un archivo, faster-whisper lo decodifica
    # con PyAV y llama a `av.open(..., metadata_errors=...)`, parámetro que NO existe en
    # el PyAV 19.0.1 que trae transformers[video] → TypeError. Con el array ya en 16 kHz
    # mono no toca PyAV, así que no hay que actualizar nada ni arriesgar esa integración.
    prepared = ingest.audio_to_16k_mono(Path(src))  # 16 kHz mono, que es lo que espera
    samples, _ = sf.read(str(prepared), dtype="float32")
    return " ".join(segmento.text.strip() for segmento in _raw_segments(samples)).strip()


def transcribe_segments(samples: np.ndarray) -> list[dict[str, Any]]:
    """Como `transcribe_audio`, pero conservando inicio y fin de cada segmento.

    Los necesita el video: se transcribe el archivo **una sola vez** y luego el texto se
    reparte entre fragmentos por su marca de tiempo. Medido el 2026-10-08 con 75 s de
    audio, la pasada completa cuesta 3,5 s.
    """
    return [
        {
            "start": float(segmento.start),
            "end": float(segmento.end),
            "text": segmento.text.strip(),
        }
        for segmento in _raw_segments(samples)
    ]


def describe_video(path: str | Path) -> dict[str, str]:
    """Texto de cada fragmento de un video, sacado de su audio.

    Devuelve `{'archivo.mp4#v1': 'lo que se dice en el tramo', ...}`. Un video mudo
    devuelve un mapa vacío: no tiene texto propio y se quedará en encontrable.
    """
    path = Path(path)
    muestras = video.audio_16k_mono(path)
    if not len(muestras):
        return {}

    segmentos = transcribe_segments(muestras)
    tramos = video.chunks(video.probe(path)["duration"], config.VIDEO_CHUNK_SECONDS)
    # Por punto medio: un segmento que cruza la frontera cae entero en un fragmento,
    # en vez de perderse por no caber en ninguno de los dos.
    reparto: dict[int, list[str]] = {numero: [] for numero in range(1, len(tramos) + 1)}
    for segmento in segmentos:
        if not segmento["text"]:
            continue
        centro = (segmento["start"] + segmento["end"]) / 2
        for numero, (inicio, fin) in enumerate(tramos, start=1):
            if inicio <= centro <= fin:
                reparto[numero].append(segmento["text"])
                break

    textos: dict[str, str] = {}
    for numero, partes in reparto.items():
        unido = " ".join(partes).strip()
        if unido:
            textos[f"{path.name}#v{numero}"] = unido
    return textos


def load_descriptions() -> dict[str, str]:
    """Descripciones guardadas, por nombre de archivo."""
    if not config.DESCRIPTIONS_FILE.exists():
        return {}

    try:
        data = json.loads(config.DESCRIPTIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    return {str(name): str(text) for name, text in data.items() if str(text).strip()}


def save_descriptions(descriptions: dict[str, str]) -> None:
    config.DESCRIPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.DESCRIPTIONS_FILE.write_text(
        json.dumps(descriptions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def video_keys(path: str | Path) -> list[str]:
    """Claves de los fragmentos de un video: `archivo.mp4#v1`, `#v2`…

    Se apoya en el mismo fragmentado que la ingesta, así que las claves y los vectores
    siempre coinciden. Devuelve una lista vacía si el contenedor no se puede leer.
    """
    try:
        tramos = video.chunks(video.probe(path)["duration"], config.VIDEO_CHUNK_SECONDS)
    except Exception:  # un archivo ilegible se reportará al generar, no al listarlo
        return []
    return [f"{Path(path).name}#v{numero}" for numero in range(1, len(tramos) + 1)]


def needs_text(path: str | Path, descriptions: dict[str, str]) -> bool:
    """¿Le falta texto a este archivo?

    El video se mira **por fragmento**: basta con que uno no tenga el suyo para que
    cuente como pendiente. Tener una entrada con el nombre del archivo no lo da por
    hecho, porque lo que se embebe es el texto de cada fragmento.
    """
    path = Path(path)
    modality = ingest.modality_of(path)
    if modality in ("image", "audio"):
        return path.name not in descriptions
    if modality == "video":
        claves = video_keys(path)
        return not claves or any(clave not in descriptions for clave in claves)
    return False


def label(key: str) -> str:
    """Nombre legible de una clave, para el panel: `video.mp4#v2` → `video.mp4 · fragmento 2`."""
    nombre, separador, fragmento = key.partition("#v")
    if separador and fragmento:
        return f"{nombre} · fragmento {fragmento}"
    return key


def generate_missing(
    paths: Iterable[str | Path],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Genera el texto que falte: de imágenes, audios y fragmentos de video.

    No pisa lo que ya está guardado: si revisaste y corrigiste una descripción, se
    respeta. Devuelve el mapa completo y los fallos, en el mismo formato que
    `ingest.ingest_paths`.
    """
    descriptions = load_descriptions()
    pending = [Path(path) for path in paths if needs_text(Path(path), descriptions)]

    errors: list[str] = []
    generated = 0
    for position, path in enumerate(pending, start=1):
        if progress:
            progress(position, len(pending))
        try:
            modality = ingest.modality_of(path)
            if modality == "image":
                nuevos = {path.name: describe_image(path)}
            elif modality == "audio":
                nuevos = {path.name: transcribe_audio(path)}
            else:  # video: una entrada por fragmento, no una por archivo
                nuevos = describe_video(path)
        except Exception as exc:  # un fallo no debe abortar el resto
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        for clave, texto in nuevos.items():
            if texto:
                descriptions[clave] = texto
                generated += 1

    save_descriptions(descriptions)
    return {"generated": generated, "errors": errors, "descriptions": descriptions}
