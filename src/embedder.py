"""Envoltura de EmbeddingGemma 2: carga perezosa, una sola vez, y encode multimodal.

Reglas que impone el modelo (model card) y que aquí se respetan:
  - El TEXTO lleva prefijo de tarea (`prompt_name`); imagen/audio/video NO llevan ninguno.
  - Precisión bfloat16 o float32. NUNCA float16: las activaciones exceden su rango
    dinámico y devuelve NaN o embeddings degradados en silencio, sin lanzar error.
  - El pipeline del modelo ya termina en `Normalize`, así que la salida es unitaria.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype() -> torch.dtype:
    """bfloat16 solo donde hay soporte nativo; float32 en el resto (la mayoría de CPUs)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class _AudioSinLibrosa:
    """Decodifica audio sin pasar por `librosa`.

    **Por qué existe.** `transformers.audio_utils.load_audio()` resuelve el audio
    llamando a `librosa.load()` (línea 279 en transformers 5.19.0). En este entorno
    —librosa 1.0.0 sobre Python 3.14— `librosa.load` revienta al importar
    `librosa.core.notation`, que no encuentra el localizador de su propio fichero:

        RuntimeError: cannot cache function '__o_fold': no locator available for
        file .../librosa/core/notation.py

    El fallo no está en el modelo ni en este proyecto: está en la decodificación previa.
    Como la llamada es a nivel de módulo (`load_audio` llama a `librosa.load` por
    nombre), sustituir ese atributo lo arregla sin tocar `site-packages` ni el modelo.
    Medido el 2026-10-09: sin el parche **toda** ingesta de audio fallaba; con él, audio
    suelto, lote mixto imagen+audio y forma interleaved devuelven vectores de norma 1.0.

    `soundfile` + `scipy.signal.resample_poly` hacen el mismo trabajo con dependencias
    que el entorno ya tiene, y `resample_poly` aplica su filtro antialias.
    """

    @staticmethod
    def load_audio(source, sampling_rate: int = 16000, backend=None):
        import io
        from math import gcd

        import numpy as np
        import soundfile as sf

        # `always_2d=True` normaliza la forma: un fichero mono llega como (n, 1).
        fuente = io.BytesIO(source) if isinstance(source, bytes) else source
        with sf.SoundFile(fuente) as fichero:
            nativo = int(fichero.samplerate)
            canales = int(fichero.channels)
            crudo = fichero.read(dtype="float32", always_2d=True)

        mono = crudo.mean(axis=1) if canales > 1 else crudo[:, 0]
        if nativo != sampling_rate:
            from scipy.signal import resample_poly

            divisor = gcd(nativo, int(sampling_rate))
            mono = resample_poly(mono, int(sampling_rate) // divisor, nativo // divisor)
        return np.asarray(mono, dtype=np.float32)


def _audio_sin_librosa() -> None:
    """Instala el decodificador de audio alternativo, una sola vez.

    **Hay que parchear DOS sitios**, y ese fue el error de la primera versión de este
    arreglo: `transformers.processing_utils` importa `load_audio` **por nombre**
    (`from .audio_utils import load_audio`), así que guarda su propia referencia. Cambiar
    solo `audio_utils.load_audio` deja intacta la copia que usa
    `apply_chat_template()`, que es justo la que se ejecuta al embeber. El síntoma es
    traicionero: una llamada directa a `audio_utils.load_audio` funciona y el pipeline
    sigue fallando.
    """
    from transformers import audio_utils

    audio_utils.load_audio = _AudioSinLibrosa.load_audio

    # Segunda puerta: la referencia local.
    try:
        from transformers import processing_utils

        processing_utils.load_audio = _AudioSinLibrosa.load_audio
    except ImportError:  # pragma: no cover - cambia de nombre entre versiones
        pass


def get_model() -> SentenceTransformer:
    """Singleton: el modelo pesa 744M parámetros y se carga una única vez."""
    global _model
    with _lock:
        if _model is None:
            _audio_sin_librosa()
            _model = SentenceTransformer(
                config.MODEL_ID,
                device=pick_device(),
                model_kwargs={"torch_dtype": pick_dtype()},
            )
        return _model


def describe() -> str:
    """Etiqueta legible para mostrar en la UI."""
    model = get_model()
    dtype = next(model.parameters()).dtype
    where = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return f"{where} · {str(dtype).replace('torch.', '')} · {config.EMBED_DIM}d"


def _encode(inputs, *, prompt_name: str | None, batch_size: int, progress: bool) -> np.ndarray:
    return np.asarray(
        get_model().encode(
            inputs,
            prompt_name=prompt_name,
            batch_size=batch_size,
            normalize_embeddings=True,  # obligatorio al truncar con matryoshka
            truncate_dim=config.EMBED_DIM,
            show_progress_bar=progress,
            convert_to_numpy=True,
        ),
        dtype="float32",
    )


def encode_text(
    texts: str | Sequence[str],
    *,
    prompt_name: str = "SearchQuery",
    batch_size: int = 32,
    progress: bool = False,
) -> np.ndarray:
    """Para hacer retrieval: `SearchQuery` en consultas, `Document` en los pasajes."""
    return _encode(texts, prompt_name=prompt_name, batch_size=batch_size, progress=progress)


def encode_media(
    items: Sequence[dict],
    *,
    batch_size: int = 8,
    progress: bool = False,
) -> np.ndarray:
    """Imagen/audio/video como dicts: `{"image": ruta}`, `{"audio": ruta}`, `{"video": ruta}`.

    Sin `prompt_name` a propósito: los prefijos de tarea aplican solo a texto.
    """
    return _encode(list(items), prompt_name=None, batch_size=batch_size, progress=progress)
