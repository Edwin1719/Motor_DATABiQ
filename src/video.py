"""Video -> fotogramas RGB, con PyAV.

El modelo **no tiene encoder de video**: la ficha oficial dice que el video *"is
processed as sampled frames through the vision encoder at 1 frame per second"*.
Esto no alimenta una torre de video, prepara fotogramas para el encoder de visión
— y por eso no hace falta ningún decodificador nuevo.

Se decodifica con **PyAV**, que trae FFmpeg embebido y ya llega con
`faster-whisper`. La vía nativa del procesador de transformers no sirve aquí: su
`fetch_videos()` fuerza `backend="torchcodec"` y, sin torchcodec, cae a
torchvision — que ya no sabe decodificar video (`read_video` se eliminó en
`torchvision` 0.26.0). Entregarle los fotogramas **ya decodificados** evita todo
eso: `load_video()` sale temprano cuando recibe un array o imágenes.

Limitación conocida: **no se aplica la matriz de rotación** del contenedor, igual
que el lector `pyav` de transformers. Un video grabado en vertical con un móvil
puede entrar girado; hay que comprobarlo con material real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Un resto más corto que esto se une al fragmento anterior en vez de generar un
# vector propio con un fotograma suelto: no aporta y ensucia el ranking.
MIN_CHUNK_SECONDS = 1.0

# Lado mayor al que se reduce cada fotograma antes de guardarlo. El procesador del
# modelo reescala de todas formas a ~161 000 píxeles por fotograma (630 parches de
# 16x16), o sea unos 536x300 en 16:9: reducir a 640 de lado mayor queda POR ENCIMA
# de ese objetivo, así que no se pierde nada y la RAM baja de ~190 MB a ~21 MB por
# fragmento de 32 s. Medido el 2026-10-08 con un 1080p.
MAX_SIDE = 640


def _target_size(alto: int, ancho: int, max_side: int) -> tuple[int, int]:
    """Tamaño reducido conservando la proporción, con dimensiones pares.

    Pares a propósito: el codificador de video y varios `pix_fmt` las exigen.
    """
    mayor = max(alto, ancho)
    if mayor <= max_side:
        return ancho, alto
    escala = max_side / mayor
    return max(2, int(ancho * escala) // 2 * 2), max(2, int(alto * escala) // 2 * 2)


def probe(path: str | Path) -> dict:
    """Duración, fps y tamaño reales del primer stream de video."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = stream.time_base
        if stream.duration is not None and base is not None:
            duracion = float(stream.duration * base)
        elif container.duration is not None:
            duracion = container.duration / av.time_base
        else:
            duracion = 0.0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        # `frames` viene a 0 en contenedores que no lo declaran: se estima.
        fotogramas = int(stream.frames) or int(round(duracion * fps))
        return {
            "duration": duracion,
            "fps": fps,
            "frames": fotogramas,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def _to_mono_float32(crudo: np.ndarray, canales: int) -> np.ndarray:
    """De lo que entrega PyAV a mono float32, que es lo que espera Whisper.

    PyAV no devuelve siempre la misma forma: en los formatos **planares** (el AAC del
    MP4, en float32) da `(canales, muestras)`, pero en los **packed** —el PCM entero de
    algunos contenedores— da **una sola fila con los canales entrelazados**. Sin
    reordenar ni normalizar, un video con audio PCM entraría a Whisper como ruido
    (dos veces más largo, con muestras de otro canal mezcladas y amplitudes de ±32768).
    """
    if crudo.ndim == 1:
        crudo = crudo.reshape(1, -1)
    if crudo.shape[0] == 1 and canales > 1:
        crudo = crudo.reshape(canales, -1)
    if crudo.dtype.kind in "iu":
        crudo = crudo.astype(np.float32) / float(np.iinfo(crudo.dtype).max)
    else:
        crudo = crudo.astype(np.float32, copy=False)
    return crudo.mean(axis=0)  # estéreo -> mono, que es como lo escucha el modelo


def audio_16k_mono(path: str | Path) -> np.ndarray:
    """Pista de audio del video en 16 kHz mono float32, que es lo que espera Whisper.

    Aquí **PyAV decodifica y scipy solo remuestrea**, al revés que con un archivo de audio
    suelto: `soundfile` no puede con un MP4 —libsndfile no reconoce el formato—, así que en
    el video la decodificación tiene que ser explícita.

    El remuestreo **no** usa `librosa.resample`: librosa 1.0.0 no importa en Python 3.14
    (ver `ingest.audio_to_16k_mono`), y esta es la segunda ruta crítica donde aparecía.
    `resample_poly` hace el mismo trabajo y su filtro antialias viene por defecto.

    Devuelve un array vacío si el video no trae pista de audio: un video mudo no tiene
    texto propio y se quedará en encontrable, no en respondible.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        stream = container.streams.audio[0]
        nativo = int(stream.rate)
        canales = int(getattr(stream.layout, "nb_channels", 1) or 1)
        crudo = np.concatenate([trozo.to_ndarray() for trozo in container.decode(stream)], axis=1)

    mono = _to_mono_float32(crudo, canales)
    if nativo == 16000 or not len(mono):
        return mono

    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(nativo, 16000)
    return np.asarray(resample_poly(mono, 16000 // divisor, nativo // divisor), dtype=np.float32)


def chunks(
    duracion: float, segundos: float, minimo: float = MIN_CHUNK_SECONDS
) -> list[tuple[float, float]]:
    """Parte la duración en tramos `(inicio, fin)` de `segundos` como máximo.

    El último tramo, si queda por debajo de `minimo`, se une al anterior: un
    video de 32,5 s con tramos de 32 s produce dos fragmentos, no uno de medio
    segundo con un solo fotograma.
    """
    if duracion <= 0 or segundos <= 0:
        return []
    tramos: list[tuple[float, float]] = []
    inicio = 0.0
    while inicio < duracion:
        fin = min(inicio + segundos, duracion)
        tramos.append((inicio, fin))
        inicio = fin
    if len(tramos) > 1 and tramos[-1][1] - tramos[-1][0] < minimo:
        anterior = tramos[-2]
        tramos[-2:] = [(anterior[0], tramos[-1][1])]
    return tramos


def frames(
    path: str | Path,
    start: float = 0.0,
    end: float | None = None,
    *,
    fps: float = 1.0,
    max_frames: int = 32,
) -> np.ndarray:
    """Fotogramas RGB `(n, alto, ancho, 3)` muestreados a `fps` dentro de `[start, end)`.

    Devuelve un array vacío (n = 0) si el tramo no tiene fotogramas decodificables,
    para que quien llama decida omitir el fragmento en vez de embeder basura.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = stream.time_base
        ventana = 1.0 / fps if fps > 0 else 0.0

        # Decodificación en varios hilos: medido el 2026-10-08 en un 1080p, este tramo
        # pasó de 11,96 s a 2,08 s (5,8x). Sin esto, decodificar domina la ingesta.
        stream.thread_type = "AUTO"

        if start > 0:
            # El offset va en unidades del `time_base` del stream. Con backward=True
            # el decodificador queda en el keyframe anterior; los fotogramas que
            # caen antes de `start` se descartan al comparar sus tiempos.
            container.seek(int(start / base), stream=stream)

        objetivo = start
        capturados: list[np.ndarray] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            momento = float(frame.pts * base)
            if end is not None and momento >= end:
                break
            if momento + 1e-6 >= objetivo:
                ancho, alto = _target_size(frame.height, frame.width, MAX_SIDE)
                if (ancho, alto) != (frame.width, frame.height):
                    frame = frame.reformat(width=ancho, height=alto)
                capturados.append(frame.to_ndarray(format="rgb24"))
                objetivo += ventana
                if len(capturados) >= max_frames:
                    break

    if not capturados:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(capturados)
