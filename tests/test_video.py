"""Video: fragmentado, muestreo de fotogramas y el texto que verá el LLM.

El candado importante: el modelo **no decodifica video** —mira fotogramas al encoder
de visión a 1 fps—, así que la decodificación es nuestra y cualquier fallo aquí deja
el fragmento sin fotogramas. Y el segundo candado, el mismo que en imagen y audio:
sin un `document` propio al LLM solo le llega el nombre del archivo, y un video se
encontraría pero no se podría responder.

Los clips de prueba los codifica PyAV en el momento: son mp4 reales y diminutos, no
bytes falsos, y no hacen falta ni red ni modelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import config, ingest, video


def _encode(path: Path, ancho: int, alto: int, fotogramas: int = 10, fps: int = 5) -> Path:
    """Codifica un mp4 real y diminuto con el mismo PyAV que después lo lee."""
    import av

    with av.open(str(path), "w") as contenedor:
        pista = contenedor.add_stream("mpeg4", rate=fps)
        pista.width, pista.height, pista.pix_fmt = ancho, alto, "yuv420p"
        for indice in range(fotogramas):
            pixeles = np.zeros((alto, ancho, 3), dtype=np.uint8)
            pixeles[:, (indice * ancho // fotogramas) : ((indice + 1) * ancho // fotogramas)] = 255
            for paquete in pista.encode(av.VideoFrame.from_ndarray(pixeles, format="rgb24")):
                contenedor.mux(paquete)
        for paquete in pista.encode():
            contenedor.mux(paquete)
    return path


@pytest.fixture
def clip(tmp_path):
    """Un mp4 real de 2 s (10 fotogramas a 5 fps) en 64x48."""
    return _encode(tmp_path / "clip.mp4", 64, 48)


# ------------------------------------------------------------------ reescalado


def test_target_size_reduce_conservando_la_proporcion():
    assert video._target_size(1080, 1920, 640) == (640, 360)


def test_target_size_no_agranda_lo_que_ya_es_pequeno():
    assert video._target_size(48, 64, 640) == (64, 48)


def test_target_size_devuelve_dimensiones_pares():
    """Impar rompería el pix_fmt y la codificación."""
    ancho, alto = video._target_size(1080, 1000, 640)
    assert ancho % 2 == 0 and alto % 2 == 0


def test_frames_reduce_los_fotogramas_grandes(tmp_path):
    """El procesador reescala de todas formas a ~161 000 px: guardar 1920x1080 solo
    consume RAM (190 MB por fragmento de 32 s frente a 21 MB)."""
    grande = _encode(tmp_path / "grande.mp4", 800, 600)

    fotogramas = video.frames(grande, 0, 2, fps=1)

    assert fotogramas.shape[1:] == (480, 640, 3)  # 800x600 -> 640x480


# ------------------------------------------------------------------ fragmentado


def test_chunks_parte_la_duracion_en_tramos():
    assert video.chunks(70, 32) == [(0, 32), (32, 64), (64, 70)]


def test_chunks_de_un_video_mas_corto_que_el_tramo():
    assert video.chunks(5, 32) == [(0, 5)]


def test_chunks_une_el_resto_diminuto_al_tramo_anterior():
    """Un resto de medio segundo generaría un vector con un solo fotograma: no aporta
    y ensucia el ranking. Se une al tramo previo."""
    assert video.chunks(32.5, 32) == [(0, 32.5)]


def test_chunks_tolera_duraciones_imposibles():
    assert video.chunks(0, 32) == []
    assert video.chunks(10, 0) == []


# ------------------------------------------------------------------- lectura


def test_probe_lee_duracion_fps_y_tamano(clip):
    datos = video.probe(clip)
    assert datos["fps"] == pytest.approx(5.0)
    assert datos["duration"] == pytest.approx(2.0, abs=0.05)
    assert (datos["width"], datos["height"]) == (64, 48)
    assert datos["frames"] == 10


def test_frames_muestrea_a_un_fotograma_por_segundo(clip):
    """Es lo que pide el modelo: 1 fps. De 2 s a 5 fps salen 2 fotogramas, no 10."""
    fotogramas = video.frames(clip, 0, 2, fps=1)
    assert fotogramas.shape == (2, 48, 64, 3)
    assert fotogramas.dtype == np.uint8


def test_frames_respeta_el_presupuesto_de_fotogramas(clip):
    """El procesador admite como mucho VIDEO_MAX_FRAMES por vector."""
    assert video.frames(clip, 0, 2, fps=5, max_frames=3).shape[0] == 3


def test_frames_sin_fotogramas_devuelve_vacio(clip):
    """Un tramo más allá del final no debe reventar: quien llama omite el fragmento."""
    assert video.frames(clip, 99, 100, fps=1).shape[0] == 0


def test_frames_del_segundo_tramo(clip):
    assert video.frames(clip, 1.0, 2.0, fps=1).shape[0] == 1


# ---------------------------------------------------- representación dual


def test_mmss_formatea_los_segundos():
    assert ingest._mmss(0) == "00:00"
    assert ingest._mmss(95) == "01:35"
    assert ingest._mmss(None) == "?"


def test_display_text_de_video_cita_el_fragmento():
    """Sin descripción propia, la cita dice QUÉ tramo coincidió, no solo el archivo."""
    metadata = {"modality": "video", "name": "demo.mp4", "start": 32.0, "end": 64.0}
    assert ingest._display_text(metadata, {}) == "demo.mp4 · fragmento 00:32–01:04"


def test_collect_de_video_crea_un_vector_por_fragmento(tmp_path, monkeypatch, clip):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)

    items: list = []
    ingest._collect(clip, "video", items, None)

    assert len(items) == 2  # 2 s partidos en tramos de 1 s
    kind, payload, metadata = items[0]
    assert kind == "media"
    assert metadata["modality"] == "video"
    assert (metadata["start"], metadata["end"]) == (0.0, 1.0)
    assert metadata["chunk"] == 1
    assert metadata["path"] == str(clip.resolve())
    assert payload["video"].shape == (1, 48, 64, 3)


def test_collect_de_video_da_un_id_por_fragmento(tmp_path, monkeypatch, clip):
    """Sin sufijo por fragmento, el segundo pisaría al primero al guardar (mismo id)."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)

    items: list = []
    ingest._collect(clip, "video", items, None)

    ids = [metadata["id"] for _kind, _payload, metadata in items]
    assert len(set(ids)) == 2
    assert ids[0] == ingest._stable_id(clip, "#v1")


def test_collect_de_video_usa_la_etiqueta_como_documento(tmp_path, monkeypatch, clip):
    """Es el punto de extensión de la fase 2: si hay descripción, gana al tramo."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)

    items: list = []
    ingest._collect(clip, "video", items, {"clip.mp4": "demostración del producto"})
    _kind, payload, metadata = items[0]

    assert ingest._display_text(metadata, payload) == "demostración del producto"


def test_collect_de_video_prefiere_la_etiqueta_del_fragmento(tmp_path, monkeypatch, clip):
    """`clip.mp4#v2` es más específico que `clip.mp4`: la transcripción del tramo debe
    ganarle a la del archivo entero, y los demás tramos caer al respaldo."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)

    items: list = []
    ingest._collect(
        clip,
        "video",
        items,
        {"clip.mp4": "el archivo entero", "clip.mp4#v2": "lo que se dice en el segundo tramo"},
    )

    primero, segundo = items
    assert ingest._display_text(primero[2], primero[1]) == "el archivo entero"
    assert ingest._display_text(segundo[2], segundo[1]) == "lo que se dice en el segundo tramo"


def test_collect_de_video_con_texto_usa_la_forma_interleaved(tmp_path, monkeypatch, clip):
    """Con transcripción, los fotogramas y el texto van en **un solo** vector: así el
    fragmento se encuentra también por lo que se DICE en él, no solo por lo que se ve.

    Sin esto, medido el 2026-10-08 con el video real, una consulta sobre lo que cuenta
    el video lo dejaba en 0,6521 —por debajo de imágenes que no tienen nada que ver—
    aunque su transcripción lo dijera todo.
    """
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)

    items: list = []
    ingest._collect(clip, "video", items, {"clip.mp4#v1": "aquí se habla de consultoría"})
    _kind, payload, metadata = items[0]

    assert payload["text"] == "aquí se habla de consultoría <|video|>"
    assert payload["video"].shape == (1, 48, 64, 3)  # los fotogramas siguen ahí
    assert ingest._display_text(metadata, payload) == "aquí se habla de consultoría"


def test_un_fragmento_intacto_no_se_decodifica(tmp_path, monkeypatch, clip, fresh_store):
    """Corregir la transcripción de UN fragmento no debe redecodificar los otros.

    Es el caso caro y real: el video entero cuesta 29,37 s medidos, y cada fragmento
    2,79 s de decodificación más 3,41 s de embedding. Con la huella, corregir una
    palabra del guion cuesta un fragmento, no tres.
    """
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 1.0)
    decodificados: list = []
    monkeypatch.setattr(
        ingest.video,
        "frames",
        lambda *args, **kwargs: (
            decodificados.append(args) or np.zeros((1, 4, 4, 3), dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(
        ingest.embedder,
        "encode_media",
        lambda items, **kwargs: np.ones((len(items), config.EMBED_DIM), dtype="float32"),
    )

    primero = ingest.ingest_paths([clip], labels={"clip.mp4#v1": "uno", "clip.mp4#v2": "dos"})
    assert (primero["ingested"], primero["intact"]) == (2, 0)
    assert len(decodificados) == 2

    segundo = ingest.ingest_paths(
        [clip], labels={"clip.mp4#v1": "uno", "clip.mp4#v2": "DOS corregido"}
    )

    assert (segundo["ingested"], segundo["intact"]) == (1, 1)
    assert len(decodificados) == 3  # el fragmento intacto no se volvió a decodificar
    assert fresh_store.total() == 2


# ----------------------------------------------------------------- pista de audio


def test_audio_16k_mono_de_un_video_sin_pista_devuelve_vacio(clip):
    """Un video mudo no puede dar texto propio: no debe reventar, solo devolver nada."""
    assert video.audio_16k_mono(clip).shape == (0,)


def test_audio_16k_mono_mezcla_canales_y_remuestrea(tmp_path):
    """PyAV decodifica (soundfile no lee MP4) y scipy remuestrea: 48 kHz estéreo entra,
    16 kHz mono float32 sale — que es lo que espera Whisper.

    Este WAV es **PCM entero**, o sea formato *packed*: PyAV lo entrega como una sola
    fila con los canales entrelazados, así que el test cubre el caso difícil.
    """
    import soundfile as sf

    estereo = tmp_path / "estereo.wav"
    un_segundo = np.zeros((48000, 2), dtype="float32")
    un_segundo[:, 0] = 0.5  # solo el canal izquierdo
    sf.write(str(estereo), un_segundo, 48000)

    muestras = video.audio_16k_mono(estereo)

    assert muestras.dtype == np.float32
    assert 15000 < len(muestras) < 17000  # ~1 s a 16 kHz
    assert np.isclose(float(muestras.max()), 0.25, atol=0.02)  # media de 0,5 y 0


def test_el_remuestreo_del_video_no_usa_librosa():
    """La segunda ruta crítica de librosa: librosa 1.0.0 no importa en Python 3.14.

    Este test existe porque el fallo **no** lo cazaba el test de arriba: al ejecutar bajo
    un sandbox donde los permisos de los temporales fallan antes, el error real quedaba
    oculto. Se comprueba sobre el AST, no sobre el texto: el docstring de la función cita
    `librosa.resample` a propósito, para explicar por qué ya no se usa.
    """
    import ast

    fuente = ast.parse(Path(video.__file__).read_text(encoding="utf-8"))
    funcion = next(
        nodo
        for nodo in fuente.body
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "audio_16k_mono"
    )
    modulos = {
        alias.name.split(".")[0]
        for nodo in ast.walk(funcion)
        if isinstance(nodo, ast.Import)
        for alias in nodo.names
    } | {
        (nodo.module or "").split(".")[0]
        for nodo in ast.walk(funcion)
        if isinstance(nodo, ast.ImportFrom)
    }

    assert "librosa" not in modulos, "el remuestreo del video no debe importar librosa"
    assert "scipy" in modulos, "debe remuestrear con scipy.signal.resample_poly"


def test_to_mono_float32_ordena_los_canales_entrelazados():
    """Los formatos *packed* (PCM entero) llegan como UNA fila con los canales
    entrelazados: sin reordenarlos, Whisper recibiría el doble de muestras mezcladas.

    Dos muestras de estéreo (izquierdo 10000, derecho 0) deben dar **dos** muestras mono
    de 10000 — no cuatro muestras mezcladas, que es lo que saldría sin reordenar.
    """
    entrelazado = np.array([[10000, 0, 10000, 0]], dtype=np.int16)  # L, R, L, R

    mono = video._to_mono_float32(entrelazado, 2)

    assert mono.dtype == np.float32
    assert mono.shape == (2,)
    assert np.allclose(mono, np.array([10000, 0], dtype=np.float32) / 32767, atol=1e-4)


def test_to_mono_float32_acepta_planar_en_float():
    """Los *planares* (el AAC del MP4) ya vienen como (canales, muestras) en float32."""
    planar = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)

    assert np.allclose(video._to_mono_float32(planar, 2), 0.25)


def test_to_mono_float32_normaliza_enteros():
    """±32767 son amplitudes de PCM entero: sin normalizar, Whisper recibe ruido."""
    planar = np.array([[32767, -32767]], dtype=np.int16)

    mono = video._to_mono_float32(planar, 1)

    assert mono.dtype == np.float32
    assert np.isclose(mono[0], 1.0, atol=1e-3)


# ---------------------------------------------- rutas que NO cargan el modelo


def test_ingest_paths_reporta_un_video_ilegible_sin_abortar(tmp_path):
    """Un mp4 corrupto se reporta y el resto de la ingesta sigue."""
    roto = tmp_path / "roto.mp4"
    roto.write_bytes(b"esto no es un video")

    resultado = ingest.ingest_paths([roto])

    assert resultado["ingested"] == 0
    assert resultado["errors"] and "roto.mp4" in resultado["errors"][0]
