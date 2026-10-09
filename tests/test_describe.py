"""Descripciones de imagen con un modelo de visión local (Ollama).

Dos cosas se fijan aquí y no son detalles:

  - **`think=False` va siempre en el payload.** Medido con `gemma4:e2b`: 101 s con
    razonamiento contra 5,2 s sin él. Sin este interruptor la ingesta de 10 imágenes
    pasa de un minuto a diecisiete.
  - **Lo ya revisado no se pisa.** Si una persona corrigió una descripción, volver a
    generar no debe sobrescribirla.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import config, describe


# ------------------------------------------------------------------ activación


def test_sin_configuracion_no_describe(monkeypatch):
    monkeypatch.setattr(config, "VISION_MODEL", "")
    monkeypatch.setattr(config, "VISION_BASE_URL", "")
    assert config.vision_enabled() is False
    assert describe.describe_image("lo-que-sea.png") == ""


@pytest.mark.parametrize(
    ("model", "url", "expected"),
    [
        ("gemma4:e2b", "http://localhost:11434", True),
        ("", "http://localhost:11434", False),
        ("gemma4:e2b", "", False),
        ("", "", False),
    ],
)
def test_vision_exige_modelo_y_url(monkeypatch, model, url, expected):
    monkeypatch.setattr(config, "VISION_MODEL", model)
    monkeypatch.setattr(config, "VISION_BASE_URL", url)
    assert config.vision_enabled() is expected


# ------------------------------------------------------------- persistencia


def test_load_vacio_sin_archivo(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", tmp_path / "no-existe.json")
    assert describe.load_descriptions() == {}


def test_load_ignora_descripciones_en_blanco(monkeypatch, tmp_path):
    archivo = tmp_path / "d.json"
    archivo.write_text(json.dumps({"a.png": "   ", "b.png": "un logo"}), encoding="utf-8")
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", archivo)
    assert describe.load_descriptions() == {"b.png": "un logo"}


def test_load_tolera_un_json_roto(monkeypatch, tmp_path):
    archivo = tmp_path / "d.json"
    archivo.write_text("{ esto no es json", encoding="utf-8")
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", archivo)
    assert describe.load_descriptions() == {}


def test_save_escribe_en_utf8_y_crea_la_carpeta(monkeypatch, tmp_path):
    archivo = tmp_path / "sub" / "d.json"
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", archivo)

    describe.save_descriptions({"logo.png": "logo azul de DATABiQ"})

    assert json.loads(archivo.read_text(encoding="utf-8")) == {"logo.png": "logo azul de DATABiQ"}


# ----------------------------------------------------------------- payload


class _Respuesta:
    def __init__(self, cuerpo: dict):
        self._cuerpo = json.dumps(cuerpo).encode()

    def read(self) -> bytes:
        return self._cuerpo

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_el_payload_apaga_el_razonamiento(monkeypatch, tmp_path):
    """`think=False` es la diferencia entre 5 s y 100 s por imagen: debe ir siempre."""
    from PIL import Image

    monkeypatch.setattr(config, "VISION_MODEL", "gemma4:e2b")
    monkeypatch.setattr(config, "VISION_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    foto = tmp_path / "foto.png"
    Image.new("RGB", (8, 8)).save(foto)

    capturado: dict = {}

    def urlopen_falso(request, timeout=None):
        capturado["payload"] = json.loads(request.data)
        capturado["url"] = request.full_url
        return _Respuesta({"response": "  un logo azul  "})

    monkeypatch.setattr(describe.urllib.request, "urlopen", urlopen_falso)

    texto = describe.describe_image(foto)

    assert capturado["payload"]["think"] is False
    assert capturado["payload"]["model"] == "gemma4:e2b"
    assert capturado["payload"]["stream"] is False
    assert capturado["payload"]["images"]  # la imagen va en base64
    assert capturado["url"].endswith("/api/generate")
    assert texto == "un logo azul"  # se recortan los espacios


# --------------------------------------------------------------- generación


def test_generate_no_pisa_lo_ya_revisado(monkeypatch, tmp_path):
    archivo = tmp_path / "d.json"
    archivo.write_text(json.dumps({"logo.png": "ya revisada"}), encoding="utf-8")
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", archivo)

    llamadas: list[str] = []
    monkeypatch.setattr(
        describe, "describe_image", lambda ruta: llamadas.append(Path(ruta).name) or "nueva"
    )

    logo = tmp_path / "logo.png"
    icono = tmp_path / "icono.png"
    for ruta in (logo, icono):
        ruta.write_bytes(b"x")

    resultado = describe.generate_missing([logo, icono])

    assert llamadas == ["icono.png"]  # solo la que no tenía descripción
    assert resultado["generated"] == 1
    assert resultado["descriptions"]["logo.png"] == "ya revisada"
    assert resultado["descriptions"]["icono.png"] == "nueva"
    assert json.loads(archivo.read_text(encoding="utf-8"))["icono.png"] == "nueva"


def test_generate_reporta_fallos_sin_abortar(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", tmp_path / "d.json")

    def que_falla(ruta):
        if Path(ruta).name == "malo.png":
            raise OSError("sin conexion con ollama")
        return "buena descripcion"

    monkeypatch.setattr(describe, "describe_image", que_falla)

    buena = tmp_path / "buena.png"
    mala = tmp_path / "malo.png"
    for ruta in (buena, mala):
        ruta.write_bytes(b"x")

    resultado = describe.generate_missing([mala, buena])

    assert resultado["generated"] == 1
    assert resultado["errors"] and "malo.png" in resultado["errors"][0]
    assert "buena.png" in resultado["descriptions"]


# --------------------------------------------------------------- transcripción


class _Segmento:
    def __init__(self, texto: str):
        self.text = texto


class _WhisperFalso:
    def __init__(self, textos: list[str]):
        self._textos = textos
        self.llamadas: list[tuple] = []

    def transcribe(self, ruta, **kwargs):
        self.llamadas.append((ruta, kwargs))
        return ([_Segmento(texto) for texto in self._textos], None)


def test_sin_modelo_de_whisper_no_transcribe(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_MODEL", "")
    assert config.transcribe_enabled() is False
    assert describe.transcribe_audio("lo-que-sea.mp3") == ""


def test_transcribe_une_los_segmentos_y_fija_el_idioma(monkeypatch, tmp_path):
    """El idioma va FIJO: con audios cortos la autodetección falla — lección de ORION.

    Y se le pasa el ARRAY, no la ruta: decodificar el archivo con PyAV revienta en el
    PyAV que trae transformers[video] (no acepta `metadata_errors`).
    """
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "es")

    falso = _WhisperFalso([" Hola, ", "soy Edwin. "])
    monkeypatch.setattr(describe, "_whisper", lambda: falso)
    monkeypatch.setattr(describe.ingest, "audio_to_16k_mono", lambda ruta: Path(ruta))
    monkeypatch.setattr(describe.sf, "read", lambda *args, **kwargs: (np.zeros(16000), 16000))

    audio = tmp_path / "pitch.mp3"
    audio.write_bytes(b"x")

    assert describe.transcribe_audio(audio) == "Hola, soy Edwin."

    muestras, opciones = falso.llamadas[0]
    assert hasattr(muestras, "shape")  # es un array, no una ruta
    assert opciones["language"] == "es"
    assert opciones["vad_filter"] is True  # un pitch tiene pausas


def test_generate_tambien_transcribe_audio_y_omite_texto(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", tmp_path / "d.json")
    monkeypatch.setattr(describe, "describe_image", lambda ruta: "descripcion")
    monkeypatch.setattr(describe, "transcribe_audio", lambda ruta: "transcripcion")

    foto = tmp_path / "foto.png"
    audio = tmp_path / "pitch.mp3"
    notas = tmp_path / "notas.md"
    for ruta in (foto, audio, notas):
        ruta.write_bytes(b"x")

    resultado = describe.generate_missing([foto, audio, notas])

    assert resultado["generated"] == 2
    assert resultado["descriptions"]["foto.png"] == "descripcion"
    assert resultado["descriptions"]["pitch.mp3"] == "transcripcion"
    assert "notas.md" not in resultado["descriptions"]  # el texto ya trae su contenido


# ------------------------------------------------------------------- video


class _SegmentoConTiempos:
    def __init__(self, texto: str, inicio: float, fin: float):
        self.text = texto
        self.start = inicio
        self.end = fin


class _WhisperConTiempos:
    def __init__(self, segmentos: list):
        self._segmentos = segmentos

    def transcribe(self, muestras, **kwargs):
        return (self._segmentos, None)


def _prepara_video(monkeypatch, duracion: float = 70.0) -> None:
    """Deja un video de mentira: 70 s, con audio, y tramos de 32 s."""
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "es")
    monkeypatch.setattr(config, "VIDEO_CHUNK_SECONDS", 32)
    monkeypatch.setattr(describe.video, "probe", lambda ruta: {"duration": duracion})
    monkeypatch.setattr(describe.video, "audio_16k_mono", lambda ruta: np.zeros(16000))


def test_describe_video_reparte_los_segmentos_por_su_punto_medio(monkeypatch, tmp_path):
    """El texto de un video sale de su audio, no de sus fotogramas: una sola pasada de
    Whisper y cada segmento cae en el fragmento donde está su punto medio, para que
    ninguno se pierda por cruzar la frontera."""
    _prepara_video(monkeypatch)
    monkeypatch.setattr(
        describe,
        "_whisper",
        lambda: _WhisperConTiempos(
            [
                _SegmentoConTiempos(" Presentación ", 0.0, 4.0),
                _SegmentoConTiempos("Servicios", 33.0, 40.0),
                _SegmentoConTiempos("Contacto", 66.0, 68.0),
            ]
        ),
    )

    textos = describe.describe_video(tmp_path / "clip.mp4")

    assert textos == {
        "clip.mp4#v1": "Presentación",
        "clip.mp4#v2": "Servicios",
        "clip.mp4#v3": "Contacto",
    }


def test_describe_video_sin_audio_no_inventa_texto(monkeypatch, tmp_path):
    """Un video mudo no tiene nada que transcribir: se quedará en encontrable."""
    _prepara_video(monkeypatch)
    monkeypatch.setattr(describe.video, "audio_16k_mono", lambda ruta: np.zeros(0, dtype=np.float32))

    assert describe.describe_video(tmp_path / "mudo.mp4") == {}


def test_needs_text_mira_el_video_por_fragmentos(monkeypatch, tmp_path):
    """Tener una entrada con el nombre del archivo no basta: lo que se embebe es el
    texto de cada fragmento."""
    _prepara_video(monkeypatch)
    clip = tmp_path / "clip.mp4"

    assert describe.needs_text(clip, {}) is True
    assert describe.needs_text(clip, {"clip.mp4": "texto del archivo entero"}) is True
    assert describe.needs_text(clip, {"clip.mp4#v1": "a", "clip.mp4#v2": "b"}) is True
    completo = {"clip.mp4#v1": "a", "clip.mp4#v2": "b", "clip.mp4#v3": "c"}
    assert describe.needs_text(clip, completo) is False


def test_label_hace_legible_la_clave_de_un_fragmento():
    assert describe.label("video.mp4#v2") == "video.mp4 · fragmento 2"
    assert describe.label("logo.png") == "logo.png"


def test_generate_tambien_transcribe_video(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DESCRIPTIONS_FILE", tmp_path / "d.json")
    _prepara_video(monkeypatch)
    monkeypatch.setattr(describe, "describe_video", lambda ruta: {"clip.mp4#v1": "lo que dice"})

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    resultado = describe.generate_missing([clip])

    assert resultado["generated"] == 1
    assert resultado["descriptions"]["clip.mp4#v1"] == "lo que dice"
