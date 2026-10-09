"""Protege el banco de validación y el arreglo de decodificación de audio.

Dos cosas que se rompen en silencio y que estos tests cazan antes que un usuario:

  1. **El parche de audio.** Si `transformers` cambia dónde guarda su referencia a
     `load_audio`, el parche deja de aplicarse y **toda** ingesta de audio vuelve a
     fallar. Es exactamente el fallo que se cometió al escribir el arreglo la primera
     vez: parchear `audio_utils` no basta, porque `processing_utils` importó el nombre.
  2. **El lector del banco.** Un `queries.yaml` mal formado daría métricas inventadas
     en vez de un error: el banco mediría cero y parecería que el sistema falla.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _cargar(nombre: str):
    """Carga un módulo de `scripts/` por ruta: esa carpeta no es un paquete.

    No se usa el fixture `conftest._load_script` porque ese está escrito para `seed_demo`
    y este módulo necesita el suyo en tiempo de importación.
    """
    ruta = ROOT / "scripts" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _cargar("bench")


# --------------------------------------------------------- lector del banco


def test_el_banco_tiene_positivas_y_negativas(bench):
    banco = bench.leer_consultas()
    assert banco["positivas"], "el banco no puede quedarse sin consultas con respuesta"
    assert banco["negativas"], "sin negativas no se mide la honestidad del sistema"


def test_toda_positiva_declara_su_resultado_esperado(bench):
    """Una positiva sin `esperado` o sin `modalidad` no se puede evaluar."""
    for caso in bench.leer_consultas()["positivas"]:
        assert caso.get("id"), f"consulta sin id: {caso}"
        assert caso.get("query"), f"consulta sin texto: {caso}"
        assert caso.get("esperado"), f"positiva sin esperado: {caso.get('id')}"
        assert caso.get("modalidad"), f"positiva sin modalidad: {caso.get('id')}"


def test_los_ids_del_banco_son_unicos(bench):
    """Dos casos con el mismo id harían ilegible el informe."""
    banco = bench.leer_consultas()
    ids = [c["id"] for c in banco["positivas"] + banco["negativas"]]
    assert len(ids) == len(set(ids))


def test_el_esperado_es_un_nombre_de_archivo_no_una_ruta(bench):
    """El banco compara contra el campo `name` del metadata, que es el nombre suelto.

    Con una ruta relativa (`audio/tono.wav`) la comparación fallaría siempre y el informe
    diría "no aparece" aunque el acierto fuera el puesto 1.
    """
    for caso in bench.leer_consultas()["positivas"]:
        assert "/" not in caso["esperado"], f"{caso['id']}: usa solo el nombre del archivo"
        assert "\\" not in caso["esperado"], f"{caso['id']}: usa solo el nombre del archivo"


def test_cada_esperado_existe_en_el_corpus(bench):
    """Si el corpus cambia y las consultas no, el banco mide otra cosa."""
    nombres = {p.name for p in (ROOT / "bench" / "corpus").rglob("*") if p.is_file()}
    if not nombres:
        pytest.skip("corpus sin generar: python scripts/make_bench_corpus.py")
    for caso in bench.leer_consultas()["positivas"]:
        assert caso["esperado"] in nombres, f"{caso['id']}: {caso['esperado']} no está en el corpus"


def test_las_negativas_no_declaran_esperado(bench):
    """Una negativa con esperado no sería una negativa, sería una positiva mal puesta."""
    for caso in bench.leer_consultas()["negativas"]:
        assert "esperado" not in caso, f"{caso['id']} declara esperado en las negativas"


# ------------------------------------------------------------------- modalidad


def test_la_modalidad_de_cada_carpeta_es_la_declarada(bench):
    """La extensión no decide la modalidad del corpus: la carpeta sí."""
    for carpeta, modalidad in {
        "audio": "audio",
        "imagen": "image",
        "documento": "text",
        "texto": "text",
    }.items():
        archivo = next(iter((ROOT / "bench" / "corpus" / carpeta).glob("*")), None)
        if archivo is None:
            pytest.skip(f"corpus sin generar (falta {carpeta}/)")
        assert bench.modalidad_de(archivo) == modalidad, f"{carpeta}/ -> {modalidad}"


# ------------------------------------------------- decodificador de audio


def test_el_parche_de_audio_esta_en_las_dos_puertas():
    """`audio_utils` **y** `processing_utils` deben apuntar al decodificador alternativo.

    Comprobarlo solo en `audio_utils` fue el fallo original: parecía arreglado y el
    pipeline seguía llamando a librosa por la copia local de `processing_utils`.
    """
    from src import embedder

    embedder._audio_sin_librosa()

    from transformers import audio_utils, processing_utils

    assert audio_utils.load_audio.__qualname__ == "_AudioSinLibrosa.load_audio"
    assert processing_utils.load_audio.__qualname__ == "_AudioSinLibrosa.load_audio"


def test_el_decodificador_alternativo_no_importa_librosa():
    """El arreglo existe justo porque librosa no carga en este entorno.

    Se analiza el **AST del método**, no el texto del fichero: el docstring de la clase
    cita `librosa.load()` a propósito, para explicar por qué existe el parche, y una
    comprobación sobre texto plano marcaría esa explicación como fallo.
    """
    from src import embedder

    arbol = ast.parse(Path(embedder.__file__).read_text(encoding="utf-8"))
    clase = next(
        nodo
        for nodo in arbol.body
        if isinstance(nodo, ast.ClassDef) and nodo.name == "_AudioSinLibrosa"
    )
    metodo = next(
        nodo
        for nodo in clase.body
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "load_audio"
    )

    modulos_importados = {
        alias.name.split(".")[0]
        for nodo in ast.walk(metodo)
        if isinstance(nodo, ast.Import)
        for alias in nodo.names
    } | {
        (nodo.module or "").split(".")[0]
        for nodo in ast.walk(metodo)
        if isinstance(nodo, ast.ImportFrom)
    }

    assert "librosa" not in modulos_importados, "el decodificador no debe importar librosa"
    assert "soundfile" in modulos_importados, "debe decodificar con soundfile"


def test_el_decodificador_alternativo_devuelve_16k_mono(tmp_path):
    """Comprueba el contrato: array float32 mono a la frecuencia pedida."""
    import soundfile as sf

    from src import embedder

    # 1 s de 440 Hz a 44,1 kHz, estéreo: obliga a mezclar canales y a remuestrear.
    t = np.linspace(0, 1, 44100, endpoint=False)
    estereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    origen = tmp_path / "440_44k_estereo.wav"
    sf.write(str(origen), estereo, 44100)

    muestras = embedder._AudioSinLibrosa.load_audio(str(origen), sampling_rate=16000)
    assert muestras.dtype == np.float32
    assert muestras.ndim == 1
    # 1 s a 16 kHz son 16 000 muestras; se tolera el redondeo del remuestreo.
    assert abs(len(muestras) - 16000) <= 8


def test_el_decodificador_alternativo_acepta_bytes(tmp_path):
    """`apply_chat_template` puede pasar bytes en vez de una ruta."""
    import soundfile as sf

    from src import embedder

    t = np.linspace(0, 0.5, 8000, endpoint=False)
    origen = tmp_path / "tono.wav"
    sf.write(str(origen), 0.5 * np.sin(2 * np.pi * 440 * t), 16000)

    desde_ruta = embedder._AudioSinLibrosa.load_audio(str(origen), sampling_rate=16000)
    desde_bytes = embedder._AudioSinLibrosa.load_audio(origen.read_bytes(), sampling_rate=16000)
    assert desde_ruta.shape == desde_bytes.shape
    assert np.allclose(desde_ruta, desde_bytes)


def test_audio_to_16k_mono_no_depende_de_librosa(tmp_path, monkeypatch):
    """La normalización de la ingesta tampoco debe pasar por librosa."""
    import soundfile as sf

    from src import config, ingest

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    origen = tmp_path / "voz.wav"
    sf.write(str(origen), 0.3 * np.sin(np.linspace(0, 100, 22050)), 22050)

    destino = ingest.audio_to_16k_mono(origen)
    assert destino.exists()
    datos, frecuencia = sf.read(str(destino))
    assert frecuencia == ingest.SAMPLE_RATE
    assert datos.ndim == 1


def test_el_silencio_no_se_remuestrea_y_conserva_su_longitud(tmp_path, monkeypatch):
    """El caso límite del RMS cero: un clip sin señal debe salir entero, no vacío."""
    import soundfile as sf

    from src import config, ingest

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    origen = tmp_path / "silencio.wav"
    sf.write(str(origen), np.zeros(16000, dtype="float32"), 16000)

    datos, frecuencia = sf.read(str(ingest.audio_to_16k_mono(origen)))
    assert frecuencia == ingest.SAMPLE_RATE
    assert len(datos) == 16000
    assert float(np.abs(datos).max()) == 0.0
