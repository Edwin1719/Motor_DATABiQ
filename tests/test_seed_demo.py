"""Etiquetas de texto del demo: lo que hace que fotos y audios sean respondibles.

`document_labels()` construye el mapa `{archivo: descripción}` que `ingest_paths()`
usa como campo `document`. Sin él, el LLM solo vería nombres de archivo y no podría
responder nada sobre una foto o un sonido.

Estos tests leen los CSV reales del demo, así que se saltan si no se ha corrido
`scripts/seed_demo.py` todavía.
"""

from __future__ import annotations

import pytest

from src import config

faltan = not (config.DATA_DIR / "esc50.csv").exists() or not (
    config.DATA_DIR / "flickr_captions.csv"
).exists()

pytestmark = pytest.mark.skipif(faltan, reason="faltan los CSV demo: corre scripts/seed_demo.py")


def test_etiqueta_los_sonidos_por_su_clase(seed_demo):
    labels = seed_demo.document_labels()
    assert labels["1-100032-A-0.wav"] == "sonido ambiental: dog"
    assert labels["1-101336-A-30.wav"] == "sonido ambiental: door wood knock"


def test_los_guiones_bajos_se_vuelven_espacios(seed_demo):
    labels = seed_demo.document_labels()
    assert "_" not in labels["1-101336-A-30.wav"]


def test_usa_el_caption_de_las_fotos(seed_demo):
    labels = seed_demo.document_labels()
    assert "garden" in labels["1009434119.jpg"]


def test_devuelve_vacio_sin_csv(seed_demo, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert seed_demo.document_labels() == {}
