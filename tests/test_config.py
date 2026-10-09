"""Configuración: cuándo se activa la generación y qué restricciones tiene el índice.

El candado importante: `generation_enabled()` exige **las tres** variables. Existe una
`DEEPSEEK_API_KEY` a nivel del sistema, heredada de otro proyecto, así que exigir solo
la key haría aparecer el botón de respuesta y fallar al pulsarlo.
"""

from __future__ import annotations

import pytest

from src import config


def siembra(monkeypatch, key: str, model: str, url: str) -> None:
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", key)
    monkeypatch.setattr(config, "AI_MODEL", model)
    monkeypatch.setattr(config, "AI_MODEL_BASE_URL", url)


def test_activa_solo_con_las_tres(monkeypatch):
    siembra(monkeypatch, "sk-x", "deepseek-flash", "https://api.deepseek.com")
    assert config.generation_enabled() is True
    assert config.generation_partial() is False


@pytest.mark.parametrize(
    ("key", "model", "url"),
    [
        ("", "deepseek-flash", "https://api.deepseek.com"),
        ("sk-x", "", "https://api.deepseek.com"),
        ("sk-x", "deepseek-flash", ""),
        ("", "", ""),
    ],
)
def test_no_se_activa_si_falta_cualquiera(monkeypatch, key, model, url):
    siembra(monkeypatch, key, model, url)
    assert config.generation_enabled() is False


@pytest.mark.parametrize(
    ("model", "url"),
    [("", "https://api.deepseek.com"), ("deepseek-flash", ""), ("", "")],
)
def test_avisa_cuando_hay_key_pero_falta_configuracion(monkeypatch, model, url):
    siembra(monkeypatch, "sk-x", model, url)
    assert config.generation_partial() is True


def test_no_hay_configuracion_parcial_sin_key(monkeypatch):
    siembra(monkeypatch, "", "", "")
    assert config.generation_partial() is False


# -------------------------------------------------------- restricciones del índice


def test_el_nombre_de_coleccion_cumple_la_regla_de_chromadb():
    """chromadb rechaza nombres de menos de 3 caracteres — nos pasó probando con 't'."""
    nombre = config.COLLECTION
    assert 3 <= len(nombre) <= 512
    assert nombre[0].isalnum() and nombre[-1].isalnum()
    assert all(caracter.isalnum() or caracter in "._-" for caracter in nombre)


def test_la_dimension_es_una_de_las_soportadas_por_matryoshka():
    assert config.EMBED_DIM in (768, 512, 256, 128)


def test_top_k_es_utilizable():
    assert config.TOP_K >= 1


def test_el_directorio_del_indice_cuelga_del_proyecto():
    assert config.CHROMA_DIR.parent == config.ROOT
