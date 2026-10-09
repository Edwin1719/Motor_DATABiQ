"""Recuperación y generación: numeración de las citas y validación de la configuración.

El candado importante: `build_context()` numera los fragmentos `[1]`, `[2]`… y esos
números son los que el LLM usa para citar. Si la numeración se desfasa respecto a
la lista de fuentes que ve el usuario, **las citas apuntan a otro archivo** y el
sistema miente sin fallar.
"""

from __future__ import annotations

import pytest

from src import config, rag


def hit(
    modality: str = "pdf",
    name: str = "paper.pdf",
    similarity: float = 0.7,
    document: str = "contenido",
    **extra,
):
    return {
        "id": "x",
        "similarity": similarity,
        "metadata": {"modality": modality, "name": name, **extra},
        "document": document,
    }


# ------------------------------------------------------------------- contexto


def test_el_contexto_numera_desde_uno():
    contexto = rag.build_context([hit(name="a"), hit(name="b"), hit(name="c")])
    assert contexto.startswith("[1]")
    assert "[2]" in contexto
    assert "[3]" in contexto


def test_el_contexto_separa_los_fragmentos_con_linea_en_blanco():
    assert rag.build_context([hit(name="a"), hit(name="b")]).count("\n\n") == 1


def test_el_contexto_incluye_la_pagina_solo_cuando_existe():
    assert "página=3" in rag.build_context([hit(page=3)])
    assert "página" not in rag.build_context([hit()])


def test_el_contexto_incluye_similitud_y_documento():
    contexto = rag.build_context([hit(similarity=0.6543, document="TEXTO_UNICO_AQUI")])
    assert "0.6543" in contexto
    assert "TEXTO_UNICO_AQUI" in contexto


def test_el_contexto_incluye_modalidad_y_archivo():
    contexto = rag.build_context([hit(modality="image", name="perro.jpg")])
    assert "modalidad=image" in contexto
    assert "archivo=perro.jpg" in contexto


def test_el_contexto_sobrevive_a_metadatos_incompletos():
    """Un fragmento sin name ni modality no debe romper la construcción del prompt."""
    contexto = rag.build_context([{"id": "x", "similarity": 0.5, "metadata": {}, "document": "d"}])
    assert "[1]" in contexto
    assert "modalidad=?" in contexto


def test_el_contexto_vacio_es_cadena_vacia():
    assert rag.build_context([]) == ""


# ------------------------------------------------- sin red y sin API key


def test_answer_sin_fragmentos_no_llama_a_la_api():
    """Sin fragmentos responde localmente: ni coste ni key involucrada."""
    assert "No hay fragmentos" in rag.answer("cualquier pregunta", [])


def test_client_exige_el_modelo(monkeypatch):
    monkeypatch.setattr(config, "AI_MODEL", "")
    monkeypatch.setattr(config, "AI_MODEL_BASE_URL", "https://api.deepseek.com")
    with pytest.raises(RuntimeError, match="AI_MODEL"):
        rag._client()


def test_client_exige_la_url(monkeypatch):
    monkeypatch.setattr(config, "AI_MODEL", "deepseek-flash")
    monkeypatch.setattr(config, "AI_MODEL_BASE_URL", "")
    with pytest.raises(RuntimeError, match="AI_MODEL"):
        rag._client()


def test_el_prompt_del_sistema_prohibe_inventar():
    """La regla anti-alucinación es parte del contrato del sistema."""
    assert "no" in rag.SYSTEM_PROMPT.lower()
    assert "fragmentos" in rag.SYSTEM_PROMPT.lower()


def test_el_prompt_del_sistema_exige_senalar_contradicciones():
    """Con fuentes en conflicto el modelo debe exponerlo, no elegir una en silencio.

    Sin esta regla el ranking decide la respuesta: el audio dice "más de 8 años" y la
    página 1 del PDF "más de 10 años", y el LLM contestaba con la que quedara arriba.
    """
    prompt = rag.SYSTEM_PROMPT.lower()
    assert "se contradicen" in prompt
    assert "no elijas uno en silencio" in prompt
    assert "ambos" in prompt
