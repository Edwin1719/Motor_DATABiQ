"""Precisión numérica: la regla que el modelo prohíbe incumplir.

EmbeddingGemma 2 no admite `float16`: sus activaciones exceden el rango dinámico de
ese formato y el modelo devuelve `NaN` o embeddings degradados **sin lanzar error**.
Es la peor clase de fallo — silencioso — así que se fija como invariante testeable.

El resto de `embedder.py` (`encode_text`, `encode_media`) exige cargar los 744M de
parámetros, así que no se testea aquí: sería un test de integración.
"""

from __future__ import annotations

import torch

from src import embedder

PERMITIDOS = (torch.bfloat16, torch.float32)


def test_la_precision_nunca_es_float16():
    precision = embedder.pick_dtype()
    assert precision in PERMITIDOS
    assert precision is not torch.float16


def test_la_precision_sigue_el_soporte_de_bfloat16():
    esperado = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    assert embedder.pick_dtype() == esperado


def test_el_dispositivo_es_cuda_o_cpu():
    assert embedder.pick_device() in ("cuda", "cpu")


def test_el_dispositivo_coincide_con_torch():
    assert embedder.pick_device() == ("cuda" if torch.cuda.is_available() else "cpu")
