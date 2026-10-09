"""Recuperación multimodal y respuesta opcional con citas.

La recuperación funciona siempre, sin API key ni llamadas externas. La generación
solo se activa si hay `DEEPSEEK_API_KEY` en el .env.
"""

from __future__ import annotations

import numpy as np

from . import config, embedder, store

SYSTEM_PROMPT = """Responde a la pregunta usando SOLO la información de los fragmentos recuperados.

Reglas:
- Cita cada afirmación con el número del fragmento entre corchetes, por ejemplo [2].
- Un fragmento puede traer texto extraído (páginas de PDF, documentos, código) o solo
  una descripción corta (el caption de una foto, la clase de un sonido).
- Si un fragmento NO trae texto de contenido, no describas lo que crees que muestra o
  suena: di únicamente que ese archivo coincidió con la búsqueda.
- Si dos fragmentos se contradicen entre sí (una cifra distinta para el mismo dato, dos
  formas de escribir un nombre, datos que no concuerdan), NO elijas uno en silencio:
  dilo con claridad, indica qué afirma cada fragmento y cita AMBOS con su número.
- Si los fragmentos no alcanzan para responder, dilo con claridad en vez de suponer.
- Responde en el idioma de la pregunta, de forma directa y concisa."""


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    modality: str | None = None,
) -> list[dict]:
    """Codifica la consulta con el prefijo `SearchQuery` y busca en la colección."""
    # `encode()` con un solo string devuelve forma (768,), no (1, 768): se aplana
    # para que la forma coincida entre una consulta suelta y un lote.
    embedding = np.asarray(
        embedder.encode_text(query, prompt_name="SearchQuery"), dtype="float32"
    ).reshape(-1)
    return store.search(embedding, n_results=top_k or config.TOP_K, modality=modality)


def build_context(hits: list[dict]) -> str:
    """Arma el contexto numerado que ve el LLM."""
    blocks = []
    for position, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        header = f"[{position}] modalidad={meta.get('modality', '?')} · archivo={meta.get('name', '?')}"
        if meta.get("page"):
            header += f" · página={meta['page']}"
        header += f" · similitud={hit['similarity']:.4f}"
        blocks.append(f"{header}\n{hit['document']}")
    return "\n\n".join(blocks)


def _client():
    from openai import OpenAI

    if not config.AI_MODEL or not config.AI_MODEL_BASE_URL:
        raise RuntimeError(
            "DEEPSEEK_API_KEY está definida, pero falta AI_MODEL o AI_MODEL_BASE_URL "
            "en el .env. Define ambas o quita la key para usar solo recuperación."
        )
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.AI_MODEL_BASE_URL)


def answer(query: str, hits: list[dict], *, uso: dict | None = None) -> str:
    """Genera una respuesta con citas a partir de los fragmentos recuperados.

    Si se pasa `uso`, se rellena con los tokens que reporta el API. Es la única cifra
    medible del coste: sin exponerla, el consumo de cada pregunta es invisible.
    """
    if not hits:
        return "No hay fragmentos en el índice para responder."

    response = _client().chat.completions.create(
        model=config.AI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Fragmentos recuperados:\n\n{build_context(hits)}\n\nPregunta: {query}",
            },
        ],
        temperature=0.2,
    )
    if uso is not None:
        contador = getattr(response, "usage", None)
        uso["prompt"] = int(getattr(contador, "prompt_tokens", 0) or 0)
        uso["completion"] = int(getattr(contador, "completion_tokens", 0) or 0)
    return response.choices[0].message.content or ""
