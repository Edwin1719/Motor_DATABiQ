"""Chroma: UNA sola colección para todas las modalidades.

La modalidad vive en el payload (`metadata`), no en colecciones separadas: como
texto, imagen y audio comparten el mismo espacio de 768 dims, una consulta de
texto puede devolver fotos, sonidos y fragmentos de PDF en un mismo ranking.

Distancia: producto punto (`ip`). Los vectores ya salen L2-normalizados del
modelo, así que `ip` es idénticamente equivalente a coseno y se ahorra normalizar
en cada búsqueda.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from . import config

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    """Crea o recupera la colección, validando que la dimensión siga coincidiendo."""
    collection = get_client().get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": "ip", "embed_dim": config.EMBED_DIM},
    )
    stored = (collection.metadata or {}).get("embed_dim")
    if stored is not None and int(stored) != config.EMBED_DIM:
        raise RuntimeError(
            f"La colección se creó con dimensión {stored} pero EMBED_DIM es "
            f"{config.EMBED_DIM}. Una query de otra dimensión no es comparable: "
            f"borra {config.CHROMA_DIR} o vuelve a EMBED_DIM={stored}."
        )
    return collection


def add_items(
    ids: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict[str, Any]],
    documents: list[str],
) -> None:
    """`upsert` para que reingerir el mismo archivo sea idempotente."""
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings.astype("float32"),
        metadatas=metadatas,
        documents=documents,
    )


def metadatos_por_ruta(path: str | Path) -> dict[str, dict[str, Any]]:
    """Metadatos guardados de los vectores que salieron de ese archivo, por id.

    Lo usa la reingesta para ver si un ítem ya está intacto: si su huella no ha cambiado,
    su vector tampoco puede haber cambiado y no hace falta volver a prepararlo.
    """
    crudo = get_collection().get(where={"path": str(path)})
    return {
        item_id: (metadata or {})
        for item_id, metadata in zip(crudo.get("ids") or [], crudo.get("metadatas") or [])
    }


def ids_for_path(path: str | Path) -> list[str]:
    """Ids de los vectores que salieron de ese archivo, buscando por su ruta resuelta."""
    return list(metadatos_por_ruta(path))


def delete_ids(ids: Iterable[str]) -> None:
    """Borra vectores por id.

    Lo usa el reindexado: al volver a ingerir un archivo se hace `upsert` de sus ids,
    pero los de una versión anterior —un PDF con menos páginas, un video más corto—
    no se producirían y quedarían en el índice respondiendo con datos que ya no existen.
    """
    ids = list(ids)
    if ids:
        get_collection().delete(ids=ids)


def inventory() -> list[dict[str, Any]]:
    """Todos los vectores con sus metadatos y su texto, para poder medir el índice.

    No es una consulta por similitud: trae la colección entera. Lo usa el panel de
    métricas, que necesita verlo todo para contar cobertura, residuo y fan-out — y por
    eso mismo carga todos los documentos en memoria: con miles de vectores sería caro.
    """
    crudo = get_collection().get(include=["metadatas", "documents"])
    return [
        {"id": item_id, "metadata": metadata or {}, "document": documento or ""}
        for item_id, metadata, documento in zip(
            crudo.get("ids") or [],
            crudo.get("metadatas") or [],
            crudo.get("documents") or [],
        )
    ]


def search(
    embedding: np.ndarray,
    *,
    n_results: int | None = None,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Consulta por vector. `modality` filtra en el payload, no cambia de colección."""
    where = {"modality": modality} if modality else None
    raw = get_collection().query(
        # reshape(1, -1): chromadb necesita una matriz (n_consultas, dim).
        # Una lista de escalares la rechaza.
        query_embeddings=embedding.reshape(1, -1).astype("float32"),
        # Sin default propio: el número de fragmentos sale del .env, no de una segunda
        # constante escondida aquí que pudiera contradecirlo.
        n_results=n_results or config.TOP_K,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    # chromadb devuelve listas anidadas (una por query); aquí siempre hay una.
    metadatas = (raw.get("metadatas") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    return [
        {
            "id": item_id,
            # Chroma reporta DISTANCIA, no similitud. Verificado contra los datos:
            # con espacio 'ip' la distancia es exactamente 1 - producto punto, y
            # como los vectores salen unitarios eso equivale a 1 - coseno. Se
            # expone como similitud para que "mayor = mejor"; el orden no cambia
            # porque Chroma ya devuelve de mejor a peor.
            "similarity": 1.0 - float(distance),
            "metadata": metadata or {},
            "document": document or "",
        }
        for item_id, metadata, document, distance in zip(ids, metadatas, documents, distances)
    ]


def counts_by_modality() -> dict[str, int]:
    """Conteo por modalidad. Trae los metadatos completos: aceptable a escala de demo."""
    collection = get_collection()
    if collection.count() == 0:
        return {}
    metadatas = (collection.get(include=["metadatas"]).get("metadatas")) or []
    return dict(Counter(meta.get("modality", "?") for meta in metadatas))


def total() -> int:
    return get_collection().count()


def reset() -> None:
    """Borra la colección completa (lo usa el script de datos demo)."""
    get_client().delete_collection(config.COLLECTION)
