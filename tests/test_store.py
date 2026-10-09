"""Chroma: espacio `ip`, conversión de distancia a similitud y filtro por modalidad.

Este archivo es el candado del hallazgo más delicado del proyecto. Chroma reporta
**distancia**, y `store.search()` la convierte a similitud con `1 - distancia`
(comprobado empíricamente: producto punto 0.658657 → distancia reportada 0.341343).

Si esa resta se invierte, o si alguien cambia la métrica a coseno/L2, **el ranking
se ordena al revés sin lanzar ningún error**: el sistema seguiría respondiendo, solo
que peor. Con vectores sintéticos de producto punto conocido, eso no puede pasar
inadvertido.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, store


def basis(index: int) -> np.ndarray:
    """Vector unitario con un 1.0 en `index` y ceros en el resto."""
    vector = np.zeros(config.EMBED_DIM, dtype="float32")
    vector[index] = 1.0
    return vector


def load(items: list[dict]) -> None:
    store.add_items(
        ids=[item["id"] for item in items],
        embeddings=np.vstack([item["vector"] for item in items]),
        metadatas=[{"modality": item["modality"], "name": item["id"]} for item in items],
        documents=[item.get("document", item["id"]) for item in items],
    )


# --------------------------------------------------------------------- espacio


def test_la_coleccion_usa_producto_punto(fresh_store):
    metadata = store.get_collection().metadata
    assert metadata["hnsw:space"] == "ip"
    assert metadata["embed_dim"] == config.EMBED_DIM


# ------------------------------------------- la fórmula distancia -> similitud


def test_vector_identico_da_similitud_uno(fresh_store):
    """Producto punto 1 → distancia 0 → similitud 1."""
    load([{"id": "a", "modality": "image", "vector": basis(0)}])
    assert store.search(basis(0))[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_vector_ortogonal_da_similitud_cero(fresh_store):
    """Producto punto 0 → distancia 1 → similitud 0."""
    load([{"id": "a", "modality": "image", "vector": basis(0)}])
    assert store.search(basis(1))[0]["similarity"] == pytest.approx(0.0, abs=1e-5)


def test_vector_opuesto_da_similitud_menos_uno(fresh_store):
    """Producto punto -1 → distancia 2 → similitud -1. Aquí se cae un signo invertido."""
    load([{"id": "a", "modality": "image", "vector": basis(0)}])
    assert store.search(-basis(0))[0]["similarity"] == pytest.approx(-1.0, abs=1e-5)


def test_el_orden_es_de_mayor_a_menor_similitud(fresh_store):
    load(
        [
            {"id": "cerca", "modality": "image", "vector": basis(0)},
            {"id": "medio", "modality": "image", "vector": (basis(0) + basis(1)) / np.sqrt(2)},
            {"id": "lejos", "modality": "image", "vector": basis(1)},
        ]
    )
    hits = store.search(basis(0))
    assert [hit["id"] for hit in hits] == ["cerca", "medio", "lejos"]
    similitudes = [hit["similarity"] for hit in hits]
    assert similitudes == sorted(similitudes, reverse=True)


# --------------------------------------------------- payload, filtro y conteos


def test_search_devuelve_metadatos_y_documento(fresh_store):
    load([{"id": "a", "modality": "pdf", "vector": basis(2), "document": "texto de la pagina"}])
    hit = store.search(basis(2))[0]
    assert hit["metadata"]["modality"] == "pdf"
    assert hit["metadata"]["name"] == "a"
    assert hit["document"] == "texto de la pagina"


def test_filtro_por_modalidad(fresh_store):
    load(
        [
            {"id": "foto", "modality": "image", "vector": basis(0)},
            {"id": "sonido", "modality": "audio", "vector": basis(1)},
        ]
    )
    assert [hit["id"] for hit in store.search(basis(0), modality="audio")] == ["sonido"]
    assert [hit["id"] for hit in store.search(basis(0), modality="image")] == ["foto"]
    assert len(store.search(basis(0))) == 2


def test_conteos_por_modalidad(fresh_store):
    assert store.counts_by_modality() == {}
    assert store.total() == 0

    load(
        [
            {"id": "a", "modality": "image", "vector": basis(0)},
            {"id": "b", "modality": "image", "vector": basis(1)},
            {"id": "c", "modality": "audio", "vector": basis(2)},
            {"id": "d", "modality": "pdf", "vector": basis(3)},
        ]
    )
    assert store.counts_by_modality() == {"image": 2, "audio": 1, "pdf": 1}
    assert store.total() == 4


def test_upsert_no_duplica_al_reingerir(fresh_store):
    item = {"id": "mismo", "modality": "image", "vector": basis(0)}
    load([item])
    load([item])
    assert store.total() == 1


def test_reset_vacia_el_indice(fresh_store):
    load([{"id": "a", "modality": "image", "vector": basis(0)}])
    store.reset()
    assert store.total() == 0


def test_desajuste_de_dimension_lanza_error(fresh_store, monkeypatch):
    """La dimensión queda fija al crear la colección: no se puede consultar con otra."""
    store.get_collection()
    monkeypatch.setattr(config, "EMBED_DIM", 512)
    with pytest.raises(RuntimeError, match="dimensión"):
        store.get_collection()


# --------------------------------------- ids por ruta, borrado y TOP_K heredado


def load_con_ruta(items: list[dict], ruta: str) -> None:
    store.add_items(
        ids=[item["id"] for item in items],
        embeddings=np.vstack([item["vector"] for item in items]),
        metadatas=[
            {"modality": item["modality"], "name": item["id"], "path": ruta} for item in items
        ],
        documents=[item["id"] for item in items],
    )


def test_ids_for_path_solo_devuelve_los_de_ese_archivo(fresh_store):
    load_con_ruta([{"id": "doc#p1", "modality": "pdf", "vector": basis(0)}], "C:/x/doc.pdf")
    load_con_ruta([{"id": "foto", "modality": "image", "vector": basis(1)}], "C:/x/foto.png")

    assert store.ids_for_path("C:/x/doc.pdf") == ["doc#p1"]
    assert store.ids_for_path("C:/x/no-existe.pdf") == []


def test_delete_ids_borra_solo_los_indicados(fresh_store):
    load_con_ruta(
        [
            {"id": "doc#p1", "modality": "pdf", "vector": basis(0)},
            {"id": "doc#p2", "modality": "pdf", "vector": basis(1)},
        ],
        "C:/x/doc.pdf",
    )
    store.delete_ids(["doc#p2"])

    assert store.ids_for_path("C:/x/doc.pdf") == ["doc#p1"]
    assert store.total() == 1


def test_delete_ids_con_lista_vacia_no_hace_nada(fresh_store):
    load_con_ruta([{"id": "doc", "modality": "pdf", "vector": basis(0)}], "C:/x/doc.pdf")
    store.delete_ids([])
    assert store.total() == 1


def test_search_sin_n_results_usa_el_top_k_de_la_configuracion(fresh_store, monkeypatch):
    """`search()` no tiene default propio: el número de fragmentos sale del `.env`, para
    que no exista una segunda constante capaz de contradecirlo."""
    monkeypatch.setattr(config, "TOP_K", 2)
    load([{"id": str(n), "modality": "image", "vector": basis(n)} for n in range(4)])
    assert len(store.search(basis(0))) == 2
