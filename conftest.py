"""Infraestructura de tests.

No toca código de producción: solo deja la raíz importable y aísla Chroma.
Ejecutar:

    pytest -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fresh_store(monkeypatch, tmp_path):
    """Un Chroma aislado de verdad por cada test, en su propio directorio.

    Ojo: `EphemeralClient()` **no sirve** para aislar. Chroma comparte una única
    instancia en memoria entre clientes, así que los vectores se filtraban de un
    test al siguiente (los conteos crecían). Un directorio distinto por test sí
    produce un cliente independiente, y además ejercita el mismo
    `PersistentClient` que usa la aplicación.
    """
    from src import config, store

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(store, "_client", None)
    yield store
    monkeypatch.setattr(store, "_client", None)


def _load_script(name: str):
    """Carga un módulo de `scripts/` por ruta: esa carpeta no es un paquete."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed_demo():
    """El módulo `scripts/seed_demo.py`, cargado sin ejecutar `main()`."""
    return _load_script("seed_demo")
