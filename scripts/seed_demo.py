"""Datos demo para el índice.

    python scripts/seed_demo.py                 # audio ESC-50 + PDF  (~4 MB)
    python scripts/seed_demo.py --images 20     # + 20 fotos de Flickr1k (140 MB)

El zip de Flickr se descarga completo porque el repo no publica imágenes sueltas;
de él solo se extraen las N que pidas, así que quedan en disco únicamente esas.

Además de descargar, este script aporta las ETIQUETAS DE TEXTO: los captions de
las fotos y las clases de sonido de ESC-50. Sin ellas, el LLM no tendría nada que
leer de una imagen o un audio, solo su nombre de archivo, y no podría responder
preguntas sobre su contenido.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, ingest, store  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0"}  # arXiv bloquea el agente por defecto
ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
FLICKR_REPO = "nlphuji/flickr_1k_test_image_text_retrieval"
FLICKR_ZIP = (
    f"https://huggingface.co/datasets/{FLICKR_REPO}/resolve/main/images_flickr_1k_test.zip"
)


def fetch(url: str, destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request) as response:
            destination.write_bytes(response.read())
    return destination


def seed_audio(limit: int) -> list[Path]:
    """Un clip por categoría de ESC-50 (50 clases de sonido ambiental)."""
    meta = fetch(f"{ESC50_RAW}/meta/esc50.csv", config.DATA_DIR / "esc50.csv")
    first_per_category: dict[str, str] = {}
    with meta.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            first_per_category.setdefault(row["category"], row["filename"])

    clips = []
    for filename in list(first_per_category.values())[:limit]:
        clips.append(fetch(f"{ESC50_RAW}/audio/{filename}", config.DATA_DIR / "esc50" / filename))
    return clips


def seed_pdf() -> list[Path]:
    return [fetch("https://arxiv.org/pdf/1706.03762", config.DATA_DIR / "attention.pdf")]


def seed_images(limit: int) -> list[Path]:
    """Extrae solo las primeras N fotos del zip, con sus nombres del CSV."""
    meta = fetch(
        f"https://huggingface.co/datasets/{FLICKR_REPO}/resolve/main/test_1k_flickr.csv",
        config.DATA_DIR / "flickr_captions.csv",
    )
    with meta.open(newline="", encoding="utf-8") as handle:
        filenames = [row["filename"] for row in csv.DictReader(handle)][:limit]

    archive = fetch(FLICKR_ZIP, config.DATA_DIR / "_cache" / "flickr.zip")
    target = config.DATA_DIR / "images"
    target.mkdir(parents=True, exist_ok=True)

    extracted = []
    with zipfile.ZipFile(archive) as zipped:
        members = {Path(name).name: name for name in zipped.namelist()}
        for filename in filenames:
            member = members.get(filename)
            if member is None:
                continue
            destination = target / filename
            if not destination.exists():
                with zipped.open(member) as source:
                    destination.write_bytes(source.read())
            extracted.append(destination)
    return extracted


def document_labels() -> dict[str, str]:
    """Texto asociado a cada archivo, para las modalidades que no lo traen dentro.

    Fotos -> el caption humano del dataset Flickr1k.
    Audios -> la clase de ESC-50 ("sonido ambiental: dog").
    """
    labels: dict[str, str] = {}

    captions = config.DATA_DIR / "flickr_captions.csv"
    if captions.exists():
        with captions.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    texts = json.loads(row.get("raw") or "[]")
                except json.JSONDecodeError:
                    continue
                if texts:
                    labels[row["filename"]] = texts[0]

    esc50 = config.DATA_DIR / "esc50.csv"
    if esc50.exists():
        with esc50.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                category = row["category"].replace("_", " ")
                labels[row["filename"]] = f"sonido ambiental: {category}"

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa datos demo.")
    parser.add_argument("--images", type=int, default=0, help="cuántas fotos de Flickr1k (0 = ninguna)")
    parser.add_argument("--audio", type=int, default=12, help="cuántos clips de ESC-50")
    parser.add_argument("--no-pdf", action="store_true", help="omitir el PDF de ejemplo")
    parser.add_argument("--reset", action="store_true", help="borrar el índice antes de cargar")
    args = parser.parse_args()

    if args.reset and store.total() > 0:
        print("Borrando el índice existente…")
        store.reset()

    paths: list[Path] = []
    print(f"Audio: {args.audio} clips de ESC-50…")
    paths += seed_audio(args.audio)

    if not args.no_pdf:
        print("PDF: Attention Is All You Need…")
        paths += seed_pdf()

    if args.images > 0:
        print(f"Imágenes: {args.images} fotos (descarga de 140 MB, solo la primera vez)…")
        paths += seed_images(args.images)

    labels = document_labels()
    print(f"\nIndexando {len(paths)} archivo(s). La primera vez descarga el modelo (~1.5 GB).")
    print(f"Etiquetas de texto para {len(labels)} archivos (captions y clases de sonido).")
    summary = ingest.ingest_paths(paths, labels=labels)

    print(f"\nVectores añadidos: {summary['ingested']}")
    for modality, count in sorted(summary["by_modality"].items()):
        print(f"  {modality}: {count}")
    if summary["skipped"]:
        print("Omitidos: " + ", ".join(summary["skipped"]))
    if summary["errors"]:
        print("Errores: " + "; ".join(summary["errors"]))
    print(f"\nTotal en el índice: {store.total()}")
    print("Listo. Arranca la interfaz con:  streamlit run app.py")


if __name__ == "__main__":
    main()
