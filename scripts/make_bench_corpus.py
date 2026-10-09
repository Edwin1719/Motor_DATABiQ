"""Genera el corpus semilla de validación: material CC0, sin descargas, reproducible.

Por qué generado y no descargado: los datos que usaba la validación original **no se
pueden redistribuir** (ESC-50 es CC BY-NC, Flickr30k no declara licencia). Un corpus
generado por este script es material original del proyecto: no hay licencia de terceros
que revisar, no depende de la red y es idéntico en cada ejecución.

**Qué mide este corpus y qué no.** Sirve para *regresión*: detecta que un cambio rompió
la plomería de una modalidad, y da una línea base reproducible con la que comparar. **No**
mide precisión sobre material del mundo real, y sus ítems son deliberadamente más fáciles
de separar que una foto o una grabación de campo. Un número alto aquí es una cota
optimista, no una promesa.

Uso:
    python scripts/make_bench_corpus.py            # genera lo que falte
    python scripts/make_bench_corpus.py --clean    # borra y regenera todo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# numpy y PIL se importan DENTRO de cada generador: cargarlos a nivel de módulo haría
# que este script no arrancara sin ellos, y `--help` no debería exigir dependencias.
# Con `TYPE_CHECKING` las anotaciones de tipo siguen siendo ciertas para el linter
# (ruff: F821) sin pagar el import en tiempo de ejecución.
if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "bench" / "corpus"
LABELS = ROOT / "bench" / "labels.json"

# Muestreo de audio del corpus: es el que espera el modelo (mono 16 kHz).
SAMPLE_RATE = 16000
SECONDS = 3.0

# Semilla fija: el mismo ruido en cada ejecución, para que el corpus sea comparable
# entre versiones. Un corpus que cambia solo invalidaría cualquier comparación.
SEED = 20261009


# --------------------------------------------------------------------------- audio


def _audio_items() -> tuple[dict[str, np.ndarray], np.random.Generator]:
    """Cinco patrones de onda, cada uno con una propiedad acústica que lo distingue.

    Se generan como muestras crudas: no se descarga ningún archivo de audio. Devuelve
    también el generador, para que el ruido sea reproducible entre ejecuciones.
    """
    import numpy as np

    rng = np.random.default_rng(SEED)
    t = np.linspace(0, SECONDS, int(SAMPLE_RATE * SECONDS), endpoint=False)

    tono_440 = 0.5 * np.sin(2 * np.pi * 440 * t)
    tono_880 = 0.5 * np.sin(2 * np.pi * 880 * t)

    # Barrido de frecuencia: un chirp lineal de 200 a 4000 Hz.
    fase = 2 * np.pi * (200 * t + (4000 - 200) * t**2 / (2 * SECONDS))
    chirp = 0.5 * np.sin(fase)

    # Amplitud modulada a 4 Hz: produce un pulso claro y periódico.
    am = 0.5 * np.sin(2 * np.pi * 300 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))

    # Silencio: existe para comprobar que un clip vacío no se pierde ni se omite.
    silencio = np.zeros_like(t)

    return {
        "audio/tono_440hz.wav": tono_440.astype("float32"),
        "audio/tono_880hz.wav": tono_880.astype("float32"),
        "audio/chirp_200_4000hz.wav": chirp.astype("float32"),
        "audio/pulso_am_4hz.wav": am.astype("float32"),
        "audio/silencio.wav": silencio.astype("float32"),
    }, rng


def _write_wav(path: Path, muestras) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), muestras, SAMPLE_RATE)


# -------------------------------------------------------------------------- imagen


def _imagen_items() -> dict[str, bytes]:
    """Cuatro imágenes de 256x256 con contenido distinguible y rotulado por color/forma."""
    import io

    from PIL import Image, ImageDraw

    salida: dict[str, bytes] = {}

    def _png(imagen) -> bytes:
        buffer = io.BytesIO()
        imagen.save(buffer, "PNG")
        return buffer.getvalue()

    # Círculo azul sobre blanco.
    im = Image.new("RGB", (256, 256), (255, 255, 255))
    ImageDraw.Draw(im).ellipse((48, 48, 208, 208), fill=(20, 60, 200))
    salida["imagen/circulo_azul.png"] = _png(im)

    # Cuadrado rojo sobre blanco.
    im = Image.new("RGB", (256, 256), (255, 255, 255))
    ImageDraw.Draw(im).rectangle((64, 64, 192, 192), fill=(200, 30, 30))
    salida["imagen/cuadrado_rojo.png"] = _png(im)

    # Triángulo verde sobre blanco.
    im = Image.new("RGB", (256, 256), (255, 255, 255))
    ImageDraw.Draw(im).polygon([(128, 40), (216, 208), (40, 208)], fill=(20, 140, 60))
    salida["imagen/triangulo_verde.png"] = _png(im)

    # Rejilla de franjas negras: distinta de las tres anteriores por textura, no por color.
    im = Image.new("RGB", (256, 256), (255, 255, 255))
    lapiz = ImageDraw.Draw(im)
    for x in range(0, 256, 32):
        lapiz.rectangle((x, 0, x + 15, 255), fill=(0, 0, 0))
    salida["imagen/franjas_negras.png"] = _png(im)

    return salida


# ----------------------------------------------------------------------------- pdf


def _pdf_items() -> dict[str, str]:
    """Tres documentos con palabras clave inequívocas.

    Ojo: son texto plano con extensión `.pdf` **a propósito**. El corpus se ingiere
    declarando la modalidad a mano (ver `labels.json`), no por extensión, así que no
    hace falta una librería de escritura de PDF para un fixture de test. La modalidad
    PDF del sistema se ejercita con los PDFs reales de `data/uploads/`.
    """
    return {
        "documento/factura_2024.pdf": (
            "FACTURA\n"
            "Numero de factura: FA-2024-0517\n"
            "Cliente: DATABiQ\n"
            "El vencimiento del pago es el 30 de noviembre de 2024.\n"
            "El importe facturado por el servicio de analitica asciende a doce mil pesos.\n"
        ),
        "documento/informe_tecnico.pdf": (
            "INFORME TECNICO DE CALIDAD\n"
            "La muestra analizada se recogio el 14 de octubre en el sector norte.\n"
            "El nivel de dureza medido fue de 320 megapascales, dentro del rango admisible.\n"
            "Se recomienda repetir la medicion a los seis meses.\n"
        ),
        "documento/acta_reunion.pdf": (
            "ACTA DE REUNION DEL COMITE\n"
            "Asistentes: cinco personas del area de operaciones.\n"
            "Se acordo priorizar la migracion del sistema financiero en el primer trimestre.\n"
            "El responsable de la migracion sera el equipo de infraestructura.\n"
        ),
    }


# ---------------------------------------------------------------------------- texto


def _texto_items() -> dict[str, str]:
    """Dos documentos de texto: uno en español y otro con un término en inglés, para
    comprobar la recuperación en los dos idiomas."""
    return {
        "texto/notas_operativas.txt": (
            "Notas operativas\n\n"
            "El inventario del almacen se revisa los lunes y los jueves. El codigo interno "
            "de cada articulo empieza por la letra Z. Cuando falta existencias se avisa al "
            "responsable de compras antes del mediodia.\n\n"
            "Palabra de control de este documento: alcachofa."
        ),
        "texto/technical_notes.txt": (
            "Technical notes\n\n"
            "The pipeline decodes media in three stages. Each stage keeps a cache keyed by "
            "the file version, so an unchanged asset is never prepared twice.\n\n"
            "Control word for this document: marmalade."
        ),
    }


# ---------------------------------------------------------------------------- main


def _etiquetas() -> dict[str, str]:
    """Texto que verá el LLM por cada ítem. Es el `document` del vector.

    Para audio e imagen es una descripción escrita a mano —humana, no generada por un
    modelo— para que el corpus no dependa de Ollama ni de Whisper y sea reproducible.
    """
    return {
        "audio/tono_440hz.wav": "Tono puro sostenido a 440 hercios, la nota La.",
        "audio/tono_880hz.wav": "Tono puro sostenido a 880 hercios, una octava mas agudo.",
        "audio/chirp_200_4000hz.wav": "Barrido de frecuencia ascendente de 200 a 4000 hercios.",
        "audio/pulso_am_4hz.wav": "Tono con amplitud modulada a cuatro pulsos por segundo.",
        "audio/silencio.wav": "Grabacion sin senal: silencio digital.",
        "imagen/circulo_azul.png": "Un circulo azul relleno centrado sobre fondo blanco.",
        "imagen/cuadrado_rojo.png": "Un cuadrado rojo relleno centrado sobre fondo blanco.",
        "imagen/triangulo_verde.png": "Un triangulo verde equilatero sobre fondo blanco.",
        "imagen/franjas_negras.png": "Franjas verticales negras y blancas a rayas.",
    }


LICENCIA = {
    "tipo": "CC0-1.0 (material original generado por scripts/make_bench_corpus.py)",
    "nota": (
        "No procede de ningun dataset de terceros. Es regenerable bit a bit con la "
        "semilla fija del script, y se puede redistribuir sin condiciones."
    ),
}


def generar(*, clean: bool = False) -> dict[str, int]:
    if clean and CORPUS.exists():
        shutil.rmtree(CORPUS)

    etiquetas = _etiquetas()
    escritos: dict[str, int] = {}

    for ruta, muestras in _audio_items()[0].items():
        destino = CORPUS / ruta
        _write_wav(destino, muestras)
        escritos["audio"] = escritos.get("audio", 0) + 1

    for ruta, contenido in _imagen_items().items():
        destino = CORPUS / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
        escritos["imagen"] = escritos.get("imagen", 0) + 1

    for ruta, cuerpo in {**_pdf_items(), **_texto_items()}.items():
        destino = CORPUS / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(cuerpo, encoding="utf-8")
        modalidad = "texto" if ruta.startswith("texto/") else "documento"
        escritos[modalidad] = escritos.get(modalidad, 0) + 1

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(
        json.dumps(
            {"licencia": LICENCIA, "etiquetas": etiquetas},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return escritos


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el corpus semilla de validación.")
    parser.add_argument("--clean", action="store_true", help="borra y regenera todo el corpus")
    args = parser.parse_args()

    escritos = generar(clean=args.clean)
    total = sum(escritos.values())
    print(f"Corpus semilla generado en {CORPUS.relative_to(ROOT)}")
    for modalidad, cantidad in sorted(escritos.items()):
        print(f"  {modalidad:<10} {cantidad}")
    print(f"  {'TOTAL':<10} {total}")
    print(f"\nEtiquetas: {LABELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
