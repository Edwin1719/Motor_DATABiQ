"""Comprobación rápida de la búsqueda cruzada, sin LLM.

    python scripts/check_retrieval.py                        # tres consultas de ejemplo
    python scripts/check_retrieval.py "tu pregunta" "otra"    # las que tú quieras

Sirve para ver, en un mismo ranking, qué modalidades compiten por una consulta. **Las
consultas que valen son las tuyas**, las que ya sabes qué archivo deberían devolver: las
de ejemplo solo comprueban que el sistema responde.

Dos diagnósticos distintos, no uno:

  - Si el archivo correcto **no aparece** en el top-3, falla la **recuperación**.
  - Si aparece y el modelo de lenguaje **no puede responder** sobre él, falla el **texto**
    que se guardó con ese vector (`document`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import rag, store  # noqa: E402

EJEMPLOS = [
    "¿de qué trata el documento principal?",
    "¿tienes algún video o audio sobre el tema?",
    "¿cuál es la capital de Francia?",  # fuera del alcance: debería quedar abajo
]

TOP = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Muestra el top-{TOP} de cada consulta.")
    parser.add_argument("consultas", nargs="*", help="tus preguntas; sin ninguna, usa ejemplos")
    args = parser.parse_args()

    total = store.total()
    if total == 0:
        print("Índice vacío: indexa archivos desde la aplicación.")
        return

    print(f"Índice con {total} vectores\n")
    for query in args.consultas or EJEMPLOS:
        print(f"Q: {query}")
        for hit in rag.retrieve(query, top_k=TOP):
            metadata = hit["metadata"]
            sufijo = ""
            if metadata.get("page"):
                sufijo = f"  página {metadata['page']}"
            elif metadata.get("start") is not None:
                sufijo = f"  {metadata['start']}-{metadata['end']} s"
            print(
                f"   {hit['similarity']:.4f}  {metadata['modality']:<5}  {metadata['name']}{sufijo}"
            )
        print()


if __name__ == "__main__":
    main()
