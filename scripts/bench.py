"""Banco de validación: convierte "parece que funciona" en números reproducibles.

Qué mide, sobre el corpus semilla de `bench/corpus/`:

  - **Recall@K** y **MRR** por modalidad y en conjunto. Un solo ranking con todas las
    modalidades compitiendo, que es como funciona el sistema.
  - **Separación positivo/negativo.** Compara la mejor similitud de una consulta fuera
    de alcance contra la peor de las consultas con respuesta conocida. Si una negativa
    puntúa por encima, el ranking no distingue "no lo tengo" de "lo tengo".
  - **Latencia** de recuperación (mediana y percentil 90).

El índice del banco vive en `bench/.index/`, **aparte del índice de trabajo**: si
compartieran colección, la métrica mentiría. No toca nada de `chroma_db/`.

Uso:
    python scripts/bench.py                     # corre y aplica el umbral de Recall@3
    python scripts/bench.py --min-recall3 0    # solo mide, no falla
    python scripts/bench.py --json salida.json # deja el resultado en un archivo
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "bench" / "corpus"
LABELS = ROOT / "bench" / "labels.json"
QUERIES = ROOT / "bench" / "queries.yaml"
INDEX = ROOT / "bench" / ".index"

K_VALORES = (1, 3, 5, 10)


# ------------------------------------------------------------------ lectura del banco


def _escalar(texto: str) -> str:
    return texto.strip().strip('"').strip("'").strip()


def leer_consultas(ruta: Path = QUERIES) -> dict[str, list[dict[str, str]]]:
    """Lector mínimo del formato de `queries.yaml`.

    No es un parser de YAML completo a propósito: el archivo usa una lista plana de
    bloques `- clave: valor`, y meter una dependencia nueva (o un parser casero de
    300 líneas) para eso sería peor que estas 25 líneas.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"no existe el banco de consultas: {ruta}")

    resultado: dict[str, list[dict[str, str]]] = {"positivas": [], "negativas": []}
    seccion: str | None = None
    actual: dict[str, str] | None = None

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        limpia = linea.strip()
        if not limpia or limpia.startswith("#"):
            continue
        if limpia.endswith(":") and not limpia.startswith("-"):
            nombre = limpia[:-1].strip()
            if nombre in resultado:
                seccion = nombre
                actual = None
            continue
        if limpia.startswith("- "):
            actual = {}
            resultado[seccion].append(actual)
            resto = limpia[2:]
            if ":" in resto:
                clave, _, valor = resto.partition(":")
                actual[clave.strip()] = _escalar(valor)
            continue
        if actual is not None and ":" in limpia:
            clave, _, valor = limpia.partition(":")
            actual[clave.strip()] = _escalar(valor)

    return resultado


# ------------------------------------------------------------------------- evaluación


def modalidad_de(ruta: Path) -> str:
    """Modalidad con la que se ingiere cada ítem del corpus.

    Los archivos de `documento/` son texto plano con extensión `.pdf` **a propósito**:
    así la extensión no decide la modalidad y se declara aquí. La modalidad PDF real
    (página renderizada) se ejercita con los PDFs de `data/uploads/`, no aquí.
    """
    carpeta = ruta.relative_to(CORPUS).parts[0] if CORPUS in ruta.parents else ruta.parent.name
    if carpeta == "audio":
        return "audio"
    if carpeta == "imagen":
        return "image"
    return "text"


def _cargar_etiquetas() -> dict[str, str]:
    if not LABELS.exists():
        raise FileNotFoundError(
            "faltan las etiquetas del corpus. Ejecuta primero: python scripts/make_bench_corpus.py"
        )
    datos = json.loads(LABELS.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in datos.get("etiquetas", {}).items()}


def preparar_indice(*, resetear: bool, reindexar: bool) -> dict:
    """Indexa el corpus semilla en la colección del banco.

    **`resetear` no es lo mismo que `reindexar`**, y confundirlos fue un error de la
    primera versión de este script: `store.reset()` borra la colección, y con ella las
    huellas guardadas, así que la ingesta posterior vuelve a embeber los 14 vectores
    aunque nada haya cambiado — justo lo contrario de lo que el salto por huella existe
    para conseguir. Por defecto el índice se **conserva** y la ingesta reutiliza lo que
    ya está al día; `--reset` solo hace falta tras cambiar el corpus o `EMBED_DIM`.

    El corpus se indexa en dos pasadas porque su forma no es la de `data/uploads/`: los
    ítems de `documento/` son texto plano con extensión `.pdf` (ver `modalidad_de`), así
    que la extensión no puede decidir la modalidad.
    """
    from src import config, ingest, store

    # El índice del banco vive aparte del de trabajo: si compartieran colección, la
    # métrica mezclaría los ítems del corpus con los archivos reales y mentiría.
    config.CHROMA_DIR = INDEX
    store._client = None

    archivos = sorted(p for p in CORPUS.rglob("*") if p.is_file())
    if not archivos:
        raise FileNotFoundError(
            "el corpus está vacío. Ejecuta primero: python scripts/make_bench_corpus.py"
        )

    if resetear and store.get_collection().count():
        store.reset()

    etiquetas = _cargar_etiquetas()
    texto_items = [p for p in archivos if modalidad_de(p) == "text"]
    media_items = [p for p in archivos if modalidad_de(p) != "text"]

    resumen_texto = _ingerir_texto(texto_items, etiquetas)
    resumen_media = (
        ingest.ingest_paths(media_items, labels=etiquetas)
        if media_items
        else {"ingested": 0, "intact": 0, "errors": []}
    )

    return {
        "archivos": len(archivos),
        "ingested": resumen_texto["ingested"] + resumen_media["ingested"],
        "intact": resumen_texto.get("intact", 0) + resumen_media.get("intact", 0),
        "errors": list(resumen_texto["errors"]) + list(resumen_media["errors"]),
        "total": store.total(),
    }


def _ingerir_texto(archivos: list[Path], etiquetas: dict[str, str]) -> dict:
    """Ingiere los ítems de texto declarando su modalidad, sin adivinarla por extensión.

    Usa los dos mecanismos que ya existen para esto en `ingest.py`:

      - `_stable_id` + `metadatos_por_ruta` + `_huella`: se pasan como "huella guardada"
        para que `_collect` produzca el ítem como **intacto** con la modalidad correcta y,
        en la primera ejecución, como texto nuevo. Es el mismo camino que usa la
        reingesta real, no un atajo paralelo.
      - `_display_text`: arma el `document` que verá el LLM con las mismas reglas que la
        aplicación, así que el banco mide el mismo texto que produce el sistema.

    Un ítem de texto no tiene representación dual (no hay imagen que conservar), así que
    `upsert` de un vector por ítem es exactamente lo que hace la ingesta de texto normal.
    """
    from src import embedder, ingest, store

    guardadas_por_archivo = {str(p.resolve()): store.metadatos_por_ruta(p) for p in archivos}

    items: list[tuple[str, object, dict]] = []
    for path in archivos:
        guardadas = guardadas_por_archivo[str(path.resolve())]
        # La huella se calcula aquí porque `_collect` la necesita para decidir si el
        # ítem sigue intacto, y en la rama de texto la calcula sobre el fragmento.
        for index, chunk in enumerate(
            ingest.text_chunks(path.read_text(encoding="utf-8", errors="replace"))
        ):
            item_id = ingest._stable_id(path, f"#{index}")
            huella = ingest._huella(ingest._version(path), chunk)
            if ingest._intacto(guardadas, item_id, huella):
                items.append(("intacto", None, {"id": item_id, "path": str(path.resolve())}))
                continue
            meta = {
                "modality": "text",
                "name": path.name,
                "path": str(path.resolve()),
                "source": str(path.resolve().parent),
                "chunk": index,
                "id": item_id,
                "fingerprint": huella,
                ingest.DOCUMENT: etiquetas.get(path.name, chunk),
            }
            items.append(("text", chunk, meta))

    pendientes = [i for i, (kind, _, _) in enumerate(items) if kind == "text"]
    intactos = [meta["id"] for kind, _, meta in items if kind == "intacto"]
    if not pendientes:
        return {"ingested": 0, "intact": len(intactos), "errors": []}

    # `_display_text` arma el `document` con las mismas reglas que la aplicación: el banco
    # tiene que medir el mismo texto que ve el LLM en producción, no una variante.
    vectores = embedder.encode_text([items[i][1] for i in pendientes], prompt_name="Document")

    errores: list[str] = []
    guardados = 0
    for posicion, indice in enumerate(pendientes):
        _, payload, meta = items[indice]
        try:
            store.add_items(
                ids=[meta["id"]],
                embeddings=vectores[posicion : posicion + 1],
                metadatas=[{k: v for k, v in meta.items() if k != "id" and not k.startswith("_")}],
                documents=[ingest._display_text(meta, payload)],
            )
            guardados += 1
        except Exception as exc:
            errores.append(f"{meta.get('name', '?')}: {type(exc).__name__}: {exc}")

    return {"ingested": guardados, "intact": len(intactos), "errors": errores}


def evaluar(*, top_k: int, resetear: bool) -> dict:
    from src import rag

    corpus = preparar_indice(resetear=resetear, reindexar=True)
    banco = leer_consultas()

    detalle: list[dict] = []
    latencias: list[float] = []
    aciertos = {k: 0 for k in K_VALORES}
    reciprocos: list[float] = []
    positivas_peor: float | None = None

    for caso in banco["positivas"]:
        arranque = time.perf_counter()
        hits = rag.retrieve(caso["query"], top_k=top_k)
        latencias.append(time.perf_counter() - arranque)

        esperado = caso["esperado"]
        nombres = [h["metadata"].get("name", "") for h in hits]
        puesto = next((i for i, n in enumerate(nombres, start=1) if n == esperado), None)

        # La modalidad se comprueba aparte del puesto: acertar el archivo por la
        # modalidad equivocada no es el mismo acierto, y el banco debe distinguirlo.
        modalidad_hit = None
        if puesto is not None:
            modalidad_hit = hits[puesto - 1]["metadata"].get("modality")

        for k in K_VALORES:
            if puesto is not None and puesto <= k:
                aciertos[k] += 1
        if puesto is not None:
            reciprocos.append(1.0 / puesto)

        similitudes = [float(h["similarity"]) for h in hits]
        # Se acumula la PEOR similitud de todas las consultas con respuesta conocida:
        # es el suelo contra el que se mide una consulta fuera de alcance. Con `top_k`
        # fragmentos recuperados, el último es el peor de esa consulta.
        if similitudes:
            positivas_peor = (
                similitudes[-1] if positivas_peor is None else min(positivas_peor, similitudes[-1])
            )

        detalle.append(
            {
                "id": caso.get("id", caso["query"][:30]),
                "query": caso["query"],
                "esperado": esperado,
                "modalidad_esperada": caso.get("modalidad", ""),
                "modalidad_hit": modalidad_hit,
                "puesto": puesto,
                "similitud_top1": round(similitudes[0], 4) if similitudes else None,
                "top3": [
                    f"{h['metadata'].get('modality', '?')}/{h['metadata'].get('name', '?')} "
                    f"({float(h['similarity']):.4f})"
                    for h in hits[:3]
                ],
            }
        )

    negativas: list[dict] = []
    for caso in banco["negativas"]:
        arranque = time.perf_counter()
        hits = rag.retrieve(caso["query"], top_k=top_k)
        latencias.append(time.perf_counter() - arranque)
        mejor = max((float(h["similarity"]) for h in hits), default=0.0)
        negativas.append(
            {
                "id": caso.get("id", caso["query"][:30]),
                "query": caso["query"],
                "mejor_similitud": round(mejor, 4),
                "top1": (
                    f"{hits[0]['metadata'].get('modality', '?')}/{hits[0]['metadata'].get('name', '?')}"
                    if hits
                    else "sin resultados"
                ),
                # Una negativa por debajo de la peor positiva está en su sitio: el
                # ranking la deja por debajo de todo lo que sí es una respuesta.
                "por_debajo_de_lo_peor_positivo": (
                    None if positivas_peor is None else mejor <= positivas_peor
                ),
            }
        )

    n = len(banco["positivas"])
    ordenadas = sorted(latencias)
    return {
        "corpus": corpus,
        "consultas_positivas": n,
        "consultas_negativas": len(banco["negativas"]),
        "recall": {f"@{k}": round(aciertos[k] / n, 4) if n else None for k in K_VALORES},
        "mrr": round(statistics.fmean(reciprocos), 4) if reciprocos else 0.0,
        "peor_similitud_positiva": round(positivas_peor, 4) if positivas_peor is not None else None,
        "latencia_mediana": round(statistics.median(latencias), 3) if latencias else None,
        "latencia_p90": round(ordenadas[int(len(ordenadas) * 0.9)], 3) if ordenadas else None,
        "detalle": detalle,
        "negativas": negativas,
    }


# ----------------------------------------------------------------------------- informe


def imprimir(resultado: dict, *, min_recall3: float) -> int:
    corpus = resultado["corpus"]
    print("=" * 78)
    print("BANCO DE VALIDACION - corpus semilla CC0 (bench/corpus/)")
    print("=" * 78)
    print(
        f"corpus: {corpus['archivos']} archivos -> {corpus['total']} vectores "
        f"({corpus['ingested']} nuevos, {corpus['intact']} al dia)"
    )
    if corpus["errors"]:
        print(f"ERRORES DE INGESTA: {corpus['errors']}")
    print(
        f"consultas: {resultado['consultas_positivas']} con respuesta · {resultado['consultas_negativas']} fuera de alcance"
    )

    recall = resultado["recall"]
    print(
        f"\nRecall@1={recall['@1']:.2f}  @3={recall['@3']:.2f}  @5={recall['@5']:.2f}  @10={recall['@10']:.2f}"
    )
    print(f"MRR={resultado['mrr']:.3f}")
    print(
        f"latencia: mediana {resultado['latencia_mediana']} s · p90 {resultado['latencia_p90']} s"
    )

    print("\n--- consultas con respuesta conocida ---")
    fallos = []
    for caso in resultado["detalle"]:
        marca = "OK " if caso["puesto"] == 1 else ("    " if caso["puesto"] else "FALLA")
        puesto = f"#{caso['puesto']}" if caso["puesto"] else "no aparece"
        print(f"[{marca}] {puesto:<12} {caso['id']:<28} {caso['query'][:44]}")
        if caso["puesto"] != 1:
            fallos.append(caso)
            print(f"          esperado: {caso['esperado']}")
            for i, top in enumerate(caso["top3"], start=1):
                print(f"          top{i}: {top}")

    print("\n--- consultas fuera de alcance ---")
    for caso in resultado["negativas"]:
        estado = "OK " if caso["por_debajo_de_lo_peor_positivo"] else "MAL"
        print(
            f"[{estado}] mejor={caso['mejor_similitud']:.4f}  {caso['id']:<28} top1={caso['top1'][:40]}"
        )

    print("\n" + "=" * 78)
    separacion = all(c["por_debajo_de_lo_peor_positivo"] for c in resultado["negativas"])
    peor_pos = resultado["peor_similitud_positiva"]
    mejor_neg = max((c["mejor_similitud"] for c in resultado["negativas"]), default=0.0)
    print(
        f"Separacion: peor positiva={peor_pos} · mejor negativa={mejor_neg} -> {'SEPARA' if separacion else 'NO SEPARA'}"
    )
    print(f"Recall@3 real {recall['@3']:.2f} contra umbral {min_recall3:.2f}")

    codigo = 0
    if recall["@3"] is None or recall["@3"] < min_recall3:
        print("\nRESULTADO: FALLA. Recall@3 por debajo del umbral.")
        codigo = 1
    if not separacion:
        print(
            "RESULTADO: el ranking no distingue 'no lo tengo' de 'lo tengo'. "
            "Es un hallazgo, no un fallo del banco: anotalo antes de tocar umbrales."
        )
        codigo = codigo or 2
    if codigo == 0:
        print("\nRESULTADO: PASA.")
    return codigo


def main() -> int:
    parser = argparse.ArgumentParser(description="Banco de validación del RAG multimodal.")
    parser.add_argument(
        "--top-k", type=int, default=10, help="fragmentos a recuperar (por defecto 10)"
    )
    parser.add_argument(
        "--min-recall3", type=float, default=0.8, help="umbral de Recall@3 para pasar"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="borra el índice del banco antes de medir (solo tras cambiar el corpus o EMBED_DIM)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="escribe el resultado completo en JSON"
    )
    args = parser.parse_args()

    resultado = evaluar(top_k=args.top_k, resetear=args.reset)
    codigo = imprimir(resultado, min_recall3=args.min_recall3)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresultado escrito en {args.json}")

    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
