"""Ingesta: clasificación por extensión, fragmentado y representación dual.

El candado importante aquí es `pdf_pages()`: devuelve **imagen + texto de la página**.
Sin ese texto el RAG recupera bien y responde mal, porque al LLM solo le llegaría el
nombre del archivo. También se fija que las etiquetas de fotos y audios acaben en el
campo `document` (lo único que lee el modelo de lenguaje).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import config, ingest

# ------------------------------------------------------------------ extensión


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("foto.jpg", "image"),
        ("FOTO.JPG", "image"),
        ("x.png", "image"),
        ("x.webp", "image"),
        ("x.tiff", "image"),
        ("s.wav", "audio"),
        ("s.mp3", "audio"),
        ("s.flac", "audio"),
        ("paper.pdf", "pdf"),
        ("notas.md", "text"),
        ("script.py", "text"),
        ("datos.csv", "text"),
        ("clip.mp4", "video"),
        ("clip.mkv", "video"),
        ("archivo.zip", None),
        ("informe.docx", None),
        ("binario.exe", None),
        ("sin_extension", None),
    ],
)
def test_modality_of(filename, expected):
    assert ingest.modality_of(Path(filename)) == expected


# ----------------------------------------------------------------- fragmentado


def test_text_chunks_devuelve_el_texto_entero_si_cabe():
    assert ingest.text_chunks("hola mundo", size=100) == ["hola mundo"]


def test_text_chunks_colapsa_los_espacios():
    assert ingest.text_chunks("a   b\n\n  c", size=100) == ["a b c"]


def test_text_chunks_devuelve_vacio_sin_contenido():
    assert ingest.text_chunks("   \n  ", size=100) == []
    assert ingest.text_chunks("", size=100) == []


def test_text_chunks_respeta_el_solape():
    texto = "abcdefghijklmnopqrstuvwxyz"
    chunks = ingest.text_chunks(texto, size=10, overlap=3)

    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    # Cada trozo comparte exactamente `overlap` caracteres con el siguiente.
    # `strict=False` a proposito: las dos listas estan desfasadas una posicion, no
    # alineadas, asi que la ultima no tiene pareja.
    for anterior, siguiente in zip(chunks, chunks[1:], strict=False):
        assert anterior[-3:] == siguiente[:3]


# ---------------------------------------------------------------------- ids


def test_stable_id_es_determinista(tmp_path):
    archivo = tmp_path / "a.txt"
    assert ingest._stable_id(archivo) == ingest._stable_id(archivo)


def test_stable_id_cambia_con_el_sufijo(tmp_path):
    archivo = tmp_path / "a.txt"
    assert ingest._stable_id(archivo, "#0") != ingest._stable_id(archivo, "#1")


# ------------------------------------------- representación dual: el texto al LLM


def test_display_text_prioriza_el_documento():
    """Si hay texto extraído, es lo que ve el LLM — aunque sea una foto o un audio."""
    metadata = {
        "modality": "audio",
        "name": "1-116765-A-41.wav",
        "_document": "sonido ambiental: chainsaw",
    }
    assert ingest._display_text(metadata, {"audio": "x.wav"}) == "sonido ambiental: chainsaw"


def test_display_text_cae_al_nombre_sin_documento():
    assert ingest._display_text({"modality": "image", "name": "foto.jpg"}, {}) == "foto.jpg"


def test_display_text_de_pdf_menciona_la_pagina():
    metadata = {"modality": "pdf", "name": "paper.pdf", "page": 3}
    assert ingest._display_text(metadata, {}) == "paper.pdf · página 3"


def test_pdf_pages_devuelve_imagen_y_texto(tmp_path, monkeypatch):
    """El corazón del diseño: se recupera por imagen, se responde con el texto."""
    import pymupdf

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    pdf = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "MULTI HEAD ATTENTION TEST CONTENT")
    document.save(str(pdf))
    document.close()

    pages = ingest.pdf_pages(pdf)

    assert len(pages) == 1
    image, text = pages[0]
    assert image.exists()
    assert image.suffix == ".png"
    assert "MULTI HEAD ATTENTION TEST" in text


def png(path: Path, mode: str = "RGB", size: tuple[int, int] = (12, 12)) -> Path:
    """Una imagen real y válida: la ingesta la abre con PIL para normalizarla."""
    from PIL import Image

    Image.new(mode, size).save(path)
    return path


def test_collect_de_imagen_usa_la_etiqueta_como_documento(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    foto = png(tmp_path / "1009434119.jpg")

    items: list = []
    ingest._collect(foto, "image", items, {"1009434119.jpg": "un perro corriendo"})

    kind, payload, metadata = items[0]
    assert kind == "media"
    assert metadata["modality"] == "image"
    assert ingest._display_text(metadata, payload) == "un perro corriendo"


def test_collect_de_imagen_sin_etiqueta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    foto = png(tmp_path / "x.jpg")

    items: list = []
    ingest._collect(foto, "image", items, None)
    _, payload, metadata = items[0]

    assert "_document" not in metadata
    assert ingest._display_text(metadata, payload) == "x.jpg"


def test_collect_de_imagen_embebe_la_copia_normalizada(tmp_path, monkeypatch):
    """El payload apunta a la copia de la caché; el metadata, al original, que es
    el archivo que la interfaz muestra al usuario."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    foto = png(tmp_path / "logo.png", mode="RGBA")

    items: list = []
    ingest._collect(foto, "image", items, None)
    _, payload, metadata = items[0]

    assert Path(payload["image"]) != foto
    assert Path(payload["image"]).suffix == ".png"
    assert metadata["path"] == str(foto.resolve())


def test_collect_de_imagen_con_etiqueta_usa_la_forma_interleaved(tmp_path, monkeypatch):
    """Con descripción, imagen y texto van en UN solo vector —la forma que documenta el
    modelo— y no en dos, que duplicaría el archivo en el ranking."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    foto = png(tmp_path / "logo.png")

    items: list = []
    ingest._collect(foto, "image", items, {"logo.png": "logo corporativo azul"})
    _, payload, metadata = items[0]

    assert payload["text"] == "logo corporativo azul <|image|>"
    assert payload["image"].endswith(".png")  # la copia normalizada, no el original
    assert ingest._display_text(metadata, payload) == "logo corporativo azul"


def test_collect_de_audio_con_etiqueta_usa_la_forma_interleaved(tmp_path, monkeypatch):
    """Con transcripción, audio y texto van en UN solo vector, igual que la imagen."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ingest, "audio_to_16k_mono", lambda ruta: ruta)

    audio = tmp_path / "pitch.mp3"
    audio.write_bytes(b"x")

    items: list = []
    ingest._collect(audio, "audio", items, {"pitch.mp3": "Hola, soy Edwin."})
    _, payload, metadata = items[0]

    assert payload["text"] == "Hola, soy Edwin. <|audio|>"
    assert payload["audio"].endswith(".mp3")
    assert ingest._display_text(metadata, payload) == "Hola, soy Edwin."


# ----------------------------------------- normalización de la imagen antes de embeber


def test_image_to_rgb_normaliza_un_gif_animado(tmp_path, monkeypatch):
    """Un GIF animado llega como pila de fotogramas `(N, alto, ancho, 3)` y el
    procesador revienta con "too many values to unpack (expected 3)".

    Es el bug real: un logo animado tumbaba la ingesta completa. Se conserva el
    primer fotograma.
    """
    from PIL import Image

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    gif = tmp_path / "animado.gif"
    frames = [Image.new("L", (24, 24), color=valor) for valor in (0, 120, 255)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=40)

    with Image.open(gif) as original:
        assert getattr(original, "n_frames", 1) > 1  # el test prueba un GIF de verdad

    normalizada = ingest.image_to_rgb(gif)

    assert normalizada.suffix == ".png"
    with Image.open(normalizada) as resultado:
        assert resultado.mode == "RGB"
        assert len(np.array(resultado).shape) == 3


def test_image_to_rgb_normaliza_escala_de_grises(tmp_path, monkeypatch):
    """El modo 'L' da un array de 2 ejes, que el encoder tampoco acepta."""
    from PIL import Image

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    gris = png(tmp_path / "gris.png", mode="L")

    with Image.open(ingest.image_to_rgb(gris)) as resultado:
        assert resultado.mode == "RGB"
        assert len(np.array(resultado).shape) == 3


def test_image_to_rgb_conserva_el_contenido_y_blanquea_el_fondo(tmp_path, monkeypatch):
    """La transparencia se compone sobre BLANCO. `convert("RGB")` a secas descarta el
    alfa y los píxeles transparentes, que llevan negro por dentro, salen negros: el logo
    de DATABiQ se aplanaba a un cuadrado negro y ni el encoder ni un modelo de visión
    podían leerlo.
    """
    from PIL import Image

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    logo = tmp_path / "logo.png"
    original = Image.new("RGBA", (16, 16), (0, 0, 0, 0))  # fondo transparente
    for x in range(4, 12):
        for y in range(4, 12):
            original.putpixel((x, y), (200, 20, 20, 255))  # cuadro rojo opaco
    original.save(logo)

    with Image.open(ingest.image_to_rgb(logo)) as resultado:
        assert resultado.mode == "RGB"
        assert resultado.getpixel((0, 0)) == (255, 255, 255)  # fondo -> blanco
        assert resultado.getpixel((8, 8)) == (200, 20, 20)  # contenido intacto


def test_image_to_rgb_todo_transparente_queda_blanco(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    vacio = tmp_path / "vacio.png"
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(vacio)

    with Image.open(ingest.image_to_rgb(vacio)) as resultado:
        assert resultado.getpixel((4, 4)) == (255, 255, 255)


def test_collect_de_texto_fragmenta_sin_tocar_el_modelo(tmp_path):
    notas = tmp_path / "notas.md"
    notas.write_text("contenido de prueba", encoding="utf-8")

    items: list = []
    ingest._collect(notas, "text", items, None)

    assert len(items) == 1
    kind, payload, metadata = items[0]
    assert kind == "text"
    assert payload == "contenido de prueba"
    assert metadata["chunk"] == 0


# ---------------------------------------------- rutas que NO cargan el modelo


def test_ingest_paths_omite_extensiones_no_soportadas(tmp_path):
    """Sin ítems que embeber, sale antes de tocar el modelo."""
    resultado = ingest.ingest_paths([tmp_path / "a.zip", tmp_path / "b.docx"])

    assert resultado["ingested"] == 0
    assert len(resultado["skipped"]) == 2
    assert resultado["errors"] == []


def test_ingest_paths_con_lista_vacia():
    resultado = ingest.ingest_paths([])
    assert resultado == {"ingested": 0, "intact": 0, "skipped": [], "errors": [], "by_modality": {}}


# ------------------------------------------------------------ salto por huella


def encode_que_cuenta(llamadas: list[int]):
    def encode(textos, **kwargs):
        llamadas.append(len(textos) if isinstance(textos, list) else 1)
        return np.ones((len(textos), config.EMBED_DIM), dtype="float32")

    return encode


def test_un_item_intacto_no_se_vuelve_a_embeber(tmp_path, monkeypatch, fresh_store):
    """La huella es lo que hace barato reindexar: lo que no cambió no se re-embebe.

    Antes, pulsar «Guardar y reindexar» volvía a embeber el corpus entero (40 s medidos)
    aunque solo se hubiera corregido una palabra.
    """
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    llamadas: list[int] = []
    monkeypatch.setattr(ingest.embedder, "encode_text", encode_que_cuenta(llamadas))
    notas = tmp_path / "notas.md"
    notas.write_text("contenido de prueba", encoding="utf-8")

    primero = ingest.ingest_paths([notas])
    assert (primero["ingested"], primero["intact"]) == (1, 0)

    segundo = ingest.ingest_paths([notas])  # nada cambió

    assert (segundo["ingested"], segundo["intact"]) == (0, 1)
    assert llamadas == [1]  # se embebió una sola vez, en la primera pasada
    assert fresh_store.total() == 1  # y el vector sigue ahí: saltar no es podar


def test_cambiar_el_archivo_fuerza_la_reindexacion(tmp_path, monkeypatch, fresh_store):
    """Misma ruta, otros bytes: la versión cambia, la huella cambia y hay que reindexar."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    llamadas: list[int] = []
    monkeypatch.setattr(ingest.embedder, "encode_text", encode_que_cuenta(llamadas))
    notas = tmp_path / "notas.md"
    notas.write_text("contenido de prueba", encoding="utf-8")
    ingest.ingest_paths([notas])

    notas.write_text("contenido distinto y bastante más largo", encoding="utf-8")
    segundo = ingest.ingest_paths([notas])

    assert (segundo["ingested"], segundo["intact"]) == (1, 0)
    assert llamadas == [1, 1]
    assert fresh_store.total() == 1


def test_el_pdf_tambien_se_salta_cuando_no_cambio(tmp_path, monkeypatch, fresh_store):
    """El PDF participa en el salto como las demás modalidades.

    Embeber una página cuesta ~0,25 s, así que un PDF de 200 páginas pagaría ~50 s en cada
    reindexado sin cambios. El PNG de la página sale de la caché por versión, así que
    comprobar la huella es barato; lo caro es embeber.
    """
    import pymupdf

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    embebidos: list[int] = []

    def encode_media(items, **kwargs):
        embebidos.append(len(items))
        return np.ones((len(items), config.EMBED_DIM), dtype="float32")

    monkeypatch.setattr(ingest.embedder, "encode_media", encode_media)

    pdf = tmp_path / "doc.pdf"
    documento = pymupdf.open()
    documento.new_page().insert_text((72, 72), "CONTENIDO DE PRUEBA")
    documento.save(str(pdf))
    documento.close()

    primero = ingest.ingest_paths([pdf])
    assert (primero["ingested"], primero["intact"]) == (1, 0)

    segundo = ingest.ingest_paths([pdf])

    assert (segundo["ingested"], segundo["intact"]) == (0, 1)
    assert embebidos == [1]  # la segunda vez no se embebió nada
    assert fresh_store.total() == 1


def test_ingest_paths_llama_al_progreso(tmp_path):
    vistos: list[tuple[int, int]] = []
    ingest.ingest_paths([tmp_path / "a.zip"], progress=lambda i, n: vistos.append((i, n)))
    assert vistos == [(1, 1)]


def test_ingest_paths_reporta_el_error_sin_abortar(tmp_path):
    """Un archivo ilegible se reporta y el resto sigue: no debe tumbar la ingesta."""
    roto = tmp_path / "roto.md"
    roto.mkdir()  # un directorio con extensión de texto: read_text() falla

    resultado = ingest.ingest_paths([roto])
    assert resultado["errors"] and "roto.md" in resultado["errors"][0]


# ----------------------------------------------- caché por versión y poda de vectores


def test_la_cache_lleva_la_version_del_archivo(tmp_path, monkeypatch):
    """El nombre de la caché incluye la versión: si el contenido cambia, no se reutiliza
    la copia vieja. Antes se llamaba solo por la ruta, así que reemplazar un PDF dejaba
    sus páginas PNG anteriores alimentando el vector nuevo."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    archivo = tmp_path / "doc.pdf"
    archivo.write_bytes(b"version uno")
    primera = ingest._cache_path(archivo, "_p001.png")

    archivo.write_bytes(b"version dos, bastante mas larga")
    segunda = ingest._cache_path(archivo, "_p001.png")

    assert primera != segunda


def test_la_poda_borra_las_copias_de_la_version_anterior(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    archivo = tmp_path / "foto.jpg"
    archivo.write_bytes(b"uno")
    vieja = ingest._cache_path(archivo, ".png")
    vieja.write_bytes(b"copia vieja")

    archivo.write_bytes(b"otro contenido, distinto")
    actual = ingest._cache_path(archivo, ".png")
    actual.write_bytes(b"copia actual")

    ingest._prune_cache(archivo)

    assert actual.exists()
    assert not vieja.exists()


def test_la_poda_limpia_el_esquema_viejo_sin_version(tmp_path, monkeypatch):
    """Las copias del esquema anterior (hash de ruta, sin versión) no coinciden con el
    glob nuevo y quedarían invisibles para siempre si no se borraran aquí.

    Eran DOS esquemas: `{prefijo}.png` para imagen y audio, y `{prefijo}_p001.png` para
    las páginas del PDF. El primero se me escapó en la primera versión de la poda.
    """
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    archivo = tmp_path / "foto.jpg"
    archivo.write_bytes(b"x")
    prefijo = ingest._stable_id(archivo)[:16]
    legado_imagen = config.CACHE_DIR / f"{prefijo}.png"
    legado_pdf = config.CACHE_DIR / f"{prefijo}_p001.png"
    for legado in (legado_imagen, legado_pdf):
        legado.parent.mkdir(parents=True, exist_ok=True)
        legado.write_bytes(b"esquema viejo")

    ingest._prune_cache(archivo)

    assert not legado_imagen.exists()
    assert not legado_pdf.exists()


def vectores_falsos(textos, **kwargs):
    """Sin modelo: un vector unitario por fragmento, del tamaño que exige la colección."""
    return np.ones((len(textos), config.EMBED_DIM), dtype="float32")


def test_reingerir_poda_los_vectores_que_ya_no_se_producen(tmp_path, monkeypatch, fresh_store):
    """Un archivo que se acorta no debe dejar vectores de su versión anterior.

    Si un PDF pierde páginas o un video se acorta, esos ids ya no se producen: sin poda
    seguirían apareciendo en las búsquedas, respondiendo con datos que ya no existen.
    """
    monkeypatch.setattr(ingest.embedder, "encode_text", vectores_falsos)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    notas = tmp_path / "notas.md"
    notas.write_text("palabra " * 400, encoding="utf-8")  # varios fragmentos

    primero = ingest.ingest_paths([notas])
    assert primero["ingested"] > 1

    notas.write_text("ahora es corto", encoding="utf-8")  # un solo fragmento
    segundo = ingest.ingest_paths([notas])

    assert segundo["ingested"] == 1
    assert fresh_store.total() == 1  # los fragmentos viejos se podaron


def test_un_archivo_que_falla_no_pierde_sus_vectores(tmp_path, monkeypatch, fresh_store):
    """La poda solo toca lo que se ingirió: un fallo no debe borrar lo que ya había."""
    monkeypatch.setattr(ingest.embedder, "encode_text", vectores_falsos)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    notas = tmp_path / "notas.md"
    notas.write_text("contenido válido", encoding="utf-8")
    ingest.ingest_paths([notas])
    assert fresh_store.total() == 1

    notas.unlink()  # ahora la lectura falla
    resultado = ingest.ingest_paths([notas])

    assert resultado["errors"]
    assert fresh_store.total() == 1
