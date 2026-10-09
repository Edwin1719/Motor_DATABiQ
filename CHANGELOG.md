# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/).
Versionado [semántico](https://semver.org/lang/es/).

## [Sin publicar]

### Corregido

- **El README se contradecía con el LICENSE.** Decía *"Licencia del código de este
  proyecto: por definir por el autor"* en la última línea, cuando el repositorio ya
  publicaba una **MIT**. Ahora la declara, y aclara que los pesos del modelo (Apache 2.0) y
  los datos de ejemplo del corpus histórico tienen licencia aparte.
- **Sección `Qué lo hace distinto` duplicada**, con dos tablas distintas. Unificada en la
  cabecera, con la fila que solo estaba en la segunda (producto punto frente a coseno).
- **El conteo de tests decía 154 y eran 159** — y ya había quedado obsoleto tres veces
  (145 → 154 → 159) mientras se construía este mismo banco. Se elimina el número fijo: el
  README remite a `pytest --collect-only -q`. Una cifra que caduca sin que nadie la revise
  es exactamente el problema que este documento combate.
- **"Dos pestañas"** en la puesta en marcha, cuando la app tiene tres desde que existe la
  de Métricas.
- Texto alternativo de la imagen de cabecera, que era el marcador por defecto de GitHub.

### Añadido

- Badge de CI en la cabecera, apuntando al workflow real del repositorio.


- **`bench/` — banco de validación reproducible.** Corpus semilla **CC0 generado por
  código** (14 ítems: 5 audio, 4 imagen, 3 documento, 2 texto), consultas con respuesta
  conocida y negativas, y `scripts/bench.py` con **Recall@K, MRR, latencia y separación
  positivo/negativo**, con código de salida para usarlo en integración continua.
  Resuelve que los resultados del README dependieran de un corpus que ya no existe y que
  no se podía redistribuir (ESC-50 es CC BY-NC; Flickr30k no declara licencia).
- **Línea base medida** sobre el banco: Recall@1 0,73 · @3 0,80 · @5 0,93 · @10 1,00 ·
  MRR 0,808. Imagen 4/4 y texto 6/6 en el puesto 1; **audio 0/4**, los cuatro dentro del
  top-6. La separación positivo/negativo **no existe** por similitud absoluta (peor
  positiva 0,5277 frente a mejor negativa 0,6167).
- `scripts/make_bench_corpus.py`: regenera el corpus bit a bit, sin red.
- `tests/test_bench.py`: 7 tests que protegen el banco y el decodificador de audio.
- `LICENSE` (MIT), `SECURITY.md`, `.gitattributes`, `pyproject.toml` (pytest + ruff) y
  `.github/workflows/ci.yml`.
- README: sección **La salida a la nube**, con la medición de extracción de texto por
  visión local frente a `deepseek-flash`, y el coste de cada opción.

### Corregido

- **`librosa` inutilizable en este entorno rompía el re-embebido de audio y de video.**
  Estaba en la ruta crítica en tres sitios: `ingest.audio_to_16k_mono()`,
  `video.audio_16k_mono()` y el `load_audio` de `transformers`, que hay que parchear en
  **dos** módulos (`audio_utils` y `processing_utils`, porque el segundo importó el nombre
  y conserva su propia referencia). Se sustituye por `soundfile` + `scipy.signal.resample_poly`,
  sin dependencias nuevas. Medido: sin el arreglo se indexaban **9 de 14** vectores del
  banco; con él, **14 de 14**.
- **El banco no conservaba el índice entre corridas** por un `reset()` implícito: destruía
  las huellas y reembebía los 14 vectores cada vez, justo lo contrario de lo que el salto
  por huella existe para conseguir. Ahora reutiliza lo que está al día
  (medido: `0 nuevos, 14 al día`).
- README: el conteo de tests decía 141 y sumaba 145; la tabla de resultados usaba un corpus
  ausente sin advertirlo; y el alias `deepseek-v4-flash` se describía como *"riesgo cero"*
  cuando un alias de compatibilidad se retira sin aviso.

### Cambiado

- README: la tabla de resultados del corpus de demostración queda **marcada como
  histórica y no reproducible**, con el motivo (licencias y corpus ausente).
- README: el diagnóstico del audio, corregido. Decía *"el audio no discrimina"*, y es
  impreciso: su espacio interno es el **más expandido** de las cuatro modalidades
  (0,1671 de amplitud par a par frente a 0,0937 de imagen). El hecho correcto es que el
  acierto de audio puntúa bajo y comprimido (0,6508–0,6614) frente a texto (hasta 0,8423).

## [0.1.0] — 2026-10-07

Primera versión congelada y validada de punta a punta.

### Añadido

- RAG multimodal sobre `google/embeddinggemma-2`: **una sola colección** de 768 dimensiones
  para texto, imagen, audio, PDF y video, con la modalidad en el payload.
- **Representación dual**: el vector sale de la modalidad (la página de un PDF se embebe
  como imagen, sin OCR) y el campo `document` guarda el texto que lee el LLM.
- **Salto por huella** `hash(versión + texto)`: lo que no cambió no se vuelve a preparar.
  Medido: reindexar sin cambios pasó de 40,6 s a 0,69 s.
- Ingesta de video con **PyAV** (FFmpeg embebido, sin dependencias del sistema), fotogramas
  a 1 fps al encoder de visión y texto desde la transcripción de su audio.
- Descripciones locales con Ollama (`think=False`: 101 s → 5 s por imagen) y transcripción
  local con `faster-whisper` en CPU e idioma fijo.
- Interfaz Streamlit con tres pestañas: Preguntar, Recuperar y Métricas.
- Generación opcional con DeepSeek, con citas numeradas y **negativa explícita** cuando los
  fragmentos no alcanzan. Sin `DEEPSEEK_API_KEY`, modo solo-recuperación.
- Suite de regresión centrada en los fallos silenciosos del sistema.
