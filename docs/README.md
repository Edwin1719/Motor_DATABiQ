# Notebook de origen

`embeddinggemma2_multimodal_retrieval.py` es el **notebook de Colab del que nació este
proyecto**, exportado a Python. Se conserva como **documentación del origen**: es el punto
de partida, no el sistema que hoy se ejecuta.

- **Origen:** [Colab · EmbeddingGemma2_Multimodal_Retrieval](https://colab.research.google.com/drive/19Ha_9EozTJt85baetLNRSc05UkjrdX3P)
- **Qué demuestra:** nueve pruebas de **recuperación** con EmbeddingGemma 2 — texto a 1 000
  fotos (Recall@1), consultas en otro idioma o usando una foto, la voz como consulta,
  sonidos ESC-50, fotogramas de video, páginas de PDF sin OCR, truncado Matryoshka y un
  modelo solo-texto.
- **Qué NO hace, para no atribuirle de más:** el notebook **solo recupera**, nunca tuvo capa
  de generación (no hay ningún LLM). Y lo que parece «comprensión» de las fotos viene de los
  **captions del propio dataset** —Flickr30k trae cinco por foto—, no del modelo, que es
  representacional y no emite texto. Los límites son suyos, no un defecto del demo.
- **Qué se llevó al proyecto y qué no:** el proyecto conserva la recuperación multimodal y le
  añadió lo que aquí no había —generación con citas, ingesta propia, video con PyAV y
  descripciones locales—. La foto y la voz **como consulta** quedaron fuera.
- **Trampas que documentaba y siguen vigentes:** no usar `float16`; el texto sí lleva prefijo
  de tarea (`SearchQuery` / `Document`) y la imagen, el audio y el video no; el audio a
  16 kHz mono.

Licencia del modelo: **Apache 2.0**. Los datos de ejemplo y sus advertencias están en el
[README principal](../README.md).
