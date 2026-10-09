# Seguridad

## Qué datos salen del equipo

Este proyecto está diseñado **local-first**, y esa promesa tiene un límite exacto que
conviene conocer antes de indexar nada:

| Componente | Dónde corre | Qué sale del equipo |
|---|---|---|
| Embeddings (texto, imagen, audio, video) | Local, tu GPU | **Nada** |
| Índice vectorial (Chroma) | Local, en disco | **Nada** |
| Descripción de imágenes (visión) | Local, Ollama | **Nada** |
| Transcripción (audio y video) | Local, Whisper | **Nada** |
| **Generación de respuestas** | **DeepSeek (nube)** | **Los fragmentos que la búsqueda devolvió** |

**Sin `DEEPSEEK_API_KEY` en el `.env`, la cuarta fila tampoco existe:** el sistema
funciona en modo solo-recuperación y no hace ninguna llamada externa. Es la configuración
para material confidencial.

## Aviso sobre la extracción de texto con visión en la nube

El README documenta una vía **opcional** (aún no implementada) para extraer texto de
imágenes usando el modelo de visión de DeepSeek. Si se activa, **el documento entero sale
del equipo**, no solo el fragmento relevante. Es una diferencia importante frente a la
generación: ahí viaja lo que la búsqueda seleccionó; aquí, la imagen original.

Antes de activarla sobre material de un cliente, la decisión es suya y debe constar por
escrito.

## Reportar una vulnerabilidad

Si encuentras un problema de seguridad:

1. **No abras un issue público.** Ábrelo en la pestaña *Security* → *Report a vulnerability*.
2. Indica versión, sistema operativo y pasos para reproducirlo.
3. Se acusará recibo antes de 7 días.

## Fuera de alcance

Este proyecto es una aplicación local de un solo usuario: **no incluye autenticación,
control de acceso ni aislamiento multiusuario**. Si lo despliegas en un servidor, esas
capas las pones tú — y el `.env` con la clave de API no debe quedar accesible por HTTP.
