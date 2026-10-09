"""Configuración central. Todo lo configurable sale del .env (única fuente de verdad)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# El .env vive en la raíz del proyecto y es la fuente de verdad: `override=True`
# hace que sus valores ganen sobre los del entorno del sistema. Importa porque
# una variable heredada de otro proyecto (p. ej. una DEEPSEEK_API_KEY global)
# cambiaría el comportamiento de esta app a espaldas del .env.
load_dotenv(ROOT / ".env", override=True)

# --- Modelo de embeddings (el objeto del proyecto, no configuración de usuario) ---
MODEL_ID = os.getenv("MODEL_ID", "google/embeddinggemma-2")

# Matryoshka: 768 (nativo), 512, 256, 128. Cambiarlo exige una colección nueva:
# la dimensión queda fijada al crear la colección y una query de otra dimensión
# no se puede comparar (lo advierte el model card).
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# --- Almacenamiento ---
CHROMA_DIR = ROOT / "chroma_db"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"  # audio normalizado a 16 kHz mono

# chromadb exige entre 3 y 512 caracteres de [a-zA-Z0-9._-]
COLLECTION = "mmrag_items"

# --- Capa de generación (opcional) ---
# Sin valores por defecto a propósito: si hay DEEPSEEK_API_KEY, entonces
# AI_MODEL y AI_MODEL_BASE_URL son obligatorios. Si no hay key, la app funciona
# igual en modo solo-recuperación, sin llamadas externas.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "").strip()
AI_MODEL_BASE_URL = os.getenv("AI_MODEL_BASE_URL", "").strip()

# --- Descripción de imágenes con un modelo de visión LOCAL (opcional) ---
# Sin estas dos variables la ingesta funciona igual: la única diferencia es que una
# foto se podrá ENCONTRAR pero no DESCRIBIR. Local por diseño, para que los datos no
# salgan del equipo.
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "").strip()

# Descripciones generadas, pendientes de revisión humana antes de indexarlas.
DESCRIPTIONS_FILE = DATA_DIR / "descripciones.json"

# --- Transcripción de audio local (opcional) ---
# Convierte un audio en texto para que sea RESPONDIBLE y no solo encontrable. Ollama
# NO sirve para esto: su API solo acepta `audio` en /api/embed, no en generate.
# CPU por defecto a propósito: ctranslate2 con CUDA exige sus propias DLLs y falla con
# "cublas64_12.dll is not found" cuando no están — es lo que pasó en el proyecto ORION.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "es").strip()

# --- Video (local, sin dependencias nuevas) ---
# El modelo NO tiene encoder de video: la ficha oficial dice que el video se
# procesa "as sampled frames through the vision encoder at 1 frame per second", o
# sea fotogramas al encoder de visión. Se decodifica con PyAV (llega con
# faster-whisper y trae FFmpeg embebido) y se le entregan YA muestreados, porque la
# vía nativa del procesador exige torchcodec + FFmpeg del sistema.
# Un video más largo que VIDEO_CHUNK_SECONDS se fragmenta: el procesador solo
# admite VIDEO_MAX_FRAMES fotogramas por vector, y cada fragmento es un vector.
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "1"))
VIDEO_CHUNK_SECONDS = float(os.getenv("VIDEO_CHUNK_SECONDS", "32"))
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "32"))


def video_enabled() -> bool:
    """El video solo necesita PyAV, que ya viene con faster-whisper.

    No es un modelo que se configure como la visión o la transcripción: es un
    decodificador. La comprobación existe para que la interfaz pueda no ofrecer
    video en vez de fallar por archivo si la dependencia faltara.
    """
    return importlib.util.find_spec("av") is not None


# --- Recuperación ---
TOP_K = int(os.getenv("TOP_K", "8"))


# --- Precio de los tokens (opcional, para el panel de métricas) ---
# Tarifas de DeepSeek verificadas el 2026-10-07, en dólares por 1M tokens y en hora
# valle (la hora peak cuesta el doble). Están configurables porque son la única forma
# de poner precio a lo que reporta el API, y así un cambio de tarifa no obliga a tocar
# código. Si `AI_MODEL` no es DeepSeek, corrígelas o ignora la cifra de coste.
PRICE_INPUT = float(os.getenv("PRICE_INPUT", "0.15"))
PRICE_OUTPUT = float(os.getenv("PRICE_OUTPUT", "0.60"))


def generation_enabled() -> bool:
    """La generación solo se activa si las TRES variables están configuradas.

    Exigir las tres evita el caso peor: que exista una API key heredada del
    entorno del sistema (de otro proyecto) y el botón aparezca para luego fallar.
    """
    return bool(DEEPSEEK_API_KEY and AI_MODEL and AI_MODEL_BASE_URL)


def generation_partial() -> bool:
    """Hay key pero falta modelo o URL: configuración incompleta, avisada en la UI."""
    return bool(DEEPSEEK_API_KEY) and not (AI_MODEL and AI_MODEL_BASE_URL)


def vision_enabled() -> bool:
    """La descripción de imágenes solo se activa con modelo y URL configurados."""
    return bool(VISION_MODEL and VISION_BASE_URL)


def transcribe_enabled() -> bool:
    """La transcripción solo se activa si hay un modelo de Whisper configurado."""
    return bool(WHISPER_MODEL)
