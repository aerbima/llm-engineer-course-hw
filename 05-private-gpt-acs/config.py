"""Все настройки проекта в одном месте.

Значения читаются из файла .env (если он есть), иначе берутся значения по умолчанию.
Так код не нужно менять, чтобы попробовать другую модель или другой Top-K.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # читает .env из текущей папки и кладёт значения в os.environ

# Закрытый контур: выключаем телеметрию библиотек. По умолчанию и Gradio,
# и ChromaDB отправляют анонимную статистику использования в интернет.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# --- Папки проекта -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # папка, где лежит config.py
DOCS_DIR = BASE_DIR / "data" / "docs"               # наши .md/.txt (попадают в Git)
RAW_DIR = BASE_DIR / "data" / "raw"                 # PDF производителей и нормы (в Git не попадают)
CHROMA_DIR = BASE_DIR / "chroma_db"                 # векторная база на диске
SESSIONS_DIR = BASE_DIR / "sessions"                # журналы диалогов в формате JSONL
RESULTS_DIR = BASE_DIR / "results"                  # результаты экспериментов

# --- Языковая модель -----------------------------------------------------
# ollama — локальная модель через Ollama; openai_compat — эмулятор школы
# или любой сервер с OpenAI-совместимым API (LM Studio, vLLM)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))   # низкая — меньше фантазий
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))                     # окно контекста для Ollama
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")              # для openai_compat
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")                # для openai_compat

# --- Эмбеддинги ----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")    # мультиязычная модель, запускается в Ollama

# --- Разбиение на чанки --------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # длина чанка в символах
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # перекрытие соседних чанков в символах

# --- Поиск ---------------------------------------------------------------
COLLECTION = os.getenv("COLLECTION", "acs_kb")          # имя коллекции в ChromaDB
TOP_K = int(os.getenv("TOP_K", "4"))                     # сколько чанков отдавать модели
# Порог близости 0…1: если лучший чанк ниже порога, ответа в базе нет и LLM не вызываем.
# Значение подбирается скриптом calibrate.py под вашу модель эмбеддингов.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.45"))

# --- Память диалога ------------------------------------------------------
# none — без памяти; window — последние MEMORY_K пар «вопрос–ответ»;
# summary_buffer — свежие сообщения целиком, старые сворачиваются в сводку
MEMORY_MODE = os.getenv("MEMORY_MODE", "summary_buffer")
MEMORY_K = int(os.getenv("MEMORY_K", "3"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "600"))
# Переписывать уточняющий вопрос в самостоятельный перед поиском
CONDENSE_QUESTION = os.getenv("CONDENSE_QUESTION", "1") == "1"