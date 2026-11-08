"""Настройки проекта: ключи берём из файла .env, а не пишем в код."""
import os

from dotenv import load_dotenv

load_dotenv()  # читает файл .env из папки проекта и кладёт значения в переменные окружения

YC_API_KEY = os.getenv("YC_API_KEY")        # секретный API-ключ сервисного аккаунта
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")    # ID каталога Yandex Cloud
YC_MODEL = os.getenv("YC_MODEL", "yandexgpt-lite/latest")  # модель; можно сменить в .env

API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

TEMPERATURE = 0.1      # низкая «креативность»: для извлечения фактов нужна точность
MAX_TOKENS = 1500      # предел длины ОТВЕТА модели в токенах
MAX_CHARS = 12_000     # предел длины одного куска документа в символах (см. методичку)
CHUNK_OVERLAP = 500    # перекрытие соседних кусков, чтобы не разрезать реквизит пополам

if not YC_API_KEY or not YC_FOLDER_ID:
    raise RuntimeError("Нет YC_API_KEY или YC_FOLDER_ID: скопируйте .env.example в .env и заполните")