"""Отправка запроса в YandexGPT через REST API (использую библиотеку requests)."""
import time

import requests

import config


def ask_yandexgpt(messages: list[dict], retries: int = 3) -> tuple[str, dict]:
    """Отправляет сообщения модели. Возвращает (текст ответа, статистика токенов)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {config.YC_API_KEY}",
        "x-folder-id": config.YC_FOLDER_ID,
    }
    body = {
        "modelUri": f"gpt://{config.YC_FOLDER_ID}/{config.YC_MODEL}",
        "completionOptions": {
            "stream": False,
            "temperature": config.TEMPERATURE,
            "maxTokens": str(config.MAX_TOKENS),
        },
        "messages": messages,
    }

    for attempt in range(1, retries + 1):
        response = requests.post(config.API_URL, headers=headers, json=body, timeout=60)
        if response.status_code == 200:
            result = response.json()["result"]
            text = result["alternatives"][0]["message"]["text"]
            return text, result.get("usage", {})
        if response.status_code in (429, 500, 503) and attempt < retries:
            wait = 2 ** attempt               # 2, 4, 8 секунд — пауза растёт
            print(f"  код {response.status_code}, повтор через {wait} c")
            time.sleep(wait)
            continue
        # остальные ошибки (401 — плохой ключ, 403 — нет роли, 400 — кривой запрос)
        raise RuntimeError(f"Ошибка API {response.status_code}: {response.text}")
    raise RuntimeError("Не удалось получить ответ")