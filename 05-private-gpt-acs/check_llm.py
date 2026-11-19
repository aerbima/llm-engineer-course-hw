"""Шаг 1 ДЗ: модель отвечает локально.

Запуск:  uv run check_llm.py
Проверка изоляции: выключите Wi-Fi / кабель и запустите ещё раз — ответ должен прийти.
"""

import time

from langchain_core.messages import HumanMessage

import config
from llm import make_embeddings, make_llm

print(f"Провайдер: {config.LLM_PROVIDER}, модель: {config.LLM_MODEL}, адрес: {config.OLLAMA_BASE_URL}")

llm = make_llm()
start = time.perf_counter()                                   # засекаем время
reply = llm.invoke([HumanMessage("Ответь одним предложением: что такое СКУД?")])
seconds = time.perf_counter() - start

print("Ответ модели:", reply.content)
print(f"Время ответа: {seconds:.1f} с")
print("Токены (если сервер их сообщил):", reply.usage_metadata)

emb = make_embeddings()
vector = emb.embed_query("считыватель на высоте 1000 мм")
print(f"Эмбеддинг: модель {config.EMBED_MODEL}, длина вектора {len(vector)}, первые числа {vector[:3]}")