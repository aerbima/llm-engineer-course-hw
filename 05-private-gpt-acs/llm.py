"""Фабрики модели и эмбеддингов.

Весь остальной код получает LLM только через make_llm(), а эмбеддинги — через
make_embeddings(). Чтобы сменить провайдера, достаточно поменять .env.
"""

import config


def make_llm(temperature: float | None = None):
    """Вернуть объект чат-модели LangChain с методом .invoke(messages)."""
    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temp,
            num_ctx=config.NUM_CTX,  # без этого Ollama может взять маленькое окно и молча обрезать промпт
            reasoning=False,         # отключаем режим «размышлений» у моделей Qwen: быстрее и без <think>
        )
    if config.LLM_PROVIDER == "openai_compat":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_MODEL,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            temperature=temp,
        )
    raise ValueError(f"Неизвестный LLM_PROVIDER: {config.LLM_PROVIDER}")


def make_embeddings():
    """Вернуть объект эмбеддингов с методами embed_documents() и embed_query()."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)