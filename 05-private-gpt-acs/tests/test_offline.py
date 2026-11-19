"""Проверка логики без Ollama: модель и эмбеддинги подменены простыми заглушками.

Запуск:  uv run pytest -q
Тест не проверяет качество ответов — только то, что части пайплайна правильно связаны.
"""

import hashlib
import math
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # чтобы видеть модули проекта
import config  # noqa: E402
import llm  # noqa: E402


class HashEmbeddings(Embeddings):
    """Заглушка: вектор из хешей трёхбуквенных сочетаний. Похожие тексты → похожие векторы."""
    # возвращает нормализованный числовой отпечаток текста длиной 256 элементов, 
    # который потом используется для поиска похожих текстов через косинусное сходство
    def _vec(self, text: str) -> list[float]:
        # 256-мерный вектор, где каждая координата — количество вхождений хешей трёхбуквенных сочетаний
        # создаётся массив из 256 нулей. Это будущий вектор признаков текста
        v = [0.0] * 256
        t = text.lower()
        # range(len(t) - 2) - чтобы последние два символа не становились началом триграммы
        # потому что берётся окно шириной 3 символа - t[i:i+3]
        # То есть всегда: range(len(строки) - размер_окна + 1), 
        # В моём коде размер окна равен 3 - range(len(t) - 3 + 1), что упрощается до: range(len(t) - 2)
        for i in range(len(t) - 2):
            # hashlib.md5(t[i:i + 3].encode()).hexdigest() - берём три символа, кодируем в байты, считаем md5-хеш,
            # hexdigest() - возвращает строку из 32 шестнадцатеричных символов. Например, 'e4d909c290d0fb1ca068ffaddf22cbd0'
            # int(..., 16) - преобразует шестнадцатеричную строк    у в целое число. Например, int('e4d909c290d0fb1ca068ffaddf22cbd0', 16) = 302668455
            # % 256 - берём остаток от деления на 256, чтобы получить индекс в диапазоне 0-255. Например, 302668455 % 256 = 55  
            # прибавить 1 к элементу вектора с индексом, равным хешу триграммы по модулю 256, напрмер, v[55] += 1.0
            v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % 256] += 1.0
        # Если текст пустой: v = [0,0,0,...], то длина: n = 0. Делить на ноль нельзя. Поэтому: 0 or 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        # Нормализация - каждый элемент делится на длину вектора
        return [x / n for x in v]

    # векторизация текста в HashEmbeddings используется для поиска похожих документов в базе данных.
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    # преобразуем запрос пользователя в вектор признаков, чтобы искать похожие документы в базе
    def embed_query(self, text):
        return self._vec(text)


class EchoLLM:
    """Заглушка LLM: переписанный вопрос, сводка или первый пункт из контекста."""

    def invoke(self, prompt):
        # если prompt — строка, то используем её, иначе берём первый элемент списка и его content
        # prompt может быть строкой или списком сообщений (например, HumanMessage, AIMessage)
        text = prompt if isinstance(prompt, str) else prompt[0].content
        # если в тексте есть "Самостоятельный вопрос:", то возвращаем AIMessage с уточнённым вопросом
        if "Самостоятельный вопрос:" in text:
            return AIMessage("Что делать, если длина линии Wiegand больше 50 м?")
        # если в тексте есть "Новое краткое содержание:", то возвращаем AIMessage с краткой сводкой 
        if "Новое краткое содержание:" in text:
            return AIMessage("Обсуждали длину линии Wiegand: 50 м.")
        # иначе берём текст после <context>, убираем лишние пробелы и переносы строк,
        # берём первые 200 символов и добавляем " [1]" в конце,
        # возвращаем AIMessage с этим текстом
        ctx = text.split("<context>")[-1] # Берётся последний элемент: текст после <context>
        return AIMessage(ctx.strip().splitlines()[2][:200] + " [1]")

 # Один раз на файл (scope="module"): Система подготовит окружение (создаст бота) только один раз перед самым первым тестом в этом .py-файле.
 #  Все остальные 9 тестов в этом файле будут использовать уже созданный экземпляр. Это сильно экономит время при запуске.
@pytest.fixture(scope="module") # Создай этот объект один раз для всего файла с тестами.
# Возвращает объект RagAssistant с подменёнными LLM и эмбеддингами.
# tmp_path_factory - встроенная fixture PyTest. Она умеет создавать временные папки
# tmp_path_factory.mktemp("chroma") - создаёт временную папку с именем "chroma" и возвращает объект Path
def bot(tmp_path_factory):
    config.CHROMA_DIR = tmp_path_factory.mktemp("chroma")
    config.SESSIONS_DIR = tmp_path_factory.mktemp("sessions")
    config.SCORE_THRESHOLD = 0.2
    # Подменяем LLM и эмбеддинги на заглушки, чтобы тесты не зависели от внешних сервисов.
    llm.make_embeddings = lambda: HashEmbeddings()
    import ingest
    # Подменяем функцию make_embeddings в модуле ingest на ту же заглушку HashEmbeddings, чтобы при загрузке документов использовались эти эмбеддинги.
    ingest.make_embeddings = llm.make_embeddings
    # Запускается индексация.
    ingest.main()
    from rag import RagAssistant
    # Возвращаем объект RagAssistant, который использует EchoLLM и HashEmbeddings вместо реальных моделей.
    return RagAssistant(llm=EchoLLM())

# тесты проверяют, что поиск и память работают корректно, а не качество ответов.
# Это тест, который проверяет:
# что документы вообще разбились на чанки;
# что у каждого чанка есть нужные метаданные;
# что чанки не превышают допустимый размер.
def test_chunks_have_metadata():
    import ingest
    # Загружаем документы и разрезаем их на чанки. Проверяем, что у каждого чанка есть метаданные: source и chunk_id.
    chunks = ingest.to_chunks(ingest.load_all())
    # Проверяем, что чанков больше 10, у каждого чанка есть source и chunk_id, и длина контента не превышает config.CHUNK_SIZE + 120 (учитывая шапку).
    assert len(chunks) > 10
    # Проверяем, что у каждого чанка есть source и chunk_id в метаданных.
    assert all(c.metadata["source"] and "chunk_id" in c.metadata for c in chunks)
    # Проверяем, что длина контента каждого чанка не превышает config.CHUNK_SIZE + 120 (учитывая шапку).
    assert max(len(c.page_content) for c in chunks) <= config.CHUNK_SIZE + 120   # +шапка

# Этот тест проверяет две вещи:
# Поиск действительно находит документ про Wiegand.
# Результаты отсортированы по релевантности (score).
# Но обратите внимание: внутри теста bot вообще не используется.
# Он нужен только для того, чтобы перед запуском теста PyTest выполнил fixture и подготовил:
# индекс ChromaDB с документами, чтобы поиск работал.
# документы; эмбеддинги; индекс.
def test_search_finds_wiegand(bot):
    from retriever import search
    # ищем по запросу "длина линии Wiegand", берём 3 лучших результата.
    hits = search("длина линии Wiegand", k=3)
    # Проверяем, что среди найденных документов есть хотя бы один, который содержит "4.1" или "Wiegand" в контенте.
    assert any("4.1" in h.doc.page_content or "Wiegand" in h.doc.page_content for h in hits)
    # Проверяем, что результаты отсортированы по убыванию score (чем выше score, тем релевантнее).
    assert hits == sorted(hits, key=lambda h: -h.score)


# Этот тест проверяет, что память диалога работает корректно в режимах "window" и "summary_buffer".
def test_memory_window_and_summary():
    from memory import ChatMemory
    m = ChatMemory(mode="window", k=2)
    for i in range(5):
        # добавляем пары "вопрос-ответ" в память. Память хранит только последние 2 пары.
        m.add_turn(f"вопрос {i}", f"ответ {i}")
    assert [x.content for x in m.history_for_prompt()] == ["вопрос 3", "ответ 3", "вопрос 4", "ответ 4"]
    s = ChatMemory(mode="summary_buffer", max_tokens=40)
    for i in range(4):
        # добавляем пары "вопрос-ответ" в память. Память хранит свежие сообщения до max_tokens, остальное сворачивается в сводку.
        # "длинный вопрос " * 5 создаёт строку "длинный вопрос длинный вопрос длинный вопрос длинный вопрос длинный вопрос"
        # llm=EchoLLM() - используем заглушку LLM, которая умеет сворачивать историю в сводку.
        # max_tokens = 40 - ограничение на количество токенов в буфере. Если буфер превышает это значение, старые сообщения сворачиваются в сводку.
        s.add_turn("длинный вопрос " * 5, "длинный ответ " * 5, llm=EchoLLM())
        # Проверяем, что после добавления 4 пар сообщений, сводка не пустая, количество вызовов LLM для сводки >= 1, и в буфере осталось не более 4 сообщений.
        # после создания summary старые сообщения должны удаляться из буфера.
    assert s.summary and s.summary_calls >= 1 and len(s.messages) <= 4

# проверить переписывание вопроса и восстановление диалога
def test_dialog_condense_and_resume(bot):
    import sessions
    from memory import ChatMemory
    sid = sessions.new_session_id()
    mem = ChatMemory(mode="summary_buffer", max_tokens=600)
    # добавляем в память пару сообщений, чтобы проверить, что история сохраняется и восстанавливается.
    bot.ask("Какая максимальная длина линии Wiegand?", mem, top_k=3, session_id=sid)
    # добавляем второй вопрос, который зависит от первого. Проверяем, что вопрос переписан с учётом истории.
    second = bot.ask("А что делать, если дальше?", mem, top_k=3, session_id=sid)
    # Проверяем, что в переписанном вопросе есть слово "Wiegand", что значит, что модель учла предыдущий контекст.
    assert "Wiegand" in second.search_query            # вопрос переписан с учётом истории
    # Проверяем, что в истории памяти осталось 4 сообщения (2 пары "вопрос-ответ").
    restored = ChatMemory(mode="summary_buffer")
    assert sessions.resume(sid, restored) == 2
    assert len(restored.messages) == len(mem.messages)