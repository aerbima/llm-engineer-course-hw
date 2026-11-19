"""Шаг 3 ДЗ: RAG-цепочка «вопрос → поиск → промпт с контекстом → локальная LLM → ответ».

Можно запустить без интерфейса:  uv run rag.py "На какой высоте ставить считыватель?"
"""

import re
import sys
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

import config
import sessions
from llm import make_llm
from memory import ChatMemory, as_lines
from prompts import CONDENSE_PROMPT, NO_ANSWER, SYSTEM_PROMPT
from retriever import Hit, search

# @dataclass - создаёт автоматически конструктор
@dataclass
class Answer:
    """Всё, что нужно показать в интерфейсе и записать в отчёт."""
    text: str # ответ модели (или NO_ANSWER)
    search_query: str                         # уточнённый вопрос для поиска (может отличаться от исходного) 
    #- запрос, который реально ушёл в поиск.
    # Важно: Датакласс вызывает эту функцию list заново для каждого нового объекта, который вы создаете. 
    # В результате каждый новый объект получает свой собственный, абсолютно независимый список в памяти.
    # field(...) позволяет управлять тем, как это поле ведет себя при создании объекта, сравнении, выводе на печать и т.д.
    hits: list[Hit] = field(default_factory=list) # Список найденных чанков (может быть пустым)
    # Что это: Флаг (переменная состояния), который означает: «Запрос отклонен пороговым значением», SCORE_THRESHOLD.
    refused_by_threshold: bool = False        # отказ без вызова LLM
    # Замеры времени.
    timings_ms: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)  # токены по данным сервера модели


def format_context(hits: list[Hit]) -> str:
    """Чанки → нумерованный текст для блока <context>."""
    return "\n\n".join(f"[{i}] ({h.label})\n{h.doc.page_content}" for i, h in enumerate(hits, start=1))


def clean(text: str) -> str:
    """Убрать блок размышлений <think>…</think>, если модель его всё же вывела."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class RagAssistant:
    def __init__(self, llm=None):
        self.llm = llm or make_llm()

    
    def condense(self, question: str, memory: ChatMemory) -> str:
        """Уточняющий вопрос («а для него?») → самостоятельный вопрос для поиска.
        Это переписывание вопроса с учётом истории c помощью LLM. Если истории нет, 
        возвращаем вопрос без изменений."""
        history = as_lines(memory.history_for_prompt()[-4:])
        if memory.summary:
            history = f"(Ранее: {memory.summary})\n{history}"
        prompt = CONDENSE_PROMPT.format(history=history, question=question)
        return clean(str(self.llm.invoke(prompt).content)) or question

    def ask(self, question: str, memory: ChatMemory, top_k: int = config.TOP_K,
            session_id: str | None = None) -> Answer:
        t = {}
        t0 = time.perf_counter()

        # 1. Вопрос для поиска: с историей переписываем, без истории — как есть
        query = question
        if config.CONDENSE_QUESTION and memory.has_history():
            # уточняем вопрос с учётом истории, чтобы поиск понял, 
            # о чём речь c помощью LLM. Если истории нет, возвращаем вопрос без изменений.
            query = self.condense(question, memory)
        t["condense"] = round((time.perf_counter() - t0) * 1000)

        # 2. Семантический поиск Top-K
        t1 = time.perf_counter()
        hits = search(query, k=top_k)
        t["retrieval"] = round((time.perf_counter() - t1) * 1000)

        # 3. Ограничитель: лучший чанк слишком далёк → ответа в базе нет, LLM не зовём
        if not hits or hits[0].score < config.SCORE_THRESHOLD:
            t["llm"] = 0
            t["total"] = round((time.perf_counter() - t0) * 1000)
            ans = Answer(NO_ANSWER, query, hits, refused_by_threshold=True, timings_ms=t)
            self._remember(question, ans, memory, session_id)
            return ans

        # 4. Промпт: системный (правила + сводка + контекст) → история → вопрос
        system = SYSTEM_PROMPT.format(
            no_answer=NO_ANSWER,
            summary=memory.summary if memory.mode == "summary_buffer" else "",
            context=format_context(hits),
        )
        messages = [SystemMessage(system), *memory.history_for_prompt(), HumanMessage(question)]

        # 5. Генерация
        t2 = time.perf_counter()
        reply = self.llm.invoke(messages)
        t["llm"] = round((time.perf_counter() - t2) * 1000)
        t["total"] = round((time.perf_counter() - t0) * 1000)

        ans = Answer(clean(str(reply.content)), query, hits, timings_ms=t,
                     # getattr(объект, "имя_атрибута", значение_по_умолчанию) 
                     # пытается достать у объекта reply атрибут с именем "usage_metadata".
                     # Если getattr вернул None (потому что в ответе модели не оказалось метаданных), 
                     # то сработает or {}, и мы получим пустой словарь {}. 
                     # Если же там был реальный объект с данными, вернется он.
                     usage=dict(getattr(reply, "usage_metadata", None) or {}))
        self._remember(question, ans, memory, session_id)
        return ans

    def _remember(self, question: str, ans: Answer, memory: ChatMemory, session_id: str | None) -> None:
        """Записать ход в память и журнал. Сводка пересчитывается здесь же."""
        pruned = memory.add_turn(question, ans.text, llm=self.llm)
        if session_id:
            sessions.append(session_id, {
                "t": "turn", "question": question, "answer": ans.text, "search_query": ans.search_query,
                "sources": [{"label": h.label, "score": round(h.score, 3)} for h in ans.hits],
                "timings_ms": ans.timings_ms, "usage": ans.usage,
            })
            if pruned:
                sessions.append(session_id, {"t": "summary", "summary": memory.summary,
                                             "kept_messages": len(memory.messages)})

# Зачем нужно: Код внутри этого блока выполнится только тогда, когда вы запускаете именно этот скрипт. 
# Если другой файл сделает import script, этот код запускаться не будет.
if __name__ == "__main__":
    # sys.argv — это список аргументов, которые вы передали скрипту при 
    # запуске из консоли. Например, если вы напишете python script.py 
    # Какая погода сегодня?, то sys.argv будет содержать ['script.py', 'Какая', 
    # 'погода', 'сегодня?'].
    # sys.argv[1:] — отрезает первый элемент (имя самого файла скрипта) 
    # и оставляет только слова вопроса.
    # " ".join(...) — склеивает эти слова обратно в единую строку через пробел.
    # or "На какой высоте устанавливают считыватель?" — если вы 
    # запустили скрипт вообще без аргументов (просто python script.py), 
    # то sys.argv[1:] будет пустым. В этом случае сработает оператор or,
    #  и программа автоматически подставит вопрос-заглушку по умолчанию.
    q = " ".join(sys.argv[1:]) or "На какой высоте устанавливают считыватель?"
    a = RagAssistant().ask(q, ChatMemory(mode="none"))
    print("Ответ:", a.text)
    for i, h in enumerate(a.hits, start=1):
        print(f"  [{i}] {h.score:.3f}  {h.label}")
    print("Время, мс:", a.timings_ms)

# Пользователь
#       │
#       ▼
#   question
#       │
#       ▼
#   condense()
#   (если есть история)
#       │
#       ▼
#    query
#       │
#       ▼
#  embed_query()
#       │
#       ▼
#  ChromaDB Search
#       │
#       ▼
#     hits
#       │
#       ├── score низкий
#       │      ▼
#       │   NO_ANSWER
#       │
#       └── score хороший
#              ▼
#       format_context() - Сообщения, которые пойдут в промпт после системного
#              ▼
#       SYSTEM_PROMPT
#              ▼
#       LLM.invoke()
#              ▼
#       Answer
#              ▼
#       _remember()
#              ▼
#       JSONL + Memory
#              ▼
#       Пользователь