"""Память диалога — то, что мы разбирали на занятии 11, на современном API LangChain.

Классы ConversationBufferWindowMemory и ConversationSummaryBufferMemory объявлены
устаревшими и вынесены в пакет langchain-classic. Здесь та же логика написана
на базовых сообщениях langchain_core — её легко повторить и в C#.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from prompts import SUMMARY_PROMPT


def as_lines(messages: list[BaseMessage]) -> str:
    """Сообщения → текст вида «Пользователь: … / Ассистент: …»."""
    names = {"human": "Пользователь", "ai": "Ассистент"}
    return "\n".join(f"{names.get(m.type, m.type)}: {m.content}" for m in messages)


class ChatMemory:
    """Хранит историю одного диалога и решает, что из неё попадёт в промпт.

    mode="none"            — модель не видит историю;
    mode="window"          — последние k пар «вопрос–ответ» (как BufferWindowMemory);
    mode="summary_buffer"  — свежие сообщения до max_tokens целиком, остальное — в сводке.
    """

    def __init__(self, mode: str = "summary_buffer", k: int = 3, max_tokens: int = 600):
        self.mode = mode
        self.k = k
        self.max_tokens = max_tokens
        self.messages: list[BaseMessage] = []   # буфер свежих сообщений
        self.summary: str = ""                  # сводка свёрнутых сообщений
        self.summary_calls = 0                  # сколько раз вызывали LLM для сводки

    def add_turn(self, question: str, answer: str, llm=None) -> list[BaseMessage]:
        """Добавить пару сообщений. Возвращает сообщения, которые свернули в сводку."""
        self.messages += [HumanMessage(question), AIMessage(answer)]
        if self.mode == "summary_buffer" and llm is not None:
            return self._prune(llm)
        return []

    def history_for_prompt(self) -> list[BaseMessage]:
        """Сообщения, которые пойдут в промпт после системного."""
        if self.mode == "none":
            return []
        if self.mode == "window":
            return self.messages[-self.k * 2:]   # k пар = 2k сообщений
        return list(self.messages)               # summary_buffer: буфер уже обрезан _prune

    def tokens(self) -> int:
        """Приблизительно: символы / 4 + служебные токены на сообщение."""
        return count_tokens_approximately(self.history_for_prompt()) + len(self.summary) // 4

    def has_history(self) -> bool:
        return bool(self.messages) and self.mode != "none"

    def _prune(self, llm) -> list[BaseMessage]:
        """Если буфер длиннее max_tokens — снять самые старые пары и дописать их в сводку."""
        pruned: list[BaseMessage] = []
        while len(self.messages) > 2 and count_tokens_approximately(self.messages) > self.max_tokens:
            pruned += self.messages[:2]          # снимаем пару целиком: вопрос + ответ
            self.messages = self.messages[2:]
        if pruned:
            prompt = SUMMARY_PROMPT.format(summary=self.summary or "(пусто)", lines=as_lines(pruned))
            self.summary = str(llm.invoke(prompt).content).strip()
            self.summary_calls += 1
        return pruned