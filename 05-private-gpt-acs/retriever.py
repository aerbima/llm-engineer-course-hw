"""Семантический поиск: вопрос → эмбеддинг → Top-K ближайших чанков."""

from dataclasses import dataclass

from langchain_core.documents import Document

from ingest import open_store

_store = None  # открываем базу один раз и переиспользуем


@dataclass
class Hit:
    """Один найденный чанк и его близость к вопросу (0…1, больше — ближе)."""
    doc: Document
    score: float

    @property
    def label(self) -> str:
        """Подпись для ссылки на источник: «файл, раздел»."""
        return f"{self.doc.metadata['source']}, {self.doc.metadata['section']}"


def search(query: str, k: int) -> list[Hit]:
    """Вернуть до k чанков, отсортированных от самого близкого к самому далёкому."""
    global _store
    if _store is None:
        _store = open_store()
    pairs = _store.similarity_search_with_relevance_scores(query, k=k)
    return [Hit(doc=d, score=float(s)) for d, s in pairs]