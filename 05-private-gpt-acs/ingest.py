"""Шаг 2 ДЗ: документы → чанки → эмбеддинги → ChromaDB.

Запуск:  uv run ingest.py           (пересобрать базу с нуля)
Читает:  data/docs/*.md, *.txt   и   data/raw/*.pdf
Пишет:   chroma_db/   (папка с векторной базой)
"""

import hashlib
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config
from llm import make_embeddings

# Делитель «по смыслу»: сначала пробует резать по пустой строке (абзац),
# потом по переводу строки, потом по точке, потом по пробелу.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Делитель Markdown по заголовкам: каждый раздел становится отдельным документом,
# а названия заголовков попадают в metadata (h1, h2, h3).
MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,  # заголовок остаётся и в тексте чанка
)


def load_markdown(path: Path) -> list[Document]:
    """Markdown/текст → список разделов с метаданными."""
    text = path.read_text(encoding="utf-8")
    sections = MD_SPLITTER.split_text(text)          # list[Document], у каждого metadata h1/h2/h3
    for doc in sections:
        doc.metadata["source"] = path.name
        doc.metadata["doc_title"] = doc.metadata.get("h1", path.stem)
        doc.metadata["section"] = doc.metadata.get("h3") or doc.metadata.get("h2") or "начало документа"
    return sections


def load_pdf(path: Path) -> list[Document]:
    """PDF → по одному документу на страницу (номер страницы — в metadata)."""
    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(path)).load()
    for doc in pages:
        doc.metadata = {
            "source": path.name,
            "doc_title": path.stem,
            "section": f"стр. {int(doc.metadata.get('page', 0)) + 1}",  # PyPDF считает страницы с 0
        }
    return pages


def load_all() -> list[Document]:
    """Собрать документы из обеих папок."""
    # docs должен быть списком (list), элементы которого — экземпляры класса Document ([Document]).
    docs: list[Document] = []
    # .glob("*") — метод поиска файлов по шаблону. Символ * означает «абсолютно всё» (любое имя и любое расширение) на 
    # первом уровне внутри этой папки. 
    # Метод возвращает генератор объектов Path для каждого найденного файла или подпапки.
    for path in sorted(config.DOCS_DIR.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.extend(load_markdown(path))
    for path in sorted(config.RAW_DIR.glob("*.pdf")):
        docs.extend(load_pdf(path))
    return docs


def to_chunks(docs: list[Document]) -> list[Document]:
    """Разрезать документы на чанки и добавить к каждому строку-«шапку».

    Шапка «[Документ | Раздел]» помогает поиску: короткий чанк «4.1. ... не более 50 м»
    без шапки не знает, что речь о СТП-СКУД-01 и кабельных линиях.
    """
    chunks = TEXT_SPLITTER.split_documents(docs)
    for i, chunk in enumerate(chunks):
        head = f"[{chunk.metadata['doc_title']} | {chunk.metadata['section']}]"
        chunk.page_content = f"{head}\n{chunk.page_content}"
        chunk.metadata["chunk_index"] = i
        # Чанк получает постоянный id: при повторной загрузке он перезапишется, а не задвоится
        chunk.metadata["chunk_id"] = hashlib.sha1(
            f"{chunk.metadata['source']}#{i}#{chunk.page_content[:50]}".encode("utf-8")
        ).hexdigest()[:16]
    return chunks


def open_store(embeddings=None) -> Chroma:
    """Открыть (или создать) коллекцию ChromaDB на диске с косинусной метрикой."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or make_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},  # близость = косинус угла
    )


def main() -> None:
    t0 = time.perf_counter()
    docs = load_all()
    chunks = to_chunks(docs)
    lengths = [len(c.page_content) for c in chunks]
    print(f"Документов-разделов: {len(docs)}, чанков: {len(chunks)}")
    print(f"Длина чанка, символов: мин {min(lengths)}, средн {sum(lengths) // len(lengths)}, макс {max(lengths)}")

    store = open_store()
    store.reset_collection()                       # пересобираем с нуля, чтобы не копить старые версии
    # Пишем пакетами: OllamaEmbeddings шлёт весь список одним запросом /api/embed,
    # и на ~1300 чанках runner Ollama падает («connection refused» на внутреннем порту).
    batch = 64
    for s in range(0, len(chunks), batch):
        part = chunks[s:s + batch]
        store.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
        print(f"  {min(s + batch, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nЗаписано в ChromaDB: {len(chunks)} чанков за {time.perf_counter() - t0:.1f} с → {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()