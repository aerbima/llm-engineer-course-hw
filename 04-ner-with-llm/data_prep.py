"""Подготовка текстов: очистка, проверка длины, разбиение на куски (chunking)."""
import re


def clean_text(text: str) -> str:
    """Убирает лишние пробелы и пустые строки, не трогая сами слова и цифры."""
    text = text.replace("\xa0", " ")          # неразрывный пробел -> обычный
    text = re.sub(r"[ \t]+", " ", text)       # несколько пробелов подряд -> один
    text = re.sub(r"\n{3,}", "\n\n", text)    # больше двух переводов строки -> два
    return text.strip()


def split_into_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    """Режет длинный текст на куски не длиннее max_chars.

    Старается резать по границе абзаца или предложения, а соседние куски
    перекрываются на overlap символов, чтобы реквизит не разрезался пополам.
    """
    if len(text) <= max_chars:
        return [text]                         # короткий документ — один кусок

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # ищем, где удобно разрезать: сначала абзац, потом конец предложения
            cut = text.rfind("\n\n", start, end)
            if cut <= start + max_chars // 2:
                cut = text.rfind(". ", start, end)
            if cut > start + max_chars // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap                 # шаг назад на overlap — перекрытие
    return chunks


# data_prep.py
def describe_lengths(docs: dict[str, str], max_chars: int, overlap: int) -> None:
   
    """Печатает таблицу: имя документа, длина, сколько кусков получится."""
    print(f"{'документ':<28}{'символов':>10}{'кусков':>8}")
    # заголовок таблицы: :<28 — выровнять влево в поле шириной 28 символов,
    # :>10 и :>8 — выровнять вправо (числа удобнее читать выровненными вправо)
    for name, text in docs.items():
        # docs — словарь «имя документа → текст»; .items() отдаёт пары (имя, текст)
        n = len(split_into_chunks(text, max_chars, overlap))
        # режем текст с тем же перекрытием и считаем, сколько кусков вышло (len списка)
        print(f"{name:<28}{len(text):>10}{n:>8}")