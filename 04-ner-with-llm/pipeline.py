"""Главный скрипт: читает документы, отправляет в YandexGPT, сохраняет JSON."""
import json
from pathlib import Path

import config
from data_prep import clean_text, describe_lengths, split_into_chunks
from llm_client import ask_yandexgpt
from llm_parser import check_grounding, parse_response
from prompts import build_messages

DOCS_DIR = Path("data/docs")
RESULTS_DIR = Path("results")


def merge_results(parts: list[dict]) -> dict:
    """Склеивает ответы по кускам одного документа в один результат."""
    merged = {"document_type": None, "contract_number": None, "parties": [],
              "dates": {"signing_date": None, "valid_from": None, "valid_until": None},
              "amounts": [], "obligation_terms": []}
    for part in parts:
        for key in ("document_type", "contract_number"):
            merged[key] = merged[key] or part.get(key)
        for key, value in (part.get("dates") or {}).items():
            merged["dates"][key] = merged["dates"].get(key) or value
        for key in ("parties", "amounts", "obligation_terms"):
            # part.get(key, []) подставляет [] только если ключа НЕТ совсем;
            # если модель вернула "parties": null, придёт None. "or []" заменяет None на [].
            for item in part.get(key) or []:
                if item not in merged[key]:      # без дублей из перекрытия кусков
                    merged[key].append(item)
    return merged


def process_document(name: str, text: str) -> dict:
    """Полный цикл для одного документа."""
    chunks = split_into_chunks(text, config.MAX_CHARS, config.CHUNK_OVERLAP)
    parts, errors, raw_answers, tokens = [], [], [], 0
    for i, chunk in enumerate(chunks, start=1):
        print(f"  кусок {i}/{len(chunks)}: {len(chunk)} символов")
        raw, usage = ask_yandexgpt(build_messages(chunk))
        raw_answers.append(raw)
        tokens += int(usage.get("totalTokens", 0))
        data, errs = parse_response(raw)
        errors += errs
        if data is not None:
            parts.append(data)
    result = merge_results(parts)
    return {
        "document": name,
        "chars": len(text),
        "chunks": len(chunks),
        "total_tokens": tokens,
        "entities": result,
        "errors": errors,
        "warnings": check_grounding(result, text),
        "raw_answers": raw_answers,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)       # создаём папку results, если её нет
    # читаем все документы из папки data/docs
    docs = {p.stem: clean_text(p.read_text(encoding="utf-8"))
            # p.stem — имя файла без расширения, p.read_text() — читаем текст из файла
            # p — это Path-объект, который мы получили из DOCS_DIR.glob("*.txt") — 
            # список всех .txt в папке data/docs
            for p in sorted(DOCS_DIR.glob("*.txt"))}
    #
    describe_lengths(docs, config.MAX_CHARS, config.CHUNK_OVERLAP)  # печатаем таблицу длин и кусков
    for name, text in docs.items():
        print(f"\n=== {name}")
        if not text:                    # пустая строка "" считается «ложью», значит not "" == True
            print("  пустой файл — пропускаю")
            continue                    # не тратим запрос к API, переходим к следующему документу
        report = process_document(name, text)
        out = RESULTS_DIR / f"{name}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  токенов: {report['total_tokens']}, ошибок: {len(report['errors'])}, "
              f"предупреждений: {len(report['warnings'])} -> {out}")


if __name__ == "__main__":
    main()