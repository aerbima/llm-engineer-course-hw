"""Журнал диалога в JSONL: одна строка — одно событие, файл только дописывается.

Тот же формат, что в плане AI-агента (раздел 14.2), только упрощённый.
По журналу можно продолжить разговор после перезапуска (resume).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from memory import ChatMemory


def new_session_id() -> str:
    """Имя без двоеточий (Windows их не любит): s-20261109-143005-ab12."""
    return f"s-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


def _path(session_id: str) -> Path:
    config.SESSIONS_DIR.mkdir(exist_ok=True)
    return config.SESSIONS_DIR / f"{session_id}.jsonl"


def append(session_id: str, event: dict) -> None:
    """Дописать одно событие в конец файла."""
    # event — словарь с ключами:
    # **event - раскрывает все пары ключ-значение, а после добавляет поле "utc"
    event = {**event, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # event["utc"] — время события в UTC, ISO 8601, без миллисекунд
    # Режим "a" означает: append = дописывать в конец файла
    with _path(session_id).open("a", encoding="utf-8") as f:
        # json.dumps(..., ensure_ascii=False) — чтобы русские буквы не превращались в \u041f\u0440\u0438\u0432\u0435\u0442
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def resume(session_id: str, memory: ChatMemory) -> int:
    """Восстановить память из журнала. Возвращает число восстановленных пар."""
    turns = 0
    for line in _path(session_id).read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue                      # оборванная последняя строка — пропускаем
        if ev.get("t") == "turn":
            # добавляем пару сообщений в память. Если в журнале есть поле "answer", значит, это пара "вопрос-ответ".
            memory.add_turn(ev["question"], ev["answer"])   # без LLM: сводку возьмём из журнала
            turns += 1
            # если в журнале есть поле "summary", значит, это событие сводки. Восстанавливаем сводку и буфер сообщений.
        elif ev.get("t") == "summary":
            memory.summary = ev["summary"]
            # если в журнале есть поле "kept_messages", значит, это событие сводки. Восстанавливаем буфер сообщений.
            # kept_messages — число сообщений, которые остались в буфере после сводки. Старые сообщения удаляются.
            # [-ev["kept_messages"]:] — берём последние ev["kept_messages"] сообщений из буфера. Если их нет, буфер очищается.
            # Если ev["kept_messages"] = 0, значит, буфер пустой. Если ev["kept_messages"] = 4, значит, в буфере осталось 4 сообщения (2 пары "вопрос-ответ").
            # Остальные сообщения удаляются из буфера.
            memory.messages = memory.messages[-ev["kept_messages"]:] if ev["kept_messages"] else []
    return turns