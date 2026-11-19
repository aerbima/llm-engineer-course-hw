"""Шаг 4 ДЗ: анализ слабых мест и влияние Top-K на качество и время.

Запуск:  uv run experiments.py
Пишет:   results/experiments.json и results/experiments.md (таблицы для README)
"""

import json
from statistics import mean

import config
from memory import ChatMemory
from prompts import NO_ANSWER
from rag import RagAssistant

# kind: base — ответ есть в базе; out — ответа нет; conflict — база расходится с общими знаниями
TESTS = [
    {"kind": "base", "q": "На какой высоте устанавливают считыватель?", "expect": ["1000"]},
    {"kind": "base", "q": "Сколько длится нормоконтроль?", "expect": ["2 рабоч"]},
    {"kind": "base", "q": "Сколько часов должна работать СКУД от аккумуляторов?", "expect": ["4 час", "4 ч"]},
    {"kind": "out", "q": "Дай рецепт яблочного пирога", "expect": None},
    {"kind": "out", "q": "Кто написал роман «Война и мир»?", "expect": None},
    {"kind": "out", "q": "Какой бюджет у проекта АБ-2026-014?", "expect": None},
    {"kind": "conflict", "q": "Какая максимальная длина линии Wiegand?", "expect": ["50 м"]},
    {"kind": "conflict", "q": "Какой длины может быть сегмент RS-485 без повторителя?", "expect": ["600"]},
]
# Диалог из двух вопросов: второй понятен только с памятью
DIALOG = ["Какая максимальная длина линии Wiegand?", "А что делать, если считыватель дальше?"]
DIALOG_EXPECT = ["RS-485"]

TOP_K_VALUES = [1, 2, 4, 8]

# text — ответ модели
# expect — список подстрок, которые должны быть в ответе (если None — ждём отказ)
def passed(text: str, expect: list[str] | None) -> bool:
    """Автопроверка: для «вне базы» ждём отказ, иначе — одну из ожидаемых подстрок."""
    if expect is None:
        return NO_ANSWER in text
    return NO_ANSWER not in text and any(e.lower() in text.lower() for e in expect)


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    bot = RagAssistant()
    rows = []

    for k in TOP_K_VALUES:
        for test in TESTS:
            ans = bot.ask(test["q"], ChatMemory(mode="none"), top_k=k)
            rows.append({
                "top_k": k, "kind": test["kind"], "question": test["q"], "answer": ans.text,
                "ok": passed(ans.text, test["expect"]), "refused_by_threshold": ans.refused_by_threshold,
                "best_score": round(ans.hits[0].score, 3) if ans.hits else None,
                "llm_ms": ans.timings_ms["llm"], "total_ms": ans.timings_ms["total"],
                "input_tokens": ans.usage.get("input_tokens"),
            })
            print(f"k={k} {test['kind']:8} ok={rows[-1]['ok']!s:5} {ans.timings_ms['total']:>6} мс  {test['q']}")
            # print("QUERY:", test["q"])
            # for i, h in enumerate(ans.hits, start=1):
            #     print(f"\n--- HIT {i} score={h.score:.3f} ---")
            #     print(h.doc.page_content[:1000])

    dialog = []
    for mode in ["none", "window", "summary_buffer"]:
        memory = ChatMemory(mode=mode, k=config.MEMORY_K, max_tokens=config.MAX_HISTORY_TOKENS)
        bot.ask(DIALOG[0], memory)
        second = bot.ask(DIALOG[1], memory)
        dialog.append({"memory": mode, "search_query": second.search_query, "answer": second.text,
                       "ok": passed(second.text, DIALOG_EXPECT)})
        print(f"диалог, память={mode}: ok={dialog[-1]['ok']}  поиск: {second.search_query}")

    (config.RESULTS_DIR / "experiments.json").write_text(
        json.dumps({"model": config.LLM_MODEL, "embed": config.EMBED_MODEL, "rows": rows, "dialog": dialog},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"Модель: `{config.LLM_MODEL}`, эмбеддинги: `{config.EMBED_MODEL}`, порог {config.SCORE_THRESHOLD}", "",
          "| Top-K | верно, из 8 | среднее время, мс | среднее LLM, мс | вход, токенов |", "|---|---|---|---|---|"]
    for k in TOP_K_VALUES:
        part = [r for r in rows if r["top_k"] == k]
        tokens = [r["input_tokens"] for r in part if r["input_tokens"]]
        # mean обычно означает среднее арифметическое
        tok = f"{mean(tokens):.0f}" if tokens else "—"      # провайдер мог не сообщить токены
        md.append(f"| {k} | {sum(r['ok'] for r in part)} | {mean(r['total_ms'] for r in part):.0f} | "
                  f"{mean(r['llm_ms'] for r in part):.0f} | {tok} |")
    md += ["", "| Память | чем искали второй вопрос | верно |", "|---|---|---|"]
    md += [f"| {d['memory']} | {d['search_query']} | {'да' if d['ok'] else 'нет'} |" for d in dialog]
    (config.RESULTS_DIR / "experiments.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()