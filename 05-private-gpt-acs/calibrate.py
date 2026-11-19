"""Подбор порога SCORE_THRESHOLD.

Запуск:  uv run calibrate.py
Печатает близость лучших чанков для вопросов «из базы» и «вне базы».
Порог ставим между ними — так отказ сработает до вызова LLM.
"""

from retriever import search

IN_BASE = [
    "На какой высоте устанавливают считыватель?",
    "Какая максимальная длина линии Wiegand?",
    "Сколько длится нормоконтроль?",
    "Как обозначать кабели СКУД?",
]
OUT_OF_BASE = [
    "Дай рецепт яблочного пирога",
    "Кто написал роман «Война и мир»?",
    "Какая погода будет завтра в Алматы?",
]


def top1(question: str) -> float:
    hits = search(question, k=3)
    print(f"  {hits[0].score:.3f}  {question}   ← {hits[0].label}")
    return hits[0].score


print("Вопросы из базы:")
inside = [top1(q) for q in IN_BASE]
print("Вопросы вне базы:")
outside = [top1(q) for q in OUT_OF_BASE]

low_in, high_out = min(inside), max(outside)
print(f"\nСамый слабый «из базы»: {low_in:.3f}; самый сильный «вне базы»: {high_out:.3f}")
if low_in > high_out:
    print(f"Рекомендуемый SCORE_THRESHOLD ≈ {(low_in + high_out) / 2:.2f} (середина зазора)")
else:
    print("Зазора нет: одним порогом не разделить. Оставьте порог ниже, а отказ поручите промпту.")