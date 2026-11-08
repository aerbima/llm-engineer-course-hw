"""Скачивает несколько текстов из датасета russian-legal-ner и кладёт их в data/docs."""
from pathlib import Path

from datasets import load_dataset
print("Скрипт запущен!")
N_SAMPLES = 3                                  # сколько документов взять
OUT_DIR = Path("data/docs")

ds = load_dataset("TryDotAtwo/russian-legal-ner", split="test")
print(ds)                                      # покажет колонки и число строк

# Колонку с текстом ищем автоматически: строковое поле с самыми длинными значениями
first = ds[0]
text_field = max((k for k, v in first.items() if isinstance(v, str)),
                 key=lambda k: len(first[k]))
print("Колонка с текстом:", text_field)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for i in range(N_SAMPLES):
    text = ds[i][text_field]
    path = OUT_DIR / f"court_{i + 1}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"{path}: {len(text)} символов")