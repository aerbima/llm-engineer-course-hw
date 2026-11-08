"""Разбор ответа модели: очистка от Markdown, json.loads, проверка структуры."""
import json
import re
from typing import Optional

# BaseModel - класс из pydantic, который позволяет описывать 
# структуру данных и проверять её, выдавать понятные ошибки
from pydantic import BaseModel, ValidationError, field_validator


# ---------- 1. Описание ожидаемой структуры (схема) ----------

class Party(BaseModel):
    role: Optional[str] = None
    name: Optional[str] = None
    inn: Optional[str] = None
    kpp: Optional[str] = None

    #эту функцию вызывать для проверки поля inn.
    @field_validator("inn")
    @classmethod
    # метод относится ко всему классу, а не к конкретному объекту.
    def check_inn(cls, v):
        if v is not None and not re.fullmatch(r"\d{10}|\d{12}", v):
            raise ValueError(f"ИНН должен состоять из 10 или 12 цифр, получено: {v!r}")
        return v

    @field_validator("kpp")
    @classmethod
    def check_kpp(cls, v):
        if v is not None and not re.fullmatch(r"\d{9}", v):
            raise ValueError(f"КПП должен состоять из 9 цифр, получено: {v!r}")
        return v


class Dates(BaseModel):
    signing_date: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class Amount(BaseModel):
    value: Optional[float] = None
    currency: Optional[str] = None
    source_text: Optional[str] = None
    purpose: Optional[str] = None


class Term(BaseModel):
    obligation: Optional[str] = None
    term: Optional[str] = None
    deadline_date: Optional[str] = None


class Extraction(BaseModel):
    document_type: Optional[str] = None
    contract_number: Optional[str] = None
    parties: list[Party] = []
    dates: Dates = Dates()
    amounts: list[Amount] = []
    obligation_terms: list[Term] = []


# ---------- 2. Очистка и разбор ----------

def clean_json_response(raw: str) -> str:
    """Достаёт JSON из ответа: убирает ```json ... ``` и текст вокруг фигурных скобок."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        text = text[first:last + 1]
    return text


def parse_response(raw: str) -> tuple[Optional[dict], list[str]]:
    """Возвращает (словарь или None, список ошибок)."""
    errors = []
    cleaned = clean_json_response(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, [f"Невалидный JSON: {e}"]
    try:
        model = Extraction.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            place = ".".join(str(p) for p in err["loc"])
            errors.append(f"Схема: {place}: {err['msg']}")
        return data, errors
    return model.model_dump(), errors


# ---------- 3. Проверка «опоры на текст» (ловим галлюцинации) ----------
# sample: re.sub(r"\D", "", "ИНН: 123-456-789")
# return -> "123456789"
def only_digits(s: str) -> str:
    return re.sub(r"\D", "", s)

# Задача функции check_grounding() именно в том, чтобы проверить:
# Все ли важные данные, которые извлекла модель, 
# действительно присутствуют в исходном документе?
def check_grounding(result: dict, source: str) -> list[str]:
    """Проверяет, что ИНН, КПП и суммы реально встречаются в исходном тексте."""
    warnings = []
    src_digits = only_digits(source)
    # p = {"name": "ООО Test", "inn": "7701234567", 
    # "kpp": "770101001"} in 
    # list[dict]
    for p in result.get("parties", []):
        for key in ("inn", "kpp"):
            v = p.get(key)
            if v and v not in src_digits:
                warnings.append(f"{key.upper()} {v} ({p.get('name')}) нет в тексте — возможна галлюцинация")
    for a in result.get("amounts", []):
        st = a.get("source_text")
        if st and only_digits(st) and only_digits(st) not in src_digits:
            warnings.append(f"Сумма «{st}» не найдена в тексте")
    return warnings