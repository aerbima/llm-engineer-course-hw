"""Шаг 4 ДЗ: веб-интерфейс на Gradio.

Запуск:  uv run app.py   → открыть http://127.0.0.1:7860
Интерфейс слушает только этот компьютер (127.0.0.1): снаружи его не видно.
"""

import config  # первым: выключает телеметрию до импорта gradio
import gradio as gr

import sessions
from memory import ChatMemory
from rag import RagAssistant

assistant = RagAssistant()  # одна модель на всё приложение

EXAMPLES = [
    "На какой высоте устанавливают считыватель?",          # по базе
    "Какая максимальная длина линии Wiegand?",              # конфликт с общими знаниями
    "А что делать, если нужно длиннее?",                    # уточнение: проверяет память
    "Дай рецепт яблочного пирога",                           # вне базы
]


def new_state(mode: str) -> dict:
    """Состояние одного пользователя: id журнала и память диалога."""
    sid = sessions.new_session_id()
    sessions.append(sid, {"t": "header", "model": config.LLM_MODEL, "embed": config.EMBED_MODEL,
                          "memory_mode": mode})
    return {"session_id": sid, "memory": ChatMemory(mode, config.MEMORY_K, config.MAX_HISTORY_TOKENS)}


def details_md(ans, memory: ChatMemory) -> str:
    """Правая панель: найденные чанки, время по этапам, состояние памяти."""
    t = ans.timings_ms
    lines = [f"**Искали по запросу:** {ans.search_query}", "", "| # | близость | источник |", "|---|---|---|"]
    lines += [f"| {i} | {h.score:.3f} | {h.label} |" for i, h in enumerate(ans.hits, start=1)]
    if ans.refused_by_threshold:
        lines += ["", f"⛔ Лучшая близость ниже порога {config.SCORE_THRESHOLD} — модель не вызывалась."]
    lines += ["", f"**Время, мс:** переписывание {t.get('condense', 0)} · поиск {t.get('retrieval', 0)} · "
                  f"LLM {t.get('llm', 0)} · всего {t.get('total', 0)}"]
    if ans.usage:
        lines.append(f"**Токены (сервер):** вход {ans.usage.get('input_tokens')} · выход {ans.usage.get('output_tokens')}")
    lines += ["", f"**Память:** режим {memory.mode}, сообщений в буфере {len(memory.messages)}, "
                  f"≈{memory.tokens()} ток., сводок {memory.summary_calls}"]
    if memory.summary:
        lines += ["", f"**Сводка:** {memory.summary}"]
    return "\n".join(lines)

# message — текст пользователя
# chat — история сообщений для отображения в Gradio
# top_k — сколько документов искать в векторной базе
# mode — режим памяти: none, window, summary_buffer 
# state — состояние текущей сессии 
def respond(message: str, chat: list, top_k: int, mode: str, state: dict | None):
    if not message.strip():
        # возврщает Tuple: пустую строку (чтобы очистить поле ввода), 
        # chat без изменений, пустой Markdown, 
        # state без изменений
        return "", chat, gr.update(), state
    # Если state ещё нет (новый пользователь) — создаём новый словарь с 
    # id сессии и памятью.
    # Если state уже есть (возврат к существующей сессии) — использу
    if state is None:
        state = new_state(mode)
    # Из словаря берётся объект памяти:
    memory: ChatMemory = state["memory"]
    memory.mode = mode                      # режим можно переключить на лету
    # Создаём объект RagAssistant, который умеет искать и вызывать LLM.
    # Возможна суммаризация истории с помощью LLM, переписывание уточняющих вопросов с помощью LLM, 
    # поиск документов в ChromaDB,
    #  сбор контекста и отправка промпта в LLM.
    ans = assistant.ask(message, memory, top_k=int(top_k), session_id=state["session_id"])
    # Формируем, что показать пользователю: 
    # текст ответа + источники (если LLM вызывалась)
    sources = "\n".join(f"[{i}] {h.label}" for i, h in enumerate(ans.hits, start=1))
    shown = ans.text if ans.refused_by_threshold else f"{ans.text}\n\n**Источники:**\n{sources}"
    # Добавляем в историю сообщений: сначала пользователь, потом ассистент.
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": shown}]
    # Функция возвращает кортеж из четырёх элементов
    # 1. пустая строка — чтобы очистить поле ввода
    # 2. chat — история сообщений для отображения в Gradio  
    # 3. Markdown с деталями поиска, времени и состояния памяти
    # 4. state — состояние текущей сессии (id журнала и память диалога)    
    return "", chat, details_md(ans, memory), state


def reset(mode: str):
    return [], "Новая сессия.", new_state(mode)

# demo - хранит всё окно
# создать чат
# создать textbox
# создать кнопку отправки
# создать кнопку очистки

# если нажали Отправить:
#     вызвать respond()

# если нажали Enter:
#     вызвать respond()

# если нажали Новая сессия:
#     вызвать reset()
with gr.Blocks(title="Private GPT · СКУД ТМ Sigur") as demo:
    # Отобразит текст сверху
    gr.Markdown("## Private GPT для проектировщика СКУД ТМ Sigur\nОтвечает только по базе знаний компании. "
                f"Модель: `{config.LLM_MODEL}`, эмбеддинги: `{config.EMBED_MODEL}` — всё локально.")
    # Всю структуру определяете вы (в коде вашей функции)
    state = gr.State(None)
    # Создаёт горизонтальную строку
    with gr.Row():
        # Создаёт вертикальную колонку (слева)- 
        # занять примерно 3 части пространства
        with gr.Column(scale=3):
            # Создаёт чат-окно (Gradio Chatbot) с высотой 520 пикселей и 
            # подписью "Диалог"
            # +---------------------------------------------------+
            # |                  Колонка (scale=3)                |
            # |                                                   |
            # |  +---------------------------------------------+  |
            # |  |                Диалог (Chatbot)             |  |
            # |  |                 высота 520px                |  |
            # |  |                                             |  |
            # |  |                                             |  |
            # |  +---------------------------------------------+  |
            # |  | Поле ввода: [ Вопрос по стандартам... ]     |  |
            # |  +---------------------------------------------+  |
            # +---------------------------------------------------+
            chat = gr.Chatbot(height=520, label="Диалог")
            box = gr.Textbox(placeholder="Вопрос по стандартам и регламентам…", label="Вопрос")
            # Создаёт горизонтальную строку с кнопками "Отправить" и "Новая сессия"
            with gr.Row():
                send = gr.Button("Отправить", variant="primary")
                clear = gr.Button("Новая сессия")
                # inputs=box — указывает, куда должен подставиться пример, 
                # когда пользователь кликнет по нему мышкой. В данном случае при клике текст примера 
                # автоматически скопируется в текстовое поле box.
            gr.Examples(EXAMPLES, inputs=box)
            # Правая панель - Занимает меньше места чем левая. Соотношение: 3 : 2
        with gr.Column(scale=2):
            # 1, 10 (диапазон) - Минимальное и максимальное значение, которое может выбрать пользователь. В данном случае можно выбрать число от 1 до 10 включительно.
            # value=config.TOP_K - Стартовое (значение по умолчанию), с которым запустится ползунок при открытии приложения. Оно берется из файла настроек (config).
            # Например, если там было записано TOP_K = 4, то ползунок сразу встанет на цифру 4
            # step=1 - Шаг изменения. Поскольку step=1, пользователь может двигать ползунок только по целым числам (1, 2, 3, 4 и т.д.), исключая дробные значения вроде 2.5.
            # label="Top-K (сколько чанков отдать модели)" -
            # Текстовая подпись к ползунку, которую видит пользователь на экране. В ней же сразу поясняется технический смысл: 
            # сколько именно кусочков текста (чанков/фрагментов документов) 
            # система найдет в базе и передаст языковой модели для формирования ответа.
            top_k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Top-K (сколько чанков отдать модели)")
            # Радиокнопки
            mode = gr.Radio(["none", "window", "summary_buffer"], value=config.MEMORY_MODE, label="Память")
            # gr.Markdown — это компонент Gradio, который умеет выводить текст с поддержкой разметки Markdown (можно делать жирный текст, списки, ссылки и т.д.). 
            # В данном случае он используется просто для вывода красивого текста.
            # Изначально: Здесь появятся найденные фрагменты...
            # Потом туда будет выводиться: details_md(ans, memory)
            details = gr.Markdown("Здесь появятся найденные фрагменты и время ответа.")

    # Читается почти как предложение: Когда нажмут кнопку send, вызови respond  и передай ей значения из [box, chat, top_k, mode, state], 
    # а затем обнови [box, chat, details, state] результатами.
    send.click(respond, [box, chat, top_k, mode, state], [box, chat, details, state])
    
    # Событие нажатия клавиши Enter в поле ввода (box) будет вызывать ту же функцию respond, что и кнопка send.
    box.submit(respond, [box, chat, top_k, mode, state], [box, chat, details, state])
    # Когда нажмут кнопку clear, вызови reset и передай ей [mode], а затем обнови [chat, details, state] результатами.
    clear.click(reset, [mode], [chat, details, state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)