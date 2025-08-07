
"""
Рекомендуемый метод: реструктуризация цепочки для сохранения состояний

Идея состоит в том, чтобы на каждом шаге не заменять результат, а добавлять его в проходящий через цепочку словарь.

Сначала исправим опечатки в именах переменных для ясности (gem -> ger, span -> spa).
"""


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter # <-- Важный импорт для извлечения данных

# --- Инициализация модели (без изменений) ---
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    base_url="https://api.vsegpt.ru/v1",
    api_key="sk-or-vv-17bd53f8f505e0a1d24a3bf0a8bb702e13edbc78c89ac9aa41f6bda7ec72270c",
)

# --- Определяем все промпты и парсер (для ясности) ---
prompt_eng_to_ger = ChatPromptTemplate.from_messages([
    ("system", "Translate English to German."),
    ("human", "{input}"),
])

prompt_ger_to_rus = ChatPromptTemplate.from_messages([
    ("system", "Translate German to Russian."),
    ("human", "{german_text}"), # <-- Ожидает ключ 'german_text'
])

prompt_rus_to_spa = ChatPromptTemplate.from_messages([
    ("system", "Translate Russian to Spanish."),
    ("human", "{russian_text}"), # <-- Ожидает ключ 'russian_text'
])

output_parser = StrOutputParser()

# --- Новая цепочка с сохранением промежуточных результатов ---

# 1. Создаем "ветку" для первого перевода
# Она возьмет исходный 'input' и вернет немецкий текст
eng_to_ger_chain = prompt_eng_to_ger | llm | output_parser

# 2. Создаем "ветку" для второго перевода
# Она ожидает на вход словарь с ключом 'german_text'
ger_to_rus_chain = prompt_ger_to_rus | llm | output_parser

# 3. Создаем "ветку" для третьего перевода
# Она ожидает на вход словарь с ключом 'russian_text'
rus_to_spa_chain = prompt_rus_to_spa | llm | output_parser


# --- Собираем всё вместе ---
full_chain = RunnablePassthrough.assign(
    german_text = eng_to_ger_chain
).assign(
    russian_text = lambda x: {"german_text": x["german_text"]} | ger_to_rus_chain
).assign(
    spanish_text = lambda x: {"russian_text": x["russian_text"]} | rus_to_spa_chain
)


# --- Выполняем цепочку ---
result = full_chain.invoke({
    "input": "I love programming."
})

print(result)

### Что делает каждая строка в новой цепочке?
"""
1.  **`full_chain = RunnablePassthrough.assign(...)`**
    *   **`RunnablePassthrough`**: Этот компонент "пробрасывает" свой вход без изменений. В данном случае он возьмет исходный словарь `{"input": "..."}`.
    *   **`.assign(german_text = eng_to_ger_chain)`**: Это ключевой шаг. Он выполняет `eng_to_ger_chain`, используя данные из проброшенного словаря (т.е. `{input}`), и **добавляет результат** в этот словарь под новым ключом `german_text`.
    *   На выходе этого шага мы получим словарь: `{'input': 'I love programming.', 'german_text': 'Ich liebe Programmieren.'}`.

2.  **`.assign(russian_text = ...)`**
    *   Этот `.assign` получает на вход результат предыдущего шага (словарь с `input` и `german_text`).
    *   **`lambda x: {"german_text": x["german_text"]}`**: Мы используем лямбда-функцию, чтобы извлечь значение по ключу `german_text` из входящего словаря `x` и передать его в нужном формате (`{"german_text": ...}`) на следующий шаг.
    *   **`| ger_to_rus_chain`**: Цепочка второго перевода выполняется с этим извлеченным значением.
    *   Результат (русский текст) присваивается новому ключу `russian_text`.
    *   На выходе этого шага мы получим словарь: `{'input': '...', 'german_text': '...', 'russian_text': 'Я люблю программировать.'}`.

3.  **`.assign(spanish_text = ...)`**
    *   Работает аналогично предыдущему шагу, но теперь извлекает `'russian_text'`, передает его в `rus_to_spa_chain` и сохраняет результат в `'spanish_text'`.

### Результат выполнения

Когда вы выполните `print(result)`, вы увидите не просто финальный перевод, а полный словарь со всеми промежуточными результатами:


{
'input': 'I love programming.',
'german_text': 'Ich liebe Programmieren.',
'russian_text': 'Я люблю программировать.',
'spanish_text': 'Me encanta programar.'
}

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
Теперь вы можете легко получить доступ к любому промежуточному результату, например, `result['russian_text']`.

### Альтернатива для отладки: Использование Callbacks

Если вам не нужно программно использовать промежуточный результат, а просто хочется посмотреть на него во время выполнения для отладки, можно использовать систему колбэков. Это не меняет структуру цепочки.

python
"""
from langchain.callbacks import StdOutCallbackHandler

# Ваша исходная, простая цепочка
simple_chain = (
    prompt_eng_to_ger
    | llm
    | output_parser
    | prompt_ger_to_rus
    | llm
    | output_parser
    | prompt_rus_to_spa
    | llm
    | output_parser
)

# Создаем обработчик, который будет печатать все в консоль
handler = StdOutCallbackHandler()

# Вызываем цепочку, передавая колбэк в конфигурации
final_output = simple_chain.invoke(
    {"input": "I love programming."},
    config={"callbacks": [handler]} # <-- Передаем обработчик здесь
)

print("\n--- Финальный результат ---")
print(final_output)

# При выполнении этого кода вы увидите в консоли подробный, цветной лог каждого шага: что вошло в каждый промпт, что вышло из каждой LLM и т.д. Это невероятно полезно для отладки, но не позволяет сохранить результат в переменную.