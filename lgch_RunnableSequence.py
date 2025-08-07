# --- Импорты ---
# Импортируем RunnableSequence и RunnableLambda для более явного кода
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import getpass

# --- Настройка окружения и LLM (без изменений) ---
if not os.environ.get("API_KEY"):
  os.environ["API_KEY"] = getpass.getpass("Enter API key for LLM: ")

if not os.environ.get("BASE_URL"):
  os.environ["BASE_URL"] = getpass.getpass("Enter BASE_URL for LLM: ")

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# --- Определяем все промпты и парсер (без изменений) ---
prompt_eng_to_ger = ChatPromptTemplate.from_messages([
    ("system", "Translate English to German."),
    ("human", "{input}"),
])

prompt_ger_to_rus = ChatPromptTemplate.from_messages([
    ("system", "Translate German to Russian."),
    ("human", "{german_text}"), # Ожидает ключ 'german_text'
])

prompt_rus_to_spa = ChatPromptTemplate.from_messages([
    ("system", "Translate Russian to Spanish."),
    ("human", "{russian_text}"), # Ожидает ключ 'russian_text'
])

output_parser = StrOutputParser()

# --- Создаем базовые цепочки для каждого шага (без изменений) ---
eng_to_ger_chain = prompt_eng_to_ger | llm | output_parser
ger_to_rus_chain = prompt_ger_to_rus | llm | output_parser
rus_to_spa_chain = prompt_rus_to_spa | llm | output_parser

# --- Собираем всё вместе, используя RunnableSequence ---

# RunnableSequence выполняет шаги последовательно.
# Вывод шага N становится вводом для шага N+1.

# Проблема: eng_to_ger_chain выводит строку, а ger_to_rus_chain ожидает на вход
# словарь вида {"german_text": "..."}.
# Решение: мы добавляем короткую lambda-функцию (которая автоматически
# преобразуется в RunnableLambda), чтобы "обернуть" строку в нужный словарь.

save_history = True  # Установите True, если хотите сохранить историю запросов
assert isinstance(save_history, bool), "save_history must be a boolean"

if not save_history:

    full_chain = RunnableSequence(
        # Шаг 1: Первая цепочка. Принимает {"input": "..."} и возвращает немецкую строку.
        eng_to_ger_chain,

        # Шаг 2: Адаптер. Принимает немецкую строку из шага 1 и создает словарь.
        RunnableLambda(lambda german_text: {"german_text": german_text}),

        # Шаг 3: Вторая цепочка. Принимает словарь от адаптера и возвращает русскую строку.
        ger_to_rus_chain,

        # Шаг 4: Еще один адаптер. Принимает русскую строку и упаковывает в словарь.
        RunnableLambda(lambda russian_text: {"russian_text": russian_text}),

        # Шаг 5: Третья цепочка. Принимает словарь и возвращает финальную испанскую строку.
        rus_to_spa_chain
    )


    # --- Выполняем цепочку ---
    # Входные данные остаются теми же
    result = full_chain.invoke({
        "input": "I love programming."
    })

    # Результатом будет только финальный вывод - испанская строка
    print(result)
    # Ожидаемый вывод: 'Me encanta programar.'

else:
    # --- Собираем всё вместе, используя RunnablePassthrough.assign ---
    # Мы создаем одну цепочку, последовательно добавляя в нее новые ключи.
    # LCEL автоматически определяет, какие ключи из словаря нужны каждой под-цепочке.

    full_chain = (
        # Шаг 1: Начинаем с Passthrough. Он примет исходный словарь.
        # .assign создаст новый ключ 'german_text'.
        # Для его вычисления будет запущена eng_to_ger_chain.
        # LCEL увидит, что этой цепочке нужен ключ 'input', и возьмет его из исходного словаря.
        RunnablePassthrough.assign(german_text=eng_to_ger_chain)
        # Шаг 2: Результат Шага 1 (словарь {'input': ..., 'german_text': ...})
        # передается дальше.
        # .assign создает новый ключ 'russian_text'.
        # ger_to_rus_chain получит ключ 'german_text' из словаря, который пришел с Шага 1.
        | RunnablePassthrough.assign(russian_text=ger_to_rus_chain)
        # Шаг 3: Результат Шага 2 (словарь с тремя ключами) передается дальше.
        # .assign создает финальный ключ 'spanish_text'.
        # rus_to_spa_chain получит ключ 'russian_text' из словаря, пришедшего с Шага 2.
        | RunnablePassthrough.assign(spanish_text=rus_to_spa_chain)
    )


    # --- Выполняем цепочку ---
    result = full_chain.invoke({
        "input": "I love programming."
    })

    # Теперь результат - это полный словарь со всеми промежуточными шагами
    print(result)

    # Ожидаемый вывод:
    # {
    #   'input': 'I love programming.',
    #   'german_text': 'Ich liebe Programmieren.',
    #   'russian_text': 'Я люблю программирование.',
    #   'spanish_text': 'Me encanta programar.'
    # }