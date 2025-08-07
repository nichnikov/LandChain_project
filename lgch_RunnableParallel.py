from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Убрали RunnablePassthrough, так как он неявно используется в .assign
# Но добавили itemgetter, который является ключевым для решения
from operator import itemgetter
from langchain_core.runnables import RunnableParallel # <-- Явно импортируем RunnableParallel
import os
import getpass


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
    # organization="...",
    # other params...
)


# --- Определяем все промпты и парсер ---
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

# --- Создаем базовые цепочки для каждого шага ---
# Это делает код чище и понятнее
eng_to_ger_chain = prompt_eng_to_ger | llm | output_parser
ger_to_rus_chain = prompt_ger_to_rus | llm | output_parser
rus_to_spa_chain = prompt_rus_to_spa | llm | output_parser

# --- Собираем всё вместе ПРАВИЛЬНЫМ способом ---

# --- Собираем всё вместе, используя RunnableParallel ---

# Шаг 1: Создаем параллельный Runnable, который выполняет первый перевод
# и сохраняет исходный ввод.
step1 = RunnableParallel({
    # "input" будет взят из первоначального вызова .invoke()
    "german_text": eng_to_ger_chain,
    "input": itemgetter("input")
})

# Шаг 2: Создаем второй Runnable, который принимает результат Шага 1.
step2 = RunnableParallel({
    # itemgetter("german_text") извлекает результат Шага 1 и передает его дальше.
    "russian_text": itemgetter("german_text") | ger_to_rus_chain,
    "german_text": itemgetter("german_text"), # Пробрасываем для итогового результата
    "input": itemgetter("input")             # Пробрасываем для итогового результата
})

# Шаг 3: Создаем финальный Runnable, который принимает результат Шага 2.
step3 = RunnableParallel({
    "spanish_text": itemgetter("russian_text") | rus_to_spa_chain,
    "russian_text": itemgetter("russian_text"), # Пробрасываем
    "german_text": itemgetter("german_text"), # Пробрасываем
    "input": itemgetter("input")             # Пробрасываем
})

# Теперь мы соединяем РАБОЧИЕ Runnable объекты в одну цепочку
full_chain = step1 | step2 | step3

# --- Выполняем цепочку ---
result = full_chain.invoke({
    "input": "I love programming."
})

print(result)