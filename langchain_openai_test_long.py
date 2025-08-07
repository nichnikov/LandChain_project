"""
https://python.langchain.com/docs/integrations/chat/openai/
"""

import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # <-- Добавлен импорт

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

# --- Шаг 1: Создаем шаблон для перевода с английского на немецкий ---
prompt_eng_to_gem = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates English to German.",
        ),
        ("human", "{input}"),
    ]
)

# --- Шаг 2: Создаем шаблон для перевода с немецкого на русский ---
# Здесь языки можно указать напрямую, так как этот шаг всегда одинаковый
prompt_gem_to_rus = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional translator that translates German to Russian."),
        ("human", "{german_output}"),
    ]
)

# --- Шаг 3: Создаем шаблон для перевода с немецкого на русский ---
# Здесь языки можно указать напрямую, так как этот шаг всегда одинаковый
prompt_rus_to_span = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional translator that translates Russian to Spanish."),
        ("human", "{russian_output}"),
    ]
)


# --- Шаг 4: Собираем полную цепочку из двух переводов ---
# Мы используем оператор | для создания последовательности (pipeline)
full_chain = (
    prompt_eng_to_gem
    | llm
    | prompt_gem_to_rus
    | llm
    | prompt_rus_to_span
    | llm
    | StrOutputParser() # <-- Снова получаем из ответа только строку
)


# --- Шаг 4: Выполняем полную цепочку и выводим конечный результат ---
final_result = full_chain.invoke(
    {
        # "input_language": "English",
        # "output_language": "German",
        "input": "I love programming.",
    }
)

print(f"Исходная фраза: I love programming.")
print(f"Конечный результат (на русском): {final_result}")
