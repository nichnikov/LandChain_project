"""
https://python.langchain.com/docs/integrations/chat/openai/
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

'''
input_text = "Смысл жизни в том, "
r = llm.invoke(input_text).content
print(r)'''

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
r2 = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
).content

print(r2)
