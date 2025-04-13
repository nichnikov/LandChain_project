"""
https://python.langchain.com/docs/integrations/chat/openai/
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-or-vv-17bd53f8f505e0a1d24a3bf0a8bb702e13edbc78c89ac9aa41f6bda7ec72270c",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="https://api.vsegpt.ru/v1",
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
