from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain


# Инициализируем OpenAI LLM
llm = OpenAI(model_name="text-davinci-003", 
             openai_api_key="sk-or-vv-17bd53f8f505e0a1d24a3bf0a8bb702e13edbc78c89ac9aa41f6bda7ec72270c", 
             openai_api_base="https://api.vsegpt.ru/v1")

# Создаем шаблон запроса
prompt_template = PromptTemplate(input_variables=["question"], template="{question}")

# Создаем цепочку для запросов к LLM
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Отправляем запрос к LLM
question = "Что такое машинное обучение?"
response = llm_chain.run(question)

print(response)