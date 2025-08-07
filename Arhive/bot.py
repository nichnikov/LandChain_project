from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub

# Инициализация LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Функция для поиска бухгалтерских материалов (заглушка)
def search(query: str) -> str:
    # В реальной реализации здесь должен быть поиск по базе знаний
    # Возвращаем примеры документов, которые могли бы быть найдены
    accounting_docs = {
        "налогообложение": "НДС составляет 20% для большинства товаров и услуг. Срок подачи декларации - до 25 числа месяца, следующего за отчетным кварталом.",
        "зарплата": "Заработная плата должна выплачиваться не реже 2 раз в месяц. НДФЛ удерживается по ставке 13% для резидентов.",
        "отчетность": "Бухгалтерская отчетность сдается раз в год до 31 марта следующего года. Включает баланс и отчет о финансовых результатах."
    }
    
    # Простой поиск по ключевым словам
    for keyword in accounting_docs:
        if keyword in query.lower():
            return accounting_docs[keyword]
    
    return "Не найдено конкретной информации по вашему запросу. Пожалуйста, уточните вопрос."

# Определяем инструменты (tools) для агента
tools = [
    Tool(
        name="Search",
        func=search,
        description="Используйте этот инструмент для поиска информации по бухгалтерским вопросам"
    )
]

# Промпт для анализа типа вопроса
classifier_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    Проанализируйте вопрос пользователя и определите его тип. Возможные типы:
    - 'приветствие' - если это приветствие или начало разговора
    - 'благодарность' - если пользователь благодарит
    - 'бухгалтерский' - если вопрос связан с бухгалтерией, налогами, отчетностью
    - 'другой' - если вопрос не относится к вышеперечисленным
    
    Вопрос: {question}
    Тип вопроса:"""
)

# Инициализация цепочки для классификации
classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt)

# Промпт для ответа на бухгалтерские вопросы
accounting_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Вы - бухгалтерский помощник. Ответьте на вопрос пользователя на основе предоставленного контекста.
    Если в контексте нет точного ответа, скажите, что не можете дать точный ответ и предложите уточнить вопрос.
    
    Контекст: {context}
    Вопрос: {question}
    Ответ:"""
)

# Инициализация цепочки для бухгалтерских вопросов
accounting_chain = LLMChain(llm=llm, prompt=accounting_prompt)

# Основной промпт для агента
agent_prompt = hub.pull("hwchase17/react-chat")

# Создание агента
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Функция для обработки вопроса пользователя
def process_question(question: str) -> str:
    # Определяем тип вопроса
    question_type = classifier_chain.run(question=question).strip().lower()
    
    if 'приветствие' in question_type:
        return "Здравствуйте! Я бухгалтерский помощник. Задайте мне ваш вопрос."
    elif 'благодарность' in question_type:
        return "Пожалуйста! Обращайтесь, если у вас будут еще вопросы."
    elif 'бухгалтерский' in question_type:
        # Используем агента для поиска и формирования ответа
        result = agent_executor.invoke({"input": question})
        return result['output']
    else:
        return "Извините, я могу отвечать только на бухгалтерские вопросы. Пожалуйста, задайте вопрос по бухгалтерии, налогам или отчетности."

# Примеры использования
print(process_question("Привет!"))  # Приветствие
print(process_question("Спасибо за помощь!"))  # Благодарность
print(process_question("Какой срок сдачи декларации по НДС?"))  # Бухгалтерский вопрос
print(process_question("Какая сейчас погода?"))  # Другой вопрос