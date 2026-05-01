from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


model=ChatGroq(model='llama-3.3-70b-versatile')

#template with placeholder slot

template=ChatPromptTemplate([
    ("system", "you are a {role}"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{question}')
])


chain=template|model

chat_history=[
    HumanMessage("my name is Abhishek"),
    AIMessage("Hello Abhishek!")
]
#storage is here, not in placeholder

result=chain.invoke({
    'role':"doctor",
    "chat_history": chat_history,
    "question":"what is fever ?"
})

print("ai:" ,result.content)