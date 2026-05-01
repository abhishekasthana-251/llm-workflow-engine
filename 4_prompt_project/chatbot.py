from langchain_groq  import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage,HumanMessage
from dotenv import load_dotenv


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')

chat_history=[
            SystemMessage(content='you are a helpful AI assistant')
]



while True:
    user_input=input('you:')
   
    if user_input == 'exit':
        break

    chat_history.append(HumanMessage(content=user_input))
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:" ,result.content)