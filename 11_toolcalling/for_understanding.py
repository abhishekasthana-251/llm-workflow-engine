from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

#tool create

@tool
def multiply(a:int, b:int) -> int:
    """Given 2 number a and b this tool returns their product"""
    return a*b

print(multiply.invoke({"a":3, "b": 4}))
print(multiply.name)
print("\n", multiply.description)
print(multiply.args,"\n")

llm=ChatGroq(model="llama-3.3-70b-versatile")

#tool binding
llm_with_tools=llm.bind_tools([multiply])


result=llm_with_tools.invoke("hi how are you ")
print(result,"\n")
#we got a content like-> hello, I'm here to help you .....
#but there no sign for the multiple sign.

print("to check it work with function or not",llm_with_tools.invoke("can you multiple 4 with 6"),"\n")
#here content="" , content is empty, and in matadata there is tool_call

print(llm_with_tools.invoke("can you multiple 4 with 9").tool_calls[0])