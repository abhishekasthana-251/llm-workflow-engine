from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()
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


#tool calling
print("to check it work with function or not",llm_with_tools.invoke("can you multiple 4 with 6"),"\n")
#here content="" , content is empty, and in matadata there is tool_call

print(llm_with_tools.invoke("can you multiple 4 with 9").tool_calls[0])

#got list of tool wrap in dictonary

result1=llm_with_tools.invoke("can you multiple 4 with 9").tool_calls[0]
print(result1)


# tool execution
# we can pass whole result1
result2=multiply.invoke({'name': 'multiply', 'args': {'a': 4, 'b': 9}, 'id': 'w19zmqmkm', 'type': 'tool_call'})

print("\n", result2) # get the tool message

result3= multiply.invoke(result1['args'])
print("\n\n\n", result3)

# connecting it as human message -> ai message-> tool message -> again LLM , got the final answer 

print("\n---------\n")

query=HumanMessage("can you multiple 3 *100")

message=[query]

Answer=llm_with_tools.invoke(message)

message.append(Answer)

# humanMessage in the Message

tool_result=multiply.invoke(Answer.tool_calls[0]) # got the tool message
message.append(tool_result)

print(llm_with_tools.invoke(message).content)