
#this is the example for LLMs , now it is old llm  and return plain text , result="India", 
# and it is paids to use  by openAI, because it is close source
# we can run this as ->"python 1.LLMs/1_llm_demo.py"
"""
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model='gpt-3.5-turbo-instruct')

result=llm.invoke("what is the capital of india ")


print(result)
"""

# NOW WE USE Grop because it is free

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm=ChatGroq(model='llama-3.3-70b-versatile')

result=llm.invoke("I'm happy today, what about you")

#print(result) by this the result come with lot of mate data 
#to handel this we use  .content

print(result.content)