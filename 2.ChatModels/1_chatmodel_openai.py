"""
# openAI(Paid)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI(model='gpt-4')
result= model.invoke("Where is prayagraj")
print(result)

"""

#groq(free)
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=0.3, max_tokens=50)
#temperature= controls creativity( 0-> very focused , 0.5->balanced, 1-> very creative)
# max_token = controls length(like 50->very short answer ) 
result =model.invoke("write the python code for adding 2 number ")
print(result.content)
