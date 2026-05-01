from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

model= ChatGroq(model="llama-3.3-70b-versatile")
#tupel style

template= ChatPromptTemplate([
    ("system", "you are a helpful {role}"),
    ("human", "Tell me about {topic}")
])

#this is normal
#result=template.invoke({
#    "role":"teacher",
 #   "topic":"LangChain"
#})

#result_final=model.invoke(result)

#print(result_final.content)


# now we use chain LCEL chain
# chain Style
chain=template|model

result=chain.invoke({
    "role":"teacher",
    "topic":"LangChain"
})


print(result.content)