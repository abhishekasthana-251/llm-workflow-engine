from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')


template=PromptTemplate(
    template='Generate two fact point about this {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
chain= template | model | parser

result=chain.invoke({'topic':'pen'}) 

print(result)

# we can  see the chain flow

chain.get_graph().print_ascii()