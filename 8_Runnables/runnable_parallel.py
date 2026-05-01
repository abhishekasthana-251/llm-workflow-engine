#RunnableParallel is a runnable primitive that allows multiple runnables to execute in parallel.
# Each runnable recevies the same input and processes it independently, producing a dictionary of outputs.

from langchain_groq import  ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')

prompt1 = PromptTemplate(
    template='generate a instagram post about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

parser =StrOutputParser()

paralle_chain=RunnableParallel({
    'insta':prompt1 | model | parser,
    'linkedin' : prompt2 | model | parser
})


result=paralle_chain.invoke({'topic' : 'is learning langchain today market is good or not for a freasher to get a job'})


print("insta -> " ,result['insta'])
print("linkedin -> ", result['linkedin'])


paralle_chain.get_graph().print_ascii()