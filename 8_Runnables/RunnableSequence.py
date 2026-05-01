#RunnableSequence is a  sequential chain of runnables in  Langchain that executes each step one after another , passing the output  of one step as the input to the next.

# it is useful when you need to compose multiple runnables together in a structure workflow.

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')

prompt1= PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template='Explain the following joke -{text}',
    input_variables=['text']
)

chain= RunnableSequence(prompt1,model, parser, prompt2 , model,parser)

print(chain.invoke({'topic':'Ai'}))