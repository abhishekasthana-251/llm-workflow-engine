#The strOutputParser is the simplest output parser in Langchain. It is used to parse the output of a language model(llm). and return it as a plain string
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()



model=ChatGroq(model='llama-3.3-70b-versatile')


 #1st prompt -> detailed report

template1=PromptTemplate(
  
    template='write a detailed report on {topic}',
    input_variables=['topic']
 )

template2=PromptTemplate(
  
    template='Write a 5 point summary on the following text. \n {text}',
    input_variables=['text']
 )


'''
prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)
'''




#the strOutputParser have mainly work with chain

parser=StrOutputParser()

chain =template1 | model | parser | template2 | model | parser

# this is a chain , where the parser helping in giving result.content , 


result= chain.invoke({'topic': 'Black Color'})

print(result)