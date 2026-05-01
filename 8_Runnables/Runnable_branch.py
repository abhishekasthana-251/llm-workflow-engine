#runnableBranch is a control flow component in langchain that allows you to conditionally route input data to different chains or runnables based on custom logics 

# if/else block of chain

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableBranch , RunnablePassthrough


load_dotenv()

prompt1= PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='summarize the following text \n {text}',
    input_variables=['text']

)

model=ChatGroq(model='llama-3.3-70b-versatile')

parser=StrOutputParser()

report_gen_chain=prompt1 | model |parser

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic': 'relation between Russia and india'}))


final_chain.get_graph().print_ascii()