#runnablePassthrough is a special runnable primitive that simply returns the input as output without modifying it.

# where we cam use it , remember the sequence runnable,where we pass a prompt(generate a joke)-> model -> parser->prompt(explain this joke)->model-> parser , so here we only get the explaination of the joke only not a joke ,to understand the which joke is explained  



from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence ,RunnableParallel, RunnablePassthrough


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

joke_gen_chain=RunnableSequence(prompt1,model , parser) # I know we can use | instead of RunnableSequence ,but I'm learning about it so that's why I'm using it

parallel_chain= RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'Elephent and ant'}))

final_chain.get_graph().print_ascii()