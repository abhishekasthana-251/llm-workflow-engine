#RunnableLambda is a runnable primitive that allows you to apply custom python functions within an AI pipeline

# it acts as a middleware between different AI components, enabling  preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow 


#eg ->
#  from langchain_core.runnables import RunnableLambda
#     # the function use in the runnable
#     def word_counter(text):
#         return len(text.split())

#     runnable_word_counter=RunnableLambda(word_counter)

#     print(runnable_word_counter.invoke('Hi there how are you today'))



from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda ,RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

model=ChatGroq(model='llama-3.3-70b-versatile')

parser =StrOutputParser()

joke_gen_chain=RunnableSequence(prompt, model, parser)

paralle_chain= RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
    #or
    #'word_count' : RunnableLambda(lambda x: len(x.split()))
})

final_chain=RunnableSequence(joke_gen_chain,paralle_chain)

result=final_chain.invoke({'topic': 'ZORO from onePieces'})

print(result['joke'])
print(result['word_count'])

final_chain.get_graph().print_ascii()