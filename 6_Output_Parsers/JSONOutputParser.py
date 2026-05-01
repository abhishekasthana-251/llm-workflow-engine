from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
load_dotenv()



model=ChatGroq(model='llama-3.3-70b-versatile')


parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me the name, age and city of a Prime Minster of india \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
####
#prompt=template.format()

#result = model.invoke(prompt)

#final_result=parser.parse(result.content)

#print(final_result)'''

# now we use chains

chain=template|model | parser

result=chain.invoke({})

print(result)
print(type(result))