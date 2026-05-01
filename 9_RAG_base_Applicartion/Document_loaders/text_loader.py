#Document Loaders are components in Langchain used to load data from various source  into a standardized format( usually as Document Object ), which can then be used for chunking, embedding , and generation.

#document (
#     page_content="the actual text content",
#     metadata={'source':"filename...."}
# )

from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')

prompt=PromptTemplate(
    template='Write a summary for the following text \n {text}',
    input_variables=['text']
)

loader=TextLoader('9_RAG_base_Applicartion/Document_loaders/test.txt', encoding='utf-8')

docs=loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

parser=StrOutputParser()

chain=prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))