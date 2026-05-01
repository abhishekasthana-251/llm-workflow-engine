#webBaseloader is a document loader in langchain used to load and extract text content from web page(urls) -> static website
#SeleninumURlLoader -> dynamic website

from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

os.environ['USER_AGENT']='myagent'
#Some websites block requests with no user agent — this tells the website who is making the request.
load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

prompt=PromptTemplate(
    template='Answer the following question \n {question} from the following text \n {text}',
    input_variables=['question' , 'text']

)

parser= StrOutputParser()

url='https://www.geeksforgeeks.org/aptitude/profit-and-loss-questions-aptitude/'

loader= WebBaseLoader(url)

docs=loader.load()

chain= prompt | model | parser

print(chain.invoke({'question':'what is a formula of profit and loss' , 'text':docs[0].page_content}))