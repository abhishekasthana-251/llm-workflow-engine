from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=45) 
#here the dimension of the vector, so 45 dimension in vector
 
#result=embedding.embed_query("Delhi is capital of india") 
#embed_query run only one query

document=[

    "today data",
    "what is happing in world",
    "what is a job market is doinh ",
    "is genai is good "
]
result=embedding.embed_documents(document)

print(result)