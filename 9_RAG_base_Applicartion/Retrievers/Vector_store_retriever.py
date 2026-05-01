#Vector Store Retriever in langchain is a most  common type of retriever that lets you search and fetch documents from a vector store based on semantic similarity using vector embeddings.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
persist_path = '9_RAG_base_Applicartion/vector_store/chrome_db'
vectorstore=Chroma(
    embedding_function=embeddings,
    persist_directory=persist_path
)


#convert vectorstore-> retriever
retriever=vectorstore.as_retriever(
    search_type="similarity", #defult
    search_kwargs={"k":3}
)


docs=retriever.invoke("tell me about cricket")

for doc in docs:
    print(doc.page_content)
