from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
#from langchain_classic.retrievers import MultiQueryRetriever  # ✅ correct!
from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever

load_dotenv()

docs = [
    Document(page_content="Cricket is a sport played with bat and ball"),
    Document(page_content="Cricket is an outdoor sport played between two teams"),
    Document(page_content="Cricket is a popular game played with bat and ball on a field"),
    Document(page_content="Virat Kohli is one of the greatest batsmen in cricket history"),
    Document(page_content="Sachin Tendulkar is known as the god of cricket"),
    Document(page_content="IPL is the biggest cricket league in the world"),
    Document(page_content="Cricket World Cup is held every four years by ICC"),
    Document(page_content="Python is a popular programming language"),
    Document(page_content="Django is a web framework built on Python"),
    Document(page_content="Machine learning is a subset of artificial intelligence"),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(docs, embeddings)

llm = ChatGroq(model="llama-3.3-70b-versatile")

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

query = "cricket"
docs_result = retriever.invoke(query)

print(f"Total unique docs fetched: {len(docs_result)}")
print("\n")
for doc in docs_result:
    print("→", doc.page_content)