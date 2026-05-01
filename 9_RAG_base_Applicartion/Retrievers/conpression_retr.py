from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

# Docs (bigger content so compression is visible!)
docs = [
    Document(page_content="""
        Cricket is a bat and ball sport.
        It is played between two teams of 11 players.
        Cricket originated in England in 16th century.
        The game involves batting, bowling and fielding.
        Players wear white uniforms in test matches.
        Cricket is most popular in India, Australia and England.
    """),
    Document(page_content="""
        Virat Kohli was born on November 5, 1988 in Delhi.
        He is one of the greatest batsmen in cricket history.
        He loves eating Italian food and spending time with family.
        Kohli has scored more than 70 international centuries.
        He is married to Bollywood actress Anushka Sharma.
        His test batting average is over 48.
    """),
    Document(page_content="""
        IPL stands for Indian Premier League.
        It is the biggest cricket league in the world.
        IPL was founded in 2008 by BCCI.
        Each team plays 14 matches in the league stage.
        IPL has made many young cricketers famous.
        The tournament is held every year in April-May.
    """),
]

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# VectorStore
vectorstore = Chroma.from_documents(docs, embeddings)

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step 1 — Base Retriever
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Step 2 — Compressor
compressor = LLMChainExtractor.from_llm(llm)

# Step 3 — Wrap both together!
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Compare both!
query = "Virat Kohli batting records"

print("📌 Normal Retriever:")
for doc in base_retriever.invoke(query):
    print(doc.page_content)
    print("---")

print("\n🗜️ Compression Retriever:")
for doc in compression_retriever.invoke(query):
    print(doc.page_content)
    print("---")