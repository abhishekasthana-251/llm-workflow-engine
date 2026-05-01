from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()



# Embedding Model
embeddings= HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    # dimension =384
)

# Connect to pinecone
pc= Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name='langchain-rag-app'

# Create index(only first time)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # must match your embedding model (MiniLM = 384)
        metric="cosine",# how similarity is measured
        spec=ServerlessSpec( # tells Pinecone to use AWS cloud, no server management needed
            cloud='aws',
            region='us-east-1' 
        )
    )
    print(" Index Created")

else:
    print("Index Loaded")

# create Document
docs=[
    Document(page_content="Cricket is a sport played with bat and ball",
             metadata={"topic": "sports", "id": "1"}),
    Document(page_content="Virat Kohli is one of the greatest batsmen",
             metadata={"topic": "sports", "id": "2"}),
    Document(page_content="IPL is the biggest cricket league in the world",
             metadata={"topic": "sports", "id": "3"}),
    Document(page_content="Python is a programming language",
             metadata={"topic": "coding", "id": "4"}),
    Document(page_content="Django is a web framework built on Python",
             metadata={"topic": "coding", "id": "5"}),

]

# store in Pinecone

vectorStore= PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name
)
print(" Docs stored in Pinecone!")

# Basic Similarity Search
query="Tell me about cricket"
results=vectorStore.similarity_search(query, k=2)

print("\n Basic Search")
for res in results:
    print(res.page_content)
    print(res.metadata)
    print("-----")

# Search with Score
results_with_score= vectorStore.similarity_search_with_score(query, k=2)

print("\n Search with Score")
for doc,score in results_with_score:
    print(f"Score:{score:.4f} ->{doc.page_content}")
    # in pinecone -> Higher score= more similar(opposite of chroma)

#metaData filter
results_filtered= vectorStore.similarity_search(
    "tell me about python",
    k=2,
    filter={"topic":"coding"}
)
print("\n Filtered(coding only)")

for res in results_filtered:
    print(res.page_content)
    print("-----")

# add new Document 
new_doc=[Document(
    page_content="Sachin Tendulkar is the god of cricket",
    metadata={"topic": "sports", "id": "6"}
)]

vectorStore.add_documents(new_doc)
print("\n new doc added")