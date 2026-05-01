from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'  # ✅ fixed
)

docs = [
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

persist_path = '9_RAG_base_Applicartion/vector_store/chrome_db'

# ✅ Create only if not exists, else load
if not os.path.exists(persist_path):
    vectorStore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_path
    )
    print("✅ Vector Store Created!")
else:
    vectorStore = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings
    )
    print("✅ Vector Store Loaded!")

print("Total docs stored:", vectorStore._collection.count())

# Basic Search
query = 'Tell me about Cricket'
results = vectorStore.similarity_search(query, k=2)

print("\n🔍 Basic Search Results:")
for res in results:
    print(res.page_content)
    print(res.metadata)
    print("----")

# Search with Score
results_with_score = vectorStore.similarity_search_with_score(query, k=2)

print("\n📊 Search with Score:")
for doc, score in results_with_score:
    print(f"Distance: {score:.4f} → {doc.page_content}")
    # lower distance = more similar

# filter by matadata

results_filtered=vectorStore.similarity_search(
    "tell me about python",
    k=2,
    filter={'topic':'coding'}  # Only search coding docs
)
print("\n Filtered Search(coding Only)")
for res in results_filtered:
    print(res.page_content)
    print(res.metadata)
    print("----")


# Curd Opertations 

#Create -> ADD new Document

new_doc=[Document(
    page_content="Sachin Tendulkar is the god of cricket",
    metadata={"topic": "sports", "id": "6"}
)]

vectorStore.add_documents(new_doc)
print("\n New doc added")
print("total doc no ", vectorStore._collection.count())

# read -Search Again(new doc appears!)

results_2=vectorStore.similarity_search("cricket legend", k=2)
print("\n After adding Sachin docs")
for res in results_2:
    print(res.page_content)


# Delete - Remove a document by id
all_data=vectorStore.get()

real_id_to_delete=None

for chrome_id, metadata in zip(all_data['ids'], all_data['metadatas']):
    if metadata.get('id')=='1':
        real_id_to_delete=chrome_id
        break

if real_id_to_delete:
    vectorStore.delete(ids=[real_id_to_delete])
    print("\n✅ Doc deleted!")
    print("Total docs now:", vectorStore._collection.count())


