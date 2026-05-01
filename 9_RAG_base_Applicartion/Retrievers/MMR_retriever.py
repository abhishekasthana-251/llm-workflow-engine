#MMR Maximal Marginal Relevance ->how can we pick results that are not only relevant to the query but also different from each other

#picking the most relevant document first
# then picking the next most relevant and least similar


# helps in rag pipeline 


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

docs = [
    # 3 very similar cricket definition docs (MMR should pick only 1)
    Document(page_content="Cricket is a sport played with bat and ball"),
    Document(page_content="Cricket is an outdoor sport played between two teams"),
    Document(page_content="Cricket is a popular game played with bat and ball on a field"),

    # 2 player docs
    Document(page_content="Virat Kohli is one of the greatest batsmen in cricket history"),
    Document(page_content="Sachin Tendulkar is known as the god of cricket"),

    # 2 tournament docs
    Document(page_content="IPL is the biggest cricket league in the world"),
    Document(page_content="Cricket World Cup is held every four years by ICC"),

    # 3 completely unrelated docs
    Document(page_content="Python is a popular programming language"),
    Document(page_content="Django is a web framework built on Python"),
    Document(page_content="Machine learning is a subset of artificial intelligence"),
]


#embedding 
embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VectorStore =Chroma.from_documents(docs, embeddings)

# normal retriever(for comparison)

normal_retriever=VectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
#MMR Retriever

mmr_retriever=VectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3, # final docs to return
        "fetch_k":10, # candidates to consider
        "lambda_mult": 0.5 # balance relevance & diversity
    }
)

# compare both 
query ="tell me about cricket"

print("normal search")
for doc in normal_retriever.invoke(query):
    print(doc.page_content)

print("\n\n\n")
print("\n MMR search")

for doc in mmr_retriever.invoke(query):
    print(doc.page_content)