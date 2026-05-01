#A retriever is a component in Langchain that fetches relevant documents from a data source in response to a user's query 
# there are multiple type of retreiever 
# All retreiever in langchain are runnables


from langchain_community.retrievers import WikipediaRetriever

# create retriever
retriever= WikipediaRetriever(
    top_k_results=2,
    doc_content_chars_max=500 # limit characters per article
    
)

docs=retriever.invoke("India")

for doc in docs:
    print(doc.page_content)
    print("----")
    print(doc.metadata)