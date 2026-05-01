from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Machine learning models can analyze large amounts of data to find patterns.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Cosine similarity measures the angle between two vectors in embedding space.",
    "Python is a popular programming language used for data science and automation.",
    "Neural networks are inspired by the structure of the human brain.",
    "The stock market fluctuates based on economic conditions and investor sentiment."
]

query = "How do we compare two embeddings using cosine similarity?"

doc_embeddings=embedding.embed_documents(documents)  # 6 vector
query_embedding=embedding.embed_query(query) # 1 one red vector

#print(cosine_similarity([query_embedding],doc_embeddings)) # find the angle between the of red vector to all the 6 vector

score=cosine_similarity([query_embedding],doc_embeddings)[0]

#enumerate give the index 
#list(enumerate(score)) # give the index to all 6 vector 

#now we sort the list and the index is attached to them so we can find which one have vector out of 6 is near the red vector

index , scores=(sorted(list(enumerate(score)),key=lambda x:x[1])[-1]) # [1]->sort on the based on vector, and [-1] give the last vaule greater one

print(query)
print(documents[index])
print("similarity score is:", scores)