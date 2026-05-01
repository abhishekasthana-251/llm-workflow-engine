from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


embeddings=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"

)

splitter= SemanticChunker(
    embeddings=embeddings
)

text = """
Cricket is a sport  played with bat and ball. 
Virat Kohli is one of the greatest batsmen of all time.
The IPL is one of the biggest cricket leagues in the world.

Python is a programming language used for many purposes.
Django is a web framework built on top of Python.
Machine learning libraries like scikit-learn are written in Python.
"""

chunks=splitter.split_text(text)

print(len(chunks))


for i, chunks in enumerate(chunks):
    print(f"chunks {i+1} ", chunks)
    print("----")