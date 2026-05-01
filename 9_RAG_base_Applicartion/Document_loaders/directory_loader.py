#directoryLoader is a document loader that lets you load multiple documents from a directory(folder) of files.

# glob Pattern

# "**/*.txt" -> all .txt files in all subfolder
# "*.pdf" -> all .pdf files in the root directory
# "data/*.csv" -> all .csv files in the data/ folders 
# "**/*" -> all files(any type, all folders )

# ** = recursive search through subfolder


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='9_RAG_base_Applicartion/Document_loaders/book',
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
#docs=loader.lazy_load()
for document in docs:
    print(document.metadata)