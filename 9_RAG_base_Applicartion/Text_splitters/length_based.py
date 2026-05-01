from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('9_RAG_base_Applicartion/Document_loaders/dl-curriculum.pdf')

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=12,
    separator=''
)

result=splitter.split_documents(docs)

print(result[0].page_content)