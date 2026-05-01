#CSVLoader is a document loader used to load CVS files into Langchain Document object- one per row ,by default.


from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path='9_RAG_base_Applicartion/Document_loaders/Social_Network_Ads.csv')

docs=loader.load()

print(len(docs))

print(docs[1])