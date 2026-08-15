from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("amazon_New.pdf")
docs = loader.load()

print(type(docs))          # <class 'list'>
print(len(docs))           # Number of pages
print(docs[0].page_content)  # Text from the first page
print(docs[0].metadata)      # Metadata (source, page number, etc.)