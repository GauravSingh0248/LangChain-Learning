from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://python.langchain.com/docs/introduction/"
)

docs = loader.load()

print(docs[0].page_content[:500])