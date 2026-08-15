from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Load PDF
loader = PyPDFLoader("cricket.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
# Store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

# Search
docs = retriever.invoke("list down the legendary players ?")

for doc in docs:
    print(doc.page_content)