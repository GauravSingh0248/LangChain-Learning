# ==========================================
# 1. Install required packages
# ==========================================

# pip install langchain langchain-community langchain-google-genai chromadb pypdf


# ==========================================
# 2. Imports
# ==========================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# ==========================================
# 3. Load PDF
# ==========================================

loader = PyPDFLoader("cricket.pdf")

docs = loader.load()

print("Total pages:", len(docs))


# ==========================================
# 4. Split PDF into chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))


# ==========================================
# 5. Create Embeddings
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ==========================================
# 6. Create Vector Store
# ==========================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_documents"
)


# ==========================================
# 7. User Query
# ==========================================

query = "Tell me about Cricket?"


# ==========================================
# 8. MMR Retrieval
# ==========================================

results = vector_store.max_marginal_relevance_search(
    query=query,
    k=4,
    fetch_k=20,
    lambda_mult=0.5
)


# ==========================================
# 9. Display Retrieved Documents
# ==========================================

print("\n===== MMR RESULTS =====\n")

for i, doc in enumerate(results, start=1):

    print(f"Document {i}")
    print("--------------------")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("\n")