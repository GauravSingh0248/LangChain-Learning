from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

docs = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

vectors = embeddings.embed_documents(docs)

print(len(vectors))      # 3
print(len(vectors[0]))   # Dimension of each embedding