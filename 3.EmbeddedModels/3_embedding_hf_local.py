from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
docs = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

vectors = embedding.embed_documents(docs)


print(str(vectors))
