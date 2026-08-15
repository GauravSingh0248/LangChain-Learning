from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

docs = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

query = "\nWhat is the capital of India?"

doc_vectors = embeddings.embed_documents(docs)
query_vector= embeddings.embed_query(query)

# convert to numpy arrays

doc_vectors=np.array(doc_vectors)
query_vector=np.array(query_vector)

# .print information about the vectors
print("Number of document embeddings:", len(doc_vectors))
print("Embedding dimension:", len(doc_vectors[0]))


# Calculate cosine similarity
similarities = cosine_similarity([query_vector], doc_vectors)
# print(similarities)
print("\nCosine Similarity Scores:")
for i, score in enumerate(similarities[0]):
    print(f"Document {i+1}: {score:.4f}")


# Find the most similar document
best_match = np.argmax(similarities)

print(query)
print("\nMost Relevant Document:")
print(docs[best_match])
print(f"Similarity Score: {similarities[0][best_match]:.4f}")