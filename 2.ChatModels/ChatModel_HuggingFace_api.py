from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    # repo_id = "TinyLlama/TinyLlama-1.1B-step-50K-105b",
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


result=model.invoke("What is the capital of Inida?")
print(result.content)