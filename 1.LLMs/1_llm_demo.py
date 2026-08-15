from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

response = llm.invoke("What is the capital of India?")

print(response.content[0]["text"])
# print(response.text())