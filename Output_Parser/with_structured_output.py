from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

load_dotenv()

# Load the LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


# Define the output schema
class Facts(BaseModel):
    fact_1: str = Field(description="First fact about the topic")
    fact_2: str = Field(description="Second fact about the topic")
    fact_3: str = Field(description="Third fact about the topic")


# Create structured model
structured_model = model.with_structured_output(Facts)

# Invoke
result = structured_model.invoke(
    "Give me 3 facts about black holes."
)

print(result)
print(type(result))