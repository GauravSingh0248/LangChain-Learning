from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)
from dotenv import load_dotenv
load_dotenv()

result=model.invoke("What is the capital of INDIA ?")
# It is an AIMessage object that contains the model's response and some additional metadata.

print(result.content[0]["text"])
""" 
AIMessage(
    content=[
        {
            'type': 'text',
            'text': 'The capital of India is **New Delhi**.'
        }
    ],
    additional_kwargs={...},
    response_metadata={...}
)
"""