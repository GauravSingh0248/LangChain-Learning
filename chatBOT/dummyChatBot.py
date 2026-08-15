from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

print("Simple ChatBot (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Ignore empty input
    if not user_input:
        print("Please enter a message.\n")
        continue

    print("AI: ", end="", flush=True)

    try:
        for chunk in model.stream(user_input):

            # Handle different chunk formats
            if hasattr(chunk, "text") and chunk.text:
                print(chunk.text, end="", flush=True)

            elif isinstance(chunk.content, str):
                print(chunk.content, end="", flush=True)

            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        print(part.get("text", ""), end="", flush=True)

        print("\n")

    except Exception as e:
        print(f"\nError: {e}\n")