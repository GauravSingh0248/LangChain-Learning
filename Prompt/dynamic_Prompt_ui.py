from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

st.title("📄 Research Paper Summarizer")

paper = st.text_area("Paste your research paper or abstract")

if st.button("Summarize"):

    prompt = f"""
    You are an expert research assistant.

    Read the following research paper and provide:

    1. A concise summary
    2. The research objective
    3. The methodology used
    4. Key findings
    5. Limitations
    6. Future work

    Research Paper:
    {paper}
    """
    result = model.invoke(prompt)

    st.write(result.content[0]["text"])