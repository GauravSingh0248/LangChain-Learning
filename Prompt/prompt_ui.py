from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

import streamlit as st

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)


st.header('Research Tool')

user_input=st.text_input('enter your prompt')

if st.button('Summarize'):
    result=model.invoke(user_input)
    st.write(result.content[0]["text"])


