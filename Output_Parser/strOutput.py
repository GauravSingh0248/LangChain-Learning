from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


# 1st prompt ---> detailed prompt

template1=PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)

# 2nd prompt ---> summary
template2=PromptTemplate(
    template='Write a 5 point summary on the following text. /n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

# entire flow or pipeline
chain=template1| model|parser|template2|model|parser

result=chain.invoke({'topic':'black hole'})

print(result)