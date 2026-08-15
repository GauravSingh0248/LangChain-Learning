from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


prompt1=PromptTemplate(
    template='Generate a detail report on the {topic}',
    input_variables=['topic']
)


prompt2=PromptTemplate(
    template='Generate a 5 pointer summary from the following text\n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

parser=StrOutputParser()

chain=prompt1 |model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'Cricket'})

print(result)
chain.get_graph().print_ascii()