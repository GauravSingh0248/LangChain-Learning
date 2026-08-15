from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
template=PromptTemplate(
    template='give me the facts about the topic {topic} \n.{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template|model|parser
final_result=chain.invoke({'topic':'black hole'})
# prompt=template.format()
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)

print(final_result)
print(type(final_result))

with open("output.json", "w") as file:
    json.dump(final_result, file, indent=4)

print("JSON saved successfully!")