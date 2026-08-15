from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()

# ------------------- LLM -------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

parser = StrOutputParser()


# ------------------- Pydantic Schema -------------------
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


# ------------------- Sentiment Classifier -------------------
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into positive or negative.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    },
)

classifier_chain = prompt1 | model | parser2


# ------------------- Response Prompts -------------------
positive_prompt = PromptTemplate(
    template="""
Write a polite response to the following positive feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="""
Write a polite response to the following negative feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)


# ------------------- Branch -------------------
branch_chain = RunnableBranch(

    (
        lambda x: x["classification"].sentiment == "positive",

        RunnableLambda(
            lambda x: {"feedback": x["feedback"]}
        )
        | positive_prompt
        | model
        | parser,
    ),

    (
        lambda x: x["classification"].sentiment == "negative",

        RunnableLambda(
            lambda x: {"feedback": x["feedback"]}
        )
        | negative_prompt
        | model
        | parser,
    ),

    RunnableLambda(lambda x: "Could not determine sentiment."),
)


# ------------------- Final Chain -------------------
chain = (
    RunnablePassthrough.assign(
        classification=classifier_chain
    )
    | branch_chain
)


# ------------------- Test -------------------
result = chain.invoke(
    {
        "feedback": "This is a beautiful phone."
    }
)

print(result)

print("\nGraph:\n")
chain.get_graph().print_ascii()