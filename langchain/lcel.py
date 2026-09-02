from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior ai engineer, answer in {limit} words"
    ),
    (
        "human",
        "{question}"
    )
])

model = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({
    "limit": 60,
    "question": "What is a good prompt?"
}))