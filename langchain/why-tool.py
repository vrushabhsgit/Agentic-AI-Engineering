from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant agent"),
    ("human", "What is 98765 multiplied by 43210?")
]) | model | StrOutputParser()

result = prompt.invoke({})

print("Answer:", result)
print("Python says",98765*43210)