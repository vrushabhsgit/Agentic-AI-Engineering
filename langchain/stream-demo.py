from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatOllama(model="mistral:latest",temperature=0)

#1. streaning directly from the model
# model_response = model.stream("Write a poem for me i am your boyfriend and you are my girlfirend")

# for chunk in model_response:
#     print(chunk.content, end="", flush=True)

#2. streaming a whole chain
full_chain = ChatPromptTemplate.from_messages([
    ( "system",
     "You are a poet"),
    ("human",
    "Write a beutifull poem of jaun elia in hindi")
]) | model | StrOutputParser()

for chunk in full_chain.stream({}):
    print(chunk, end="", flush=True)
