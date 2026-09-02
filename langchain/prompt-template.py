from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

prompt = ChatPromptTemplate([
    (
        "system",
        "You are a {language} trainer, keep answers under {limit} words"
    ),(
        "human",
        "Explain {topic} to a beginner"
    )
])

filled = prompt.invoke({
    "language":"python",
    "limit":40,
    "topic":"harness engineering"
})
print(filled.messages)
print()

response = model.invoke(filled)
print(response.content)