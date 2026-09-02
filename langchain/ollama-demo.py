from langchain_ollama import ChatOllama

model = ChatOllama(
    model="mistral:latest",
    temperature=0
)

print(
    model.invoke(
        "What is Ollama tell me in short"
    ).content
)