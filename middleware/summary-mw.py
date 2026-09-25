from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv()


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt="You are a support agent for the online store.",
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",  # model used to summarize
            trigger=("messages", 6),     # summarize at 6 messages
            keep=("messages", 2),        # keep latest 2 messages untouched
        )
    ],
)


conversation = [
    "Hi, my name is Asha and my customer id is C-9087.",
    "I ordered a mechanical keyboard last week, order ORD-1002.",
    "My pin code is 560034, in Bengaluru.",
    "I also have an older order, ORD-1001, a wireless mouse.",
    "The keyboard is a gift, so the date matters to me.",
    "Tell me everything you remember about me and my orders.",
]


messages = []


for question in conversation:
    messages.append({
        "role": "user",
        "content": question
    })

    result = agent.invoke({
        "messages": messages
    })

    messages = result["messages"]


print("Customer:", question)
print("Agent:", result["messages"][-1].content[:160])
print(f"History now holds {len(messages)} messages")