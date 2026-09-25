from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call
from langchain.agents import create_agent

load_dotenv()
STRONG = init_chat_model("openai:gpt-4o")
SMALL = init_chat_model("openai:gpt-4o-mini")


HARD_WORDS = (
    "hard",
    "difficult",
    "complex",
    "complicated",
    "challenging",
    "compare",
    "explain why",
    "analyze",
    "risk",
    "legal",
)


def latest_question(messages):
    """
    Get the customer's most recent message.
    """

    for message in reversed(messages):
        if type(message).__name__ == "HumanMessage":
            return str(message.content).lower()

    return ""


@wrap_model_call
def route_model(request, handler):

    question = latest_question(request.messages)

    hard = (
        any(word in question for word in HARD_WORDS)
        or len(question) > 100
    )

    if hard:
        chosen = STRONG
        model_name = "gpt-4o"
    else:
        chosen = SMALL
        model_name = "gpt-4o-mini"

    print(f"routed to {model_name}")

    request = request.override(
        model=chosen
    )

    return handler(request)


agent = create_agent(
    model=SMALL,
    tools=[],
    system_prompt="You are the support agent for the online store.",
    middleware=[route_model],
)


def ask(question):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print(result["messages"][-1].content)


ask("Where is my order ORD-1002?")

ask(
    "Compare buying a mechanical keyboard now against waiting for the sale, "
    "and explain why."
)