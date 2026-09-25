from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ToolRetryMiddleware


load_dotenv()


attempts = {"count": 0}


@tool
def order_status(order_id: str) -> str:
    """Get the order status using its id."""

    attempts["count"] += 1

    print(
        f"{attempts['count']} attempt made "
        "for getting the order status"
    )

    # deliberately fail the first two attempts
    # to simulate an outage
    if attempts["count"] < 3:
        raise RuntimeError("Order service is unavailable")

    return f"{order_id} is packed and will ship tomorrow"


agent = create_agent(
    model="openai:gpt-4o-mini",

    tools=[
        order_status
    ],

    system_prompt=(
        "You are a support agent for an online delivery platform. "
        "Use the available tools to answer order-related questions."
    ),

    middleware=[
        ToolRetryMiddleware(
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=1.5,
        )
    ],
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Where is my order order-404?"
            }
        ]
    }
)


print(result["messages"][-1].content)
print("Attempts:", attempts["count"])