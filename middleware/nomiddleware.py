import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

ORDER_DATABASE = {
    "order-404": {
        "item": "wireless mouse",
        "status": "shipped",
        "price": 2000,
        "quantity": 3,
    },

    "order-405": {
        "item": "durex ultra thin",
        "status": "not shipped",
        "price": 230,
        "quantity": 2300,
    },
}


@tool
def order_status(order_number: str) -> str:
    """Use this tool for checking the order status of the customer and return the correct result."""

    if not order_number.startswith("order-"):
        order_number = f"order-{order_number}"

    order = ORDER_DATABASE.get(order_number)

    if order is None:
        return "no order found with this number"

    return f"the status of your order is {order['status']}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[order_status],
    system_prompt=(
        "You are an online delivery agent. Your job is to help customers "
        "by solving their queries. Never guess. Use the appropriate tool "
        "when information needs to be looked up."
    ),
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the status of my order number 404 dude?"
        }
    ]
})

print(result["messages"][-1].content)