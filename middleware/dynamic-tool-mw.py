from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call

load_dotenv()


ORDER_DATABASE = {
    "order-404": {
        "item": "wireless mouse",
        "status": "shipped",
        "price": 2000,
        "quantity": 3,
        "delivery_date": "21/12/2027",
    },

    "order-405": {
        "item": "durex ultra thin",
        "status": "not shipped",
        "price": 230,
        "quantity": 2300,
        "delivery_date": "21/08/2027",
    },
}


@tool
def order_status(order_id: str) -> str:
    """Use this tool to get the order status using the order_id."""

    order = ORDER_DATABASE.get(order_id)

    if order is None:
        return f"We didn't find any order with order id {order_id}"

    return (
        f"Order status for {order_id}: {order['status']}"
    )


@tool
def delivery_estimate(order_id: str) -> str:
    """Use this tool to get the delivery estimate using the order_id."""

    order = ORDER_DATABASE.get(order_id)

    if order is None:
        return f"We didn't find any order with order id {order_id}"

    return f"Your delivery date is {order['delivery_date']}"


@tool
def cancel_order(order_id: str) -> str:
    """Use this tool to cancel an order using the order_id."""

    order = ORDER_DATABASE.get(order_id)

    if order is None:
        return f"We didn't find any order with order id {order_id}"

    order["status"] = "cancelled"

    return f"Your order {order_id} has been cancelled."


@tool
def start_refund(order_id: str) -> str:
    """Use this tool to initiate a refund using the order_id."""

    order = ORDER_DATABASE.get(order_id)

    if order is None:
        return f"We didn't find any order with order id {order_id}"

    order["status"] = "refund initiated"

    return (
        f"Your order {order_id} has been cancelled "
        "and the refund has been initiated."
    )


DELIVERY_TOOLS = [
    order_status,
    delivery_estimate,
]

MONEY_TOOLS = [
    order_status,
    start_refund,
    cancel_order,
]

MONEY_WORDS = (
    "refund",
    "cancel",
    "money",
    "return",
)


def latest_question(messages):
    """
    Get the customer's most recent message.

    We don't simply use messages[-1] because during an agent run
    the last message could be a tool result, model response,
    or another internal message.

    We specifically want the latest HumanMessage.
    """

    for message in reversed(messages):
        if type(message).__name__ == "HumanMessage":
            return str(message.content).lower()

    return ""


@wrap_model_call
def pick_tool(request, handler):

    question = latest_question(request.messages)

    if any(word in question for word in MONEY_WORDS):
        request = request.override(
            tools=MONEY_TOOLS
        )
    else:
        request = request.override(
            tools=DELIVERY_TOOLS
        )

    print(
        "offered:",
        [tool.name for tool in request.tools]
    )

    return handler(request)


agent = create_agent(
    model="openai:gpt-4o-mini",

    tools=[
        order_status,
        delivery_estimate,
        cancel_order,
        start_refund,
    ],

    middleware=[
        pick_tool
    ],

    system_prompt=(
        "You are a support agent. "
        "Use tools to get the correct result. "
        "Never make a guess."
    )
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I want a refund for order-404, it is too late for me."
        }
    ]
})


print(result["messages"][-1].content)