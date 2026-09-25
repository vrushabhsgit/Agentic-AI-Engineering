import sys
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@dataclass
class Customer:
    """What we know about the person, before model sees anything"""
    name:str
    plan:str
    language:str


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
def order_status(order_id:str)->str:
    """Use this tool to get the order status and to get the order status use the order_id"""
    order = ORDER_DATABASE.get(order_id)
    if order is None:
        return f'We didnt found any order with this oredr id {order}'
    return f'Get you order status for the order id {order_id} status = {order["status"]}'

@dynamic_prompt
def support_prompt(request):

    # gives us the Customer object we passed in at agent.invoke()
    customer = request.runtime.context

    lines = [
        f"you are a support agent give the answer to the customer {customer.name}",
        "never make a guess give the actual reply",
        "give the reply in the chat format dont use the email signature",
        f"Reply in language {customer.language}"
    ]

    if customer.plan == "premium":
        lines.append("this is premium customer, appologies to the customer and request a call if any issue happen")
    else:
        lines.append("this a free plan customer, give a normal reply to this customer")

    # lines.append(f"Reply in laungage {customer.language}")

    prompt = " ".join(lines)
    print("[prompt used]", prompt)
    print()
    return prompt

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[order_status],
    system_prompt="You are a support agent",
    middleware=[support_prompt],
    context_schema=Customer
)

question = {"messages":[{"role":"user","content":"Where is my order order-404?"}]}

for customer in [
    Customer(name="vrushabh", plan="premium", language="english"),
    Customer(name="musafir", plan="premium", language="spanish"),
]:
    result = agent.invoke(question,context=customer)
    print(f"To {customer.name} ({customer.plan}):")
    print(result["messages"][-1].content)
    


