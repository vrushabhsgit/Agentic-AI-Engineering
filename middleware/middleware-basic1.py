import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call
)
from langchain.tools import tool
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
def order_status(order_id:str)->str:
    """Use this tool to get the order status and to get the order status use the order_id"""
    order = ORDER_DATABASE.get(order_id)
    if order is None:
        return f'We didnt found any order with this oredr id {order}'
    return f'Get you order status for the order id {order_id} status = {order["status"]}'

# runs before the model is called
@before_model
def show_before(state, runtime):
    print(f"[before_model] {len(state['messages'])} messages going to the model")
    return None # None means: I am not changing the state

# around the model call
@wrap_model_call
def time_the_model(request, handler):
    print(f"[wrap_model_call] tools offered: {[t.name for t in request.tools]}")
    response = handler(request)
    # <-- the model runs here
    print("[wrap_model_call] model has replied")
    return response

# runs immediately after the model responds
@after_model
def show_after(state, runtime):
    last = state["messages"][-1]
    asked = [call["name"] for call in getattr(last, "tool_calls", []) or []]
    print(f"[after_model] asked for: {asked or 'nothing, this is the final answer'}")
    return None

# sits around every tool execution
@wrap_tool_call
def show_tool(request, handler):
    print(f"[wrap_tool_call] running {request.tool_call['name']} "
    f"with {request.tool_call['args']}")
    result = handler(request)
    # <-- the tool runs here
    print("[wrap_tool_call] tool finished")
    return result

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[order_status],
    system_prompt="you are a support agent",
    middleware=[show_before, time_the_model, show_after, show_tool],
)

result = agent.invoke({
    "messages":[{"role":"user","content":"What is the status of my order order-404"}]
})

print("Answer : ",result["messages"][-1].content) 

      




