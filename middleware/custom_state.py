from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import (
    AgentState,
    AgentMiddleware,
    dynamic_prompt,
)

load_dotenv()


class SupportState(AgentState):
    """The usual messages, plus two keys of our own."""

    tool_uses: int
    refund_started: bool


@tool
def order_status(order_id: str) -> str:
    """Get the status of an order using its id."""
    return f"{order_id.upper()}: packed, ships tomorrow"


@tool
def delivery_estimate(pin_code: str) -> str:
    """Estimate delivery days for an Indian pin code."""
    return "5 days"


@tool
def start_refund(order_id: str, reason: str) -> str:
    """Start a refund for an order."""
    return f"Refund started for {order_id.upper()}, reason: {reason}"


@dynamic_prompt
def prompt_with_budget(request):
    used = request.state.get("tool_uses", 0)

    prompt = (
        "You are a support agent. "
        "Look things up with the tools, never guess."
    )

    if used >= 3:
        prompt += (
            " You have already used three tools. "
            "Answer now with what you have."
        )

    return prompt


class TrackWork(AgentMiddleware):
    """Track tool usage and whether a refund has been started."""

    state_schema = SupportState

    def after_model(self, state, runtime):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", []) or []

        if not calls:
            return None

        update = {
            "tool_uses": state.get("tool_uses", 0) + len(calls)
        }

        if any(call["name"] == "start_refund" for call in calls):
            update["refund_started"] = True

        print(
            f"\n[state] tool_uses={update['tool_uses']} "
            f"refund_started="
            f"{update.get('refund_started', state.get('refund_started', False))}"
        )

        return update


agent = create_agent(
    model="gpt-4o-mini",
    tools=[
        order_status,
        delivery_estimate,
        start_refund,
    ],
    middleware=[
        prompt_with_budget,
        TrackWork(),
    ],
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Where is ORD-1002, when does it reach pin 560034, "
                    "and refund it because it is too late."
                ),
            }
        ],
        "tool_uses": 0,
        "refund_started": False,
    }
)


print("Answer:", result["messages"][-1].content)
print("tool_uses:", result["tool_uses"])
print("refund_started:", result["refund_started"])

if result["refund_started"]:
    print(
        "Refund has been started. "
        "Notify the finance team and support team."
    )