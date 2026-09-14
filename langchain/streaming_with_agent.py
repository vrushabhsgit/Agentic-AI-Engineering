from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
# Tool 1: Calculate hotel price
@tool
def hotel_ticket_price(city: str, nights: int) -> str:
    """Calculate the total hotel price for a city based on number of nights."""

    cities = {
        "new-york": 3200,
        "california": 34000,
        "wellington": 4200,
    }

    price_per_night = cities.get(city.lower())

    if price_per_night is None:
        return f"Hotel pricing is not available for {city}."

    total_price = price_per_night * nights

    return (
        f"Hotel price in {city} for {nights} nights "
        f"is ₹{total_price}."
    )


# Tool 2: Calculate flight price
@tool
def flight_ticket_price(destination: str) -> str:
    """Get the flight ticket price for a destination."""

    ticket_prices = {
        "new-york": 3200,
        "california": 34000,
        "wellington": 4200,
    }

    price = ticket_prices.get(destination.lower())

    if price is None:
        return f"Flight pricing is not available for {destination}."

    return f"The flight ticket price to {destination} is ₹{price}."


# Create agent
agent = create_agent(
    model="openai:gpt-4o-mini",

    tools=[
        hotel_ticket_price,
        flight_ticket_price,
    ],

    system_prompt="""
    You are a travel planning agent.

    Help users with:
    - Hotel prices
    - Flight ticket prices
    - Basic travel planning

    When the user asks about hotel or flight prices,
    always use the appropriate tool instead of guessing.

    If both hotel and flight prices are requested,
    use both tools.
    """
)


# User question
question = """
I want to travel to New-York for 3 nights.
Tell me the hotel cost and flight ticket price.
"""


# Stream each agent step
for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },
    stream_mode="updates",
):

   for node, update in chunk.items():
    for message in update["messages"]:
        if getattr(message, "tool_calls", None):
            print(f"[{node}] wants: {[c['name'] for c in message.tool_calls]}")
        elif type(message).__name__ == "ToolMessage":
            print(f"[{node}] {message.name} returned: {message.content}")
        elif message.content:
            print(f"[{node}] says: {message.content}")


  