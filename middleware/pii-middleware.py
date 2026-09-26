import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMatch, PIIMiddleware, PIIDetectionError

load_dotenv()


def indian_phone(text: str) -> list[PIIMatch]:
    """Find 10-digit Indian mobile numbers, with or without the country code."""
    matches = []

    for found in re.finditer(r"(?:\+91[\s-]?)?[6-9]\d{9}", text):
        matches.append(
            PIIMatch(
                type="phone",
                value=found.group(),
                start=found.start(),
                end=found.end(),
            )
        )

    return matches


agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="You are the support agent. Confirm what the customer told you in one line.",
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        PIIMiddleware("credit_card", strategy="mask"),
        PIIMiddleware(
            "phone_number",
            detector=indian_phone,
            strategy="redact",
        ),
    ],
)

message = (
    "Hi, I am Asha. Mail me at asha.k@example.com or call 9876543210. "
    "I paid with card 4111 1111 1111 1111."
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": message}
        ]
    }
)

print("What customer typed:")
print(" ", message)

print()

print("What the model received:")
print(" ", result["messages"][0].content)

print()

print("Answer:")
print(" ", result["messages"][-1].content)
print()
print()

strict_agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="You are the support agent",
    middleware=[
        PIIMiddleware("credit_card", strategy="block"),
    ],
)

try:
    strict_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hey, my name is Vrushabh and my credit card number is 4111 1111 1111 1111",
                }
            ]
        }
    )

except PIIDetectionError:
    print(
        "Blocked:",
        "Hey Vrushabh, please don't send your credit card details to the model because it is sensitive information.",
    )