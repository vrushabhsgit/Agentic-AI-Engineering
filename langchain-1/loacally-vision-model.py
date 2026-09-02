import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

image_path = Path(__file__).parent / "sample.jpg"

encoded = base64.b64encode(
    image_path.read_bytes()
).decode("utf-8")

local_message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe this image in two lines."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            }
        }
    ]
)

print(model.invoke([local_message]).content)