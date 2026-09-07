import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

client = OpenAI()
HERE = Path(__file__).parent

def save(result, filename):
    """The image comes back as base64 text, so decode it and write the file."""
    path = HERE / filename
    path.write_bytes(base64.b64decode(result.data[0].b64_json))
    print("Saved:", path.name)

#1. langchain chain writes the image prompt
writer = ChatPromptTemplate.from_messages([
    ("system", "You write short, visual image prompts. One sentence, no preamble."),
    ("human", "Draw a image of a beautiful woman")
]) | model | StrOutputParser()

# 2. the OpenAI SDK turns that prompt into a picture
result = client.images.generate(
    model="gpt-image-1-mini",
    prompt=writer.invoke({}),
    size="1024x1024",
    quality="low",
)

save(result,"women.png")