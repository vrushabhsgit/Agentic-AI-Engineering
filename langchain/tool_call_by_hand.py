from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Input(BaseModel):
    """Input for the word count"""
    word:str = Field(description="word for the word count")

@tool("word_count",args_schema=Input)
def word_count(word:str)->int:
    count = len(word.split())
    return count

model_with_tool = model.bind_tools([word_count])

reply = model_with_tool.invoke("How many words in this sentence: Sam altman is really a cool guy")

if reply.tool_calls:
    tool_call = reply.tool_calls[0]
    print("tool name: ", tool_call["name"])
    print("arguments: ", tool_call["args"])
else:
    print("tool is not calling")

