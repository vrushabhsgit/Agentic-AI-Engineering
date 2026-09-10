from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@tool
def multiply(a:int,b:int)->int:
    """Use this tool when someone asks for the multiplication of two numbers"""
    return a*b

@tool
def word_count(text:str)->int:
    """Use this tool when someone asks for the count words"""
    return len(text.split())

model_with_tools = model.bind_tools([multiply,word_count])

#1. ex of model needed a tool
response = model_with_tools.invoke("What is the multiplicaton of 12*5 ?")
print("Response1: ", repr(response.content))
print("Tool calling:",response.tool_calls)

#2. needed no tool
response2 = model_with_tools.invoke("Tell me in one why python is famous?")
print("Response2: ", repr(response2.content))
print("Tool calling:",response2.tool_calls)


#model decides which model to choose
response3 = model_with_tools.invoke("How many word in this sentence: I Love You Baby")
print("Response3: ", repr(response3.content))
print("Tool calling:",response3.tool_calls)