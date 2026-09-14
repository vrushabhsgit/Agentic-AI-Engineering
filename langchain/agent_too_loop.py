import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# Just makes the console support UTF-8 characters.
sys.stdout.reconfigure(encoding="utf-8")

@tool
def get_the_current_time(city:str):
    """Use this tool for getting the current time of the city"""
    zones = {
        "mumbai": "Asia/Kolkata",
        "london": "Europe/London",
        "new york": "America/New_York"
    }

    zone = zones.get(city.lower())

    if zone is None:
        return "i dont have any context of the city : {city}"

    return datetime.now(ZoneInfo(zone)).strftime("%d %B %Y, %I:%M %p")

@tool
def multiply(num1:int,num2:int)->int:
    """Use this tool for multiplying two numbers"""
    return num1*num2

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_the_current_time,multiply],
    system_prompt=(
        "You are a helpfull asistant"
        "choose the agent what fits in"
    ),
)

response = agent.invoke(
   {
      "messages":[
          {
              "role":"user",
              "content":"What is the current time in mumbai and what is the answer of 2*2?"
          }
      ]
   }
)
print("response:", response["messages"][-1].content)
print("Final Answer")
