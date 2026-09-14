import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

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

#put the tools in the list
tools_list = [get_the_current_time,multiply]

#create a dictorny with the track of the tool names
tool_names = {tool.name: tool for tool in tools_list}

# model_with_tools = model.bind_tools(tools_list)

model = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools(tools_list)


#create the conversation
messages = [
    SystemMessage(
        content="You are the assitant"
    ),
    HumanMessage(
        content="What is the current time in mumbai, and what is the multiplication of the numbers 2*2"
    )
]

step = 1

while True:
    response = model.invoke(messages)

    #keep the models own reply in the conversation
    #so the next turn knows what was the prevoius context
    messages.append(response)

    if not response.tool_calls:
        print("\nFinal answer:")
        print(response.content)
        break

    print(f"Step {step}: the model asked for {len(response.tool_calls)} tool calls")

    for call in response.tool_calls:
        #  print("What is call",call)
         

        # call contains: tool name, arguments, tool call id
         tool_to_run = tool_names[call['name']]
        # passing the whole call back gives a ToolMessage that is already
        # tied to this request by its id
         tool_message = tool_to_run.invoke(call)
          
         print(f"{step}{call['name']}({call['args']}) -> {tool_message.content}")
         messages.append(tool_message)
step += 1
print("\nMessages in the conversation:", [type(m).__name__ for m in messages])

