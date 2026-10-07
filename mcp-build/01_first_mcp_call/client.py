#all the importst that are needed
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import sys
from pathlib import Path
import asyncio

#load the env
load_dotenv()

#path of the file
SERVER_PATH = Path(__file__).parent / "server.py"

#parametrs of the servers to call the server basically
server_params = {
    "greetings": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_PATH)]
    }
}


async def main():
    #client created 
    client = MultiServerMCPClient(server_params)
     #got the tools from the server
    tools = await client.get_tools()
    #llm needed to call
    llm = ChatOpenAI(model="gpt-4o-mini")
    #bind the tools that we got from the mcp servers
    llm_with_tools = llm.bind_tools(tools)
    #maintain the history
    history = [
        HumanMessage(
            content="Print Vrushabh in unique way default times"
        )
    ]
    #call the tools for the messages we get from the user
    response = await llm_with_tools.ainvoke(history)
    #append to the hisotry becuase it includes the ai message as well
    history.append(response)
   #create a loopup tool by name
    tools_by_name = {
        tool.name: tool
        for tool in tools
    }
    #tool call loop
    for tool_call in response.tool_calls:
        #get the tool from the loopup
        tool = tools_by_name[tool_call["name"]]
        #call the tool with the tool name
        tool_message = await tool.ainvoke(tool_call)

        print(tool_message.content)

        history.append(tool_message)

    final_response = await llm_with_tools.ainvoke(history)

    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())