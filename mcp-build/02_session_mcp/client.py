
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

SERVER_PATH = Path(__file__).resolve().parent / "server.py"

print("Python:", sys.executable)
print("Server exists:", SERVER_PATH.is_file())

mcp_client = MultiServerMCPClient({
    "cart_server": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_PATH)],
        "cwd": str(SERVER_PATH.parent),
        "env": os.environ.copy()
    }
})



async def main():
    async with mcp_client.session("cart_server") as session:
        tools = await load_mcp_tools(session)

        tools_by_name = {}
        for tool in tools:
            tools_by_name[tool.name] = tool

        result = await tools_by_name["add_to_cart"].ainvoke({"item": "apple"})
        print(result)

        result = await tools_by_name["add_to_cart"].ainvoke({"item": "banana"})
        print(result)

        result = await tools_by_name["add_to_cart"].ainvoke({"item": "grapes"})
        print(result)

        result = await tools_by_name["show_cart"].ainvoke({})
        print(result)

        result = await tools_by_name["get_session_id"].ainvoke({})
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
