import asyncio
import sys
from pathlib import Path
from fastmcp import Client
from fastmcp.exceptions import ToolError

SERVER_PATH = Path(__file__).parent / "server.py"

async def try_tool(session, tool_name: str, arguments: dict):
    print(f"\nCalling {tool_name}({arguments})")
    try:
        result = await session.call_tool(tool_name, arguments)
        content = result.content[0].text if result.content else ""
        print(" OK ->", content)
    except ToolError as e:
        print(" ERROR ->", e)
    except Exception as e:
        print(" ERROR ->", e)

async def main():
    async with Client(SERVER_PATH) as client:
        await try_tool(client, "divide", {"num1": 10, "num2": 2})
        await try_tool(client, "divide", {"num1": 10, "num2": 0})
        await try_tool(client, "read_the_secret_file", {})

if __name__ == "__main__":
    asyncio.run(main())