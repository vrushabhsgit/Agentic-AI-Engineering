import sys
from pathlib import Path
from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport

mcp = FastMCP("mymcpserver")

@mcp.tool
def square(n:int)->int:
    """Multiply the n and return"""
    return n*n

@mcp.tool
async def square_then_double(number:int)->int:
    """Double n using the tool on the stdio server and square it"""
    stdio_server = PythonStdioTransport(Path(__file__).parent / "stdio_server.py", python_cmd=sys.executable)
    async with Client(stdio_server) as stdio:
        doubleDude = await stdio.call_tool("double",{"number":number})
        return square(doubleDude.data)

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8002)