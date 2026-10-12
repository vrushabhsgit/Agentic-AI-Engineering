import asyncio
import sys
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

stdio_server = PythonStdioTransport(
    Path(__file__).parent/"stdio_server.py", python_cmd=sys.executable
)
http_server = "http://127.0.0.1:8002/mcp"

async def main():
    async with Client(stdio_server) as stdio, Client(http_server) as http:
        result1 = await stdio.call_tool("add_then_double",{"number1":3,"number2":4})
        print(f'1. {result1.data}')

        result2 = await stdio.call_tool("sum_of_squares",{"num1":3,"num2":4})
        print(f'2. {result2.data}')

        result3 = await http.call_tool("square_then_double",{"number":3})
        print(f'3. {result3.data}')

if __name__ == "__main__":
    asyncio.run(main())