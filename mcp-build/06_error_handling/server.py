from pathlib import Path
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# mask_error_details = for UNEXPECTED errors, the client only gets a
# generic message. The real error stays on the server.
mcp = FastMCP("StdioServer", mask_error_details=True)

@mcp.tool
def divide(num1:int,num2:int)->float:
    """Use this tool to divide the two numbers"""
    if num2 == 0:
        raise ToolError(f"Cannot divide by zero please provide a valid number. " \
        "please use diffrent number for num2 : {num2}")
    return num1/num2

@mcp.tool
def read_the_secret_file()->str:
    """a tool with bug it tries to read the file that does not exists"""
    return open(Path(__file__).parent/"passwords.txt").read()

if __name__ == "__main__":
    mcp.run(transport="stdio",show_banner=False)


