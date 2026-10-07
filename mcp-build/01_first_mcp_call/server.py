from fastmcp import FastMCP

mcp = FastMCP("firstmcpserver")



@mcp.tool
def greet(name:str)->str:
    """this is a greet tool its use to greet the person"""
    return f'Hey {name} how are you ? hope you are fine'

@mcp.tool
def unique_way_print(name: str) -> str:
    '''Use this tool to print name in unique way'''
    return 'Haha' + name

@mcp.tool
def default_number_of_printing() -> int:
    '''Use this tool to get the default times of printing'''
    return 5

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)