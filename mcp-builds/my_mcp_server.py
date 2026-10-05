from fastmcp import FastMCP

my_mcp_server = FastMCP("first_mcp_server")

@my_mcp_server.tool
def greet()->str:
    """This is the greeting tool use this to greet someone"""
    return "Hey hi hello i am a vrushabh's ai agent how can i help you"

@my_mcp_server.tool
def add_two_numbers(num1,num2)->str:
    """This is the tool for adding two numbers"""
    result = num1 + num2
    return f'addition = {result}'

def main():
    my_mcp_server.run(transport="stdio")

if __name__ == "__main__":
    main()

    


