from fastmcp import Client, FastMCP

my_mcp_client = FastMCP("MyMcpServer")

@my_mcp_client.tool
def double(number:int)->int:
    """Multiply the number by 2 and return after that"""
    return number*2

@my_mcp_client.tool
def add_then_double(number1:int,number2:int)->int:
    """Add the two number and then call the double tool/function"""
    sum = number1 + number2
    return double(sum)

@my_mcp_client.tool
async def sum_of_squares(num1:int,num2:int)->int:
    """return the sum of num1*num1 and num2*num2 by calling the tool from the http server"""
    async with Client("http://127.0.0.1:8002/mcp") as http:
        square_num1 = await http.call_tool("square",{"n": num1})
        square_num2 = await http.call_tool("square",{"n":num2})
    return square_num1.data + square_num2.data

if __name__ == "__main__":
    my_mcp_client.run(transport="stdio",show_banner=False)


