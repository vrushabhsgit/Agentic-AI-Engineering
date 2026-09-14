
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()


# our data --> imagine it is our order database
ORDERS = {
    "ORDER-101":{"item":"Sony WH-CH520","status":"shipped","Amount":44499},
    "ORDER-102":{"item":"Keychron Keyboard Mechanical","status":"in-cart","price":11000}
}

# imagine we have warehouse stock
STOCK = {
    "Sony WH-CH520":15,"Keychron Keyboard Mechanical":0,"Cetaphil-oily-skin-faewash":2
}

@tool
def get_status(order_id:str)->str:
    """Get the order status and the amount of the order using its order_id ex:order_101"""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return f"No order found with the {order_id}"
    return f"pick your order : order name : {order["item"]}, order status : {order["status"]}, order amount : {order["price"]}"

@tool
def check_stock(item:str)->str:
    """check of how many units of stock left in the warehouse"""
    count = STOCK.get(item)
    if count is None:
        return f"This item : {item} is not in stock"
    return f"the count of the {item} in the stock is {count}"

@tool
def apply_discount(amount:float,percent:float)->float:
    """apply the discount on the existing amount and return the new amont"""
    return round(amount - (amount * percent / 100), 2)

@tool
def dilivery_days(pin_code:str)->str:
    """Check the dilivery days acording to the given the given pin_code"""
    metro_cities_pin_code = {"4001","4002","4003"}
    if pin_code not in metro_cities_pin_code:
        return "You wil recive your dilivery in 5 days"
    return "You will recive your dilivery in 2 days"


agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_status,check_stock,apply_discount,dilivery_days],
    system_prompt=(
        "Your are a helpfull asistant working for a online ecoomerce platform"
        "give the answers of the queries wahtever the user will ask example:status of the order"
        "stock avilabilty, and doing work for applying discount, giving infomration about dilivery days"
        "dont give the random answers give the original one using tools"
    )
)

def ask_questions(que):
    print("-" * 60)
    print('question: ', que)
    answer = agent.invoke({"messages":[{"role":"user","content":que}]})
    print("answer: ", answer["messages"][-1].content)
    print("-" * 60)

ask_questions("I want to purchase the mechanical keyboard check if avilable or not")
