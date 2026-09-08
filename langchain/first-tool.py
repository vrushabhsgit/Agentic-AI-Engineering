from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

@tool
def get_city_time(city:str)->str:
    """Get the current time in a city. Use it whenever the user asks about time."""
    zones = {
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "london": "Europe/London",
    "new york": "America/New_York",
    }

    zone = zones.get(city.lower())
    if zone is None:
     return f"I do not know the timezone for {city}."
    return datetime.now(ZoneInfo(zone)).strftime("%d %B %Y, %I:%M %p")

@tool
def multiply(numbers:list[int])->int:
   """Get the multiplication of two numbers. Use it whenver user asks about the multiplication of the numbers"""
   result = 1

   for n in numbers:
      result = result * n

   return result

print(get_city_time.invoke({"city": "Mumbai"}))
print(multiply.invoke({"numbers":[98765,43210]}))