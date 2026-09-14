from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

@tool
def book_seat(name:str,seat:str)->str:
    """Book a seat for the intersteller movie"""
    return f"your seat is booked for the intrsteller movie let grab popcorn and enjoy :)"

agent = create_agent(
    model="gpt-4o-mini",
    tools=[book_seat],
    system_prompt=(
        "you are a movie seat booking agent and helping people to book seat for the movie?"
    ),
    checkpointer=InMemorySaver()#it saves the converation of each step
)

def ask(question,thread_id):
    result = agent.invoke({'messages':{"role":"user","content":question}},
    config={"configurable": {"thread_id": thread_id}},
    )
    print(f"[{thread_id}] Q: {question}")
    print(f"[{thread_id}] A: {result['messages'][-1].content}")
    print()

ask("Hey hi",thread_id="vinay")
ask("Book me a seat A12.",thread_id="vinay")
ask("What is my name and what seat did i book",thread_id="vinay")
# ask("What is my name and what seat did i book",thread_id="nimish")