from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

image_url = "https://m.media-amazon.com/images/M/MV5BMDU0Y2UxMmUtN2ZmOS00MWM1LTg2ODUtYWZmNDZiMmY5MDVjXkEyXkFqcGc@._V1_QL75_UX328_.jpg"

url_message = HumanMessage(content=[
    {
       "type":"text",
       "text":"Describe this picture in two sentences. also talk about the chracters in this image." 
    },{
        "type":"image",
        "url":image_url
    }
])

print(model.invoke([url_message]).content)