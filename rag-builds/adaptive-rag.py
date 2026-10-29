import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DATA = HERE / "data"

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
documents = [
      Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
      for path in sorted(DATA.glob("*.md"))
]

chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(documents)

class Route(BaseModel):
    needs_retrivel:bool

router = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Route)

routing_prompt = (
    "Decide whether this message needs a search of the company documents. "
    "The documents cover refunds, shipping, delivery, warranty, checkout "
    "error codes, the leave policy and internal security rules.\n"
    "Answer false only for greetings, thanks and small talk that ask for "
    "nothing.\n"
    "When in doubt, answer true. Answering from memory instead of the "
    "documents is the worst thing this system can do."
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="the_adaptive_rag_demo"
)


def preview(text, width=58):
    return " ".join(text.split())[:width]

question = "When will my order recive?"


route = router.invoke(f"{routing_prompt}\n\nMessage: {question}")


if not route.needs_retrivel:
    print("No retrieval needed. This is just small talk.")
    
else:
    print("Retrieval needed. Searching the vector store...")
    
    
    results = vector_store.similarity_search(question, k=3)
    
    for rank, doc in enumerate(results, start=1):
        print(f"  {rank}. [{doc.metadata['source']}] {preview(doc.page_content)}")

