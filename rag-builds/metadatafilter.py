import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DATA_PATH = HERE/"data"

METADATA = {
    "refunds.md": {"team": "support", "visibility": "public"},
    "shipping.md": {"team": "support", "visibility": "public"},
    "warranty.md": {"team": "support", "visibility": "public"},
    "error_codes.md": {"team": "engineering", "visibility": "internal"},
    "leave_policy.md": {"team": "hr", "visibility": "internal"},
    "security_policy.md": {"team": "engineering", "visibility": "internal"},
}

documents = [
    Document(page_content=path.read_text(encoding="utf-8"),metadata={"source":path.name})
    for path in sorted(DATA_PATH.glob("*md"))
]

chunks = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50).split_documents(documents)

store = Chroma.from_documents(
    chunks,embedding=OpenAIEmbeddings(model="text-embedding-3-small"),collection_name="metadata_filtering"
)

class Search_Plan(BaseModel):
    query: str = Field(
        description="The part of the question to search for, without the "
        "team or location constraint")
    team: Literal["support", "engineering", "hr", "any"] = Field(
        description="Which team owns the document. Use 'any' when the "
        "question does not say"
        )
planner = ChatOpenAI(model="gpt-4o-mini",
temperature=0).with_structured_output(Search_Plan)

PLANNER_PROMPT="""Turn the user's question into a search plan.

Teams that exist: support (refunds, shipping, warranty), engineering (error
codes, security), hr (leave policy). Use 'any' unless the question clearly
points at one team.

Keep the query focused on what to look for, not on where to look."""

def preview(text, width=58):
    """
    Make a retrieved chunk short enough to display nicely.
    """

    return " ".join(
        text.split()
    )[:width]

def search(question):
    #1. LLM Produces a validated plan
    plan = planner.invoke(f'{PLANNER_PROMPT} \n\n Question:{question}')

    #2. Our code turns out that plan into the filter
    where = None if plan.team == "any" else {"team":plan.team}

    #3. The vector search narrows down if we have the filter
    results = store.similarity_search(plan.query,k=3,filter=where)

    print(f"plan : query={plan.query!r} team={plan.team}")
    for doc in results:
        print(f'[Source:{doc.metadata["source"]}], preview: {preview(doc.page_content)}')

search("What did HR say about carrying leaves forward?")
search("What does error ERR-500 mean?")
search("How long does a refund take?")
search("What is the name of the owner of this RAG application?")
