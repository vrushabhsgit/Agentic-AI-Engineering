import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

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

class Grade(BaseModel):
    sufficient: bool
    reason: str 

grader = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Grade)
rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0)
answer_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

GRADER_PROMPT = (
    "Decide whether these chunks are good enough to answer the question."
)
    
REWRITE_PROMPT = (
    "The search did not find what was needed. Rewrite the query to search "
    "again, using different wording."
)

ANSWER_PROMPT = (
    "Answer the question based on the provided context."
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="the_corrective_rag_demo"
)

def preview(text, width=58):
    return " ".join(text.split())[:width]

def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

question = "When will my order recive?"
search_query = question

for attempt in range(2):
    results = vector_store.similarity_search(search_query, k=3)
    context = build_context(results)
    
    grade = grader.invoke(f"{GRADER_PROMPT}\n\nQuestion: {question}\n\nChunks:\n{context}")
    
    if grade.sufficient:
        final_answer = answer_model.invoke(
            f"{ANSWER_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        ).content
        print(f"\nFinal Answer:\n{final_answer}")
        sys.exit(0)
        
    print(f"Attempt {attempt + 1}: Not sufficient, rewriting query...")
    search_query = rewriter.invoke(f"{REWRITE_PROMPT}\n\nOriginal question: {question}").content

print("I could not find this information in the documents.")