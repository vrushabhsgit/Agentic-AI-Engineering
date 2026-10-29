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


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="the_rerank_demo"
)

class Rating(BaseModel):
    index: int = Field(description="Which candidate chunk this rates")
    score: int = Field(description="0 to 10 how useful for the question")

class Ratings(BaseModel):
    ratings: list[Rating]


reranker = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Ratings)

PROMPT = (
    "Rate how useful each chunk is for answering the question.\n"
    "10 means the chunk directly answers it.\n"
    "5 means related but does not answer it.\n"
    "0 means unrelated.\n"
    "Rate every chunk and be strict."
)

def preview(text, width=58):
    return " ".join(text.split())[:width]

def rerank(question, candidates, keep=3, minimum_score=3):
    numbered = "\n".join(
        f"[{number}] {' '.join(doc.page_content.split())}"
        for number, doc in enumerate(candidates, start=1)
    )

   
    result = reranker.invoke(f"{PROMPT}\n\nQuestion: {question}\n\nChunks:\n{numbered}")

    scored = []
    for rating in result.ratings:
        if 1 <= rating.index <= len(candidates):
            scored.append((rating.score, candidates[rating.index - 1]))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [pair for pair in scored[:keep] if pair[0] >= minimum_score]


question = "What does error ERR-500 mean?"

candidates = vector_store.similarity_search(question, k=8)
top = rerank(question, candidates, keep=3)

print(f"Loaded {len(chunks)} chunks from {DATA}")
for score, doc in top:
    print(f"Score {score} | [{doc.metadata['source']}] {preview(doc.page_content)}")