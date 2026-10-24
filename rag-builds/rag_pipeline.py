import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import (
    CharacterTextSplitter
)

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DATA_PATH = HERE / "data"

metadata = {
    "refunds.md": {"team": "support", "visibility": "public"},
    "shipping.md": {"team": "support", "visibility": "public"},
    "warranty.md": {"team": "support", "visibility": "public"},
    "error_codes.md": {"team": "engineering", "visibility": "internal"},
    "leave_policy.md": {"team": "hr", "visibility": "internal"},
    "security_policy.md": {"team": "engineering", "visibility": "internal"},
}

# Only chunks with this visibility are searched. Set to None to search everything.
VISIBILITY = "public"

# ---------- 1. INDEXING (runs once) ----------
documents = []

for path in sorted(DATA_PATH.glob("*.md")):
    documents.append(
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, **metadata.get(path.name, {})},
        )
    )


# chunks = RecursiveCharacterTextSplitter(
#     chunk_size=350,
#     chunk_overlap=50,
# ).split_documents(documents)

def fixed_size(docs):
    """Split each document into chunks of up to 350 characters."""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=350,
        chunk_overlap=0,
    )
    return splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chunks = fixed_size(documents)

print("\nRefund policy chunks:")
for i, chunk in enumerate(
    (c for c in chunks if c.metadata.get("source") == "refunds.md"),
    start=1,
):
    print(f"\n--- Chunk {i} ({len(chunk.page_content)} characters) ---")
    print(chunk.page_content)

store = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="rag-pipeline")

# ---------- 2. RETRIEVAL (per question) ----------
search_filter = {"visibility": VISIBILITY} if VISIBILITY else None
retriever = store.as_retriever(search_kwargs={"k": 3, "filter": search_filter})


def format_context(found):
    """Turn retrieved Documents into one block of text."""
    return "\n\n".join(
        f"[{doc.metadata['source']}] {doc.page_content}"
        for doc in found
    )


# ---------- 3. GENERATION (per question) ----------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Answer using only the context below. If the context does not contain "
         "the answer, say you do not have that information. Keep it to three "
         "lines.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)

chain = (
    {"context": retriever | format_context, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

question = "How long does a refund take after I return the item?"
# question = "what is the secuirty policy of your company ?"

print(chain.invoke(question))
