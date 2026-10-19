"""Show what happens when the original ingestion pattern runs repeatedly.

Run this script twice. It reads the same Markdown files, makes the same
chunks, and calls Chroma.from_documents without supplying IDs, just like
emb-chromavs.py. This demo uses its own persistent store folder.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

HERE = Path(__file__).parent
DATA_PATH = HERE / "data"
DEMO_STORE_PATH = HERE / "duplicate_demo_store"

# Load the same Markdown source files as the original script.
documents = [
    {
        "page_content": path.read_text(encoding="utf-8"),
        "metadata": {"source": path.name},
    }
    for path in sorted(DATA_PATH.glob("*.md"))
]

# Split the source documents into chunks using the original settings.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

# Convert dictionaries to LangChain Documents, then split them.
from langchain_core.documents import Document

chunks = splitter.split_documents(
    [Document(**document) for document in documents]
)

print(f"Loaded {len(documents)} documents; created {len(chunks)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Same pattern as the original: no IDs are supplied here.
store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="support_docs",
    persist_directory=str(DEMO_STORE_PATH),
)

print(f"Chunks in this persistent demo store: {len(store.get()['ids'])}")
print(f"Demo store folder: {DEMO_STORE_PATH}")
