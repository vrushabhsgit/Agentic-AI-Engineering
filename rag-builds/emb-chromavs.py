from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Set the input folder and the folder where Chroma will save the vector store.
HERE = Path(__file__).parent
DATA_PATH = HERE / "data"
STORE_PATH = HERE / "chroma_store"

# Stage 1: Load each Markdown file as a document, keeping its filename as metadata.
documents = [
    Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": path.name},
    )
    for path in sorted(DATA_PATH.glob("*.md"))
]

# Split documents into smaller, overlapping chunks for more focused retrieval.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents; created {len(chunks)} chunks")

# Stage 2: Turn chunks into embeddings and add them to the Chroma vector store.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="support_docs",
    persist_directory=str(STORE_PATH),
)

# Stage 3: Search the stored chunks for the two most relevant to the question.
# question = "how long until i get my money back"
question = "my laptop screen stopped working after two month"
results = store.similarity_search(question, k=2)

# Display each matching chunk with its source filename.
# for result in results:
#     print(f"[{result.metadata['source']}]")
#     print(result.page_content[:300])

# def preview(text, width=80):
#     """Return a one-line preview of a chunk, keeping the output readable."""
#     return " ".join(text.split())[:width]

# print(" ")
# print("With distances:")
# for result, distance in store.similarity_search_with_score(question, k=3):
#     print(
#         f" {distance:.3f} [{result.metadata['source']}] "
#         f"{preview(result.page_content, 60)}"
#     )

# print("1. similarity, k=2")
# for result in store.similarity_search(question, k=2):
#     print(f" [{result.metadata['source']}] {preview(result.page_content, 70)}")
# store.add_documents(
#     [
#         Document(
#             page_content=(
#                 "Refunds for cash on delivery orders "
#                 "are sent to your bank account."
#             ),
#             metadata={"source": "refunds.md"},
#         )
#     ]
# )

# print(" ")
# print("After adding one document:", len(store.get()["ids"]), "chunks in the store")