from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def preview(text, width=80):
    """Return a one-line preview of a chunk, keeping the output readable."""
    return " ".join(text.split())[:width]

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

# print("----------------------------------------------------------------")
# print("1. similarity, k=2")
# for result in store.similarity_search(question, k=2):
#     print(f" [{result.metadata['source']}] {preview(result.page_content, 70)}")

# print("----------------------------------------------------------------")
# print("2. filtered to shipping.md")
# #searching chunks whose metadata contains shipping.md
# for result in store.similarity_search(
#     question,
#     k=3,
#     filter={"source":"shipping.md"}
# ):
#     print(f"[{result.metadata['source']}], {preview(result.page_content,70)}")


# relevant result + diverse result --> variety of useful info, not duplicates
# fetch_k --> gets 6 chunks --> k=3 --> from those 6, pick 3 relevant + diverse
# varied = store.as_retriever(search_type="mmr",
#                             search_kwargs={"k":3, "fetch_k":6})
# for result in varied.invoke(question):
#     print(f" [{result.metadata['source']}] {preview(result.page_content)}")

for text in ["warranty claim process", "who won the cricket match"]:
             scored = store.similarity_search_with_relevance_scores(text, k=3)
             print(f" scores for '{text}': {[round(score, 2) for _, score in scored]}")