import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever


load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

HERE = Path(__file__).parent
DATA_PATH = HERE / "data"

documents = [
    Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
    for path in sorted(DATA_PATH.glob("*.md"))
]

chunks = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30).split_documents(documents)

vector_store = Chroma.from_documents(
    chunks, embedding=embeddings, collection_name="hybrid_search"
)


keyword_search = BM25Retriever.from_documents(chunks)
keyword_search.k = 4

vector_search = vector_store.as_retriever(search_kwargs={"k": 4})

def preview(text, width=60):
    """Convert a chunk into one short readable line."""
    return " ".join(text.split())[:width]

def add_fuse(result_list, k=60, top_n=4):
    """Reciprocal rank fusion over several ranked documents"""
    scores = {}  # chunk text -> combined score
    seen = {}    # chunk text -> the original Document object

    for result in result_list:
        for rank, doc in enumerate(result, start=1):
            key = doc.page_content
            seen[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(seen[key], score) for key, score in ranked[:top_n]]


question = "ERR-5090"
question = "what if the courier cannot find my area?"

# fused = add_fuse(
#     [
#         vector_search.invoke(question),
#         keyword_search.invoke(question)
#     ]
# )


# for doc, score in fused:
#     print(f"{score:.5f} | [{doc.metadata['source']}] {preview(doc.page_content)}")

def compare(question):
    print("=" * 75)
    print(f"QUESTION: {question}")
    print("=" * 75)

    # 1. Vector Search Only
    vector_results = vector_search.invoke(question)
    print("\n--- 1. VECTOR SEARCH ---")
    for rank, doc in enumerate(vector_results, start=1):
        print(f"  {rank}. [{doc.metadata['source']}] {preview(doc.page_content)}")

    # 2. Keyword Search (BM25) Only
    keyword_results = keyword_search.invoke(question)
    print("\n--- 2. KEYWORD SEARCH (BM25) ---")
    for rank, doc in enumerate(keyword_results, start=1):
        print(f"  {rank}. [{doc.metadata['source']}] {preview(doc.page_content)}")

    # 3. Fused (Hybrid Search with RRF)
    fused_results = add_fuse([vector_results, keyword_results])
    print("\n--- 3. FUSED (RRF) ---")
    for rank, (doc, score) in enumerate(fused_results, start=1):
        print(f"  {rank}. ({score:.5f}) [{doc.metadata['source']}] {preview(doc.page_content)}")
    print("\n")


# Run the comparison on both questions!
# compare("what if the courier cannot find my area?")
compare("ERR-5000")