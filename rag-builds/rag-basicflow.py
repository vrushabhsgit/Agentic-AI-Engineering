from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i} ({len(chunk.page_content)} characters) — {chunk.metadata['source']}")
    print(chunk.page_content)

# print(f"Loaded {len(documents)} documents; created {len(chunks)} chunks")

store = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="rag_demo"
)

retriver = store.as_retriever(search_kwargs={"k": 3})

def format_context(found):
    """Turn retrieved Documents into one block of text."""
    return "\n\n".join(
        f"[{doc.metadata['source']}] {doc.page_content}"
        for doc in found
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support agent. Answer only from the context below. "
            "If the context does not cover the question, say you do not have that "
            "information and offer to connect a human. Keep it to three lines.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {
     "context":retriver | format_context,
     "question": RunnablePassthrough() # keep the original question unchanged
    }
    | prompt # insert context + question into the template
    | model # send the completed prompt to the LLM
    | StrOutputParser() # convert the response into a normal string
)

question = "How long does a refund take after I return the item?"
chain.invoke(question)

