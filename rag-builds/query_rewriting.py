import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")


HERE = Path(__file__).parent
DATA_PATH = HERE/"data"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Stage 1: Load each Markdown file as a document, keeping its filename as metadata.
documents = [
    Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": path.name},
    )
    for path in sorted(DATA_PATH.glob("*.md"))
]

# Split documents into smaller, overlapping chunks for more focused retrieval.
chunks = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=50).split_documents(documents)

# store the chunks in the vectore store and create the emeddings
store = Chroma.from_documents(chunks,embedding=embeddings,collection_name="query_rewriting_practice")

#. step 2 is for the retrivel
retriever = store.as_retriever(search_kwargs={"k":3})

model = ChatOpenAI(model="gpt-4o-mini")

def preview(text, width=62):
    return " ".join(text.split())[:width]

def show(label, found):
    print(f"   {label}")
    for doc in found:
        print(f"      [{doc.metadata['source']:18}] {preview(doc.page_content)}")


rewrite = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's last message as one standalone search query. "
         "Use the conversation for context. Return the query only, no quotes, "
         "no explanation."),
        ("human", "Conversation so far:\n{history}\n\nLast message: {question}"),
    ])
    | model
    | StrOutputParser()
)
# "and what if I spilled water on it"
# "what about that one"
# "is it covered"

# these questions may make sesne in the conversation, but the vector store does not automatically understand the conversation

history = """customer: My keyboard stopped working, I bought it two months ago.
agent: Sorry to hear that. It is covered under warranty for manufacturing defects."""
follow_up = "and what if I spilled water on it"

standalone = rewrite.invoke({

    "history": history,

    "question": follow_up

}).strip()

print(
    "Follow up question:",
    follow_up
)

print(
    "Rewritten as      :",
    standalone
)

print()
# Search with the raw vague question.
show(
    "retrieved with the raw question:",
    retriever.invoke(follow_up)
)


# Then:
#
# Search with the rewritten standalone question.
show(
    "retrieved with the rewritten query:",
    retriever.invoke(standalone)
)
