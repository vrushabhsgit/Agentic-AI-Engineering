#loaded the document
from pathlib import Path
from langchain_core.documents import Document

HERE = Path(__file__).parent
documents = []

for path in sorted((HERE / "data").glob("*.md")):
    documents.append(
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
    )

print(f"Loaded {len(documents)} documents")

for document in documents:
    filename = document.metadata["source"]
    character_count = len(document.page_content)
    print(f"{filename:<15} {character_count:>5} characters")

#2. splitting work
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)
print(f"{len(documents)} documents became {len(chunks)} chunks")