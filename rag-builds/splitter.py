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

print(f"Loaded {len(documents)} documents\n")
print(f"{'File':<20} {'Characters':>10}")
print("-" * 31)

for document in documents:
    filename = document.metadata["source"]
    character_count = len(document.page_content)
    print(f"{filename:<20} {character_count:>10}")