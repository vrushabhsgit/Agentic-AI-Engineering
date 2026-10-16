from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

sentences = [
    "Refunds take 5 to 7 working days.",
    "Delivery to metro cities takes 2 days.",
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = embeddings.embed_documents(sentences)
print("OpenAI text-embedding-3-small")
print(" vectors :", len(vectors)) # 2
print(" dims :", len(vectors[0])) # 1536
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# embeddings = HuggingFaceEmbeddings(
# model_name="sentence-transformers/all-MiniLM-L6-v2")
# every one of them has the same two methods
# embeddings.embed_query("python")
# embeddings.embed_documents(sentences)
