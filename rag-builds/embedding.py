import sys 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embedding_model.embed_query("I love to build")

print('Type: ', type(vector).__name__)
print('Dimensions: ', len(vector))
print("First five dimensions: ",[round(number,4) for number in vector[:5]])

def cosine_similarity(first,second):
    """How close two vectors point. 1 means same direction, 0 means unrelated."""
    # Multiply numbers at the same positions and add the results.
    dot = sum(a * b for a, b in zip(first, second))

    #Calculate the magnitude/length of the first vector.
    size_first = sum(a * a for a in first) ** 0.5

    # Calculate the magnitude/length of the second vector.
    size_second = sum(b * b for b in second) ** 0.5

    # similarity search final formula
    return dot / (size_first * size_second)


question = "I want my money back"

candidates = [
    "How do refunds work?",
    "Refunds are processed in 5 to 7 working days.",
    "When will my order be delivered?",
    "The capital of France is Paris.",
]

# embed_documents takes a list and returns one vector per item, in one API call
question_vector = embedding_model.embed_query(question)
candidates_vector = embedding_model.embed_documents(candidates)

scored = sorted(
    zip(candidates, candidates_vector),
    key=lambda pair: cosine_similarity(question_vector, pair[1]),
    reverse=True,
)

for text, candidates_vector in scored:
    score = cosine_similarity(question_vector, candidates_vector)
    print(f" {score:.3f} {text}")

    