
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

query = "How can I launch an OpenAI-compatible API?"

documents = [
    "Start the OpenAI-compatible server using vllm serve.",
    "Dynamically load LoRA adapters using an HTTP endpoint.",
    "A risotto is prepared with rice and vegetable broth.",
]

embeddings = model.encode(
    [query] + documents,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

query_embedding = embeddings[0]
document_embeddings = embeddings[1:]

scores = document_embeddings @ query_embedding

ranking = sorted(
    zip(scores, documents),
    key=lambda item: float(item[0]),
    reverse=True,
)

for score, document in ranking:
    print(f"{float(score):.3f} | {document}")
