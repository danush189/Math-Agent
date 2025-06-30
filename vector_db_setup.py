# Minimal example: Store GSM8K questions in Qdrant with embeddings

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# 1. Load a small subset of GSM8K
subset = load_dataset('gsm8k', 'main', split='train[:10]')
questions = [item['question'] for item in subset]

# 2. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions)

# 3. Connect to Qdrant (local, default port)
client = QdrantClient("localhost", port=6333)

# 4. Create a collection (if not exists)
collection_name = "gsm8k_questions"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

# 5. Upload data
client.upload_collection(
    collection_name=collection_name,
    vectors=embeddings,
    payload=[{"question": q} for q in questions],
    ids=None,  # auto-generate ids
    batch_size=10
)

print(f"Uploaded {len(questions)} questions to Qdrant!")
