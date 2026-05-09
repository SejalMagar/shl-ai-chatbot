import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load catalog
with open("data/shl_catalog.json", "r") as f:

    catalog = json.load(f)

documents = []

for item in catalog:

    text = f"""
    Assessment: {item['name']}
    Type: {item['test_type']}
    """

    documents.append(text)

# Create embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Save vector database
faiss.write_index(index, "data/shl.index")

print("Embeddings created successfully.")