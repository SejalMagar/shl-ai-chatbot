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
    Assessment Name: {item.get('name', '')}

    Description: {item.get('description', '')}

    Test Type: {item.get('test_type', '')}

    Category: {item.get('category', '')}

    Skills: {item.get('skills', '')}

    Job Level: {item.get('job_level', '')}

    Remote Testing: {item.get('remote_testing', '')}

    Adaptive Support: {item.get('adaptive_support', '')}
    """

    documents.append(text)

# Create embeddings
embeddings = model.encode(documents)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save vector database
faiss.write_index(index, "data/shl.index")

print("Improved embeddings created successfully.")