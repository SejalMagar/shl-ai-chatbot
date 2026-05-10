
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

    skills = item.get("skills", [])

    if isinstance(skills, list):
        skills = " ".join(skills)

    description = item.get("description", "")

    if isinstance(description, list):
        description = " ".join(description)

    category = item.get("category", "")

    if isinstance(category, list):
        category = " ".join(category)

    text = f"""
    Assessment Name: {item.get('name', '')}

    Description:
    {description}

    Skills:
    {skills}

    Category:
    {category}

    Test Type:
    {item.get('test_type', '')}
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
