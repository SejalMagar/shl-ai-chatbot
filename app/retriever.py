import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
index = faiss.read_index("data/shl.index")

# Load catalog data
with open("data/shl_catalog.json", "r") as f:

    catalog = json.load(f)

def retrieve_assessments(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Search similar vectors
    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(catalog[idx])

    return results

# Test search
if __name__ == "__main__":

    query = "Java developer assessment"

    results = retrieve_assessments(query)

    print("\nResults:\n")

    for item in results:

        print(item["name"])
        print(item["url"])
        print()

def find_assessment_by_name(name):

    name = name.lower()

    for item in catalog:

        if name in item["name"].lower():

            return item

    return None