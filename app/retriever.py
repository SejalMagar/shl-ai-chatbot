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
        top_k * 3
    )

    query_lower = query.lower()

    scored_results = []

    for idx in indices[0]:

        item = catalog[idx]

        score = 0

        text = (
            item["name"] + " " +
            item.get("description", "") + " " +
            item.get("test_type", "")
        ).lower()

        # Keyword boosting
        keywords = query_lower.split()

        for word in keywords:

            if word in text:
                score += 2

        # Skill boosting
        important_words = [
            "java",
            "python",
            "developer",
            "software",
            "coding",
            "programming",
            "communication",
            "stakeholder",
            "leadership",
            "manager",
            "analyst"
        ]

        for word in important_words:

            if word in query_lower and word in text:
                score += 5

        scored_results.append(
            (score, item)
        )

    # Sort by score
    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    final_results = [
        item for score, item
        in scored_results[:top_k]
    ]

    return final_results


def find_assessment_by_name(name):

    name = name.lower()

    for item in catalog:

        if name in item["name"].lower():

            return item

    return None


# Test search
if __name__ == "__main__":

    query = "Java developer assessment"

    results = retrieve_assessments(query)

    print("\nResults:\n")

    for item in results:

        print(item["name"])
        print(item["url"])
        print()