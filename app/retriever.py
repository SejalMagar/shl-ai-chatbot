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


# Skill keyword boosting
skill_keywords = {
    "java": ["java", "backend", "developer"],
    "python": ["python", "developer"],
    "communication": ["communication", "stakeholder"],
    "leadership": ["leadership", "manager"],
    "sales": ["sales", "account manager"],
    "coding": ["coding", "developer", "technical"]
}


# Reranking function
def rerank_results(query, results):

    query_words = set(query.lower().split())

    scored = []

    for item in results:

        text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            item.get("skills", "")
        ).lower()

        overlap = len([
            word for word in query_words
            if word in text
        ])

        scored.append((overlap, item))

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [x[1] for x in scored]


# Main retrieval function
def retrieve_assessments(query, top_k=5):

    query_lower = query.lower()

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Search vectors
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k * 3
    )

    boosted_results = []

    for idx in indices[0]:

        item = catalog[idx]

        score = 0

        combined_text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            item.get("skills", "") + " " +
            item.get("category", "")
        ).lower()

        # Keyword boosting
        for key, words in skill_keywords.items():

            if key in query_lower:

                for word in words:

                    if word in combined_text:
                        score += 2

        # Technical filtering boost
        technical_keywords = [
            "java",
            "python",
            "developer",
            "coding",
            "software"
        ]

        if any(
            word in query_lower
            for word in technical_keywords
        ):

            if "technical" in combined_text:
                score += 5

        boosted_results.append((score, item))

    # Sort by boosted score
    boosted_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = [
        x[1]
        for x in boosted_results[:top_k]
    ]

    # Final reranking
    results = rerank_results(query, results)

    return results


# Find assessment by name
def find_assessment_by_name(name):

    name = name.lower()

    for item in catalog:

        if name in item["name"].lower():

            return item

    return None


# Local test
if __name__ == "__main__":

    query = "Java developer communication"

    results = retrieve_assessments(query)

    print("\nResults:\n")

    for item in results:

        print(item["name"])
        print(item["url"])
        print()