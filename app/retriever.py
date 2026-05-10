
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
    "coding": ["coding", "developer", "technical"],
    "spring": ["spring", "backend", "java"],
    "backend": ["backend", "api", "developer"],
    "software": ["software", "developer", "technical"]
}


# Keyword score
def keyword_score(query, text):

    query_words = set(query.lower().split())

    text = text.lower()

    score = 0

    for word in query_words:

        if word in text:
            score += 1

    return score


# Reranking function
def rerank_results(query, results):

    query_words = set(query.lower().split())

    scored = []

    for item in results:

        skills = item.get("skills", [])

        if isinstance(skills, list):
            skills = " ".join(skills)

        description = item.get("description", "")

        if isinstance(description, list):
            description = " ".join(description)

        category = item.get("category", "")

        if isinstance(category, list):
            category = " ".join(category)

        text = (
            item.get("name", "") + " " +
            description + " " +
            skills + " " +
            category
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

    # Create query embedding
    query_embedding = model.encode([query])

    # Search FAISS
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k * 5
    )

    boosted_results = []

    technical_keywords = [
        "java",
        "python",
        "developer",
        "backend",
        "software",
        "coding",
        "spring",
        "api"
    ]

    technical_query = any(
        word in query_lower
        for word in technical_keywords
    )

    for idx in indices[0]:

        item = catalog[idx]

        score = 0

        skills = item.get("skills", [])

        if isinstance(skills, list):
            skills = " ".join(skills)

        description = item.get("description", "")

        if isinstance(description, list):
            description = " ".join(description)

        category = item.get("category", "")

        if isinstance(category, list):
            category = " ".join(category)

        combined_text = (
            item.get("name", "") + " " +
            description + " " +
            skills + " " +
            category
        ).lower()

        # Keyword boosting
        for key, words in skill_keywords.items():

            if key in query_lower:

                for word in words:

                    if word in combined_text:
                        score += 3

        # Hybrid keyword retrieval
        keyword_boost = keyword_score(
            query,
            combined_text
        )

        score += keyword_boost * 2

        # Technical boosting
        if technical_query:

            technical_match = any(
                word in combined_text
                for word in technical_keywords
            )

            if technical_match:
                score += 8
            else:
                score -= 2

        boosted_results.append((score, item))

    # Sort by score
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

        if name in item.get("name", "").lower():

            return item

    return None
