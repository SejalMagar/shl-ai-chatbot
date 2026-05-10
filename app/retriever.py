import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer

model = None
index = None
catalog = []


def load_resources():

    global model
    global index
    global catalog

    if model is None:

        print("Loading model...")

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    if index is None:

        print("Loading FAISS index...")

        index = faiss.read_index("data/shl.index")

    if not catalog:

        print("Loading catalog...")

        with open("data/shl_catalog.json", "r") as f:

            catalog.extend(json.load(f))


def retrieve_assessments(query, top_k=5):

    load_resources()

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(catalog[idx])

    return results


def find_assessment_by_name(name):

    load_resources()

    name = name.lower()

    for item in catalog:

        if name in item["name"].lower():

            return item

    return None