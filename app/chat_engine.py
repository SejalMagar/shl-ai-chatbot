
from app.retriever import (
    retrieve_assessments,
    find_assessment_by_name
)


# Check vague queries
def needs_clarification(text):

    vague_queries = [
        "assessment",
        "test",
        "need assessment",
        "help"
    ]

    text = text.lower().strip()

    if len(text.split()) < 4:
        return True

    if text in vague_queries:
        return True

    return False


# Detect refinement queries
def is_refinement(text):

    refinement_words = [
        "actually",
        "also",
        "add",
        "include",
        "instead",
        "remove",
        "change",
        "leadership",
        "personality",
        "technical",
        "coding",
        "behavioral"
    ]

    text = text.lower()

    return any(
        word in text
        for word in refinement_words
    )


# Off-topic protection
def is_off_topic(text):

    blocked_topics = [
        "salary",
        "legal",
        "lawsuit",
        "politics",
        "medical",
        "ignore instructions",
        "system prompt",
        "reveal prompt",
        "hack"
    ]

    text = text.lower()

    return any(
        topic in text
        for topic in blocked_topics
    )


# Comparison feature
def compare_assessments(text):

    text = text.lower()

    if "difference" not in text and "compare" not in text:
        return None

    words = text.split()

    found = []

    for item in words:

        result = find_assessment_by_name(item)

        if result:
            found.append(result)

    if len(found) >= 2:

        a = found[0]
        b = found[1]

        return {
            "reply": (
                f"{a['name']} is a {a.get('test_type', 'Assessment')} assessment, "
                f"while {b['name']} is a {b.get('test_type', 'Assessment')} assessment."
            ),
            "recommendations": [
                {
                    "name": a["name"],
                    "url": a["url"],
                    "test_type": a["test_type"]
                },
                {
                    "name": b["name"],
                    "url": b["url"],
                    "test_type": b["test_type"]
                }
            ],
            "end_of_conversation": False
        }

    return None


# Main chatbot logic
def process_chat(messages):

    latest_message = messages[-1]["content"]

    full_text = " ".join([
        m["content"]
        for m in messages
    ])

    # Scope restriction
    allowed_keywords = [
        "assessment",
        "developer",
        "hiring",
        "skills",
        "java",
        "python",
        "coding",
        "manager",
        "sales",
        "technical",
        "communication"
    ]

    if not any(
        word in latest_message.lower()
        for word in allowed_keywords
    ):

        return {
            "reply": (
                "I can only help with SHL assessments "
                "and hiring-related recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # Off-topic refusal
    if is_off_topic(full_text):

        return {
            "reply": (
                "I can only assist with "
                "SHL assessment recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # Comparison handling
    comparison = compare_assessments(full_text)

    if comparison:
        return comparison

    # Clarification handling
    if needs_clarification(latest_message):

        return {
            "reply": (
                "Could you share the role, "
                "experience level, and key skills?"
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # Refinement handling
    if is_refinement(full_text):

        enhanced_query = full_text

        if "personality" in full_text.lower():
            enhanced_query += " personality behavioral"

        if "leadership" in full_text.lower():
            enhanced_query += " leadership manager"

        if "coding" in full_text.lower():
            enhanced_query += " technical coding developer"

        results = retrieve_assessments(
            enhanced_query,
            top_k=5
        )

    else:

        results = retrieve_assessments(
            full_text,
            top_k=5
        )

    recommendations = []

    for item in results:

        recommendations.append({
            "name": item.get("name", "Unknown"),
            "url": item.get("url", "#"),
            "test_type": item.get("test_type", "Assessment")
        })

    return {
        "reply": (
            f"Here are {len(recommendations)} recommended SHL assessments."
        ),
        "recommendations": recommendations,
        "end_of_conversation": False
    }