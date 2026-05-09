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
        "include"
    ]

    text = text.lower()

    return any(
        word in text
        for word in refinement_words
    )


# Off-topic and prompt injection protection
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

    if "difference" not in text:
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
                f"{a['name']} and "
                f"{b['name']} are different SHL assessments. "
                f"Both are used for hiring evaluation."
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
            "end_of_conversation": True
        }

    return None


# Main chatbot logic
def process_chat(messages):

    # Combine conversation
    full_text = " ".join([
        m["content"]
        for m in messages
    ])

    # Off-topic refusal
    if is_off_topic(full_text):

        return {
            "reply": (
                "I can only assist with "
                "SHL assessment recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": True
        }

    # Comparison handling
    comparison = compare_assessments(full_text)

    if comparison:

        return comparison

    # Clarification handling
    if needs_clarification(full_text):

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

        results = retrieve_assessments(
            full_text,
            top_k=5
        )

        recommendations = []

        for item in results:

            recommendations.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"]
            })

        return {
            "reply": (
                "Updated recommendations "
                "based on your new requirements."
            ),
            "recommendations": recommendations,
            "end_of_conversation": True
        }

    # Normal recommendation flow
    results = retrieve_assessments(
        full_text,
        top_k=5
    )

    recommendations = []

    for item in results:

        recommendations.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"]
        })

    return {
        "reply": (
            f"Here are {len(recommendations)} "
            "recommended SHL assessments."
        ),
        "recommendations": recommendations,
        "end_of_conversation": True
    }


# Local testing
if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": (
                "Hiring Java developer "
                "with stakeholder communication skills"
            )
        }
    ]

    response = process_chat(messages)

    print(response)