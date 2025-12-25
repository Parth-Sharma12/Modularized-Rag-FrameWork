def build_context(ranked_docs: list[tuple[str, float]]) -> str:
    context_lines = []
    for doc, score in ranked_docs:
        context_lines.append(f"- {doc}")
    return "\n".join(context_lines)
