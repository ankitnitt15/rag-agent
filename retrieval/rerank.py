from common.gemini_client import generate_content
from generation import prompts
from shared.models import PassageCandidate, RerankScore


def rerank(query: str, candidates: list[PassageCandidate], top_k: int = 10) -> list[PassageCandidate]:
    if not candidates:
        return []

    prompt = prompts.build_rerank_prompt(query, candidates)
    response = generate_content(
        prompt,
        config={"response_mime_type": "application/json", "response_schema": list[RerankScore]},
    )
    relevance_by_chunk = {score.chunk_id: score.relevance for score in response.parsed}

    for candidate in candidates:
        candidate.rerank_score = relevance_by_chunk.get(candidate.chunk_id, 0.0)

    candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
    return candidates[:top_k]
