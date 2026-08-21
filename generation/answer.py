import uuid

from common.gemini_client import generate_content
from generation import claims, prompts, query_understanding, vote
from retrieval import bm25, rerank, rrf
from storage import vector_store
from shared.embedder import embed_query
from shared.models import AnswerResponse, PassageCandidate, PassageRef

CONFIDENCE_FLOOR = 0.4


def _refused_response(reason: str, query_id: str) -> AnswerResponse:
    return AnswerResponse(
        answer="Sorry, I could not find anything relevant.",
        refused=True,
        refusal_reason=reason,
        passages=[],
        confidence=0.0,
        query_id=query_id,
    )


def answer(
    query: str,
    qdrant_client,
    bm25_index: bm25.BM25Index,
    use_voting: bool = False,
) -> AnswerResponse:
    query_id = str(uuid.uuid4())

    # --- retrieval: dense (Qdrant) + sparse (BM25), fused by rank (RRF) ---
    filters = query_understanding.extract_filters(query, bm25_index.known_tags())
    print(f"[debug] extracted filters: {filters}")
    query_vector = embed_query(query)
    dense_hits = vector_store.search(qdrant_client, query_vector, top_k=40, filters=filters)
    sparse_hits = bm25_index.search(query, top_k=40, filters=filters)

    dense_lookup = {hit["chunk_id"]: hit for hit in dense_hits}
    sparse_lookup = dict(sparse_hits)
    dense_ranked = [(hit["chunk_id"], hit["score"]) for hit in dense_hits]

    fused = rrf.rrf_fusion(dense_ranked, sparse_hits)

    candidates = []
    for chunk_id, fused_score in fused:
        if chunk_id in dense_lookup:
            hit = dense_lookup[chunk_id]
            text, doc_id = hit["text"], hit["doc_id"]
        else:
            # sparse-only hit: hydrate text from BM25's own store instead of Qdrant.
            # doc_id is derived from our "<doc_id>:<position>" chunk_id convention.
            text = bm25_index.get_text(chunk_id)
            doc_id = chunk_id.split(":")[0]

        candidates.append(
            PassageCandidate(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                dense_score=dense_lookup.get(chunk_id, {}).get("score"),
                sparse_score=sparse_lookup.get(chunk_id),
                fused_score=fused_score,
            )
        )

    # --- rerank + confidence floor ---
    ranked = rerank.rerank(query, candidates, top_k=10)

    if not ranked or ranked[0].rerank_score <= CONFIDENCE_FLOOR:
        return _refused_response("low_retrieval_confidence", query_id)

    top_passages = ranked[:5]

    # --- generation ---
    prompt = prompts.build_generation_prompt(query, top_passages)
    raw_answer = generate_content(prompt, config={"temperature": 0}).text

    if raw_answer.strip().startswith("INSUFFICIENT_CONTEXT"):
        return _refused_response("no_relevant_passages", query_id)

    # --- claim-by-claim fact-checking against the retrieved passages ---
    atomic_claims = claims.extract_atomic_claims(raw_answer)
    if use_voting:
        verifications = [vote.verify_claim_with_voting(claim, top_passages) for claim in atomic_claims]
    else:
        verifications = [claims.verify_claim_against_passages(claim, top_passages) for claim in atomic_claims]

    if not all(v.verdict == "ENTAILED" for v in verifications):
        return _refused_response("unverified_claim", query_id)

    passage_refs = [
        PassageRef(doc_id=p.doc_id, chunk_id=p.chunk_id, text=p.text, score=p.rerank_score)
        for p in top_passages
    ]
    confidence = sum(p.rerank_score for p in top_passages) / len(top_passages)

    return AnswerResponse(
        answer=raw_answer,
        refused=False,
        refusal_reason=None,
        passages=passage_refs,
        confidence=confidence,
        query_id=query_id,
    )
