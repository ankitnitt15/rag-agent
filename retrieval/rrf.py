def rrf_fusion(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    # Ignores the raw dense/sparse scores (they live on different, incomparable
    # scales -- cosine similarity vs. BM25) and fuses on rank position instead.
    fused_scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_hits):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(sparse_hits):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)

    fused = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused
