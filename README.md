# RAGAgent — RAG Prototype

A runnable prototype of a production RAG system design goal: "RAG with 10M
docs, zero hallucinations" — hybrid retrieval, a confidence-gated refusal
path, and claim-by-claim fact-checking so the system never asserts anything
it can't trace back to a retrieved passage.

This prototype proves out the *mechanisms* (chunking, hybrid retrieval,
refusal gating, claim verification, incremental reindexing, near-dup
detection) at toy scale (a dozen hand-authored docs), not the production
*scale* (10M docs, 200-250 QPS) the design targets. See
[Known simplifications](#known-simplifications) for exactly where the two
diverge.

## Infra substitutions

No external services are installed. Each production component is swapped for
a lightweight substitute with a comparable API, so the code mirrors the real
architecture without requiring Qdrant/Postgres/Redis/S3/Elasticsearch servers:

| Production component | Substitute | Where |
|---|---|---|
| S3 | Local filesystem, content-addressable path | `storage/filestore.py` |
| PostgreSQL (doc/chunk registry) | SQLite | `storage/registry.py` |
| Redis (query/preview cache) | In-memory dict + manual TTL | `storage/cache.py` |
| Qdrant | `qdrant-client`, embedded/local mode (real API, real HNSW) | `storage/vector_store.py` |
| Elasticsearch / BM25 | Hand-rolled BM25 (`k1=1.2`, `b=0.75`) | `retrieval/bm25.py` |
| Cross-encoder reranker | One batched Gemini call | `retrieval/rerank.py` |
| NLI entailment / fact-check | LLM-as-judge verifier | `generation/claims.py` |
| AWS Lambda (realtime trigger) | Plain Python function call | `ingestion/ingest.py::reindex_document` |
| Batch API (~50% cheaper) | In-process loop | `ingestion/ingest.py::batch_ingest` |
| Distributed sharding, alias-based zero-downtime rebuild | Out of scope | — |

## Directory layout

Organized by responsibility:

```
systems/RAGAgent/
    main.py                    # runnable end-to-end demo
    corpus.py                  # synthetic 11-doc corpus + near-dup pairs
    test_failure_scenarios.py  # refusal / injection / crash-consistency checks
    README.md

    shared/                    # cross-cutting, no dependency on the other folders
        models.py              # every Pydantic schema
        embedder.py            # Gemini embeddings wrapper
        hashing.py             # content_hash, tokenize, shingles

    storage/                   # the S3 / Postgres / Redis / Qdrant substitutes
        filestore.py
        registry.py
        cache.py
        vector_store.py

    retrieval/                 # hybrid search: dense + sparse + fusion + rerank
        bm25.py
        rrf.py
        rerank.py

    ingestion/                 # the write path
        chunking.py
        minhash.py
        ingest.py              # index_document / reindex_document / batch_ingest

    generation/                # the LLM-facing half of the read path
        answer.py              # answer() -- the full orchestrator
        claims.py               # atomic claim extraction + NLI-style verification
        vote.py                 # optional majority-voting verification
        prompts.py               # every prompt builder, incl. injection defenses
        query_understanding.py   # extract_filters()

    data/                      # created at runtime: docs/, registry.db, qdrant_local/
```

Internal imports are rooted at this folder (`from storage import ...`,
`from shared.models import ...`), and there's a local `common/` copy of the
Gemini client wrapper -- this folder has no dependency on anything outside
itself, so it runs the same way whether it's part of a larger checkout or
its own standalone repo.

## Write flow — ingesting, chunking, and reindexing a document

```mermaid
sequenceDiagram
    participant Caller as main.py / batch_ingest
    participant Ingest as ingestion/ingest.py
    participant Files as storage/filestore.py
    participant Reg as storage/registry.py
    participant MH as ingestion/minhash.py
    participant Chunk as ingestion/chunking.py
    participant Embed as shared/embedder.py
    participant Vec as storage/vector_store.py
    participant BM25 as retrieval/bm25.py

    Caller->>Ingest: index_document(doc)
    Ingest->>Files: store_document(text)
    Files-->>Ingest: content_hash
    Ingest->>Reg: should_reindex(doc_id, content_hash)
    Reg-->>Ingest: bool

    alt content unchanged
        Ingest-->>Caller: IndexResult(status="skipped_unchanged")
    else new or changed content
        Ingest->>MH: find_near_duplicates(doc_id, text)
        MH-->>Ingest: [(existing_doc_id, similarity), ...]
        Ingest->>Chunk: chunk_text(text)
        Chunk-->>Ingest: [chunk_text, ...]
        Ingest->>Embed: embed_texts(chunk_texts)
        Embed-->>Ingest: [vector, ...]
        Ingest->>Vec: upsert_chunks(chunks, vectors)
        Ingest->>BM25: add(chunk_id, text, metadata) [per chunk]
        Ingest->>MH: add(doc_id, text)
        Ingest->>Reg: register_chunks(doc_id, chunk_ids, content_hash, version=1)
        Ingest-->>Caller: IndexResult(status="indexed" | "flagged_near_duplicate")
    end
```

**Reindex (update) path** — `reindex_document`, used for the ~1000 docs/day
realtime-update scenario. Same pipeline, but starts by deleting the
document's *previous* chunks from both stores before writing the new ones,
so a doc whose chunk count changes (e.g. a paragraph is added) doesn't leave
stale chunks behind:

```mermaid
sequenceDiagram
    participant Caller as main.py
    participant Ingest as ingestion/ingest.py
    participant Files as storage/filestore.py
    participant Reg as storage/registry.py
    participant Vec as storage/vector_store.py
    participant BM25 as retrieval/bm25.py
    participant Chunk as ingestion/chunking.py
    participant Embed as shared/embedder.py

    Caller->>Ingest: reindex_document(doc_id, new_text)
    Ingest->>Files: store_document(new_text)
    Files-->>Ingest: content_hash
    Ingest->>Reg: should_reindex(doc_id, content_hash)
    Reg-->>Ingest: True (content changed)
    Ingest->>Reg: get_active_chunk_ids(doc_id)
    Reg-->>Ingest: [old_chunk_id, ...]
    Ingest->>BM25: get_metadata(old_chunk_ids[0])
    BM25-->>Ingest: domain/tags/source/timestamp
    Ingest->>Vec: delete_chunks(old_chunk_ids)
    Ingest->>BM25: remove(chunk_id) [per old chunk]
    Ingest->>Reg: mark_superseded(doc_id)
    Ingest->>Chunk: chunk_text(new_text)
    Chunk-->>Ingest: [chunk_text, ...] (count may differ from before)
    Ingest->>Embed: embed_texts(new_chunk_texts)
    Ingest->>Vec: upsert_chunks(new_chunks, vectors)
    Ingest->>BM25: add(chunk_id, text, metadata) [per new chunk] + build()
    Ingest->>Reg: register_chunks(doc_id, new_chunk_ids, content_hash, version=next_version)
    Ingest-->>Caller: IndexResult(status="indexed")
```

**Doc preview (cache-through read)** — `storage/cache.py`'s `TTLCache`
substitutes for Redis, sitting in front of the filestore/registry pair that
the write flow above just populated. First lookup for a `doc_id` is a cache
miss and reads through to disk; subsequent lookups within the TTL window are
served from memory:

```mermaid
sequenceDiagram
    participant Caller
    participant Cache as storage/cache.py
    participant Reg as storage/registry.py
    participant Files as storage/filestore.py

    Caller->>Cache: get_doc_preview(doc_id)
    Cache->>Cache: preview_cache.get(doc_id)
    alt cache hit
        Cache-->>Caller: cached preview text
    else cache miss
        Cache->>Reg: get_current_content_hash(doc_id)
        Reg-->>Cache: content_hash
        Cache->>Files: load_document(content_hash)
        Files-->>Cache: full text
        Cache->>Cache: preview_cache.set(doc_id, preview)
        Cache-->>Caller: preview text
    end
```
> Note: `get_doc_preview` is implemented and covered by the mechanism above,
> but `main.py`'s demo run doesn't currently call it — no citation-hover UI
> exists in this prototype to trigger it. See [Known simplifications](#known-simplifications).

## Query flow — `answer()`, the full read path

```mermaid
sequenceDiagram
    participant Caller as main.py
    participant Answer as generation/answer.py
    participant QU as generation/query_understanding.py
    participant Embed as shared/embedder.py
    participant Vec as storage/vector_store.py
    participant BM25 as retrieval/bm25.py
    participant RRF as retrieval/rrf.py
    participant Rerank as retrieval/rerank.py
    participant Gemini as Gemini (generate_content)
    participant Claims as generation/claims.py
    participant Vote as generation/vote.py

    Caller->>Answer: answer(query)
    Answer->>QU: extract_filters(query, known_tags)
    QU-->>Answer: QueryFilters (domain/date range/tags/source)
    Answer->>Embed: embed_query(query)
    Embed-->>Answer: query_vector

    par dense retrieval
        Answer->>Vec: search(query_vector, filters, top_k=40)
        Vec-->>Answer: dense_hits
    and sparse retrieval
        Answer->>BM25: search(query, filters, top_k=40)
        BM25-->>Answer: sparse_hits
    end

    Answer->>RRF: rrf_fusion(dense_hits, sparse_hits)
    RRF-->>Answer: fused candidates
    Answer->>Rerank: rerank(query, candidates, top_k=10)
    Rerank->>Gemini: score all candidates (one batched call)
    Gemini-->>Rerank: relevance per chunk_id
    Rerank-->>Answer: ranked candidates

    alt top score <= confidence floor (0.4)
        Answer-->>Caller: refused("low_retrieval_confidence")
    else
        Answer->>Gemini: build_generation_prompt(query, top 5) -> generate_content(temp=0)
        Gemini-->>Answer: raw_answer

        alt raw_answer starts with INSUFFICIENT_CONTEXT
            Answer-->>Caller: refused("no_relevant_passages")
        else
            Answer->>Claims: extract_atomic_claims(raw_answer)
            Claims-->>Answer: [AtomicClaim, ...]
            loop each claim
                alt use_voting
                    Answer->>Vote: verify_claim_with_voting(claim, passages)
                    Vote->>Claims: verify_claim_against_passages() x N samples
                    Vote-->>Answer: majority-vote ClaimVerification
                else
                    Answer->>Claims: verify_claim_against_passages(claim, passages)
                    Claims-->>Answer: ClaimVerification
                end
            end

            alt any verdict != ENTAILED
                Answer-->>Caller: refused("unverified_claim")
            else
                Answer-->>Caller: AnswerResponse(answer, passages, confidence)
            end
        end
    end
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in GEMINI_API_KEY
python main.py                     # end-to-end demo: ingest, query, filter, reindex
python test_failure_scenarios.py   # refusal / injection / crash-consistency checks
```

Both scripts make real calls to the Gemini API (embeddings + generation) --
nothing is mocked. Re-running `main.py` a second time without deleting
`data/` should report `skipped_unchanged` for every doc (the content-hash
gate from `registry.should_reindex`).

## Known simplifications

Relative to both the target production design and the original
implementation plan, the following are deliberate simplifications, not bugs:

1. **Single-process, single embedded Qdrant instance** — no sharding,
   replicas, or alias-based zero-downtime rebuild.
2. **Reranker is one batched LLM call**, not a cross-encoder — latency/cost
   tradeoff noted in `retrieval/rerank.py`.
3. **No highlight-span / grounding logic** — `PassageRef` carries a score but
   not `highlight_start`/`highlight_end`; the original plan's `highlight.py`
   was never built.
4. **"Batch" ingestion is an in-process loop**, not a real Batch API job (no
   polling, no SLA, no cost discount) — see the comment in `batch_ingest`.
5. **Realtime path is a direct function call**, not a serverless (Lambda)
   trigger — `reindex_document` is the same function either way.
6. **No atomic transaction spans the Qdrant write and the registry write** —
   demonstrated, not solved, in `test_failure_scenarios.py`'s crash test.
7. **Embeddings are truncated to 768 dimensions** for toy-scale locality.
8. **BM25 and MinHash are hand-rolled** for pedagogical fidelity to their
   underlying algorithms, not a production recommendation over
   Elasticsearch/`datasketch`.
9. **Chunk overlap only applies within a single long paragraph's
   sentence-packing fallback**, not between two independent paragraph-chunks
   of the same doc — a claim spanning a pronoun/reference across two
   paragraphs (e.g. "Lincoln was president... He was a leader...") can still
   land in separate, non-overlapping chunks. Known, deliberately deferred.
10. **`enrichment.py` (LLM-driven metadata extraction) and `tracing.py`
    (OTel-style span recording) were never built** — `domain`/`tags` are
    hand-authored directly on `RawDoc` in `corpus.py` rather than inferred by
    an LLM enrichment step, and there's no per-request trace log.
11. **`get_doc_preview` (the Redis-substitute caching path) is implemented
    but not wired into `main.py`'s demo** — nothing currently calls it at
    runtime; see the write-flow diagram's note above.
12. **10M docs / 200-250 QPS / p50 1.5s / p99 5s are the production targets**
    this design is for — the 11-doc toy corpus proves the mechanisms
    (content-hash skip, near-dup flagging, RRF fusion, refusal gating,
    registry versioning) work, not that they scale.
