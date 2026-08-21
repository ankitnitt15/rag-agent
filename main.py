from corpus import generate_corpus
from retrieval import bm25
from ingestion import ingest, minhash
from storage import registry, vector_store
from generation.answer import answer


def print_index_results(results) -> None:
    print("Ingestion summary:")
    for result in results:
        line = f"  - {result.doc_id}: {result.status}"
        if result.near_duplicate_of:
            line += f" (near-duplicate of {result.near_duplicate_of})"
        print(line)
    print("-" * 60)


def print_result(query: str, result) -> None:
    print(f"Query: {query}")
    print(f"refused={result.refused} reason={result.refusal_reason} confidence={result.confidence:.3f}")
    print(f"Answer: {result.answer}\n")
    if result.passages:
        print("Passages used:")
        for passage in result.passages:
            preview = passage.text[:80]
            print(f"  - [{passage.doc_id}] score={passage.score:.3f} :: {preview}...")
    print("-" * 60)


def main() -> None:
    docs = generate_corpus()

    qdrant_client = vector_store.get_client()
    vector_store.ensure_collection(qdrant_client)
    bm25_index = bm25.BM25Index()
    minhash_index = minhash.MinHashIndex()
    conn = registry.get_connection()

    results = ingest.batch_ingest(docs, qdrant_client, bm25_index, minhash_index, conn)
    print_index_results(results)

    answerable_query = "What were Acme's Q3 2026 financial results?"
    result = answer(answerable_query, qdrant_client, bm25_index)
    print_result(answerable_query, result)

    out_of_corpus_query = "What is the weather in Paris today?"
    result = answer(out_of_corpus_query, qdrant_client, bm25_index)
    print_result(out_of_corpus_query, result)

    # Reindexing test
    updated_support_policy = (
        "Acme's support policy guarantees a response within 24 hours for "
        "all paid tiers via email support.\n\n"
        "Enterprise customers also get access to a dedicated Slack "
        "channel monitored during business hours for faster turnaround.\n\n"
        "For critical production outages, Enterprise customers can page "
        "on-call support directly from the status page, with a target "
        "acknowledgement time of 15 minutes."
    )
    reindex_result = ingest.reindex_document(
        "product-004", updated_support_policy, qdrant_client, bm25_index, minhash_index, conn
    )
    print(f"[reindex] product-004: {reindex_result.status}")
    print(f"[reindex] active chunks now: {registry.get_active_chunk_ids(conn, 'product-004')}")
    print("-" * 60)

    followup_query = "What is Acme's target acknowledgement time for a critical outage page?"
    result = answer(followup_query, qdrant_client, bm25_index)
    print_result(followup_query, result)


if __name__ == "__main__":
    main()
