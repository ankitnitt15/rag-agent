from ingestion import chunking, minhash
from retrieval import bm25
from storage import filestore, registry, vector_store
from shared.embedder import embed_texts
from shared.models import ChunkMetadata, IndexResult, RawDoc


def _build_chunks(doc: RawDoc, content_hash: str) -> list[ChunkMetadata]:
    pieces = chunking.chunk_text(doc.text)
    return [
        ChunkMetadata(
            doc_id=doc.doc_id,
            chunk_id=f"{doc.doc_id}:{position}",
            position=position,
            text=piece,
            domain=doc.domain,
            timestamp=doc.timestamp,
            tags=doc.tags,
            source=doc.source,
            doc_content_hash=content_hash,
        )
        for position, piece in enumerate(pieces)
    ]


def index_document(
    doc: RawDoc,
    qdrant_client,
    bm25_index: bm25.BM25Index,
    minhash_index: minhash.MinHashIndex,
    conn,
) -> IndexResult:
    content_hash = filestore.store_document(doc.text)

    if not registry.should_reindex(conn, doc.doc_id, content_hash):
        return IndexResult(doc_id=doc.doc_id, status="skipped_unchanged")

    # Checked against already-indexed docs only, before this doc joins the
    # index itself -- still indexed either way (see note in main.py), just tagged.
    near_dupes = minhash_index.find_near_duplicates(doc.doc_id, doc.text)

    chunks = _build_chunks(doc, content_hash)
    vectors = embed_texts([chunk.text for chunk in chunks])
    vector_store.upsert_chunks(qdrant_client, chunks, vectors)

    for chunk in chunks:
        bm25_index.add(chunk.chunk_id, chunk.text, metadata=chunk.model_dump())

    minhash_index.add(doc.doc_id, doc.text)
    registry.register_chunks(conn, doc.doc_id, [chunk.chunk_id for chunk in chunks], content_hash)

    if near_dupes:
        nearest_id, _similarity = max(near_dupes, key=lambda pair: pair[1])
        return IndexResult(doc_id=doc.doc_id, status="flagged_near_duplicate", near_duplicate_of=nearest_id)

    return IndexResult(doc_id=doc.doc_id, status="indexed")


def reindex_document(
    doc_id: str,
    new_text: str,
    qdrant_client,
    bm25_index: bm25.BM25Index,
    minhash_index: minhash.MinHashIndex,
    conn,
) -> IndexResult:
    content_hash = filestore.store_document(new_text)

    if not registry.should_reindex(conn, doc_id, content_hash):
        return IndexResult(doc_id=doc_id, status="skipped_unchanged")

    old_chunk_ids = registry.get_active_chunk_ids(conn, doc_id)
    # Only the text is changing here -- reuse domain/tags/source/timestamp
    # from the old chunk rather than asking the caller to repeat them.
    old_metadata = bm25_index.get_metadata(old_chunk_ids[0])

    vector_store.delete_chunks(qdrant_client, old_chunk_ids)
    for chunk_id in old_chunk_ids:
        bm25_index.remove(chunk_id)
    registry.mark_superseded(conn, doc_id)

    doc = RawDoc(
        doc_id=doc_id,
        text=new_text,
        source=old_metadata["source"],
        tags=old_metadata["tags"],
        timestamp=old_metadata["timestamp"],
        domain=old_metadata["domain"],
    )
    new_chunks = _build_chunks(doc, content_hash)
    vectors = embed_texts([chunk.text for chunk in new_chunks])
    vector_store.upsert_chunks(qdrant_client, new_chunks, vectors)

    for chunk in new_chunks:
        bm25_index.add(chunk.chunk_id, chunk.text, metadata=chunk.model_dump())
    bm25_index.build()  # chunk count changed (removed old, added new) -- stats are stale otherwise

    minhash_index.add(doc_id, new_text)

    new_version = registry.next_version(conn, doc_id)
    registry.register_chunks(
        conn, doc_id, [chunk.chunk_id for chunk in new_chunks], content_hash, version=new_version
    )

    return IndexResult(doc_id=doc_id, status="indexed")


def batch_ingest(
    docs: list[RawDoc],
    qdrant_client,
    bm25_index: bm25.BM25Index,
    minhash_index: minhash.MinHashIndex,
    conn,
) -> list[IndexResult]:
    # Loops index_document() in-process. A real Batch API (OpenAI/Gemini batch
    # endpoints, ~50% cheaper, minutes-to-24h SLA) would instead submit one job
    # for the whole corpus and poll for completion, rather than embedding each
    # doc synchronously in a loop as this prototype does.
    results = [
        index_document(doc, qdrant_client, bm25_index, minhash_index, conn) for doc in docs
    ]
    bm25_index.build()  # must run once, after every add() -- needs corpus-wide stats
    return results
