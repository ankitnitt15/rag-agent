import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from shared.models import ChunkMetadata, QueryFilters
from shared.embedder import EMBEDDING_DIM

COLLECTION_NAME = "rag_prototype_chunks"
DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "qdrant_local")


def get_client(path: str = DATA_PATH) -> QdrantClient:
    # embedded mode: no server, everything lives on disk at `path`.
    # Qdrant builds an HNSW graph over these vectors internally (see session2/hnsw.md)
    # so we get approximate nearest-neighbor search without writing that algorithm ourselves.
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, dim: int = EMBEDDING_DIM) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def _point_id(chunk_id: str) -> str:
    # Qdrant only accepts unsigned ints or UUIDs as point ids, not arbitrary strings.
    # We derive a stable UUID from our human-readable chunk_id, and still keep the
    # original chunk_id inside the payload so we can read it back after a search.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def delete_chunks(client: QdrantClient, chunk_ids: list[str]) -> None:
    point_ids = [_point_id(chunk_id) for chunk_id in chunk_ids]
    client.delete(collection_name=COLLECTION_NAME, points_selector=point_ids)


def chunk_exists(client: QdrantClient, chunk_id: str) -> bool:
    points = client.retrieve(collection_name=COLLECTION_NAME, ids=[_point_id(chunk_id)])
    return len(points) > 0


def upsert_chunks(client: QdrantClient, chunks: list[ChunkMetadata], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=_point_id(chunk.chunk_id),
            vector=vector,
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def _build_filter(filters: QueryFilters | None) -> Filter | None:
    if filters is None:
        return None

    conditions = []
    if filters.domain:
        conditions.append(FieldCondition(key="domain", match=MatchValue(value=filters.domain)))
    if filters.source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=filters.source)))
    if filters.tags:
        conditions.append(FieldCondition(key="tags", match=MatchAny(any=filters.tags)))
    if filters.date_from or filters.date_to:
        conditions.append(
            FieldCondition(
                key="timestamp",
                range=DatetimeRange(gte=filters.date_from, lte=filters.date_to),
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def search(
    client: QdrantClient, query_vector: list[float], top_k: int = 5, filters: QueryFilters | None = None
) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=_build_filter(filters),
    ).points

    return [
        {
            "chunk_id": point.payload["chunk_id"],
            "doc_id": point.payload["doc_id"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in results
    ]
