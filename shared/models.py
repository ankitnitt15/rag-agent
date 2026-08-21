from pydantic import BaseModel
from typing import Literal

Domain = Literal["financial", "legal", "product"]
Verdict = Literal["ENTAILED", "NOT_ENTAILED", "UNVERIFIABLE"]
RefusalReason = Literal["low_retrieval_confidence", "no_relevant_passages", "unverified_claim"]
IndexStatus = Literal["indexed", "skipped_unchanged", "flagged_near_duplicate"]

class RawDoc(BaseModel):
    doc_id: str
    text: str
    source: str
    tags: list[str]
    timestamp: str
    domain: Domain

class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_id: str
    position: int
    text: str # kept in payload for now
    domain: Domain
    timestamp: str
    tags: list[str]
    source: str
    doc_content_hash: str

class QueryFilters(BaseModel):
    domain: Domain | None = None
    date_from: str | None = None
    date_to: str | None = None
    tags: list[str] = []
    source: str | None = None

class PassageCandidate(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

class RerankScore(BaseModel):
    chunk_id: str
    relevance: float

class AtomicClaim(BaseModel):
    claim_id: str
    text: str

class ClaimVerification(BaseModel):
    claim_id: str
    verdict: Verdict
    reasoning: str

class PassageRef(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None
    passages: list[PassageRef]
    confidence: float
    query_id: str

class IndexResult(BaseModel):
    doc_id: str
    status: IndexStatus
    near_duplicate_of: str | None = None
