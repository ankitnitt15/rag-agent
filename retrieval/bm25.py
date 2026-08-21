import math
from collections import Counter

from shared.hashing import tokenize
from shared.models import QueryFilters

K1_DEFAULT = 1.2
B_DEFAULT = 0.75


class BM25Index:
    def __init__(self, k1: float = K1_DEFAULT, b: float = B_DEFAULT):
        self.k1 = k1
        self.b = b
        self._term_freqs: dict[str, Counter] = {}   # chunk_id -> {term: count}
        self._doc_lengths: dict[str, int] = {}       # chunk_id -> token count
        self._texts: dict[str, str] = {}             # chunk_id -> original text
        self._metadata: dict[str, dict] = {}         # chunk_id -> ChunkMetadata.model_dump()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def add(self, chunk_id: str, text: str, metadata: dict | None = None) -> None:
        tokens = tokenize(text)
        self._term_freqs[chunk_id] = Counter(tokens)
        self._doc_lengths[chunk_id] = len(tokens)
        self._texts[chunk_id] = text
        self._metadata[chunk_id] = metadata or {}

    def build(self) -> None:
        n_docs = len(self._term_freqs)
        if n_docs == 0:
            self._avgdl = 0.0
            return

        self._avgdl = sum(self._doc_lengths.values()) / n_docs

        doc_freq: Counter = Counter()
        for term_counts in self._term_freqs.values():
            for term in term_counts:
                doc_freq[term] += 1

        # Lucene-style "+1 inside the log" IDF:
        self._idf = {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for term, df in doc_freq.items()
        }

    def _score(self, query_terms: list[str], chunk_id: str) -> float:
        term_counts = self._term_freqs[chunk_id]
        doc_length = self._doc_lengths[chunk_id]
        score = 0.0
        for term in query_terms:
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / self._avgdl)
            score += idf * (numerator / denominator)
        return score

    def _passes_filters(self, chunk_id: str, filters: QueryFilters | None) -> bool:
        if filters is None:
            return True

        meta = self._metadata.get(chunk_id, {})

        if filters.domain and meta.get("domain") != filters.domain:
            return False
        # ISO date strings ("YYYY-MM-DD") sort the same lexicographically as
        # chronologically, so plain string comparison works as a range check.
        if filters.date_from and meta.get("timestamp", "") < filters.date_from:
            return False
        if filters.date_to and meta.get("timestamp", "") > filters.date_to:
            return False
        if filters.tags and not set(filters.tags) & set(meta.get("tags", [])):
            return False
        if filters.source and meta.get("source") != filters.source:
            return False

        return True

    def search(
        self, query: str, top_k: int = 40, filters: QueryFilters | None = None
    ) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scored = [
            (chunk_id, self._score(query_terms, chunk_id))
            for chunk_id in self._term_freqs
            if self._passes_filters(chunk_id, filters)
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def get_text(self, chunk_id: str) -> str:
        return self._texts[chunk_id]

    def get_metadata(self, chunk_id: str) -> dict:
        return self._metadata[chunk_id]

    def remove(self, chunk_id: str) -> None:
        # Caller must call build() again afterward -- avgdl/IDF are corpus-wide
        # stats that are now stale once a chunk's term counts are gone.
        self._term_freqs.pop(chunk_id, None)
        self._doc_lengths.pop(chunk_id, None)
        self._texts.pop(chunk_id, None)
        self._metadata.pop(chunk_id, None)

    def known_tags(self) -> list[str]:
        # The actual tag vocabulary in use, derived from whatever's indexed --
        # used to constrain filter extraction so the LLM can't invent tags
        # (e.g. company names pulled from the query) that no chunk could ever match.
        tags: set[str] = set()
        for meta in self._metadata.values():
            tags.update(meta.get("tags", []))
        return sorted(tags)
