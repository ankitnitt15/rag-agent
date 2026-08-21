import hashlib
import random

from shared.hashing import shingles, tokenize

NUM_HASHES = 32
DEFAULT_THRESHOLD = 0.6
# A prime slightly above 2**32, per the permutation-function convention (there it's a tiny example prime; we need a much
# larger one so real shingle hashes don't collide constantly).
LARGE_PRIME = 4294967311


def _string_to_int(shingle: str) -> int:
    # hashlib (not Python's builtin hash()) because builtin hash() is
    # randomized per-process by default -- it would give different MinHash
    # signatures for the same text on every run.
    return int(hashlib.md5(shingle.encode()).hexdigest(), 16)


def make_hash_functions(num_hashes: int = NUM_HASHES, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [
        (rng.randrange(1, LARGE_PRIME - 1), rng.randrange(0, LARGE_PRIME - 1))
        for _ in range(num_hashes)
    ]


def minhash_signature(token_set: set[str], hash_fns: list[tuple[int, int]]) -> list[int]:
    if not token_set:
        return [0] * len(hash_fns)

    shingle_ints = [_string_to_int(shingle) for shingle in token_set]
    return [min((a * x + b) % LARGE_PRIME for x in shingle_ints) for a, b in hash_fns]


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


class MinHashIndex:
    def __init__(self, num_hashes: int = NUM_HASHES, threshold: float = DEFAULT_THRESHOLD, seed: int = 0):
        self.threshold = threshold
        self._hash_fns = make_hash_functions(num_hashes, seed)
        self._signatures: dict[str, list[int]] = {}

    def add(self, doc_id: str, text: str) -> None:
        shingle_set = shingles(tokenize(text))
        self._signatures[doc_id] = minhash_signature(shingle_set, self._hash_fns)

    def find_near_duplicates(self, doc_id: str, text: str) -> list[tuple[str, float]]:
        shingle_set = shingles(tokenize(text))
        candidate_sig = minhash_signature(shingle_set, self._hash_fns)

        matches = []
        for existing_id, existing_sig in self._signatures.items():
            if existing_id == doc_id:
                continue
            similarity = jaccard_estimate(candidate_sig, existing_sig)
            if similarity >= self.threshold:
                matches.append((existing_id, similarity))

        return matches
