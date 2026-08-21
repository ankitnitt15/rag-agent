import hashlib
import re


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(tokens: list[str], k: int = 3) -> set[str]:
    # k-gram shingles: sliding windows of k consecutive tokens, joined into one
    # string per window. Two documents sharing many shingles share whole phrases,
    # not just individual words -- a much stronger near-duplicate signal.
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}
