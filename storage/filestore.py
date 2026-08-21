from pathlib import Path

from shared import hashing

ROOT = Path(__file__).resolve().parent.parent / "data" / "docs"


def _path_for(content_hash: str) -> Path:
    # git-style hash-prefix folder layout: first 2 hex chars fan out into subfolders
    # so no single directory ends up with millions of files.
    return ROOT / content_hash[:2] / f"{content_hash}.md"


def store_document(text: str) -> str:
    digest = hashing.content_hash(text)
    path = _path_for(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return digest


def load_document(content_hash: str) -> str:
    return _path_for(content_hash).read_text()
