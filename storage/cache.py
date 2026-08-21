import time

from storage import filestore, registry


class TTLCache:
    def __init__(self, ttl_seconds: float = 300):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[key]
            return None

        return value

    def set(self, key: str, value) -> None:
        self._store[key] = (time.time() + self.ttl_seconds, value)


preview_cache = TTLCache(ttl_seconds=300)


def get_doc_preview(doc_id: str, conn, max_chars: int = 200) -> str | None:
    cached = preview_cache.get(doc_id)
    if cached is not None:
        return cached

    content_hash = registry.get_current_content_hash(conn, doc_id)
    if content_hash is None:
        return None

    text = filestore.load_document(content_hash)
    preview = text[:max_chars]
    preview_cache.set(doc_id, preview)
    return preview
