import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.db"


def get_connection(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # Direct translation of the doc_chunk_registry table from RagProduction.md,
    # with Postgres's TIMESTAMPTZ swapped for TEXT (sqlite has no native datetime type).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunk_registry (
            doc_id          TEXT NOT NULL,
            chunk_id        TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1,
            indexed_at      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY (doc_id, chunk_id, version)
        )
    """)
    conn.commit()


def register_chunks(
    conn: sqlite3.Connection,
    doc_id: str,
    chunk_ids: list[str],
    content_hash: str,
    version: int = 1,
) -> None:
    indexed_at = datetime.now(timezone.utc).isoformat()
    rows = [(doc_id, chunk_id, content_hash, version, indexed_at) for chunk_id in chunk_ids]
    conn.executemany(
        """
        INSERT INTO doc_chunk_registry (doc_id, chunk_id, content_hash, version, indexed_at, status)
        VALUES (?, ?, ?, ?, ?, 'active')
        """,
        rows,
    )
    conn.commit()


def get_current_content_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM doc_chunk_registry WHERE doc_id = ? AND status = 'active' LIMIT 1",
        (doc_id,),
    ).fetchone()
    return row[0] if row else None


def should_reindex(conn: sqlite3.Connection, doc_id: str, new_content_hash: str) -> bool:
    current_hash = get_current_content_hash(conn, doc_id)
    return current_hash != new_content_hash


def get_active_chunk_ids(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM doc_chunk_registry WHERE doc_id = ? AND status = 'active'",
        (doc_id,),
    ).fetchall()
    return [row[0] for row in rows]


def mark_superseded(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute(
        "UPDATE doc_chunk_registry SET status = 'superseded' WHERE doc_id = ? AND status = 'active'",
        (doc_id,),
    )
    conn.commit()


def next_version(conn: sqlite3.Connection, doc_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) FROM doc_chunk_registry WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    current_max = row[0]  # None if doc_id has no rows yet
    return (current_max or 0) + 1
