import re

MAX_CHARS_DEFAULT = 300
OVERLAP_CHARS_DEFAULT = 60


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_sentences(text: str) -> list[str]:
    # naive: split after '.', '!', '?' followed by whitespace -- good enough
    # for hand-authored prose, not a real sentence boundary detector.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _pack_sentences(paragraph: str, max_chars: int, overlap_chars: int) -> list[str]:
    # Fallback for a paragraph that's too long on its own: pack sentences
    # into chunks up to max_chars, carrying the tail of one chunk into the
    # start of the next so retrieval doesn't lose context at a chunk boundary.
    chunks = []
    current = ""
    for sentence in split_sentences(paragraph):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        elif current:
            chunks.append(current)
            current = current[-overlap_chars:] + " " + sentence
        else:
            current = sentence  # a single sentence longer than max_chars -- keep as-is

    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str, max_chars: int = MAX_CHARS_DEFAULT, overlap_chars: int = OVERLAP_CHARS_DEFAULT
) -> list[str]:
    # Recursive splitting per RagProduction.md: try paragraphs first (one
    # paragraph = one chunk, if it fits); only fall back to sentence-level
    # packing for a paragraph that's too long on its own.
    chunks = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            chunks.extend(_pack_sentences(paragraph, max_chars, overlap_chars))
    return chunks
