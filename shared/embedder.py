from common.gemini_client import embed_content

EMBEDDING_DIM = 768


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embed_content(
        contents=texts,
        config={"output_dimensionality": EMBEDDING_DIM},
    )
    return [embedding.values for embedding in response.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
