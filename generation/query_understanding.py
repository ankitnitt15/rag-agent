from common.gemini_client import generate_content
from generation import prompts
from shared.models import QueryFilters


def extract_filters(raw_query: str, known_tags: list[str]) -> QueryFilters:
    prompt = prompts.build_filter_extraction_prompt(raw_query, known_tags)
    response = generate_content(
        prompt,
        config={"response_mime_type": "application/json", "response_schema": QueryFilters},
    )
    return response.parsed
