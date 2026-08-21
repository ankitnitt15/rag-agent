from common.gemini_client import generate_content
from generation import prompts
from shared.models import AtomicClaim, ClaimVerification, PassageCandidate


def extract_atomic_claims(answer_text: str) -> list[AtomicClaim]:
    prompt = prompts.build_claim_extraction_prompt(answer_text)
    response = generate_content(
        prompt,
        config={"response_mime_type": "application/json", "response_schema": list[AtomicClaim]},
    )
    return response.parsed


def verify_claim_against_passages(
    claim: AtomicClaim, passages: list[PassageCandidate], temperature: float | None = None
) -> ClaimVerification:
    prompt = prompts.build_nli_verification_prompt(claim, passages)
    config = {"response_mime_type": "application/json", "response_schema": ClaimVerification}
    if temperature is not None:
        config["temperature"] = temperature
    response = generate_content(prompt, config=config)
    verification = response.parsed
    verification.claim_id = claim.claim_id  # the model doesn't see claim_id, so stamp it ourselves
    return verification


def all_claims_entailed(
    answer_text: str, passages: list[PassageCandidate]
) -> tuple[bool, list[ClaimVerification]]:
    claims = extract_atomic_claims(answer_text)
    verifications = [verify_claim_against_passages(claim, passages) for claim in claims]
    all_entailed = all(v.verdict == "ENTAILED" for v in verifications)
    return all_entailed, verifications
