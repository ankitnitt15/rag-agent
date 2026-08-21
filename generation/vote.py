from collections import Counter

from generation import claims
from shared.models import AtomicClaim, ClaimVerification, PassageCandidate

VOTE_TEMPERATURE = 0.7


def verify_claim_with_voting(
    claim: AtomicClaim, passages: list[PassageCandidate], n_samples: int = 3
) -> ClaimVerification:
    # latency cost: n_samples LLM calls instead of 1, per claim.
    # benefit: reduces single-sample judge noise. Off by default in answer().
    samples = [
        claims.verify_claim_against_passages(claim, passages, temperature=VOTE_TEMPERATURE)
        for _ in range(n_samples)
    ]

    votes = Counter(sample.verdict for sample in samples)
    top_verdict, top_count = votes.most_common(1)[0]

    # A tie means the judge isn't confident either way -- fail closed, since
    # the system's guarantee is zero unverified claims, not "best guess."
    if list(votes.values()).count(top_count) > 1:
        top_verdict = "NOT_ENTAILED"

    reasoning = next(
        (sample.reasoning for sample in samples if sample.verdict == top_verdict),
        f"{top_count}/{n_samples} samples voted {top_verdict}",
    )

    return ClaimVerification(claim_id=claim.claim_id, verdict=top_verdict, reasoning=reasoning)
