from shared.models import RawDoc

#
# Convention: `timestamp` is the reporting *period* the doc is about (e.g. a
# quarter's end date), not its publish date -- a quarterly report for Q3 is
# typically published a few days into Q4, so filtering on publish date would
# exclude it from a "Q3" date-range query. Period-end keeps date filtering
# aligned with how people actually ask these questions.
DOCS = [
    RawDoc(
        doc_id="fin-001",
        text="Acme Corp reported Q3 2026 revenue of $42M, up 8% year over year, "
             "driven by strong growth in its cloud subscription business.",
        source="acme-quarterly-report",
        tags=["financial", "earnings"],
        timestamp="2026-09-30",
        domain="financial",
    ),
    RawDoc(
        doc_id="fin-002",
        text="In Q1 2026, Acme Corp's operating margin improved to 18%, up from "
             "15% in the prior quarter, as the company cut hosting costs.",
        source="acme-quarterly-report",
        tags=["financial", "earnings"],
        timestamp="2026-03-31",
        domain="financial",
    ),
    RawDoc(
        doc_id="fin-003",
        text="Acme Corp's board approved a $50M share buyback program in "
             "September 2026, funded from existing cash reserves.",
        source="acme-press-release",
        tags=["financial", "capital-allocation"],
        timestamp="2026-09-18",
        domain="financial",
    ),
    RawDoc(
        doc_id="legal-001",
        text="Under Section 4.2 of the Master Services Agreement, either party "
             "may terminate the contract with 30 days written notice if the "
             "other party fails to cure a material breach.",
        source="msa-template",
        tags=["legal", "contract"],
        timestamp="2025-06-01",
        domain="legal",
    ),
    RawDoc(
        doc_id="legal-002",
        text="The liability cap in the standard vendor agreement is limited to "
             "the total fees paid in the preceding 12 months, excluding claims "
             "arising from gross negligence or willful misconduct.",
        source="vendor-agreement-template",
        tags=["legal", "contract"],
        timestamp="2025-08-14",
        domain="legal",
    ),
    RawDoc(
        doc_id="product-001",
        text="Version 3.2 of the Acme dashboard adds dark mode, a redesigned "
             "settings page, and fixes a bug where exported CSVs dropped the "
             "last row.",
        source="acme-changelog",
        tags=["product", "release-notes"],
        timestamp="2026-02-20",
        domain="product",
    ),
    RawDoc(
        doc_id="product-002",
        text="The Acme mobile app now supports offline mode: users can view "
             "previously loaded dashboards without an internet connection.",
        source="acme-changelog",
        tags=["product", "release-notes"],
        timestamp="2026-05-30",
        domain="product",
    ),
    RawDoc(
        doc_id="product-003",
        text="Acme's API rate limit was raised from 100 to 500 requests per "
             "minute for all paid tiers starting in June 2026.",
        source="acme-changelog",
        tags=["product", "api"],
        timestamp="2026-06-10",
        domain="product",
    ),
    # Near-duplicate pair 1: same changelog entry re-posted with a corrected
    # version number -- minimal edit, meant for minhash.py to catch.
    RawDoc(
        doc_id="product-001-corrected",
        text="Version 3.3 of the Acme dashboard adds dark mode, a redesigned "
             "settings page, and fixes a bug where exported CSVs dropped the "
             "last row.",
        source="acme-changelog",
        tags=["product", "release-notes"],
        timestamp="2026-02-21",
        domain="product",
    ),
    # Near-duplicate pair 2: same clause reissued with a different notice
    # period -- minimal edit, meant for minhash.py to catch.
    RawDoc(
        doc_id="legal-001-amended",
        text="Under Section 4.2 of the Master Services Agreement, either party "
             "may terminate the contract with 45 days written notice if the "
             "other party fails to cure a material breach.",
        source="msa-template",
        tags=["legal", "contract"],
        timestamp="2025-07-15",
        domain="legal",
    ),
    # Multi-paragraph doc -- the only one that actually exercises chunking.py's
    # splitter (2 chunks today). adding a
    # 3rd paragraph later should produce 3 chunks, not 2.
    RawDoc(
        doc_id="product-004",
        text="Acme's support policy guarantees a response within 24 hours for "
             "all paid tiers via email support.\n\n"
             "Enterprise customers also get access to a dedicated Slack "
             "channel monitored during business hours for faster turnaround.",
        source="acme-support-policy",
        tags=["product", "support"],
        timestamp="2026-01-10",
        domain="product",
    ),
]


def generate_corpus() -> list[RawDoc]:
    return DOCS
