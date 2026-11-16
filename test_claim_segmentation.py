
import re

from enhancements.claim_categorization import load_rank_one_records


def split_claim_into_segments(claim):
    """
    Initial rule-based segmentation.

    Split at sentence boundaries and colons followed
    by an uppercase letter.

    This does not guarantee fully atomic claims.
    """
    claim = " ".join(claim.split())

    segments = re.split(
        r'(?<=[.!?])\s+(?=[A-Z*])|:\s+(?=[A-Z])',
        claim
    )

    return [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]


records = load_rank_one_records(
    "data/verification/categorization_input_v2.jsonl"
)

# Examine the same two examples from previous steps.
for record in records[:2]:
    claim = record["claim"]
    segments = split_claim_into_segments(claim)

    print("\n" + "=" * 60)
    print("ORIGINAL CLAIM:")
    print(claim)

    print("\nEXTRACTED SEGMENTS:")

    for i, segment in enumerate(segments, start=1):
        print(f"{i}. {segment}")

    print("TOTAL SEGMENTS:", len(segments))
