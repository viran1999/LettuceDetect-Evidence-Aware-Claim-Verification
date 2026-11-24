
import re


# Structured fields whose meaning we can identify reliably.
STRUCTURED_FIELD_PATTERN = re.compile(
    r"(?P<field>"
    r"RestaurantsTakeOut|RestaurantsReservations|"
    r"RestaurantsGoodForGroups|OutdoorSeating|WiFi|"
    r"BusinessParking(?:_[A-Za-z]+)?|"
    r"Ambience(?:_[A-Za-z]+)?"
    r")\s*:\s*"
    r"(?:True|False|None|free|paid|no)\.?",
    re.IGNORECASE,
)


def assess_evidence_relevance(span_context, evidence):
    """
    Check whether a single structured evidence passage
    concerns the attribute targeted by a suspicious span.

    Returns:
        related: Matching attribute.
        irrelevant: Clearly different structured attribute.
        uncertain: Cannot establish relevance safely.

    This is not a support/contradiction classifier.
    """

    attribute = span_context.get("attribute_lookup")

    if (
        span_context.get("span_match_status") != "matched"
        or not isinstance(attribute, dict)
        or attribute.get("match_status") != "matched"
        or not attribute.get("lookup")
    ):
        return {"status": "uncertain"}

    lookup = attribute["lookup"]

    field = lookup["field"].casefold()
    subfield = lookup.get("subfield")

    # Only assess simple, recognized key-value passages.
    # Do not guess the meaning of free-text evidence.
    match = STRUCTURED_FIELD_PATTERN.fullmatch(
        evidence.strip()
    )

    if match is None:
        return {"status": "uncertain"}

    evidence_field = match.group("field").casefold()

    expected_field = field
    if subfield:
        expected_field += "_" + subfield.casefold()

    if evidence_field == expected_field:
        return {"status": "related"}

    # A child parking/ambience field can be related to
    # its parent attribute, without necessarily proving
    # the entire claim.
    if (
        subfield is None
        and evidence_field.startswith(field + "_")
    ):
        return {"status": "related"}

    return {"status": "irrelevant"}
