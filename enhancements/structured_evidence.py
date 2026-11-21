
import re


TAKEOUT_PATTERN = re.compile(
    r"\bRestaurantsTakeOut\s*:\s*(True|False)\b\.?",
    re.IGNORECASE
)


def normalize_structured_evidence(evidence):
    """
    Convert recognized structured evidence fields
    into natural-language statements for NLI.

    Unrecognized fields remain unchanged.
    """

    if not isinstance(evidence, str):
        raise TypeError("Evidence must be a string")

    def replace_takeout(match):
        value = match.group(1).lower()

        if value == "true":
            return "The restaurant offers takeout services."

        return "The restaurant does not offer takeout services."

    return TAKEOUT_PATTERN.sub(
        replace_takeout,
        evidence
    )
