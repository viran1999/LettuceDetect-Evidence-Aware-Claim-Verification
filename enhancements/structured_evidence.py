
import re


# Natural-language interpretations of supported structured fields.
FIELD_SENTENCES = {
    "restaurantstakeout": {
        "true": "The restaurant offers takeout services.",
        "false": "The restaurant does not offer takeout services.",
    },
    "restaurantsreservations": {
        "true": "The restaurant accepts reservations.",
        "false": "The restaurant does not accept reservations.",
    },
    "outdoorseating": {
        "true": "The restaurant has outdoor seating.",
        "false": "The restaurant does not have outdoor seating.",
    },
    "restaurantsgoodforgroups": {
        "true": "The restaurant is suitable for groups.",
        "false": "The restaurant is not suitable for groups.",
    },
    "wifi": {
        "free": "The restaurant offers free Wi-Fi.",
        "paid": "The restaurant offers paid Wi-Fi.",
        "no": "The restaurant does not offer Wi-Fi.",
    },
}

# Context-aware interpretations of nested Data2Txt fields.
NESTED_FIELD_SENTENCES = {
    "businessparking_garage": {
        "true": "The restaurant offers garage parking.",
        "false": "The restaurant does not offer garage parking.",
    },
    "businessparking_street": {
        "true": "The restaurant offers street parking.",
        "false": "The restaurant does not offer street parking.",
    },
    "businessparking_lot": {
        "true": "The restaurant has a parking lot.",
        "false": "The restaurant does not have a parking lot.",
    },
    "businessparking_valet": {
        "true": "The restaurant offers valet parking.",
        "false": "The restaurant does not offer valet parking.",
    },
    "businessparking_validated": {
        "true": "The restaurant offers validated parking.",
        "false": "The restaurant does not offer validated parking.",
    },
    "ambience_casual": {
        "true": "The restaurant has a casual atmosphere.",
        "false": "The restaurant does not have a casual atmosphere.",
    },
    "ambience_romantic": {
        "true": "The restaurant has a romantic atmosphere.",
        "false": "The restaurant does not have a romantic atmosphere.",
    },
    "ambience_classy": {
        "true": "The restaurant has a classy atmosphere.",
        "false": "The restaurant does not have a classy atmosphere.",
    },
    "ambience_intimate": {
        "true": "The restaurant has an intimate atmosphere.",
        "false": "The restaurant does not have an intimate atmosphere.",
    },
}

FIELD_SENTENCES.update(NESTED_FIELD_SENTENCES)


# Automatically recognize every field in FIELD_SENTENCES.
# This avoids manually updating the regex whenever a field is added.
FIELD_NAMES = "|".join(
    re.escape(field)
    for field in sorted(FIELD_SENTENCES, key=len, reverse=True)
)

FIELD_PATTERN = re.compile(
    rf"\b({FIELD_NAMES})\s*:\s*(True|False|free|paid|no)\b\.?",
    re.IGNORECASE,
)


def normalize_structured_evidence(evidence):
    """
    Convert recognized Data2Txt key-value attributes into
    natural-language evidence for NLI verification.

    Preserve unsupported fields unchanged.
    """
    if not isinstance(evidence, str):
        raise TypeError("Evidence must be a string")

    def replace_field(match):
        field = match.group(1).casefold()
        value = match.group(2).casefold()

        return FIELD_SENTENCES.get(field, {}).get(
            value,
            match.group(0)
        )

    return FIELD_PATTERN.sub(replace_field, evidence)
