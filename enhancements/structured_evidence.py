
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
        "no": "The restaurant does not offer Wi-Fi.",
    },
}


FIELD_PATTERN = re.compile(
    r"\b(RestaurantsTakeOut|RestaurantsReservations|"
    r"OutdoorSeating|RestaurantsGoodForGroups|WiFi)"
    r"\s*:\s*(True|False|free|no)\b\.?",
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
