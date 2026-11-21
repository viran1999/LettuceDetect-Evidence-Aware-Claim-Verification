import re

def lookup_structured_attribute(source_info, field, subfield=None):
    """
    Look up a Data2Txt attribute without guessing its meaning.

    Returns:
        status: known, unknown, missing, or unavailable
        field: requested top-level attribute
        subfield: optional nested attribute
        value: original source value
    """

    result = {
        "status": "unavailable",
        "field": field,
        "subfield": subfield,
        "value": None,
    }

    if not isinstance(source_info, dict):
        return result

    attributes = source_info.get("attributes")

    if not isinstance(attributes, dict):
        return result

    if field not in attributes:
        result["status"] = "missing"
        return result

    value = attributes[field]

    if subfield is not None:
        if value is None:
            result["status"] = "unknown"
            return result

        if not isinstance(value, dict):
            return result

        if subfield not in value:
            result["status"] = "missing"
            return result

        value = value[subfield]

    result["value"] = value
    result["status"] = "unknown" if value is None else "known"

    return result


# Recognized Data2Txt expressions and their source attributes.
SPAN_ATTRIBUTE_ALIASES = {
    "business parking": ("BusinessParking", None),
    "garage parking": ("BusinessParking", "garage"),
    "street parking": ("BusinessParking", "street"),
    "valet parking": ("BusinessParking", "valet"),
    "parking lot": ("BusinessParking", "lot"),
    "validated parking": ("BusinessParking", "validated"),
    "wi-fi": ("WiFi", None),
    "wifi": ("WiFi", None),
    "outdoor seating": ("OutdoorSeating", None),
    "takeout": ("RestaurantsTakeOut", None),
    "reservations": ("RestaurantsReservations", None),
    "hipster ambiance": ("Ambience", "hipster"),
    "casual ambiance": ("Ambience", "casual"),
    "classy ambiance": ("Ambience", "classy"),
    "romantic ambiance": ("Ambience", "romantic"),
}


def lookup_span_attribute(source_info, suspicious_span):
    """
    Match a suspicious span to a recognized Data2Txt attribute.

    Does not determine whether a claim is true or false.
    Ambiguous and unsupported expressions are not guessed.
    """

    def normalize(text):
        return " ".join(
            re.findall(r"[a-z0-9]+", text.casefold())
        )

    span_text = f" {normalize(suspicious_span)} "

    candidates = set()

    for expression, target in SPAN_ATTRIBUTE_ALIASES.items():
        expression_text = f" {normalize(expression)} "

        if expression_text in span_text:
            candidates.add(target)

    if not candidates:
        return {
            "match_status": "no_match",
            "lookup": None
        }

    if len(candidates) > 1:
        return {
            "match_status": "ambiguous",
            "lookup": None
        }

    field, subfield = next(iter(candidates))

    lookup = lookup_structured_attribute(
        source_info,
        field,
        subfield
    )

    return {
        "match_status": "matched",
        "lookup": lookup
    }
