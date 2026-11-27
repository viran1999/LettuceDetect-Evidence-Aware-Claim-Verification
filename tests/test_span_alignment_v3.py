from enhancements.span_alignment_v3 import align_span, choose_atomic_claims, recover_span
from enhancements.span_verification_v3 import _verify_one_atomic, aggregate_atom_results


def test_clipped_word_recovered_from_offsets():
    answer = "Preheat the oven to 425 degrees."
    assert recover_span(answer, {"start": 0, "end": 3, "text": "Pre"}) == "Preheat"


def test_multiple_facts_in_one_span():
    claim = "The hotel has no pool or spa."
    a = ["The hotel has no pool.", "The hotel has no spa."]
    res = choose_atomic_claims("no pool or spa", a, claim)
    assert res["status"] == "matched" and set(res["claims"]) == set(a)


def test_wifi_alias():
    c = "The restaurant offers free Wi-Fi."
    a = ["The restaurant offers free WiFi."]
    res = choose_atomic_claims("free Wi-Fi", a, c)
    assert res["status"] == "matched"


def test_numeric_hallucination_rejected():
    res = choose_atomic_claims("27 holes", ["There are 26 holes."], "There are 27 holes.")
    assert res["status"] == "needs_review"


def test_omitted_span_stays_review():
    res = choose_atomic_claims(
        "quiet", ["The hotel has a pool."], "The hotel is quiet and has a pool."
    )
    assert res["status"] == "needs_review"


def test_duplicate_atoms_deduplicated():
    res = choose_atomic_claims(
        "free WiFi",
        ["The restaurant has free WiFi.", "The restaurant has free WiFi."],
        "The restaurant has free WiFi.",
    )
    assert res["status"] == "matched" and len(res["claims"]) == 1


def test_targeted_fallback_flags_prevent_acceptance():
    claim = "The restaurant is quiet."

    def bad_fallback(*args: object):
        return {
            "proposed_claims": ["The restaurant is quiet."],
            "flags": [{"flag": "suspected_invention"}],
        }

    res = align_span(
        claim,
        {"start": 18, "end": 23, "text": "quiet"},
        claim,
        ["The restaurant has a pool."],
        bad_fallback,
    )
    assert res["status"] == "needs_review"


def test_targeted_fallback_can_recover_grounded_missing_fact():
    claim = "The restaurant is quiet."

    def fallback(*args: object):
        return {"proposed_claims": ["The restaurant is quiet."], "flags": []}

    res = align_span(
        claim,
        {"start": 18, "end": 23, "text": "quiet"},
        claim,
        ["The restaurant has a pool."],
        fallback,
    )
    assert res["status"] == "matched" and res["method"] == "targeted_qwen_validated"


def test_neutral_not_forcibly_reclassified():
    assert aggregate_atom_results([{"category": "Unverifiable"}])["category"] == "Unverifiable"


def test_mixed_supported_and_unverifiable():
    assert (
        aggregate_atom_results([{"category": "Supported"}, {"category": "Unverifiable"}])[
            "category"
        ]
        == "Partially Supported"
    )


def test_mixed_contradiction_abstains():
    assert (
        aggregate_atom_results([{"category": "Contradicted"}, {"category": "Supported"}])["status"]
        == "needs_review"
    )


class StubProb:
    """Mock probability scores for testing."""

    def __init__(self, values):
        """Initialize probability values."""
        self.values = values

    def argmax(self):
        """Return the index of the highest probability."""
        return max(range(len(self.values)), key=self.values.__getitem__)

    def __iter__(self):
        """Iterate over probability values."""
        return iter(self.values)


class StubNLI:
    """Mock NLI model for testing."""

    model = type(
        "M",
        (),
        {
            "config": type(
                "C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}}
            )()
        },
    )()

    def predict(self, pairs, apply_softmax=True):
        """Return simulated NLI predictions."""
        assert apply_softmax
        return [
            StubProb([0.95, 0.01, 0.04]) if "not offer" in e else StubProb([0.01, 0.01, 0.98])
            for e, _ in pairs
        ]


def test_structured_wifi_nli():
    passages = [{"rank": 1, "evidence": "WiFi: no"}]
    ctx = {
        "span_match_status": "matched",
        "attribute_lookup": {
            "match_status": "matched",
            "lookup": {"field": "WiFi", "subfield": None},
        },
    }
    res = _verify_one_atomic("The restaurant has free WiFi.", passages, "Data2Txt", StubNLI(), ctx)
    assert res["category"] == "Contradicted"


def test_unrelated_attribute_evidence_not_used():
    passages = [{"rank": 1, "evidence": "RestaurantsReservations: True"}]
    ctx = {
        "span_match_status": "matched",
        "attribute_lookup": {
            "match_status": "matched",
            "lookup": {"field": "WiFi", "subfield": None},
        },
    }
    res = _verify_one_atomic("The restaurant has free WiFi.", passages, "Data2Txt", StubNLI(), ctx)
    assert res["status"] == "needs_review"
