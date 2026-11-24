from enhancements.span_alignment_candidate import propose_alignment


def selected(result):
    return [item["claim"] for item in result["candidates"]]


def test_targeted_wifi_claim():
    result = propose_alignment(
        " free WiFi",
        ["The restaurant has outdoor seating.", "The restaurant has free WiFi."],
    )
    assert result["status"] == "candidate_single"
    assert selected(result) == ["The restaurant has free WiFi."]


def test_compound_span_can_select_multiple_claims():
    result = propose_alignment(
        "Benefits include improved digestion, increased energy, and weight loss.",
        ["Benefits include improved digestion.", "Benefits include increased energy.", "Benefits include weight loss."],
    )
    assert result["status"] == "candidate_multi"
    assert len(selected(result)) == 3


def test_missing_fact_does_not_get_guessed():
    result = propose_alignment(
        " including 27 holes",
        ["Spieth wanted to win the Masters.", "He achieved 26 birdies."],
    )
    assert result["status"] == "unresolved"


def test_ambiguous_short_span_not_forced():
    result = propose_alignment(
        " spot",
        ["A popular spot for drinks.", "A relaxing spot for dinner."],
    )
    assert result["status"] == "unresolved"


def test_original_claims_deduplicated():
    result = propose_alignment(
        "free WiFi",
        ["The restaurant has free WiFi.", "The restaurant has free WiFi."],
    )
    assert len(selected(result)) == 1


def test_numerical_mismatch_not_aligned():
    result = propose_alignment("27 holes", ["The course has 26 holes."])
    assert result["status"] == "unresolved"


def test_preserves_negation_in_short_span():
    result = propose_alignment(
        "no parking available",
        ["The restaurant has parking available.", "The hotel has no parking available."],
    )
    assert result["status"] == "candidate_single"
    assert selected(result) == ["The hotel has no parking available."]


def test_no_content_words_not_assigned():
    result = propose_alignment(" a", ["A good restaurant."])
    assert result["status"] == "unresolved"
