import io
import json
from unittest.mock import patch

import pytest

from enhancements.span_targeted_decomposer import (
    assess_generated_claims,
    meaningful_tokens,
    targeted_decompose,
)


def test_returns_distinct_claims():
    res = assess_generated_claims(
        "The hotel has no pool or spa.",
        "no pool or spa",
        ["The hotel has no pool.", "The hotel has no spa.", "The hotel has no pool."],
    )
    assert len(res["proposed_claims"]) == 2
    assert res["status"] == "requires_human_review"


def test_flags_invented_numbers():
    res = assess_generated_claims(
        "Most dishes cost $15 to $30.", "$15 to $30", ["Most dishes cost $15 to $300."]
    )
    assert any(f["flag"] == "new_number_not_in_source" for f in res["flags"])


def test_flags_irrelevant_fact():
    res = assess_generated_claims(
        "The hotel has no pool and offers free breakfast.",
        "no pool",
        ["The hotel serves breakfast."],
    )
    assert any(f["flag"] == "no_lexical_overlap_with_span" for f in res["flags"])


def test_empty_responses_not_silently_labeled():
    res = assess_generated_claims("Hi.", ":", [])
    assert res["status"] == "requires_human_review"
    assert any(f["flag"] == "no_targeted_claims_returned" for f in res["flags"])


def test_whitespace_dedup():
    res = assess_generated_claims(
        "The hotel has no pool.", "no pool", [" The hotel has no pool. ", "the  hotel has no pool."]
    )
    assert res["proposed_claims"] == ["The hotel has no pool."]


def test_bad_model_result_rejected():
    with pytest.raises(ValueError):
        assess_generated_claims("A", "A", ["claim", None])


def test_wifi_normalization():
    assert meaningful_tokens("Wi-Fi") == meaningful_tokens("WiFi")


def test_mocked_ollama_contract():
    output = {
        "message": {
            "content": json.dumps({"target_atomic_claims": ["Most dishes cost $15 to $30."]})
        }
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: object):
            self.close()

    with patch(
        "enhancements.span_targeted_decomposer.urllib.request.urlopen",
        return_value=Response(json.dumps(output).encode("utf-8")),
    ) as mocked:
        res = targeted_decompose("Most dishes cost $15 to $30.", "$15 to $30")
    assert res["proposed_claims"] == ["Most dishes cost $15 to $30."]
    assert res["status"] == "requires_human_review"
    assert mocked.call_args.kwargs["timeout"] == 300
