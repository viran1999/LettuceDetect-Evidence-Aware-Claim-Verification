"""Offline tests; do not download the real LettuceDetect model."""
import json

import pytest

from run_baseline_evidence_integration import (
    analyze_record, claim_containing_span, evidence_as_text, extract_question,
    load_record, normalize_spans,
)


class FakeDetector:
    def __init__(self, spans):
        self.spans = spans
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.spans


def test_serialization_preserves_boolean():
    assert "WiFi: False" in evidence_as_text({"WiFi": False, "Music": None})
    assert "Music" not in evidence_as_text({"WiFi": False, "Music": None})


def test_question_extraction():
    assert extract_question({"question": "Where is Paris?"}) == "Where is Paris?"
    assert extract_question("question: Where is Paris?\npassages: ...)" ) == "Where is Paris?"
    assert extract_question({"prompt": "Full source context"}) == ""


def test_exact_offset_and_claim():
    a = "Paris is the capital. France has 69 million people."
    start = a.index("69")
    span = normalize_spans(a, [{"start": start, "end": start + 2, "text": "69", "confidence": 0.9}])[0]
    assert span["text"] == "69"
    assert claim_containing_span(a, start, start + 2) == "France has 69 million people."


def test_invalid_offsets_rejected():
    with pytest.raises(ValueError):
        normalize_spans("abc", [{"start": -1, "end": 3}])


def test_baseline_predictions_used_and_retained():
    answer = "The capital is Paris. The population is 69 million."
    start = answer.index("69 million")
    detector = FakeDetector([{"start": start, "end": start + 10,
                             "text": "69 million", "confidence": 0.92}])
    calls = []

    def ranker(claim, source, task, k):
        calls.append((claim, source, task, k))
        return [{"rank": 1, "evidence": "Population: 67 million", "score": 0.8}]

    result = analyze_record(detector=detector, answer=answer, question="How many?",
                            source_info={"population": "67 million"}, task="QA", ranker=ranker)
    assert len(detector.calls) == 1
    assert detector.calls[0]["output_format"] == "spans"
    assert detector.calls[0]["context"] == ["population: 67 million"]
    assert result["baseline_spans"][0]["start"] == start
    assert result["baseline_spans"][0]["confidence"] == 0.92
    assert calls[0][0] == "The population is 69 million."
    assert result["claims_with_evidence"][0]["verification_status"] == "NOT_IMPLEMENTED"


def test_duplicate_claims_ranked_once():
    answer = "France has 69 million people."
    detector = FakeDetector([{"start": 11, "end": 13}, {"start": 14, "end": 21}])
    calls = []
    def ranker(*args):
        calls.append(args)
        return []
    result = analyze_record(detector=detector, answer=answer, question="", source_info="France has 67 million people.", task="QA", ranker=ranker)
    assert len(result["claims_with_evidence"]) == 2
    assert len(calls) == 1


def test_empty_predictions_do_not_fabricate_hallucinations():
    detector = FakeDetector([])
    result = analyze_record(detector=detector, answer="Paris is in France.", question="",
                            source_info="Paris is in France.", task="QA", ranker=lambda *args: [])
    assert not result["baseline_example_prediction"]
    assert result["claims_with_evidence"] == []


def test_source_lookup_not_gold_labels(tmp_path):
    folder = tmp_path / "ragtruth"
    folder.mkdir()
    (folder / "response.jsonl").write_text(json.dumps({"id": 5661, "source_id": 13599,
                                                         "response": "There is free WiFi", "labels": [{"start": 9, "end": 18}]}) + "\n")
    (folder / "source_info.jsonl").write_text(json.dumps({"source_id": 13599, "source_info": "WiFi: no"}) + "\n")
    response, source = load_record("5661", folder)
    assert response["response"] == "There is free WiFi"
    assert source["source_info"] == "WiFi: no"


def test_v2_adapter_with_stubbed_embeddings(monkeypatch):
    """v2 uses the predicted claim and extracts WiFi: no without loading model weights."""
    import sys
    import types
    import numpy as np
    from run_baseline_evidence_integration import create_ranker

    class StubEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, strings, **kwargs):
            embeddings = []
            for s in strings:
                lo = s.lower()
                embeddings.append([
                    float("wifi" in lo or "wi-fi" in lo),
                    float("restroom" in lo),
                    float("seating" in lo),
                ])
            return np.asarray(embeddings)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=StubEmbedding),
    )
    ranker = create_ranker("v2")
    passages = ranker(
        "The restaurant offers free WiFi.",
        "name: Restaurant\nWiFi: no\nOutdoorSeating: True\nRestroom needs attention.",
        "Data2Txt", 3,
    )
    assert passages
    assert passages[0]["evidence"].lower().startswith("wifi: no")
    assert passages[0]["rank"] == 1
