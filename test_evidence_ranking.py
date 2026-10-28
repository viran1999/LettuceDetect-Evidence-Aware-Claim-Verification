"""Quick local test for the LettuceDetect evidence-ranking enhancement.

Run from the LettuceDetect repository root:
    python test_evidence_ranking.py
"""

import json
from pathlib import Path

from enhancements.evidence_ranking import rank_evidence


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ragtruth"
RESPONSES = DATA_DIR / "response.jsonl"
SOURCES = DATA_DIR / "source_info.jsonl"
CLAIM = "Anne Frank died on February 7, 2022."


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def main():
    # Select an actual record rather than relying on a hard-coded ID or ID type.
    response = next(
        (item for item in read_jsonl(RESPONSES)
         if "Anne Frank" in str(item.get("response", "")) and item.get("labels")),
        None,
    )
    if response is None:
        print("No annotated Anne Frank example found in response.jsonl")
        return

    target_id = str(response.get("source_id", "")).strip()
    print("Response source_id:", repr(target_id))
    print("Task label from response:", response.get("task_type", "not included"))
    print("Claim:", CLAIM)

    source = next(
        (item for item in read_jsonl(SOURCES)
         if str(item.get("source_id", "")).strip() == target_id),
        None,
    )
    if source is None:
        print("\nNo matching source_id was found in source_info.jsonl.")
        print("First five IDs in the source file:")
        for item in list(read_jsonl(SOURCES))[:5]:
            print(" ", repr(item.get("source_id")))
        print("Check whether both files belong to the same RAGTruth dataset release.")
        return

    evidence = source_text(source.get("source_info"))
    print("Task type:", source.get("task_type"))
    print("Evidence length:", len(evidence), "characters")
    if not evidence.strip():
        print("Source evidence is empty; inspect source_info.jsonl")
        return

    results = rank_evidence(CLAIM, evidence, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\nRank {i} | similarity: {result['score']}")
        print(result["evidence"])
    if not results:
        print("No passages returned by the ranking module.")


if __name__ == "__main__":
    main()
