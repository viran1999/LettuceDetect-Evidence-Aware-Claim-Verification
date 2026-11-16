"""Semantic evidence ranking prototype for the LettuceDetect enhancement.

Run from the project root:
    python semantic_evidence_ranking.py

This prototype uses a manually written claim to isolate the ranking component.
Do not use ground-truth labels or annotation explanations as ranking input.
"""

import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_FILE = Path("data/ragtruth/source_info.jsonl")
SOURCE_ID = "15592"
CLAIM = "Anne Frank died on February 7, 2022."


def split_passages(source_text):
    """Create sentence-length passages from the original source text."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if s.strip()]


def load_source(source_id):
    with SOURCE_FILE.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if str(item.get("source_id")) == str(source_id):
                source = item.get("source_info")
                if not isinstance(source, str):
                    raise TypeError("This test expects a text source_info field")
                return source
    raise ValueError(f"Source {source_id} not found in {SOURCE_FILE}")


def rank_evidence(claim, source_text, model, top_k=3):
    passages = split_passages(source_text)
    if not passages:
        return []

    # Normalized embeddings allow a dot product to represent cosine similarity.
    claim_embedding = model.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    evidence_embeddings = model.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.dot_score(claim_embedding, evidence_embeddings)[0]
    ranked_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [(passages[i], float(scores[i])) for i in ranked_indices]


def main():
    print("Loading source evidence...")
    source = load_source(SOURCE_ID)
    print(f"Source ID: {SOURCE_ID} | Evidence length: {len(source)} characters")
    print(f"Claim: {CLAIM}\n")
    print(f"Loading model: {MODEL_NAME}")
    print("Note: the first run downloads model files; later runs reuse the cache.\n")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    for rank, (passage, score) in enumerate(rank_evidence(CLAIM, source, model), 1):
        print(f"Rank {rank} | cosine similarity: {score:.4f}")
        print(passage)
        print()


if __name__ == "__main__":
    main()
