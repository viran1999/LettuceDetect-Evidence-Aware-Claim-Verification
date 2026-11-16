
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_passages(source_text):
    """Split source evidence into sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        if sentence.strip()
    ]


def rank_evidence(claim, source_text, top_k=3):
    """Rank source passages by relevance to a claim."""

    passages = split_into_passages(source_text)

    if not passages:
        return []

    # Represent the claim and passages using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        [claim] + passages
    )

    # Calculate similarity between claim and each passage
    scores = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    ).flatten()

    # Sort passages by highest similarity
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "evidence": passages[index],
            "score": round(float(scores[index]), 4)
        })

    return results
