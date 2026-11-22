
from sentence_transformers import CrossEncoder
from enhancements.claim_categorization import load_rank_one_records

# Load the actual V2 dataset
records = load_rank_one_records(
    "data/verification/categorization_input_v2.jsonl"
)

# Select the first stock example
record = records[0]
evidence = record["evidence"]

# Manually extracted atomic claims
atomic_claims = [
    "A stock appreciates when its value becomes higher than the price paid.",

    "Buying a stock for $100 and selling it for $150 produces a $50 profit per share.",

    "Stock prices fluctuate, so the final profit is not known until the stock is sold."
]

print("Loading local NLI model...")

model = CrossEncoder(
    "cross-encoder/nli-MiniLM2-L6-H768",
    device="cpu"
)

labels = model.model.config.id2label

# Compare each atomic claim against Rank 1 evidence
pairs = [
    (evidence, claim)
    for claim in atomic_claims
]

scores = model.predict(
    pairs,
    apply_softmax=True
)

for i, (claim, probabilities) in enumerate(
    zip(atomic_claims, scores), start=1
):
    predicted_id = int(probabilities.argmax())

    print("\n" + "=" * 55)
    print(f"ATOMIC CLAIM {i}:")
    print(claim)
    print("PREDICTION:", labels[predicted_id])

    for j, probability in enumerate(probabilities):
        print(f"{labels[j]}: {float(probability):.4f}")
        

# Combine actual NLI predictions into one category
from enhancements.claim_categorization import categorize_atomic_claims

predictions = [
    str(labels[int(probabilities.argmax())]).lower()
    for probabilities in scores
]

result = categorize_atomic_claims(predictions)

print("\n" + "=" * 55)
print("FINAL ATOMIC VERIFICATION RESULT")
print("Predictions:", predictions)
print("Category:", result["category"])
print("Status:", result["status"])
print("Supported:", result["supported_count"])
print("Contradicted:", result["contradicted_count"])
print("Neutral:", result["neutral_count"])

