
import json
import urllib.request
from sentence_transformers import CrossEncoder
from enhancements.claim_categorization import categorize_atomic_claims

from enhancements.claim_categorization import load_rank_one_records

MODEL = "qwen2.5:0.5b-instruct"
# MODEL = "qwen2.5:1.5b-instruct"

records = load_rank_one_records(
    "data/verification/categorization_input_v2.jsonl"
)


def decompose_claim(claim):
    
    prompt = f"""
You are an atomic claim decomposition system.

Convert the input into independent, verifiable factual statements.

STRICT RULES:
1. Each atomic claim must express only ONE distinct fact.
2. Separate definitions, examples, and additional factual assertions.
3. Separate different factual assertions joined by "and" or commas.
4. Repeat the relevant subject in every atomic claim.
5. Every atomic claim must be a complete sentence.
6. Preserve all numbers, names, conditions, and uncertainty.
7. Do not invent facts or change the original meaning.
8. Do not evaluate whether a statement is true.
9. Return only JSON with an "atomic_claims" array.

Example input:
A camera has a 24 MP sensor, records video at 60 fps, and costs $500.

Example output:
{{
  "atomic_claims": [
    "The camera has a 24 MP sensor.",
    "The camera records video at 60 fps.",
    "The camera costs $500."
  ]
}}

Example input:
If a ticket costs $40 and you buy two tickets, the total is $80.
Ticket prices can change over time.

Example output:
{{
  "atomic_claims": [
    "If a ticket costs $40 and you buy two tickets, the total is $80.",
    "Ticket prices can change over time."
  ]
}}

Now decompose the following input into the smallest
independently verifiable factual statements.

Input:
{claim}

Output:
"""


    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "atomic_claims": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["atomic_claims"]
        },
        "options": {
            "temperature": 0
        },
        "keep_alive": 0
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)

    content = result["message"]["content"]
    parsed = json.loads(content)

    atomic_claims = parsed["atomic_claims"]

    if (
        not isinstance(atomic_claims, list)
        or not atomic_claims
        or not all(
            isinstance(item, str) and item.strip()
            for item in atomic_claims
        )
    ):
        raise ValueError("Invalid atomic claims returned")

    return atomic_claims

decomposition_cases = []

for index, record in enumerate(records[:2], start=1):
    print("\n" + "=" * 60)
    print(f"EXAMPLE {index}")
    print("ORIGINAL CLAIM:")
    print(record["claim"])

    print("\nGENERATING ATOMIC CLAIMS...")

    atomic_claims = decompose_claim(record["claim"])

    print("\nDECOMPOSED CLAIMS:")
    for number, atomic in enumerate(atomic_claims, start=1):
        print(f"{number}. {atomic}")

    print("TOTAL ATOMIC CLAIMS:", len(atomic_claims))
    
    decomposition_cases.append((record, atomic_claims))


print("\nLoading local MiniLM NLI model...")

nli_model = CrossEncoder(
    "cross-encoder/nli-MiniLM2-L6-H768",
    device="cpu"
)

labels = {
    int(k): str(v).lower()
    for k, v in nli_model.model.config.id2label.items()
}

for index, (record, atomic_claims) in enumerate(
    decomposition_cases, start=1
):
    evidence = record["evidence"]

    # Verify each atomic claim against Rank 1 evidence.
    pairs = [
        (evidence, atomic_claim)
        for atomic_claim in atomic_claims
    ]

    scores = nli_model.predict(
        pairs,
        apply_softmax=True
    )

    predictions = []

    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULT — EXAMPLE {index}")

    for number, (claim, probabilities) in enumerate(
        zip(atomic_claims, scores), start=1
    ):
        predicted_id = int(probabilities.argmax())
        relation = labels[predicted_id]

        predictions.append(relation)

        print(f"\nATOMIC CLAIM {number}: {claim}")
        print("NLI RELATION:", relation)
        print(
            "MODEL SCORE:",
            round(float(probabilities[predicted_id]), 4)
        )

    # Combine atomic-level predictions.
    result = categorize_atomic_claims(predictions)

    print("\nFINAL VERIFICATION")
    print("Category:", result["category"])
    print("Status:", result["status"])
    print("Supported:", result["supported_count"])
    print("Contradicted:", result["contradicted_count"])
    print("Neutral:", result["neutral_count"])
