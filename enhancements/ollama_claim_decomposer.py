
import json
import os
import urllib.request


DEFAULT_MODEL = "qwen2.5:0.5b-instruct"
# DEFAULT_MODEL = "qwen2.5:1.5b-instruct"

OLLAMA_URL = "http://localhost:11434/api/chat"


def decompose_claim(claim, model_name=None):
    """
    Decompose a complex claim into atomic factual statements.

    Args:
        claim: Original claim text.
        model_name: Optional Ollama model override.

    Returns:
        List of atomic claim strings.
    """

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must be a non-empty string")

    model = (
        model_name
        or os.getenv("DECOMPOSITION_MODEL")
        or DEFAULT_MODEL
    )

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
        "model": model,
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
        "options": {"temperature": 0},
        "keep_alive": 0
    }

    request = urllib.request.Request(
        OLLAMA_URL,
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
        raise ValueError("Invalid atomic claims returned by model")

    return [item.strip() for item in atomic_claims]
