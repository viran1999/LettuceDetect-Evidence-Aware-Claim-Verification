import json
from pathlib import Path

import torch
from lettucedetect.models.inference import HallucinationDetector

# Limit CPU threads to keep the laptop responsive.
torch.set_num_threads(2)

MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

detector = HallucinationDetector(
    method="transformer",
    model_path=MODEL_ID,
)

contexts = [
    "France is a country in Europe. "
    "The capital of France is Paris. "
    "The population of France is 67 million."
]

question = "What is the capital of France and its population?"

examples = [
    {
        "name": "Supported answer",
        "answer": (
            "The capital of France is Paris. "
            "The population of France is 67 million."
        ),
    },
    {
        "name": "Answer containing incorrect information",
        "answer": (
            "The capital of France is Paris. "
            "The population of France is 69 million."
        ),
    },
]

results = []

for example in examples:
    print(f"\n--- {example['name']} ---", flush=True)
    print("Answer:", example["answer"], flush=True)

    predictions = detector.predict(
        context=contexts,
        question=question,
        answer=example["answer"],
        output_format="spans",
    )

    print(
        "Predicted hallucinated spans:",
        json.dumps(predictions, indent=2),
        flush=True,
    )

    results.append({
        **example,
        "predicted_spans": predictions,
    })

output = {
    "model_id": MODEL_ID,
    "context": contexts,
    "question": question,
    "examples": results,
}

output_path = Path("results/baseline_demo.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

print(f"\nResults saved to {output_path}", flush=True)