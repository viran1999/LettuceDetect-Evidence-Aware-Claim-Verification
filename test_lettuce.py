from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
)

contexts = [
    "France is a country in Europe. The capital of France is Paris. "
    "The population of France is 67 million."
]

question = "What is the capital of France? What is the population of France?"

answer = "The capital of France is Paris. The population of France is 69 million."

predictions = detector.predict(
    context=contexts,
    question=question,
    answer=answer,
    output_format="spans"
)

print("Predictions:", predictions)
