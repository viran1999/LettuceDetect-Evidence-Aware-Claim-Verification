
# Evidence-Aware Atomic Claim Verification

## Overview

This module extends the LettuceDetect hallucination detection pipeline with evidence-aware claim verification.

It uses Rank 1 evidence from Evidence Ranking V2 to classify claims into four categories:

- Supported
- Contradicted
- Partially Supported
- Unverifiable

All predictions are provisional until formally evaluated.

## Models Used

### 1. Qwen2.5-Instruct — Atomic Claim Decomposition

Splits complex claims into smaller, independently verifiable factual statements.

Local development model:
`qwen2.5:0.5b-instruct`

The model is accessed through Ollama. The model name can be changed using the `DECOMPOSITION_MODEL` environment variable.

### 2. MiniLM NLI — Evidence Verification

Model:
`cross-encoder/nli-MiniLM2-L6-H768`

Uses Natural Language Inference (NLI) to compare each atomic claim against Rank 1 evidence.

Possible relations:
- Entailment
- Contradiction
- Neutral

## Execution Flow

1. Load claim records from Evidence Ranking V2.
2. Extract the highest-ranked evidence passage.
3. Decompose the original claim using Qwen.
4. Verify each atomic claim using MiniLM NLI.
5. Combine atomic predictions into an overall category.
6. Save verification results as JSONL.

## Main Files

- `run_atomic_rank1_categorization.py` — Enhanced pipeline entry point.
- `enhancements/ollama_claim_decomposer.py` — Qwen decomposition.
- `enhancements/atomic_claim_verifier.py` — Atomic NLI verification.
- `enhancements/claim_categorization.py` — Rank 1 loading and category logic.
- `run_rank1_categorization.py` — Original whole-claim NLI baseline.

## Setup

Activate the project's Python environment and install the NLI dependency:

```bash
source .venv/bin/activate
python -m pip install sentence-transformers
```

Install Ollama separately, then start its service and download the local model:

```bash
sudo systemctl start ollama
ollama pull qwen2.5:0.5b-instruct
```

No OpenAI API key is required.

## Input Data

The runner expects:

`data/verification/categorization_input_v2.jsonl`

This file must be generated or supplied separately from Evidence Ranking V2. It is not included with this module.

## Run

```bash
python run_atomic_rank1_categorization.py
```

The runner currently processes the first five claim records.

To process all available records, change:

```python
selected_records = records[:5]
```

to:

```python
selected_records = records
```

## Output

Results are saved to:

`results/categorization/atomic_rank1_predictions.jsonl`

Each record includes the overall category, atomic claims, NLI predictions, model information and Rank 1 evidence.

## Current Limitations

- Only Rank 1 evidence is verified.
- Automatic claim decomposition can introduce duplicates or alter wording.
- Context-dependent claims may lack their original subject.
- Mixed entailment and contradiction cases can return `needs_review` with no category.
- NLI scores are not calibrated factual confidence estimates.
- Formal classification accuracy has not yet been established.

The module has passed local functionality tests on five real claim records and controlled examples covering all four categories.
