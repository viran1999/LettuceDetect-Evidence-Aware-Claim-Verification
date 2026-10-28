 <div align="center">

# Enhancing LettuceDetect with Evidence-Aware Claim Verification

### Evidence Ranking and Hallucination Categorization for RAG Systems

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Research](https://img.shields.io/badge/Project-Research-purple)
![Baseline](https://img.shields.io/badge/Baseline-Reproduced-success)
![Enhancement](https://img.shields.io/badge/Enhancements-In%20Progress-orange)

**An extension of LettuceDetect for more interpretable hallucination analysis in Retrieval-Augmented Generation (RAG) systems.**

</div>

---

## Overview

Retrieval-Augmented Generation (RAG) systems combine external knowledge with Large Language Models (LLMs) to generate more informed responses. However, these systems can still produce inaccurate or unsupported information.

**LettuceDetect** is an existing hallucination detection framework that identifies potentially hallucinated text spans in generated responses.

This research project aims to extend LettuceDetect with two additional components:

- **Evidence Ranking:** Identify and rank relevant evidence passages for detected hallucination spans.
- **Hallucination Categorization:** Analyze suspicious claims against source evidence to identify different types of hallucinations.

The goal is to improve the interpretability of hallucination detection and investigate whether evidence-aware verification improves detection performance.

## Proposed System Architecture

```text
       RAGTruth Dataset
              |
              v
       Source Evidence
       + Generated Answer
              |
              v
     +--------------------+
     |   LettuceDetect   |
     | Hallucination     |
     | Span Detection    |
     +--------------------+
              |
              v
     +--------------------+
     | Evidence Ranking   |
     | (Proposed)         |
     +--------------------+
              |
              v
     +--------------------+
     | Hallucination      |
     | Categorization     |
     | (Proposed)         |
     +--------------------+
              |
              v
       Final Analysis
```

## Proposed Enhancements

### 1. Evidence Ranking

The evidence ranking module will compare detected suspicious claims with available source passages and rank those passages according to their relevance.

The module is intended to return the most relevant evidence for further verification.

### 2. Hallucination Categorization

The categorization module will analyze detected claims against ranked evidence to identify their relationship to the source.

Initial categories will be based on RAGTruth's existing hallucination annotations, including:

- **Evident Conflict:** Information that contradicts the source evidence.
- **Baseless Information:** Information not supported by the available source evidence.

Additional categories may be explored during implementation.

## Baseline Reproduction

The original LettuceDetect base model has been evaluated using the RAGTruth dataset.

### Example-Level Evaluation Results

| Metric | Reproduced Result |
|---|---:|
| Precision | 76.64% |
| Recall | 75.50% |
| F1-score | 76.07% |
| AUROC | 0.8886 |

### Span-Level Evaluation

The project also reproduces character-level span evaluation using the existing LettuceDetect evaluation script.

Performance is measured across three RAGTruth task types:

- **Question Answering:** Generating answers from provided evidence.
- **Data-to-Text:** Converting structured data into natural-language text.
- **Summarization:** Producing shorter versions of source documents.

## Dataset

This research uses the [RAGTruth dataset](https://github.com/ParticleMedia/RAGTruth), which provides generated responses, source information, and human-annotated hallucination spans.

Dataset files are prepared locally for evaluation.

## Getting Started

### Requirements

- Python 3.11 or later
- Git
- Required dependencies from the original LettuceDetect project

### Clone the Repository

```bash
git clone https://github.com/viran1999/LettuceDetect-Evidence-Aware-Claim-Verification.git
cd LettuceDetect-Evidence-Aware-Claim-Verification
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -e .
```

### Run Baseline Evaluation

After preparing the RAGTruth dataset:

```bash
python scripts/evaluate.py \
  --model_path KRLabsOrg/lettucedect-base-modernbert-en-v1 \
  --data_path data/ragtruth/ragtruth_data.json \
  --evaluation_type example_level \
  --batch_size 4
```

## Project Progress

| Component | Status |
|---|---|
| Baseline reproduction | Completed |
| Example-level evaluation | Completed |
| Character-level span evaluation | Completed |
| Evidence ranking | Planned |
| Hallucination categorization | Planned |
| Enhanced system evaluation | Planned |

## Research Objectives

1. Reproduce the baseline performance of LettuceDetect.
2. Implement evidence ranking for suspicious claims.
3. Develop an evidence-aware hallucination categorization module.
4. Evaluate the proposed enhancements against the original baseline.
5. Analyze performance, interpretability, and computational requirements.

## Original Research and Acknowledgment

This project builds upon the original [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) research and implementation.

**Original paper:** [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125)

The original framework is used as the baseline for this academic research. Credit for the original research and implementation belongs to its respective authors and contributors.

The original license and applicable copyright notices are retained.

---

<div align="center">

**Academic Research Project |
