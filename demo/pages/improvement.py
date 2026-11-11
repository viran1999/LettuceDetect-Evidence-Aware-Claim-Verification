
# import streamlit as st

# st.set_page_config(page_title="Improvement", page_icon="✨")

# st.title("Improve Your Answer")
# st.write("Use this page to improve answers and make them more reliable.")

# st.subheader("Context")
# context = st.text_area(
#     "Enter the reference context",
#     value="Jupiter is the largest planet in our solar system. It has a Great Red Spot, which is a giant storm. Jupiter takes approximately 12 Earth years to complete one orbit around the Sun.",
#     height=150,
#     key="improvement_context"
# )

# st.subheader("Question")
# question = st.text_area(
#     "Enter your question",
#     value="Which is the largest planet in our solar system, and how long does it take to orbit the Sun?",
#     height=100,
#     key="improvement_question"
# )

# st.subheader("Original Answer")
# answer = st.text_area(
#     "Enter the answer",
#     value="Saturn is the largest planet in our solar system. It takes approximately 29 Earth years to complete one orbit around the Sun.",
#     height=150,
#     key="improvement_answer"
# )

# if st.button("Improve Answer"):
#     st.info("Improvement functionality will be implemented in the next step.")



import streamlit as st
import streamlit.components.v1 as components

from html import escape
from typing import Any

from lettucedetect.models.inference import HallucinationDetector


st.set_page_config(page_title="Improvement", page_icon="✨")

CATEGORY_COLORS = {
    "contradiction": "rgba(255, 99, 71, 0.35)",
    "unsupported_addition": "rgba(255, 193, 7, 0.35)",
    "fabricated_reference": "rgba(156, 39, 176, 0.30)",
}

DEFAULT_SPAN_COLOR = "rgba(255, 99, 71, 0.30)"

MODEL_OPTIONS = {
    "Typed spans (v2 encoder cascade)": {
        "model_path": "KRLabsOrg/lettucedect-v2-mmbert-base",
        "taxonomy_head": "KRLabsOrg/lettucedect-v2-taxonomy-head",
    },
    "Binary spans (ModernBERT)": {
        "model_path": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    },
}


verification_result = {
    "task": "QA",
    "response_id": "15844",
    "source_id": "15561",
    "baseline_span": {
        "start": 119,
        "end": 245,
        "text": " This means that if you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share.",
        "confidence": 0.9996097683906555
    },
    "claim": "When a stock you own appreciates in value: This means that if you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share. However, since stock prices fluctuate constantly, you won't know exactly how much you'll make until you sell.",
    "verification": {
        "category": "Partially Supported",
        "status": "provisional",
        "supported_count": 1,
        "contradicted_count": 0,
        "neutral_count": 2,
        "total_atomic_claims": 3,
        "atomic_verifications": [
            {
                "atomic_claim": "When a stock you own appreciates in value, it means you can sell it for a profit.",
                "nli_relation": "neutral",
                "nli_scores": {
                    "contradiction": 0.2123,
                    "entailment": 0.0481,
                    "neutral": 0.7396
                }
            },
            {
                "atomic_claim": "If you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share.",
                "nli_relation": "neutral",
                "nli_scores": {
                    "contradiction": 0.2085,
                    "entailment": 0.0313,
                    "neutral": 0.7602
                }
            },
            {
                "atomic_claim": "However, stock prices fluctuate constantly, so you won't know exactly how much you'll make until you sell.",
                "nli_relation": "entailment",
                "nli_scores": {
                    "contradiction": 0.0074,
                    "entailment": 0.949,
                    "neutral": 0.0436
                }
            }
        ],
        "scope": "rank_1_only",
        "evidence_rank": 1,
        "evidence": "The first way is when a stock you own appreciates in value-that is, when people who want to buy the stock decide that a share is worth more than you paid for it.ince stock prices fluctuate constantly when the market is open, you never really know how much you're going to make until you sell. The second way is when the company that owns the stock issues dividends-a payout that companies sometimes make to shareholders.",
        "ranking_score": 0.6905,
        "decomposition_model": "qwen2.5:0.5b-instruct",
        "nli_model": "cross-encoder/nli-MiniLM2-L6-H768"
    }
}



def create_interactive_text(
    text: str, spans: list[dict[str, Any]]
) -> str:

    html_text = escape(text)

    for span in sorted(
        spans, key=lambda item: item["start"], reverse=True
    ):
        start = int(span["start"])
        end = int(span["end"])

        span_text = escape(text[start:end])
        category = span.get("category")
        subcategory = span.get("subcategory")
        confidence = span.get("confidence")

        color = CATEGORY_COLORS.get(
            str(category), DEFAULT_SPAN_COLOR
        )

        tooltip_parts = []

        if category:
            tooltip_parts.append(f"Category: {category}")

        if subcategory:
            tooltip_parts.append(f"Subcategory: {subcategory}")

        if isinstance(confidence, int | float):
            tooltip_parts.append(f"Confidence: {confidence:.3f}")

        tooltip = escape(
            " | ".join(tooltip_parts) or "Hallucination",
            quote=True,
        )

        highlighted_span = (
            f'<span class="hallucination" '
            f'style="background-color: {color}" '
            f'title="{tooltip}">{span_text}</span>'
        )

        html_text = (
            html_text[:start]
            + highlighted_span
            + html_text[end:]
        )

    return f"""
    <style>
        .container {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            padding: 20px;
        }}
        .hallucination {{
            padding: 2px;
            border-radius: 3px;
            cursor: help;
        }}
        .hallucination:hover {{
            filter: saturate(1.5);
        }}
    </style>
    <div class="container">{html_text}</div>
    """


def create_legend_html() -> str:

    items = "".join(
        (
            '<span style="display: inline-flex; '
            'align-items: center; margin-right: 1rem">'
            f'<span style="background: {color}; '
            'width: 0.8rem; height: 0.8rem; '
            'display: inline-block; margin-right: 0.3rem">'
            f'</span>{escape(category)}</span>'
        )
        for category, color in CATEGORY_COLORS.items()
    )

    return (
        '<div style="margin: 0.5rem 0 1rem">'
        f'<strong>Categories:</strong> {items}</div>'
    )


@st.cache_resource
def load_detector(selected_model: str) -> HallucinationDetector:

    return HallucinationDetector(
        method="transformer",
        **MODEL_OPTIONS[selected_model]
    )


# --------------------------------------------------
# IMPROVEMENT PAGE
# --------------------------------------------------

st.title("Our Proposed Improvement")

st.write(
    "An extended hallucination detection and "
    "claim-level verification approach."
)

st.sidebar.subheader("Detector")

model_label = st.sidebar.selectbox(
    "Select Detector",
    list(MODEL_OPTIONS)
)

detector = load_detector(model_label)


# Input fields

st.subheader("Context")

context = st.text_area(
    "Enter the reference context",
    value=(
        "Jupiter is the largest planet in our solar system. "
        "It has a Great Red Spot, which is a giant storm. "
        "Jupiter takes approximately 12 Earth years to "
        "complete one orbit around the Sun."
    ),
    height=150,
    key="improvement_context"
)


st.subheader("Question")

question = st.text_area(
    "Enter your question",
    value=(
        "Which is the largest planet in our solar system, "
        "and how long does it take to orbit the Sun?"
    ),
    height=100,
    key="improvement_question"
)


st.subheader("Original Answer")

answer = st.text_area(
    "Enter the answer",
    value=(
        "Saturn is the largest planet in our solar system. "
        "It takes approximately 29 Earth years to complete "
        "one orbit around the Sun."
    ),
    height=150,
    key="improvement_answer"
)


# --------------------------------------------------
# ORIGINAL HALLUCINATION DETECTION
# --------------------------------------------------

if st.button("Detect Hallucinations"):

    with st.spinner("Detecting hallucinations..."):

        predictions = detector.predict(
            context=[context],
            question=question,
            answer=answer,
            output_format="spans"
        )

    st.subheader("Hallucination Detection Results")

    if any(prediction.get("category") for prediction in predictions):
        st.markdown(
            create_legend_html(),
            unsafe_allow_html=True
        )

    html_content = create_interactive_text(
        answer,
        predictions
    )

    components.html(
        html_content,
        height=200,
        scrolling=True
    )

    if predictions:

        rows = [
            {
                "start": prediction.get("start"),
                "end": prediction.get("end"),
                "text": prediction.get(
                    "text",
                    answer[
                        prediction["start"]:prediction["end"]
                    ]
                ),
                "category": prediction.get(
                    "category", "untyped"
                ),
                "subcategory": prediction.get(
                    "subcategory", ""
                ),
                "confidence": prediction.get("confidence"),
            }
            for prediction in predictions
        ]

        st.subheader("Detected Span Details")

        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("No hallucinated spans detected.")


        # st.subheader("Raw Detection Output")

        # st.json(predictions)


# --------------------------------------------------
# STATIC CLAIM VERIFICATION REPORT
# --------------------------------------------------

verification = verification_result["verification"]
baseline = verification_result["baseline_span"]

st.divider()
st.header("Detailed Verification Report")

# Summary metrics
st.subheader("Overall Verification")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Supported", verification["supported_count"])
col2.metric("Contradicted", verification["contradicted_count"])
col3.metric("Neutral", verification["neutral_count"])
col4.metric("Total Claims", verification["total_atomic_claims"])

st.write(
    "**Verification Category:**",
    verification["category"]
)

st.write(
    "**Verification Status:**",
    verification["status"].title()
)

# Claim-level verification
st.subheader("Atomic Claim Verification")

claim_rows = []

for item in verification["atomic_verifications"]:
    scores = item["nli_scores"]

    claim_rows.append({
        "Atomic Claim": item["atomic_claim"],
        "NLI Relation": item["nli_relation"].title(),
        "Contradiction (%)": round(scores["contradiction"] * 100, 2),
        "Entailment (%)": round(scores["entailment"] * 100, 2),
        "Neutral (%)": round(scores["neutral"] * 100, 2),
    })

st.dataframe(
    claim_rows,
    hide_index=True,
    use_container_width=True
)
    
# Baseline span
st.subheader("Baseline Span")

st.write("**Baseline Text:**")
st.info(baseline["text"])

st.write(
    f"**Detection Confidence:** "
    f"{baseline['confidence'] * 100:.2f}%"
)

# Original claim
st.subheader("Original Claim")

st.write(verification_result["claim"])

# Evidence
st.subheader("Retrieved Evidence / Proof Text")

st.info(verification["evidence"])

# Additional information
with st.expander("Additional Verification Details"):

    st.write("**Task:**", verification_result["task"])
    st.write("**Response ID:**", verification_result["response_id"])
    st.write("**Source ID:**", verification_result["source_id"])
    st.write("**Evidence Rank:**", verification["evidence_rank"])
    st.write("**Ranking Score:**", verification["ranking_score"])
    st.write("**Scope:**", verification["scope"])
    st.write("**Decomposition Model:**", verification["decomposition_model"])
    st.write("**NLI Model:**", verification["nli_model"])
    
    # else:
    #     st.success("No hallucinated spans detected.")
