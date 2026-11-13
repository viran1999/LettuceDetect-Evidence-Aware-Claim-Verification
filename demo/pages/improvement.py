import streamlit as st
import streamlit.components.v1 as components

import json
from pathlib import Path

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


# verification_result = {
#     "task": "QA",
#     "response_id": "15844",
#     "source_id": "15561",
#     "baseline_span": {
#         "start": 119,
#         "end": 245,
#         "text": " This means that if you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share.",
#         "confidence": 0.9996097683906555
#     },
#     "claim": "When a stock you own appreciates in value: This means that if you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share. However, since stock prices fluctuate constantly, you won't know exactly how much you'll make until you sell.",
#     "verification": {
#         "category": "Partially Supported",
#         "status": "provisional",
#         "supported_count": 1,
#         "contradicted_count": 0,
#         "neutral_count": 2,
#         "total_atomic_claims": 3,
#         "atomic_verifications": [
#             {
#                 "atomic_claim": "When a stock you own appreciates in value, it means you can sell it for a profit.",
#                 "nli_relation": "neutral",
#                 "nli_scores": {
#                     "contradiction": 0.2123,
#                     "entailment": 0.0481,
#                     "neutral": 0.7396
#                 }
#             },
#             {
#                 "atomic_claim": "If you buy a stock for $100 and it increases in value to $150, you can sell it for a profit of $50 per share.",
#                 "nli_relation": "neutral",
#                 "nli_scores": {
#                     "contradiction": 0.2085,
#                     "entailment": 0.0313,
#                     "neutral": 0.7602
#                 }
#             },
#             {
#                 "atomic_claim": "However, stock prices fluctuate constantly, so you won't know exactly how much you'll make until you sell.",
#                 "nli_relation": "entailment",
#                 "nli_scores": {
#                     "contradiction": 0.0074,
#                     "entailment": 0.949,
#                     "neutral": 0.0436
#                 }
#             }
#         ],
#         "scope": "rank_1_only",
#         "evidence_rank": 1,
#         "evidence": "The first way is when a stock you own appreciates in value-that is, when people who want to buy the stock decide that a share is worth more than you paid for it.ince stock prices fluctuate constantly when the market is open, you never really know how much you're going to make until you sell. The second way is when the company that owns the stock issues dividends-a payout that companies sometimes make to shareholders.",
#         "ranking_score": 0.6905,
#         "decomposition_model": "qwen2.5:0.5b-instruct",
#         "nli_model": "cross-encoder/nli-MiniLM2-L6-H768"
#     }
# }



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

# ****************************************************************************
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# if st.button("Detect Hallucinations"):

#     with st.spinner("Detecting hallucinations..."):

#         predictions = detector.predict(
#             context=[context],
#             question=question,
#             answer=answer,
#             output_format="spans"
#         )

#     st.subheader("Hallucination Detection Results")

#     if any(prediction.get("category") for prediction in predictions):
#         st.markdown(
#             create_legend_html(),
#             unsafe_allow_html=True
#         )

#     html_content = create_interactive_text(
#         answer,
#         predictions
#     )

#     components.html(
#         html_content,
#         height=200,
#         scrolling=True
#     )

#     if predictions:

#         rows = [
#             {
#                 "start": prediction.get("start"),
#                 "end": prediction.get("end"),
#                 "text": prediction.get(
#                     "text",
#                     answer[
#                         prediction["start"]:prediction["end"]
#                     ]
#                 ),
#                 "category": prediction.get(
#                     "category", "untyped"
#                 ),
#                 "subcategory": prediction.get(
#                     "subcategory", ""
#                 ),
#                 "confidence": prediction.get("confidence"),
#             }
#             for prediction in predictions
#         ]

#         st.subheader("Detected Span Details")

#         st.dataframe(
#             rows,
#             hide_index=True,
#             use_container_width=True
#         )
#     else:
#         st.success("No hallucinated spans detected.")

# *******************************************************************************************
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# --------------------------------------------------
# ORIGINAL HALLUCINATION DETECTION + SAVE RESULTS
# --------------------------------------------------


if st.button("Detect Hallucinations"):

    try:
        with st.spinner("Detecting hallucinations..."):

            predictions = detector.predict(
                context=[context],
                question=question,
                answer=answer,
                output_format="spans",
            )

            # Prepare baseline results
            output = {
                "model_id": MODEL_OPTIONS[model_label]["model_path"],
                "context": [context],
                "question": question,
                "examples": [
                    {
                        "name": "Current answer",
                        "answer": answer,
                        "predicted_spans": predictions,
                    }
                ],
            }

            # Save baseline results
            project_root = Path(__file__).resolve().parents[2]
            output_path = project_root / "results" / "baseline_demo.json"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text(
                json.dumps(output, indent=2),
                encoding="utf-8",
            )
            # verification_result = json.loads(
            #     output_path.read_text(encoding="utf-8")
            # )

        # Success message
        st.success(f"Baseline results saved to: {output_path}")

        # Display detection results
        st.subheader("Hallucination Detection Results")

        if any(prediction.get("category") for prediction in predictions):
            st.markdown(
                create_legend_html(),
                unsafe_allow_html=True,
            )

        html_content = create_interactive_text(
            answer,
            predictions,
        )

        components.html(
            html_content,
            height=200,
            scrolling=True,
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
                        ],
                    ),
                    "category": prediction.get("category", "untyped"),
                    "subcategory": prediction.get("subcategory", ""),
                    "confidence": prediction.get("confidence"),
                }
                for prediction in predictions
            ]

            st.subheader("Detected Span Details")

            st.dataframe(
                rows,
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.success("No hallucinated spans detected.")

        # verification = verification_result["verification"]
        # baseline = verification_result["baseline_span"]

        # example = verification_result["examples"][0]

        # report_spans = example.get("predicted_spans", [])
        # report_context = verification_result.get("context", [""])[0]
        # report_question = verification_result.get("question", "")
        # report_answer = example.get("answer", "")

        # st.divider()
        # st.header("Detailed Verification Report")

        # # Summary metrics
        # st.subheader("Overall Verification")

        # col1, col2, col3, col4 = st.columns(4)

        # col1.metric("Supported", verification["supported_count"])
        # col2.metric("Contradicted", verification["contradicted_count"])
        # col3.metric("Neutral", verification["neutral_count"])
        # col4.metric("Total Claims", verification["total_atomic_claims"])

        # st.write(
        #     "**Verification Category:**",
        #     verification["category"]
        # )

        # st.write(
        #     "**Verification Status:**",
        #     verification["status"].title()
        # )

        # # Claim-level verification
        # st.subheader("Atomic Claim Verification")

        # claim_rows = []

        # for item in verification["atomic_verifications"]:
        #     scores = item["nli_scores"]

        #     claim_rows.append({
        #         "Atomic Claim": item["atomic_claim"],
        #         "NLI Relation": item["nli_relation"].title(),
        #         "Contradiction (%)": round(scores["contradiction"] * 100, 2),
        #         "Entailment (%)": round(scores["entailment"] * 100, 2),
        #         "Neutral (%)": round(scores["neutral"] * 100, 2),
        #     })

        # st.dataframe(
        #     claim_rows,
        #     hide_index=True,
        #     use_container_width=True
        # )
            
        # # Baseline span
        # st.subheader("Baseline Span")

        # st.write("**Baseline Text:**")
        # st.info(baseline["text"])

        # st.write(
        #     f"**Detection Confidence:** "
        #     f"{baseline['confidence'] * 100:.2f}%"
        # )

        # # Original claim
        # st.subheader("Original Claim")

        # st.write(verification_result["claim"])

        # # Evidence
        # st.subheader("Retrieved Evidence / Proof Text")

        # st.info(verification["evidence"])

        # # Additional information
        # with st.expander("Additional Verification Details"):

        #     st.write("**Task:**", verification_result["task"])
        #     st.write("**Response ID:**", verification_result["response_id"])
        #     st.write("**Source ID:**", verification_result["source_id"])
        #     st.write("**Evidence Rank:**", verification["evidence_rank"])
        #     st.write("**Ranking Score:**", verification["ranking_score"])
        #     st.write("**Scope:**", verification["scope"])
        #     st.write("**Decomposition Model:**", verification["decomposition_model"])
        #     st.write("**NLI Model:**", verification["nli_model"])
            


        # --------------------------------------------------
        # DYNAMIC HALLUCINATION DETECTION REPORT
        # --------------------------------------------------

        st.divider()
        st.header("Detailed Hallucination Detection Report")

        # 1. Input Information
        st.subheader("Input Information")

        st.write("**Question:**")
        st.text(question)

        st.write("**Reference Context:**")
        st.text(context)

        st.write("**Original Answer:**")
        st.text(answer)

        # 2. Process detected spans
        valid_spans = [
            span for span in predictions
            if isinstance(span, dict)
        ]

        category_counts = {}

        report_rows = []

        for span in valid_spans:

            category = span.get("category") or "untyped"
            subcategory = span.get("subcategory") or "N/A"
            confidence = span.get("confidence")

            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

            if isinstance(confidence, (int, float)):
                confidence_display = round(confidence * 100, 2)
            else:
                confidence_display = None

            start = span.get("start")
            end = span.get("end")

            detected_text = span.get("text")

            if not isinstance(detected_text, str):
                if (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start <= end <= len(answer)
                ):
                    detected_text = answer[start:end]
                else:
                    detected_text = "N/A"

            report_rows.append({
                "Detected Text": detected_text,
                "Category": category,
                "Subcategory": subcategory,
                "Confidence (%)": confidence_display,
                "Start Position": start,
                "End Position": end,
            })

        # 3. Overall Summary
        st.subheader("Overall Detection Summary")

        total_spans = len(valid_spans)
        total_categories = len(category_counts)

        col1, col2 = st.columns(2)

        col1.metric("Total Detected Spans", total_spans)
        col2.metric("Detection Categories", total_categories)

        # 4. Category Breakdown
        if category_counts:

            st.write("**Hallucination Category Breakdown**")

            category_rows = [
                {
                    "Category": category,
                    "Count": count,
                }
                for category, count in category_counts.items()
            ]

            st.dataframe(
                category_rows,
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.success("No hallucinated spans detected.")

        # 5. Detailed Span Report
        st.subheader("Detected Claim Details")

        if report_rows:

            st.dataframe(
                report_rows,
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info("No detected hallucination spans to display.")

        # 6. Additional Details
        with st.expander("Additional Detection Details"):

            st.write("**Selected Model:**", MODEL_OPTIONS[model_label]["model_path"])

            st.write("**Total Characters in Answer:**", len(answer))

            st.write("**Total Words in Answer:**", len(answer.split()))

            st.write("**Total Detected Spans:**", total_spans)

            st.write("**Total Categories:**", total_categories)


    except Exception as e:
        st.error(f"Detection or saving failed: {e}")
        st.exception(e)

#         # st.subheader("Raw Detection Output")

#         # st.json(predictions)


# --------------------------------------------------
# STATIC CLAIM VERIFICATION REPORT
# --------------------------------------------------


    # else:
    #     st.success("No hallucinated spans detected.")
