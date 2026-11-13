import streamlit as st
import streamlit.components.v1 as components

import json
from pathlib import Path

from html import escape
from typing import Any
from lettucedetect.models.inference import HallucinationDetector


st.set_page_config(page_title="Enhancement", page_icon="✨")

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

st.title("Enhancement")

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
        # st.success(f"Baseline results saved to: {output_path}")

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
