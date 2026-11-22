import streamlit as st
import streamlit.components.v1 as components

import json
from pathlib import Path

from html import escape
from typing import Any
from lettucedetect.models.inference import HallucinationDetector
import sys

# --------------------------------------------------
# FULL PIPELINE SETUP (added by Nayem)
# Lets this page use the team's code:
#   - Hybrid Evidence Ranking v2 (Viran)
#   - Evidence-aware claim verification (Nilanga)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERDICT_COLORS = {
    "Supported": "green",
    "Partially Supported": "orange",
    "Contradicted": "red",
    "Unverifiable": "gray",
}

RELATION_LABELS = {
    "entailment": ("🟢", "agrees with the evidence"),
    "neutral": ("🟡", "not mentioned in the evidence"),
    "contradiction": ("🔴", "disagrees with the evidence"),
}


def safe_md(text) -> str:
    # Streamlit treats $ as a math sign, so we escape it
    return str(text).replace("$", "\\$")


@st.cache_resource(show_spinner="Loading Hybrid Evidence Ranking v2 (first time only)...")
def load_evidence_ranker():
    from run_baseline_evidence_integration import create_ranker
    return create_ranker("v2")


def run_full_pipeline(detector, context, question, answer, task):
    # LettuceDetect -> Hybrid v2 evidence -> claim verification
    from run_baseline_evidence_integration import analyze_record
    from enhancements.v2_claim_verification import add_claim_verification

    result = analyze_record(
        detector=detector,
        answer=answer,
        source_info=context,
        task=task,
        question=question,
        ranker=load_evidence_ranker(),
        top_k=3,
    )
    return add_claim_verification(result)


def show_claim_card(number, item):
    span = item.get("baseline_span", {})
    verification = item.get("verification") or {}

    with st.container(border=True):
        st.subheader(f"Claim {number}")

        st.markdown(f"🖍️ **Suspicious part:** :red[{safe_md(span.get('text', '').strip())}]")
        confidence = span.get("confidence")
        if isinstance(confidence, (int, float)):
            st.write(f"Detector confidence: {confidence:.0%}")

        st.markdown(f"✂️ **Claim:** {safe_md(item.get('claim', ''))}")

        ranked = item.get("ranked_evidence") or []
        if ranked:
            st.markdown("📄 **Best evidence (Hybrid v2)**")
            st.info(safe_md(ranked[0].get("evidence", "")))
            if len(ranked) > 1:
                with st.expander("See other evidence (rank 2 and 3)"):
                    for passage in ranked[1:]:
                        st.markdown(f"**#{passage.get('rank')}** — {safe_md(passage.get('evidence', ''))}")
        else:
            st.warning("No evidence was found for this claim.")

        st.markdown("👨‍⚖️ **Verdict**")
        if not verification:
            st.warning(f"No verdict: {item.get('verification_status', 'not verified')}")
            return

        category = verification.get("category") or "Needs review"
        st.badge(category, color=VERDICT_COLORS.get(category, "blue"))
        st.write(
            f"{verification.get('supported_count', 0)} of "
            f"{verification.get('total_atomic_claims', 0)} small claims are supported by the evidence."
        )

        for piece in verification.get("atomic_verifications", []):
            relation = piece.get("nli_relation", "")
            icon, meaning = RELATION_LABELS.get(relation, ("⚪", relation))
            score = piece.get("nli_scores", {}).get(relation, 0)
            st.markdown(f"{icon} {safe_md(piece.get('atomic_claim', ''))} — *{meaning}* ({score:.0%} sure)")

        if verification.get("status") == "provisional":
            st.caption("Note: this verdict is provisional (still being improved).")


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

# Task type is used by Hybrid Evidence Ranking v2 (added by Nayem)
task_type = st.selectbox(
    "Task type",
    ["QA", "Summary", "Data2Txt"],
    key="improvement_task"
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


        # --------------------------------------------------
        # EVIDENCE-AWARE CLAIM VERIFICATION (added by Nayem)
        # Uses the team's model: Hybrid v2 + claim verification
        # --------------------------------------------------
        st.divider()
        st.header("Evidence-Aware Claim Verification")
        st.caption(
            "LettuceDetect finds suspicious parts → Hybrid Evidence Ranking v2 "
            "finds the best evidence → claim verification gives a verdict."
        )

        if predictions:
            try:
                with st.spinner("Finding evidence and verifying claims... (can take up to a minute)"):
                    pipeline_result = run_full_pipeline(
                        detector, context, question, answer, task_type
                    )

                claim_items = pipeline_result.get("claims_with_evidence", [])

                verdict_counts = {}
                for item in claim_items:
                    verdict = (item.get("verification") or {}).get("category") or "Needs review"
                    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

                summary_columns = st.columns(len(verdict_counts) + 1)
                summary_columns[0].metric("Claims checked", len(claim_items))
                for column, (verdict, count) in zip(summary_columns[1:], verdict_counts.items()):
                    column.metric(verdict, count)

                for number, item in enumerate(claim_items, start=1):
                    show_claim_card(number, item)

            except Exception as pipeline_error:
                error_text = str(pipeline_error)
                if "11434" in error_text or "Connection refused" in error_text or "urlopen" in error_text:
                    st.error(
                        "Claim verification could not reach Ollama. "
                        "Open the Ollama app and click the button again."
                    )
                else:
                    st.error(f"Claim verification failed: {pipeline_error}")
                st.exception(pipeline_error)
        else:
            st.info("No suspicious parts, so there are no claims to verify.")

    except Exception as e:
        st.error(f"Detection or saving failed: {e}")
        st.exception(e)
