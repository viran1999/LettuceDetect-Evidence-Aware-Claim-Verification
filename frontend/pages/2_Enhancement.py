import json
import streamlit as st


# Colors for the final verdict
VERDICT_COLORS = {
    "Supported": "green",
    "Partially Supported": "orange",
    "Contradicted": "red",
    "Unverifiable": "gray",
}

# Icons and words for each small piece (atomic claim)
RELATION_LABELS = {
    "entailment": ("🟢", "agrees with the evidence"),
    "neutral": ("🟡", "not mentioned in the evidence"),
    "contradiction": ("🔴", "disagrees with the evidence"),
}


def safe(text):
    # Streamlit treats $ as a math sign, so we escape it
    return str(text).replace("$", "\\$")


def read_results(text):
    # Reads results whether each claim is on one line
    # or spread over many lines ("pretty" JSON)
    decoder = json.JSONDecoder()
    results = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        item, position = decoder.raw_decode(text, position)
        if isinstance(item, list):
            results.extend(item)
        else:
            results.append(item)
    return results


st.title("✨ Enhancement: Evidence-Aware Verification")
st.write("For each suspicious claim: the best evidence (Hybrid v2) and the verdict.")

uploaded = st.file_uploader("Upload results file", type=["jsonl", "json"])

if uploaded:
    text = uploaded.read().decode("utf-8")

    results = read_results(text)

    st.write("Number of claims:", len(results))

    for number, item in enumerate(results, start=1):
        span = item.get("baseline_span", {})
        verification = item.get("verification", {})

        with st.container(border=True):
            # Card title
            st.subheader(f"Claim {number}")
            st.caption(f"{item.get('task', '')} · Response #{item.get('response_id', '')}")

            # Part 1: what LettuceDetect found
            st.markdown(f"🖍️ **Suspicious part:** :red[{safe(span.get('text', '').strip())}]")
            st.write(f"Detector confidence: {span.get('confidence', 0):.0%}")

            # Part 2: the full claim
            st.markdown(f"✂️ **Claim:** {safe(item.get('claim', ''))}")

            # Part 3: the evidence from Hybrid v2
            st.markdown(f"📄 **Evidence #{verification.get('evidence_rank', 1)} (Hybrid v2)**")
            st.info(safe(verification.get("evidence", "No evidence found.")))
            st.caption(f"Ranking score: {verification.get('ranking_score', 0):.2f}")

            # Part 4: the verdict (the Judge)
            st.markdown("👨‍⚖️ **Verdict**")
            category = verification.get("category", "Unknown")
            st.badge(category, color=VERDICT_COLORS.get(category, "blue"))
            st.write(
                f"{verification.get('supported_count', 0)} of "
                f"{verification.get('total_atomic_claims', 0)} small claims are supported by the evidence."
            )

            # Part 5: each small piece of the claim
            for piece in verification.get("atomic_verifications", []):
                relation = piece.get("nli_relation", "")
                icon, meaning = RELATION_LABELS.get(relation, ("⚪", relation))
                score = piece.get("nli_scores", {}).get(relation, 0)
                st.markdown(f"{icon} {safe(piece.get('atomic_claim', ''))} — *{meaning}* ({score:.0%} sure)")

            if verification.get("status") == "provisional":
                st.caption("Note: this verdict is provisional (still being improved).")