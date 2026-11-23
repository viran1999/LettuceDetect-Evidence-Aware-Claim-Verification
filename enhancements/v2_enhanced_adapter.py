"""Backend-only compatibility layer for the existing Streamlit enhancement.py.

The public entry point remains enhancements.v2_claim_verification.add_claim_verification.
Enable enhanced v2 with LETTUCE_VERIFIER=enhanced after testing. No frontend edit.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _ranker():
    # One ranking encoder per app process for the v3 path.
    from run_baseline_evidence_integration import create_ranker
    return create_ranker("v2")


def _run_enhanced(result: dict, source_info: Any, ranker):
    from enhancements.span_verification_v3 import add_spanaware_verification
    return add_spanaware_verification(result, source_info, ranker)


def _best_check(atom: dict) -> dict | None:
    checks = [x for x in atom.get("evidence_checks", []) if isinstance(x, dict)]
    if not checks:
        return None
    relation = {"Supported": "entailment", "Contradicted": "contradiction"}.get(atom.get("category"))
    if relation:
        preferred = [x for x in checks if x.get("nli_relation") == relation]
        if preferred:
            return max(preferred, key=lambda x: float((x.get("nli_scores") or {}).get(relation, 0.0)))
    # For neutral and unresolved decisions, present highest-ranked checked passage.
    return min(checks, key=lambda x: (x.get("rank") is None, x.get("rank") or 99999))


def adapt_enhanced_output_for_frontend(result: dict) -> dict:
    """Populate the existing v2 UI fields from enhanced decisions without altering verdicts.

    Keeps original parent-claim evidence in `parent_ranked_evidence` and exposes
    inspected atomic-claim evidence through the pre-existing `ranked_evidence` UI field.
    """
    for review in result.get("claims_with_evidence", []):
        verification = review.get("verification")
        if not isinstance(verification, dict):
            continue
        atom_results = verification.get("atomic_verifications_v3", [])
        frontend_atoms = []
        selected_passages = []
        used_evidence = set()
        for atom in atom_results:
            if not isinstance(atom, dict):
                continue
            chosen = _best_check(atom)
            frontend_atoms.append({
                "atomic_claim": atom.get("atomic_claim", ""),
                "nli_relation": chosen.get("nli_relation", "") if chosen else "",
                "nli_scores": (chosen.get("nli_scores") or {}) if chosen else {},
                "category": atom.get("category"),
                "decision_reason": atom.get("reason"),
                "status": atom.get("status"),
            })
            # The claim card displays only up to three passages. Ensure they
            # actually were checked for at least one of the selected atoms.
            if chosen and chosen.get("evidence") and chosen["evidence"] not in used_evidence:
                used_evidence.add(chosen["evidence"])
                selected_passages.append({
                    "rank": len(selected_passages) + 1,
                    "evidence": chosen["evidence"],
                    "score": chosen.get("ranking_score"),
                    "verified_atomic_claim": atom.get("atomic_claim", ""),
                })
        verification["atomic_verifications"] = frontend_atoms
        verification["total_atomic_claims"] = len(frontend_atoms)
        verification["supported_count"] = sum(a.get("category") == "Supported" for a in atom_results)
        verification["contradicted_count"] = sum(a.get("category") == "Contradicted" for a in atom_results)
        verification["verification_backend"] = "v2_spanaware_enhanced"
        if selected_passages:
            # Never lose the earlier claim-level ranking (useful for comparisons).
            review["parent_ranked_evidence"] = review.get("ranked_evidence", [])
            review["ranked_evidence"] = selected_passages[:3]
            first = selected_passages[0]
            verification["evidence"] = first["evidence"]
            verification["evidence_rank"] = first["rank"]
            # Older JSON-viewing frontend formats this numerically.
            verification["ranking_score"] = float(first["score"]) if first["score"] is not None else 0.0
            verification["ranking_score_available"] = first["score"] is not None
    return result


def add_enhanced_verification_for_frontend(result: dict, source_info=None) -> dict:
    """Same input/output contract as add_claim_verification(..., source_info=...)."""
    if source_info is None:
        raise ValueError("Enhanced v2 requires source_info; original v2 remains available")
    verified = _run_enhanced(result, source_info, _ranker())
    return adapt_enhanced_output_for_frontend(verified)
