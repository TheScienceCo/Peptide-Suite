"""
What each goal computes, and what it refuses to claim.

Lives in core rather than in `api.py` because the boundary CI job installs no
third-party packages on purpose, and importing the FastAPI app to read a label
is what broke it. The catalogue is data about the engine's own capabilities, so
the engine is where it belongs; the HTTP layer serves it and adds nothing.

The labels here were audited. "Binding affinity" implied a computed dissociation
constant and what is computed is the size of a physicochemical perturbation, so
the lane says so. `computes` and `does_not_compute` are structured fields rather
than prose, because prose is what an interface truncates first.
"""

from __future__ import annotations

from typing import Dict


# The four kinds of reasoning that could bear on a binding question, and which
# of them this deployment can actually do. Presenting all four as one "binding
# affinity prediction" is the overstatement the label audit exists to fix: a
# sequence-derived perturbation size and a structure-supported interaction
# prediction are not the same claim and must not share a number.
BINDING_REASONING_KINDS = [
    {
        "kind": "sequence_derived",
        "label": "Sequence-derived perturbation",
        "available": True,
        "what_it_is": (
            "How large a physicochemical change the substitution is: charge at the "
            "requested pH by Henderson-Hasselbalch, and hydrophobicity by Kyte-Doolittle."),
        "what_it_is_not": (
            "Not a statement about whether the change helps or hurts binding. A large "
            "perturbation at a position that does not touch the receptor changes nothing."),
    },
    {
        "kind": "experimental_mutation",
        "label": "Experimentally measured mutation effect",
        "available": False,
        "what_it_is": (
            "A published measurement of this substitution in this peptide against this "
            "receptor."),
        "what_it_is_not": (
            "Not available: no empirical variant evidence store is populated in this "
            "build, and PubMed is unreachable from this environment."),
    },
    {
        "kind": "structure_supported",
        "label": "Structure-supported interaction prediction",
        "available": False,
        "what_it_is": (
            "What the substitution does to a resolved contact, from an experimental "
            "structure of the complex."),
        "what_it_is_not": (
            "Not available: the structure-template gate has supplied no interface "
            "structure, so no contact can be evaluated."),
    },
    {
        "kind": "learned_model",
        "label": "Learned model prediction",
        "available": False,
        "what_it_is": "A model trained on measured binding data.",
        "what_it_is_not": (
            "Not available: no model in this repository is trained on real labels."),
    },
]


def goal_catalog() -> Dict:
    """
    Goals the substitution predictor has a real scoring path for.

    Each goal declares what it computes and what it does not. A goal absent from
    this list has no scoring path, and is absent rather than present-and-empty:
    an empty capability list reads as a capability that was measured at zero.
    """
    return {
    "binding_reasoning_kinds": BINDING_REASONING_KINDS,
    "goals": [
        {
            "id": "protease_resistance",
            "label": "Protease resistance (documented P1 motifs)",
            "short_label": "Protease resistance",
            "quick_win": True,
            "description": (
                "Highest-confidence lane. Scored by matching residues against documented "
                "protease P1 specificities, so the evidence is a real motif match rather "
                "than an estimate."
            ),
            "computes": [
                "Whether the wild-type residue matches a documented P1 specificity",
                "Whether the mutant introduces a new P1 site",
            ],
            "does_not_compute": [
                "kcat/KM, or any rate constant",
                "Half-life in plasma or any other medium",
                "Cleavage at sites outside the reference specificity set",
            ],
            "strongest_evidence": "DIRECT_EXPERIMENTAL",
        },
        {
            "id": "binding_affinity",
            "label": "Binding-interface perturbation (not a predicted Kd)",
            "short_label": "Binding perturbation",
            "quick_win": False,
            "description": (
                "Charge state (Henderson-Hasselbalch) and hydrophobicity change are computed, "
                "but the DIRECTION of the effect on binding is not predicted without a "
                "receptor structure. Magnitude reflects perturbation size only."
            ),
            "computes": [
                "Change in effective residue charge at the requested pH",
                "Change in Kyte-Doolittle hydrophobicity",
                "How large a perturbation those amount to",
            ],
            "does_not_compute": [
                "Kd, Ki, IC50, EC50 or any affinity constant",
                "Whether affinity goes up or down — the direction is not determined",
                "Selectivity between a primary receptor and a cross-reactive one",
            ],
            "strongest_evidence": "BIOCHEMICAL_PRINCIPLE",
            "reasoning_kinds": BINDING_REASONING_KINDS,
        },
        {
            "id": "generic_improvement",
            "label": "No goal specified",
            "short_label": "No goal",
            "quick_win": False,
            "description": (
                "No primary benefit is claimed; ranking reflects off-target cost only."
            ),
            "computes": ["Off-target cost only"],
            "does_not_compute": ["Any benefit — none is claimed without a goal"],
            "strongest_evidence": "INFERENCE_ONLY",
        },
    ]
}
