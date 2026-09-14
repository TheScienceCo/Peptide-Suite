"""
Core module: shared infrastructure for peptide analysis.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class EvidenceTier(Enum):
    """Hierarchical evidence confidence levels (Bayesian-flavored)."""
    DIRECT_EXPERIMENTAL = 1.0
    HOMOLOG_EXPERIMENTAL = 0.8
    BIOCHEMICAL_PRINCIPLE = 0.6
    INFERENCE_ONLY = 0.3


class ConfidenceLevel(Enum):
    """Coarse confidence labels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Effect:
    """
    Represents a predicted effect of a substitution.

    Attributes:
        category: "primary" or "off_target"
        description: Plain-language effect (e.g., "adds negative charge")
        evidence_tier: EvidenceTier enum
        confidence: ConfidenceLevel enum
        score: 0-1 numeric confidence score
        reasoning: Detailed justification including sources
        equation_refs: Which equations (if any) informed this
    """
    category: str  # "primary" or "off_target"
    description: str
    evidence_tier: EvidenceTier
    confidence: ConfidenceLevel
    score: float  # 0-1
    reasoning: str
    equation_refs: List[int] = None  # e.g., [43, 11, 1] for Shannon, HH, Coulomb
    sources: List[str] = None  # Literature citations, database lookups


@dataclass
class SubstitutionRecommendation:
    """
    Top-level recommendation for a single substitution.

    Attributes:
        position: 0-indexed position in peptide
        wild_type_aa: Original amino acid (1-letter code)
        mutant_aa: Proposed amino acid (1-letter code or name)
        is_noncancical: True if mutant is a non-classical AA (filled in later)
        target_category: e.g., "protease_resistance", "binding_affinity", "activation"
        primary_effect: Effect object
        off_target_effects: List of Effect objects
        net_recommendation: "recommend" / "recommend_with_caveats" / "not_recommended"
        overall_confidence: Weighted average confidence across all effects
        net_score: Primary benefit score minus off-target penalty
        ranking_rationale: Why this ranked where it did
    """
    position: int
    wild_type_aa: str
    mutant_aa: str
    is_noncanonical: bool = False
    target_category: str = ""
    primary_effect: Optional[Effect] = None
    off_target_effects: List[Effect] = None
    net_recommendation: str = "not_recommended"
    overall_confidence: float = 0.0
    net_score: float = 0.0
    ranking_rationale: str = ""

    def __post_init__(self):
        if self.off_target_effects is None:
            self.off_target_effects = []


@dataclass
class PeptideContext:
    """Metadata about the peptide being analyzed."""
    sequence: str
    name: str = ""
    inferred_function: str = ""
    confirmed_goal: str = ""
    target_organism: str = "human"
    known_homologs: List[str] = None
    conservation_entropy: Dict[int, float] = None  # position -> Shannon entropy
