"""
Core module: shared infrastructure for peptide analysis.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class EvidenceTier(Enum):
    """
    Hierarchical evidence confidence levels.

    The member values are names, not numbers. The tier a claim sits in is a
    fact about where the claim came from and belongs in the engine; how much
    that tier is worth is a fitted quantity and belongs in the policy. Carrying
    the weight as the enum value conflated the two, and made the weight
    impossible to change without editing the engine.
    """
    DIRECT_EXPERIMENTAL = "DIRECT_EXPERIMENTAL"
    HOMOLOG_EXPERIMENTAL = "HOMOLOG_EXPERIMENTAL"
    BIOCHEMICAL_PRINCIPLE = "BIOCHEMICAL_PRINCIPLE"
    INFERENCE_ONLY = "INFERENCE_ONLY"


# Feature-family key for each tier's weight, declared by peptide_suite.policy.
_TIER_POLICY_KEY = {
    EvidenceTier.DIRECT_EXPERIMENTAL:   "evidence.tier_direct_experimental",
    EvidenceTier.HOMOLOG_EXPERIMENTAL:  "evidence.tier_homolog_experimental",
    EvidenceTier.BIOCHEMICAL_PRINCIPLE: "evidence.tier_biochemical_principle",
    EvidenceTier.INFERENCE_ONLY:        "evidence.tier_inference_only",
}


def evidence_weight(tier: "EvidenceTier") -> float:
    """What the active policy says a claim at this tier is worth."""
    from ..runtime import weight
    return weight(_TIER_POLICY_KEY[tier])


class ConfidenceLevel(Enum):
    """Coarse confidence labels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Effect:
    """
    Represents a predicted effect of a substitution.

    `score` and `magnitude` are deliberately separate axes and must not be
    collapsed into one another: `score` answers "how sure are we this effect is
    real", `magnitude` answers "how much does it matter if it is". A trivial
    effect we are certain about must not outrank a large effect we are less
    certain about.

    Attributes:
        category: "primary" or "off_target"
        description: Plain-language effect (e.g., "adds negative charge")
        evidence_tier: EvidenceTier enum
        confidence: ConfidenceLevel enum
        score: 0-1 confidence that this effect is real
        magnitude: 0-1 size of the effect if real (benefit for primary, cost for off-target)
        reasoning: Detailed justification including sources
        equation_refs: Which equations (if any) informed this
        term_key: Stable identifier for which term this effect reports, so a
            consumer can find one term among several without matching on prose.
            Descriptions are written for a reader and change freely; a key does
            not.
        computed: False when the term exists but had no input data to compute
            from. Such an effect still appears, carrying magnitude 0 and an
            explanation, but a consumer must render it as absent rather than as
            zero: "no conservation data" and "conservation cost is nil" are
            different statements and must not share a cell.
    """
    category: str  # "primary" or "off_target"
    description: str
    evidence_tier: EvidenceTier
    confidence: ConfidenceLevel
    score: float  # 0-1 confidence
    reasoning: str
    magnitude: float = 0.5  # 0-1 effect size
    equation_refs: List[int] = None  # e.g., [43, 11, 1] for Shannon, HH, Coulomb
    sources: List[str] = None  # Literature citations, database lookups
    term_key: Optional[str] = None
    computed: bool = True


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
    score_breakdown: Dict = None
    # What the variant-evidence store knows about this exact change, if
    # anything. Absent rather than empty when no lookup was made, so that "not
    # searched" and "searched and found nothing" stay distinguishable -- they
    # render identically as an empty list and mean opposite things.
    experimental_precedent: Optional[Dict] = None

    def __post_init__(self):
        if self.off_target_effects is None:
            self.off_target_effects = []

    @property
    def has_experimental_precedent(self) -> bool:
        return bool((self.experimental_precedent or {}).get("has_precedent"))


@dataclass
class CellTypeHit:
    """A cell type or tissue implicated in a queried biological function."""
    name: str
    ontology_id: str = ""          # GO / CL identifier where known
    relationship: str = ""         # How it relates to the function
    source: str = ""               # Which resource answered
    confidence: float = 0.0


@dataclass
class PeptideCandidate:
    """A peptide/protein proposed as relevant to a queried function."""
    name: str
    gene: str = ""
    uniprot: str = ""
    rationale: str = ""
    expression_note: str = ""      # What expression data says, or that none was retrieved
    literature_note: str = ""      # What the literature search returned
    expression_score: Optional[float] = None   # None when not retrieved — never invented
    literature_score: Optional[float] = None
    combined_score: float = 0.0
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    evidence_tier: EvidenceTier = EvidenceTier.INFERENCE_ONLY
    citations: List[str] = None
    requires_verification: bool = True

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


@dataclass
class FindPeptidesResult:
    """Output of Workflow 2."""
    query: str
    cell_types: List[CellTypeHit] = None
    known_answer_found: bool = False
    known_answers: List[PeptideCandidate] = None
    candidates: List[PeptideCandidate] = None
    data_notes: List[str] = None
    methodology_note: str = ""

    def __post_init__(self):
        for field_name in ("cell_types", "known_answers", "candidates", "data_notes"):
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])


# Shannon entropy over too few sequences carries no conservation information (a
# single sequence is 0 at every position by construction), so it is reported as
# unavailable rather than as a computed value. How few is a policy question.
def min_homologs_for_conservation() -> int:
    from ..runtime import int_threshold
    return int_threshold("conservation.min_distinct_sequences")


@dataclass
class PeptideContext:
    """Metadata about the peptide being analyzed."""
    sequence: str
    name: str = ""
    inferred_function: str = ""
    confirmed_goal: str = ""
    target_organism: str = "human"
    known_homologs: List[str] = None
    # Where the homologs came from. Conservation entropy is only as good as
    # this field: the same number means something different if the alignment
    # came from a database than if it came from a file in this repository.
    homolog_source: str = ""
    homolog_source_detail: str = ""
    conservation_entropy: Dict[int, float] = None  # position -> Shannon entropy
    homolog_count: int = 0
    conservation_available: bool = False
    data_notes: List[str] = None  # Plain-language limitations to surface to the user

    def __post_init__(self):
        if self.data_notes is None:
            self.data_notes = []
