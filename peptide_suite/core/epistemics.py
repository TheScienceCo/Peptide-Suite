"""
Epistemic contract.

Every claim this system emits declares how it was arrived at. The three classes
are not interchangeable and are never collapsed:

    RETRIEVED  someone measured or published this; a citation is required
    COMPUTED   this system calculated it; the method and tier are required
    INFERRED   this system reasoned to it; the inference step is required

A computed value is never presented as an experimental one. Absent binding data,
performance claims are explicitly labelled untested.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ClaimType(Enum):
    RETRIEVED = "RETRIEVED"
    COMPUTED = "COMPUTED"
    INFERRED = "INFERRED"


class AssumptionKind(Enum):
    """
    Assumption categories that silently change conclusions when wrong.
    These are surfaced as structured fields rather than buried in prose.
    """
    PROTONATION_STATE = "protonation_state"
    CONFORMER = "assumed_conformer"
    RECEPTOR_ACCESSORY_SUBUNITS = "receptor_accessory_subunits"
    FORMULATION_CONDITIONS = "formulation_conditions"
    NATIVE_CONTEXT = "native_context"
    OTHER = "other"


@dataclass
class Assumption:
    """A stated premise that, if wrong, invalidates the claim resting on it."""
    kind: AssumptionKind
    statement: str
    impact_if_wrong: str = ""


@dataclass
class Claim:
    """
    A single assertion with its provenance.

    Construct via the classmethods rather than directly: they enforce that a
    RETRIEVED claim carries a citation and a COMPUTED claim names its method
    and tier.
    """
    text: str
    claim_type: ClaimType
    method: str = ""            # COMPUTED: what calculated it, at which tier
    tier: Optional[int] = None  # COMPUTED: physics tier
    citations: List[str] = field(default_factory=list)   # RETRIEVED
    inference_step: str = ""    # INFERRED: the reasoning applied
    experimentally_tested: bool = False

    @classmethod
    def retrieved(cls, text: str, citations: List[str], tested: bool = True) -> "Claim":
        if not citations:
            raise ValueError(
                "A RETRIEVED claim requires at least one citation. If no source can "
                "be named, the claim is INFERRED, not RETRIEVED."
            )
        return cls(text=text, claim_type=ClaimType.RETRIEVED,
                   citations=citations, experimentally_tested=tested)

    @classmethod
    def computed(cls, text: str, method: str, tier: int) -> "Claim":
        if not method:
            raise ValueError(
                "A COMPUTED claim requires a named method. An unnamed calculation "
                "cannot be audited and must not be reported."
            )
        return cls(text=text, claim_type=ClaimType.COMPUTED, method=method,
                   tier=tier, experimentally_tested=False)

    @classmethod
    def inferred(cls, text: str, inference_step: str) -> "Claim":
        if not inference_step:
            raise ValueError(
                "An INFERRED claim requires the inference to be stated. "
                "Unstated reasoning is indistinguishable from assertion."
            )
        return cls(text=text, claim_type=ClaimType.INFERRED,
                   inference_step=inference_step, experimentally_tested=False)

    def render(self) -> str:
        """One-line rendering that always carries the provenance tag."""
        if self.claim_type is ClaimType.RETRIEVED:
            return f"[RETRIEVED] {self.text} ({'; '.join(self.citations)})"
        if self.claim_type is ClaimType.COMPUTED:
            return f"[COMPUTED · {self.method} · Tier {self.tier}] {self.text} (untested)"
        return f"[INFERRED] {self.text} — inference: {self.inference_step}"


# Scaffold classes whose marketed analogs are heavily represented in training
# corpora. A recommendation that reproduces a known clinical modification on one
# of these is flagged: it may reflect memorisation of the answer rather than
# derivation of it, and that distinction matters when judging the method.
KNOWN_ANALOG_SCAFFOLDS: Dict[str, Dict] = {
    "glp1": {
        "match_motifs": ["HAEGTFTSDV", "HGEGTFTSDL"],
        "marketed_analogs": ["semaglutide", "liraglutide", "dulaglutide", "exenatide"],
        "known_modifications": [
            {
                "description": "Aib at position 2 (DPP-4 evasion)",
                "match_any": [["aib"], ["aminoisobutyric"]],
            },
            {
                "description": "C18 diacid acylation at Lys26 via a gamma-Glu-2xOEG linker (albumin binding)",
                "match_any": [["acylation"], ["lipidation"], ["diacid"], ["c18"]],
            },
            {
                "description": "Lys34->Arg (directs acylation to a single site)",
                "match_any": [["lys34", "arg"], ["k34r"]],
            },
        ],
    },
    "insulin": {
        "match_motifs": ["GIVEQCC", "FVNQHLCGSHLVEAL"],
        "marketed_analogs": ["insulin lispro", "insulin aspart", "insulin glargine", "insulin degludec"],
        "known_modifications": [
            {
                "description": "B28-B29 inversion (Lispro; monomer-favouring)",
                "match_any": [["b28", "b29"], ["lispro"]],
            },
            {
                "description": "B28 Pro->Asp (Aspart; monomer-favouring)",
                "match_any": [["b28", "asp"], ["aspart"]],
            },
            {
                "description": "C16 acylation at B29 (Detemir/Degludec; albumin binding)",
                "match_any": [["acylation"], ["lipidation"], ["c16"]],
            },
        ],
    },
    "gip": {
        "match_motifs": ["YAEGTFISDY"],
        "marketed_analogs": ["tirzepatide"],
        "known_modifications": [
            {
                "description": "Aib at positions 2 and 13",
                "match_any": [["aib"], ["aminoisobutyric"]],
            },
            {
                "description": "C20 diacid acylation at Lys20",
                "match_any": [["acylation"], ["lipidation"], ["diacid"]],
            },
        ],
    },
}


@dataclass
class LeakageFlag:
    """Raised when a recommendation coincides with a known marketed modification."""
    scaffold: str
    marketed_analogs: List[str]
    matching_modification: str
    note: str


def check_corpus_leakage(sequence: str, move_description: str) -> Optional[LeakageFlag]:
    """
    Flag a recommendation that matches a documented modification of a marketed
    analog on the same scaffold.

    This is not an accusation that the recommendation is wrong — these
    modifications are in the clinic precisely because they work. It is a warning
    that the system may be reproducing a well-known answer rather than deriving
    it, which changes how much the agreement counts as validation.
    """
    seq = sequence.upper()
    desc = move_description.lower()

    for scaffold, spec in KNOWN_ANALOG_SCAFFOLDS.items():
        if not any(motif in seq for motif in spec["match_motifs"]):
            continue

        for known_mod in spec["known_modifications"]:
            # Each modification declares the token sets that identify it. Keeping
            # these explicit rather than deriving them from the prose description
            # keeps the matcher from hinging on filler words.
            matched = any(
                all(token in desc for token in token_set)
                for token_set in known_mod["match_any"]
            )
            if matched:
                return LeakageFlag(
                    scaffold=scaffold,
                    marketed_analogs=spec["marketed_analogs"],
                    matching_modification=known_mod["description"],
                    note=(
                        f"This move closely matches a documented modification in marketed "
                        f"{scaffold.upper()} analogs ({', '.join(spec['marketed_analogs'])}): "
                        f"{known_mod['description']}. Agreement with clinical practice is encouraging but is "
                        f"weak evidence that the method derived it, since this modification is "
                        f"well represented in the literature. Treat as corroboration, not validation."
                    ),
                )

    return None
