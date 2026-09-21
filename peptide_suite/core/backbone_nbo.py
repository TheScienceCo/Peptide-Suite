"""
n->pi* backbone analysis as a reporting requirement.
[Addendum 2 section 4, post-6i]

NBO second-order perturbation analysis is REQUIRED for any proposal involving
Aib, other alpha,alpha-disubstituted residues, N-methylation, or proline-rich
segments. Not recommended -- required, and the reason is specific:

  It is the mechanistic account of WHY such substitutions stabilise helices.
  It is not captured by classical force fields.

Those two facts together are what make it a requirement rather than a nicety. A
force field asked about Aib reports the steric consequence of two methyls on
C-alpha and says nothing about the n->pi* donation from one carbonyl lone pair
into the next amide's antibonding orbital, which is the interaction actually
doing the stabilising. A proposal that says "Aib rigidifies the backbone" and
stops there has given the observation in place of the mechanism, and the
observation was already known.

No NBO engine exists in this deployment, so the requirement is enforced the only
honest way available: proposals that trigger it are marked as owing a
mechanistic account they have not got. The alternative -- staying silent -- would
let the steric hand-wave stand as the explanation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class NBOTrigger(Enum):
    """What makes a proposal owe an n->pi* account."""
    ALPHA_ALPHA_DISUBSTITUTED = "alpha_alpha_disubstituted"
    N_METHYLATION = "n_methylation"
    PROLINE_RICH = "proline_rich"

    @property
    def why(self) -> str:
        return {
            NBOTrigger.ALPHA_ALPHA_DISUBSTITUTED:
                "Alpha,alpha-disubstitution restricts phi/psi, and the helical preference "
                "that follows is usually attributed to sterics alone. The n->pi* donation "
                "between consecutive amides is the part a force field cannot see.",
            NBOTrigger.N_METHYLATION:
                "N-methylation removes a backbone hydrogen-bond donor and shifts the "
                "cis/trans amide equilibrium. Both effects run through the amide "
                "electronic structure, which is what the analysis describes.",
            NBOTrigger.PROLINE_RICH:
                "Proline-rich segments hold their conformation through n->pi* donation "
                "along the backbone rather than through side-chain packing, so a "
                "sterics-only account of a proline substitution describes the wrong "
                "interaction.",
        }[self]


_PATTERNS = {
    NBOTrigger.ALPHA_ALPHA_DISUBSTITUTED: re.compile(
        r"\bAib\b|alpha-aminoisobutyric|alpha,alpha-disubstituted|"
        r"\balpha-methyl|\bdiethylglycine\b|\bDeg\b", re.I),
    NBOTrigger.N_METHYLATION: re.compile(r"N-methylat|\bN-Me\b|N-methyl", re.I),
}


def proline_rich_fraction() -> float:
    from ..runtime import threshold
    return threshold("nbo.proline_rich_fraction")


@dataclass
class NBORequirement:
    """
    A proposal's outstanding n->pi* obligation.

    `satisfied` is never true in this build. Kept as a field rather than assumed
    so that wiring an engine in later changes one value rather than the shape of
    every consumer.
    """
    triggers: List[NBOTrigger] = field(default_factory=list)
    satisfied: bool = False
    analysis: Optional[dict] = None
    required_tooling: List[str] = field(default_factory=list)

    @property
    def is_required(self) -> bool:
        return bool(self.triggers)

    @property
    def is_outstanding(self) -> bool:
        return self.is_required and not self.satisfied

    def statement(self) -> str:
        if not self.is_required:
            return ""
        if self.satisfied:
            return "n->pi* analysis reported below."
        reasons = " ".join(t.why for t in self.triggers)
        return (
            f"This proposal requires an n->pi* backbone analysis and does not have one. "
            f"{reasons} No NBO engine is available in this deployment, so the mechanistic "
            f"account is outstanding: what follows explains the steric consequence, which "
            f"is the observation rather than the mechanism."
        )


def requirement_for(description: str, sequence: str = "") -> NBORequirement:
    """
    Whether a proposal owes an n->pi* account, and why.

    Proline richness is read from the sequence rather than from the proposal
    text, because a proposal that modifies a proline-rich segment need not
    mention proline at all.
    """
    text = f"{description or ''}"
    triggers = [trigger for trigger, pattern in _PATTERNS.items() if pattern.search(text)]

    sequence = (sequence or "").upper()
    if sequence:
        fraction = sequence.count("P") / len(sequence)
        if fraction >= proline_rich_fraction():
            triggers.append(NBOTrigger.PROLINE_RICH)

    return NBORequirement(
        triggers=triggers,
        satisfied=False,
        required_tooling=(["NBO second-order perturbation analysis (NBO 7, or an "
                           "equivalent implementation), on an optimised geometry"]
                          if triggers else []),
    )


class NBOAnalysisNotAvailable(NotImplementedError):
    """Raised when an n->pi* analysis is requested and cannot be produced."""


def analyse(description: str, sequence: str = ""):
    """
    Run the analysis. Not implemented; raises rather than returning a number.

    A stabilisation energy invented here would be indistinguishable from a
    computed one, and it would be attached to exactly the proposals whose whole
    argument rests on it.
    """
    requirement = requirement_for(description, sequence)
    raise NBOAnalysisNotAvailable(
        f"No n->pi* analysis was produced. {requirement.statement() or 'Not required here.'} "
        f"Needs: {', '.join(requirement.required_tooling) or 'nothing'}."
    )
