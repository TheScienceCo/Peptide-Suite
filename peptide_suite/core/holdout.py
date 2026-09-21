"""
Holdout protocols and rank-based reporting.  [Addendum 2 section 11, build step 6i]

The named-entity exclusion protocol is insufficient, and the reason is specific
rather than general. Removing every document mentioning semaglutide leaves
liraglutide intact, which carries Arg34 and gamma-Glu-linked acylation at Lys26,
and taspoglutide, which is [Aib8, Aib35]-GLP-1. The union of those two published
drugs is the complete answer. What such an experiment demonstrates is correct
retrieval and recombination of established engineering motifs -- a real and
useful capability, and one that should be claimed as exactly that. It is not
evidence of de novo discovery, and a reviewer who knows the GLP-1 literature
takes the stronger claim apart in under a minute.

Three protocols replace it:

  MOTIF_ABLATION   Exclude by chemistry rather than by name. All lipidated
                   peptides, all Aib-containing peptides, all gamma-Glu
                   constructs, each as a separate run.
  TEMPORAL         Freeze the corpus at a cutoff year and predict modifications
                   first published afterward. The only protocol that yields an
                   honest hit rate.
  DECOY            Run on peptides where the motif is known not to help. A
                   system that proposes it anyway has a prior, not a prediction.

And a reporting contract that is the real deliverable: never report a hit as
binary. Report the rank of the true modification within the full proposal list
and the count of proposals ranked above it.

"Predicted two of three modifications" is not supportable. "The true
modification ranked 2nd of 47 proposals under motif-level ablation, with a 0.11
false-positive rate on decoys" is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_DATA = Path(__file__).parent.parent / "data"


class Protocol(Enum):
    NAMED_ENTITY = "named_entity"        # insufficient; retained so it can be labelled
    MOTIF_ABLATION = "motif_ablation"
    TEMPORAL = "temporal"
    DECOY = "decoy"

    @property
    def is_clean_holdout(self) -> bool:
        """
        Whether a result under this protocol may be called a holdout at all.

        Named-entity exclusion may not. A result obtained under it is reported
        as such and never as a clean holdout, because the chemistry survives in
        other molecules under other names.
        """
        return self is not Protocol.NAMED_ENTITY

    @property
    def description(self) -> str:
        return {
            Protocol.NAMED_ENTITY:
                "Documents naming the target drug were excluded. Insufficient on its own: "
                "the engineering motifs survive in other drugs under other names.",
            Protocol.MOTIF_ABLATION:
                "Excluded by chemistry rather than by name -- every peptide carrying the "
                "motif, whatever it is called.",
            Protocol.TEMPORAL:
                "The corpus was frozen at a cutoff year and the modification was first "
                "published afterward.",
            Protocol.DECOY:
                "Run on a peptide where the motif is known not to help, to measure how "
                "often the system proposes it regardless.",
        }[self]


class ReportingViolation(Exception):
    """Raised when a result is reported in a way the contract forbids."""


@dataclass
class RankedOutcome:
    """
    Where the true modification landed in the proposal list.

    Rank and total are both required. A rank without a total says nothing: 2nd
    of 3 and 2nd of 47 are different results, and the second is the interesting
    one.
    """
    modification: str
    protocol: Protocol
    rank: Optional[int]                  # 1-based; None means it was never proposed
    total_proposals: int
    proposals_ranked_above: int = 0
    ablated_motifs: List[str] = field(default_factory=list)
    cutoff_year: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.rank is not None:
            self.proposals_ranked_above = self.rank - 1
        if self.protocol is Protocol.MOTIF_ABLATION and not self.ablated_motifs:
            raise ReportingViolation(
                "A motif-ablation result must name the motifs that were ablated. Without "
                "them the run cannot be distinguished from a named-entity exclusion, which "
                "is the protocol this one replaces.")
        if self.protocol is Protocol.TEMPORAL and self.cutoff_year is None:
            raise ReportingViolation(
                "A temporal holdout must state its cutoff year. The claim is that the "
                "modification was first published after the cutoff, and without the year "
                "there is no claim.")

    @property
    def was_proposed(self) -> bool:
        return self.rank is not None

    def report(self) -> str:
        """
        The contract's required form. Never binary.

        Deliberately has no boolean 'hit' anywhere in it: a system whose output
        can be reduced to yes-or-no will be, and 2nd of 47 will be written up as
        a hit alongside 2nd of 3.
        """
        if not self.was_proposed:
            base = (f"'{self.modification}' was not proposed at all among "
                    f"{self.total_proposals} proposals")
        else:
            base = (f"'{self.modification}' ranked {_ordinal(self.rank)} of "
                    f"{self.total_proposals} proposals, with "
                    f"{self.proposals_ranked_above} ranked above it")
        qualifier = f" under {self.protocol.value}"
        if self.ablated_motifs:
            qualifier += f" (ablated: {', '.join(sorted(self.ablated_motifs))})"
        if self.cutoff_year:
            qualifier += f" (corpus frozen at {self.cutoff_year})"
        if not self.protocol.is_clean_holdout:
            qualifier += (". This is NOT a clean holdout: named-entity exclusion leaves the "
                          "same chemistry present in other molecules")
        return base + qualifier + "."


@dataclass
class DecoyOutcome:
    """
    How often a motif was proposed where it is known not to help.

    Reported alongside every hit. A hit rate without this is half a result: a
    system that proposes lipidation for everything will "predict" it correctly
    whenever lipidation happens to be the answer.
    """
    motif: str
    trials: int
    times_proposed: int

    @property
    def false_positive_rate(self) -> Optional[float]:
        return self.times_proposed / self.trials if self.trials else None

    def report(self) -> str:
        if not self.trials:
            return (f"No decoy trials were run for '{self.motif}', so no false-positive "
                    f"rate is available and none is assumed.")
        return (f"'{self.motif}' was proposed in {self.times_proposed} of {self.trials} "
                f"decoy trials (false-positive rate "
                f"{self.false_positive_rate:.2f}).")


@dataclass
class HoldoutReport:
    """A complete result: the ranked outcomes and the decoy controls together."""
    outcomes: List[RankedOutcome] = field(default_factory=list)
    decoys: List[DecoyOutcome] = field(default_factory=list)

    def summary(self) -> str:
        if not self.outcomes:
            return "No holdout runs recorded."
        lines = [o.report() for o in self.outcomes]
        if self.decoys:
            lines.extend(d.report() for d in self.decoys)
        else:
            lines.append(
                "No decoy controls were run. A hit rate without them is half a result: a "
                "system that proposes a motif for everything will appear to predict it "
                "whenever it happens to be the answer.")
        return "\n".join(lines)

    def permits_discovery_claim(self) -> bool:
        """
        Whether these results support a claim of de novo discovery.

        Requires a clean protocol and decoy controls. Correct retrieval and
        recombination of established motifs is a real capability and should be
        claimed as that; it is a different claim.
        """
        return (bool(self.outcomes)
                and all(o.protocol.is_clean_holdout for o in self.outcomes)
                and bool(self.decoys))

    def claim_guidance(self) -> str:
        if self.permits_discovery_claim():
            return ("Protocols are clean and decoy controls are present. Report the ranks "
                    "and the false-positive rate; do not reduce them to a hit count.")
        reasons = []
        if any(not o.protocol.is_clean_holdout for o in self.outcomes):
            reasons.append("a result rests on named-entity exclusion")
        if not self.decoys:
            reasons.append("no decoy controls were run")
        return ("These results support a claim of correct retrieval and recombination of "
                "established engineering motifs, which is real and useful. They do not "
                "support a claim of de novo discovery"
                + (f", because {' and '.join(reasons)}" if reasons else "") + ".")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class GoldenSet:
    """The drugs the protocols are run against, and the motifs they carry."""

    _drugs: Optional[Dict[str, Dict]] = None

    @classmethod
    def _raw(cls) -> Dict:
        return json.loads((_DATA / "holdout_golden_set.json").read_text())

    @classmethod
    def drugs(cls) -> Dict[str, Dict]:
        if cls._drugs is None:
            cls._drugs = dict(cls._raw()["drugs"])
        return cls._drugs

    @classmethod
    def motifs_of(cls, drug: str) -> List[str]:
        return list(cls.drugs().get(drug, {}).get("motifs", []))

    @classmethod
    def drugs_carrying(cls, motif: str) -> List[str]:
        return sorted(name for name, entry in cls.drugs().items()
                      if motif in entry.get("motifs", []))

    @classmethod
    def named_entity_leakage(cls, target: str) -> Dict[str, List[str]]:
        """
        What a named-entity exclusion of `target` leaves behind.

        This is the function that makes the argument concrete rather than
        rhetorical: for semaglutide it returns the other drugs still carrying
        each of its motifs, and every motif is still covered.
        """
        leakage = {}
        for motif in cls.motifs_of(target):
            others = [d for d in cls.drugs_carrying(motif) if d != target]
            if others:
                leakage[motif] = others
        return leakage

    @classmethod
    def decoys(cls) -> List[Dict]:
        return list(cls._raw()["decoys"]["entries"])
