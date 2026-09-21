"""
Partner-peptide module and the RAMP schema requirement.
[Addendum 2 section 8, build step 6f]

Triggered when the contact classifier returns ESSENTIAL / PARTNER_DEPENDENT, or
when retrieval surfaces a synergy, potentiation or co-secretion claim.

The output is a co-agent proposal with a REQUIRED mode field. The mode is not a
label on an otherwise identical proposal: the four modes need different
evidence, ship as different products, and fail in different ways. Emitting one
without a mode would let "these two peptides work well together" stand in for
four distinct claims.

The RAMP requirement is the part that silently corrupts a training set if
skipped. CLR with RAMP1 is the CGRP receptor; CLR with RAMP2 is AM1. Same
receptor gene, different pharmacology. So "the receptor" is an underspecified
entity, and affinities measured against different accessory complexes are not
the same quantity and must not be pooled.

That is enforced here at the schema level rather than described, and the
enforcement turns on a distinction the original ReceptorComplex did not make:
an accessory list that is empty because the receptor has no accessory subunit
is a different statement from one that is empty because nobody recorded it.
Treating them alike is how measurements from three different pharmacologies end
up averaged into one number with nothing to indicate it.

Terminology, enforced throughout: a PARTNER PEPTIDE forms a BINARY COMPLEX.
Never "dipeptide" -- that word means a two-residue peptide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_DATA = Path(__file__).parent.parent / "data"


class PartnerMode(Enum):
    """
    How a second peptide participates. Required on every co-agent proposal.

    Different evidence, different products, different failure modes. An obligate
    heterodimer shipped as a combination is two inactive peptides; a synergistic
    pair shipped as a fixed covalent construct throws away the ratio that made
    it work.
    """
    OBLIGATE_HETERODIMER = "OBLIGATE_HETERODIMER"
    SYNERGISTIC = "SYNERGISTIC"
    ACCESSORY_DEPENDENT = "ACCESSORY_DEPENDENT"
    CHIMERIC_CANDIDATE = "CHIMERIC_CANDIDATE"

    @property
    def description(self) -> str:
        return {
            PartnerMode.OBLIGATE_HETERODIMER:
                "Neither component is active alone.",
            PartnerMode.SYNERGISTIC:
                "Each component is active alone; the combination is superadditive.",
            PartnerMode.ACCESSORY_DEPENDENT:
                "Activity is determined by a non-peptide partner at the receptor.",
            PartnerMode.CHIMERIC_CANDIDATE:
                "The two activities are plausibly fusable into a single sequence.",
        }[self]

    @property
    def shipping_requirement(self) -> str:
        """What a proposal in this mode must specify to be actionable."""
        return {
            PartnerMode.OBLIGATE_HETERODIMER:
                "Ship as two peptides at a fixed stoichiometry, or as a covalent construct "
                "with a linker specification. A stoichiometry is required either way.",
            PartnerMode.SYNERGISTIC:
                "Ship as a combination with a ratio and a measured synergy index. An "
                "unmeasured synergy claim is an assertion that the combination is better, "
                "which is the thing being claimed.",
            PartnerMode.ACCESSORY_DEPENDENT:
                "Name the accessory protein. Without it the receptor is underspecified and "
                "the proposal targets an entity that does not uniquely exist.",
            PartnerMode.CHIMERIC_CANDIDATE:
                "State the fusion topology and what evidence supports fusability. Two "
                "activities being desirable together does not make them fusable.",
        }[self]


class SchemaError(Exception):
    """Raised when a receptor record is too incomplete to pool measurements under."""


# The distinction that makes the RAMP rule enforceable. An accessory list that
# is empty because the receptor has none is a claim; one that is empty because
# nobody looked is an absence. A shared sentinel keeps them from collapsing.
class AccessoryStatus(Enum):
    DECLARED_NONE = "declared_none"      # checked: this receptor has no accessory subunit
    DECLARED = "declared"                # accessory subunits named
    UNSPECIFIED = "unspecified"          # nobody recorded it; blocks pooling


@dataclass(frozen=True)
class ReceptorRecord:
    """
    A receptor identity complete enough to pool measurements under.

    `accessory` must be given explicitly, including as an empty tuple meaning
    "checked, none". Defaulting it to empty is what lets an unrecorded record
    pool with a genuinely accessory-free one, which is the silent corruption the
    section warns about.
    """
    receptor: str
    accessory: Optional[Tuple[str, ...]]     # None means UNSPECIFIED, not "none"
    species: str = ""
    identity: str = ""                       # e.g. "CGRP receptor"

    @property
    def status(self) -> AccessoryStatus:
        if self.accessory is None:
            return AccessoryStatus.UNSPECIFIED
        return AccessoryStatus.DECLARED_NONE if not self.accessory else AccessoryStatus.DECLARED

    @property
    def is_complete(self) -> bool:
        return self.status is not AccessoryStatus.UNSPECIFIED

    @property
    def identifier(self) -> str:
        parts = [self.receptor]
        if self.accessory:
            parts.extend(sorted(self.accessory))
        elif self.accessory is None:
            parts.append("accessory:UNSPECIFIED")
        label = "+".join(parts)
        return f"{label} ({self.species})" if self.species else label

    def poolable_with(self, other: "ReceptorRecord") -> bool:
        """
        Whether measurements against these two records may be pooled.

        An unspecified accessory field never pools -- not even with another
        unspecified one. Two records that both failed to say are not thereby
        known to agree, and pooling them is how three pharmacologies become one
        number with nothing to indicate it.
        """
        if not (self.is_complete and other.is_complete):
            return False
        return (self.receptor == other.receptor
                and self.species == other.species
                and set(self.accessory or ()) == set(other.accessory or ()))

    def require_complete(self) -> None:
        if not self.is_complete:
            raise SchemaError(
                f"Receptor record '{self.receptor}' has no accessory-protein field. "
                f"Same receptor gene with different accessory subunits is different "
                f"pharmacology -- CLR+RAMP1 is the CGRP receptor and CLR+RAMP2 is AM1 -- "
                f"so a record without the field is incomplete and cannot be used to pool "
                f"measurements. Declare it explicitly, including as an empty list meaning "
                f"'checked, none'."
            )


def pool_measurements(records: Sequence[Tuple[ReceptorRecord, float]]) -> Dict[str, List[float]]:
    """
    Group measurements by receptor complex, refusing incomplete records.

    Raises rather than dropping the bad ones. A silently smaller pool looks like
    a normal result, and the parameter budget downstream would then be computed
    against a measurement count that was never true.
    """
    pools: Dict[str, List[float]] = {}
    for record, value in records:
        record.require_complete()
        pools.setdefault(record.identifier, []).append(value)
    return pools


@dataclass
class PartnerProposal:
    """
    A co-agent proposal. The mode is required and carries the shipping contract.

    No score. Like a research request, a partner proposal ranked alongside
    single-peptide proposals would read as a comparable alternative, and it is
    not -- it changes what the product is.
    """
    primary: str
    partner: str
    mode: PartnerMode
    rationale: str
    stoichiometry: str = ""
    engagement_order: str = ""
    synergy_index: str = ""
    accessory_protein: str = ""
    citation: str = ""
    citation_precision: str = ""
    independently_verified: bool = False
    confidence: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def unmet_requirements(self) -> List[str]:
        """
        What this proposal still has to specify for its mode.

        Checked per mode rather than uniformly: a stoichiometry is meaningless
        for a chimeric candidate and mandatory for an obligate heterodimer.
        """
        missing = []
        if self.mode is PartnerMode.OBLIGATE_HETERODIMER and not self.stoichiometry:
            missing.append(
                "Stoichiometry not stated. An obligate pair shipped without one is not a "
                "specification, and shipping it as a plain combination gives two inactive "
                "peptides.")
        if self.mode is PartnerMode.SYNERGISTIC:
            if not self.stoichiometry:
                missing.append("Ratio not stated; synergy is ratio-dependent.")
            if not self.synergy_index:
                missing.append(
                    "No measured synergy index. Superadditivity is the claim being made, so "
                    "asserting it without a measurement asserts the conclusion.")
        if self.mode is PartnerMode.ACCESSORY_DEPENDENT and not self.accessory_protein:
            missing.append(
                "Accessory protein not named. Without it the receptor is underspecified "
                "and the proposal targets an entity that does not uniquely exist.")
        if self.mode is PartnerMode.CHIMERIC_CANDIDATE and "fus" not in self.rationale.lower():
            missing.append(
                "No statement of fusion topology or evidence for fusability. Two "
                "activities being desirable together does not make them fusable.")
        return missing

    @property
    def is_actionable(self) -> bool:
        return not self.unmet_requirements()

    def summary(self) -> str:
        return (f"{self.primary} + {self.partner} — {self.mode.value}: "
                f"{self.mode.description}")


# Retrieval language that triggers the module, per the section's trigger clause.
_TRIGGER_TERMS = ("synergy", "synergistic", "superadditive", "potentiat",
                  "co-secret", "cosecret", "two-component", "heterodimer",
                  "combination therapy", "dual agonist", "co-administ")


def triggers_partner_analysis(text: str) -> bool:
    """Whether a retrieved claim should open the partner module."""
    lowered = (text or "").lower()
    return any(term in lowered for term in _TRIGGER_TERMS)


class PartnerCases:
    """The seed set from Addendum 2 section 8."""

    _cases: Optional[Dict[str, PartnerProposal]] = None
    _ramps: Optional[Dict[str, ReceptorRecord]] = None

    @classmethod
    def _raw(cls) -> Dict:
        return json.loads((_DATA / "partner_peptides.json").read_text())

    @classmethod
    def all(cls) -> Dict[str, PartnerProposal]:
        if cls._cases is None:
            raw = cls._raw()
            cls._cases = {}
            for key, entry in raw["cases"].items():
                components = entry["components"]
                cls._cases[key] = PartnerProposal(
                    primary=components[0],
                    partner=" + ".join(components[1:]) if len(components) > 1 else components[0],
                    mode=PartnerMode[entry["mode"]],
                    rationale=entry["claim"],
                    stoichiometry=entry.get("stoichiometry", ""),
                    engagement_order=entry.get("engagement_order", ""),
                    synergy_index=entry.get("synergy_magnitude", ""),
                    citation=entry.get("citation", ""),
                    citation_precision=entry.get("citation_precision", ""),
                    independently_verified=entry.get("independently_verified", False),
                    confidence=0.5 if not entry.get("independently_verified") else 0.9,
                    notes=[v for k, v in entry.items()
                           if k.startswith("why_") and isinstance(v, str)]
                    + ([f"Magnitude from: {entry['magnitude_source']}"]
                       if entry.get("magnitude_source") else [])
                    + ([f"Related: {', '.join(entry['related'])}"]
                       if entry.get("related") else []),
                )
        return cls._cases

    @classmethod
    def ramp_complexes(cls) -> Dict[str, ReceptorRecord]:
        if cls._ramps is None:
            raw = cls._raw()["ramp_complexes"]
            cls._ramps = {
                key: ReceptorRecord(
                    receptor=entry["receptor"],
                    accessory=tuple(entry["accessory"]),
                    identity=entry["identity"],
                )
                for key, entry in raw.items() if not key.startswith("_")
            }
        return cls._ramps

    @classmethod
    def for_peptide(cls, name: str) -> List[PartnerProposal]:
        """
        Partner cases that apply to this peptide.

        Matched against each case's explicit `applies_to` list, for the same
        reason as the native-contact set: matching on the free-text description
        attributes a case to any name that happens to contain a component's.
        """
        if not name:
            return []
        from .contact_classifier import matches_any_alias
        return [cls.all()[key] for key, entry in cls._raw()["cases"].items()
                if matches_any_alias(name, entry.get("applies_to", []))]
