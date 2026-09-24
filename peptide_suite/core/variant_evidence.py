"""
What was actually measured about a modified peptide.  [Phase 3]

The system can say a substitution is a large physicochemical perturbation. It
cannot say whether anyone has ever made that substitution and measured what
happened. Those are different claims and the second one is better, so it goes
first wherever it exists.

THE SCHEMA SEPARATES TWO THINGS THAT LOOK LIKE ONE. A `Modification` is what
someone changed. A `MeasuredOutcome` is what they measured. A record holding
the first without the second is a design, not evidence, and it is stored as a
design: `has_evidence` is False, `outcomes` is empty, and nothing downstream
may use it to support a claim about effect. The distinction exists because a
store of modification names reads like a store of results -- "semaglutide:
Aib8, lipidation" sits in a table next to a column headed Evidence and acquires
the column's meaning.

NO CITATION, NO DIRECT_EXPERIMENTAL. Enforced through `Provenance.max_tier`,
the same gate the biological-context layer uses, rather than re-implemented
here. An outcome typed in from memory caps at BIOCHEMICAL_PRINCIPLE however
well known it is, because nothing in this deployment can check it.

A MEASUREMENT NEEDS A COMPARATOR. "Three-fold more potent" is not a fact until
it says three-fold more potent *than what*, in *what assay*. Both are required
fields and a numeric effect without them is refused at construction, because
the number survives being copied into a summary and the missing baseline does
not.

COMBINATION CHANGES ARE CONFOUNDED, AND SAYING SO IS THE FEATURE. The
long-acting GLP-1 analogues are the standing example: a marketed analogue
differs from its parent at several places at once -- a backbone substitution, a
sequence substitution, an attached fatty-acid chain -- and its measured
half-life is a property of that whole construct. Attributing it to the
substitution is the single most available error in this domain, because the
substitution is the part that looks like the other rows in the table. A record
with more than one modification is marked unattributable and stays that way;
the confounding note names every change so a reader sees what else moved.

THE STORE IS EMPTY, AND THAT IS THE HONEST STATE. Every outcome worth storing
needs a citation, and no host this deployment can reach serves one: PubMed,
UniProt, EBI and RCSB are all unreachable from here. Populating it from memory
would produce exactly the artefact this module exists to prevent -- a
citation-shaped store whose citations nobody can follow. So the schema, the
retrieval, the ranking and the confounding logic are real and tested, and they
currently return "no precedent found", which is true. The same shape as the
NCAA registry, which reports itself empty rather than promoting a residue on
three of six stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import EvidenceTier
from .biological_context import Provenance, SourceKind

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "variant_evidence.json"


class ModificationKind(Enum):
    """What sort of change was made. Not all of them are substitutions."""

    SUBSTITUTION = "SUBSTITUTION"
    NCAA_SUBSTITUTION = "NCAA_SUBSTITUTION"
    LIPIDATION = "LIPIDATION"
    PEGYLATION = "PEGYLATION"
    CYCLIZATION = "CYCLIZATION"
    TERMINAL_MODIFICATION = "TERMINAL_MODIFICATION"
    TRUNCATION = "TRUNCATION"
    EXTENSION = "EXTENSION"
    DISULFIDE_CHANGE = "DISULFIDE_CHANGE"
    BACKBONE_MODIFICATION = "BACKBONE_MODIFICATION"
    OTHER = "OTHER"

    @property
    def is_point_change(self) -> bool:
        """
        Whether this change is confined to one position.

        A lipidation attached at a named lysine is still not a point change:
        the chain is a large addition whose effects are not local to that
        residue, and treating it as one would let it be compared against a
        substitution table.
        """
        return self in (ModificationKind.SUBSTITUTION,
                        ModificationKind.NCAA_SUBSTITUTION)


@dataclass(frozen=True)
class Modification:
    """One change, relative to a named parent molecule."""

    kind: ModificationKind
    description: str
    position: Optional[int] = None      # 1-indexed in the parent's numbering
    wild_type: str = ""
    mutant: str = ""

    def __post_init__(self):
        if not self.description.strip():
            raise ValueError("A modification with no description cannot be read back.")

    @property
    def label(self) -> str:
        if self.kind.is_point_change and self.position and self.wild_type and self.mutant:
            return f"{self.wild_type}{self.position}{self.mutant}"
        return self.description

    def matches_substitution(self, position: int, wild_type: str, mutant: str) -> bool:
        """Whether this is the same point change, in the parent's numbering."""
        return (self.kind.is_point_change
                and self.position == position
                and self.wild_type.upper() == wild_type.upper()
                and self.mutant.upper() == mutant.upper())

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "description": self.description,
                "position": self.position, "wild_type": self.wild_type,
                "mutant": self.mutant, "label": self.label}


class OutcomeMeasure(Enum):
    """What was measured. The unit is part of the measure, not a free field."""

    AFFINITY_KD = "AFFINITY_KD"
    AFFINITY_KI = "AFFINITY_KI"
    POTENCY_EC50 = "POTENCY_EC50"
    POTENCY_IC50 = "POTENCY_IC50"
    PLASMA_HALF_LIFE = "PLASMA_HALF_LIFE"
    PROTEASE_STABILITY = "PROTEASE_STABILITY"
    THERMAL_STABILITY = "THERMAL_STABILITY"
    SOLUBILITY = "SOLUBILITY"
    AGGREGATION = "AGGREGATION"
    IMMUNOGENICITY = "IMMUNOGENICITY"
    RECEPTOR_SELECTIVITY = "RECEPTOR_SELECTIVITY"
    IN_VIVO_EFFECT = "IN_VIVO_EFFECT"

    @property
    def is_affinity(self) -> bool:
        return self in (OutcomeMeasure.AFFINITY_KD, OutcomeMeasure.AFFINITY_KI)

    @property
    def needs_a_receptor(self) -> bool:
        """A measure that is meaningless without naming what it was measured against."""
        return self.is_affinity or self in (
            OutcomeMeasure.POTENCY_EC50, OutcomeMeasure.POTENCY_IC50,
            OutcomeMeasure.RECEPTOR_SELECTIVITY)


class Direction(Enum):
    """
    Which way the measure moved.

    NOT_DETERMINED is a real answer and is kept separate from UNCHANGED. A
    study that did not resolve the direction and a study that found no change
    are different results, and collapsing them turns the first into the second.
    """

    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    UNCHANGED = "UNCHANGED"
    NOT_DETERMINED = "NOT_DETERMINED"


class EvidenceSchemaError(ValueError):
    """Raised when a record would store something that cannot be read back."""


@dataclass(frozen=True)
class MeasuredOutcome:
    """
    One measurement, with what it was measured against and how.

    `comparator` and `assay` are required. "Three-fold more potent" is not a
    fact until it says than what and in what, and a fold change survives being
    copied into a summary while its missing baseline does not.
    """

    measure: OutcomeMeasure
    direction: Direction
    comparator: str                      # what the variant was measured against
    assay: str                           # how
    provenance: Provenance
    fold_change: Optional[float] = None  # relative to the comparator
    value: Optional[float] = None
    unit: str = ""
    target: str = ""                     # receptor or enzyme, where one applies
    note: str = ""

    def __post_init__(self):
        if not self.comparator.strip():
            raise EvidenceSchemaError(
                f"{self.measure.value} has no comparator. A relative measurement without "
                f"a baseline is not a measurement: the number survives being quoted and "
                f"the missing baseline does not.")
        if not self.assay.strip():
            raise EvidenceSchemaError(
                f"{self.measure.value} has no assay. Two numbers for the same measure "
                f"from different assays are not comparable, and without the assay nobody "
                f"can tell that they came from different ones.")
        if self.value is not None and not self.unit.strip():
            raise EvidenceSchemaError(
                f"{self.measure.value} has an absolute value of {self.value} and no unit.")
        if self.measure.needs_a_receptor and not self.target.strip():
            raise EvidenceSchemaError(
                f"{self.measure.value} needs the receptor or enzyme it was measured "
                f"against; an affinity with no target names no interaction.")
        if self.fold_change is not None and self.fold_change <= 0:
            raise EvidenceSchemaError("A fold change must be positive.")
        if (self.direction is Direction.UNCHANGED
                and self.fold_change is not None and abs(self.fold_change - 1.0) > 0.5):
            raise EvidenceSchemaError(
                f"Direction is UNCHANGED with a {self.fold_change}-fold change recorded. "
                f"One of the two is wrong, and which one cannot be guessed here.")

    @property
    def tier(self) -> EvidenceTier:
        """
        The strongest tier this outcome may be reported at.

        Delegated to `Provenance.max_tier`, which is where "no citation, no
        DIRECT_EXPERIMENTAL" is enforced. Re-implementing the rule here would
        be a second place for it to drift.
        """
        return self.provenance.max_tier

    @property
    def is_quantitative(self) -> bool:
        return self.fold_change is not None or self.value is not None

    def describe(self) -> str:
        bits = [self.measure.value.replace("_", " ").lower(),
                self.direction.value.lower()]
        if self.fold_change is not None:
            bits.append(f"{self.fold_change:g}-fold vs {self.comparator}")
        elif self.value is not None:
            bits.append(f"{self.value:g} {self.unit} vs {self.comparator}")
        else:
            bits.append(f"vs {self.comparator}, magnitude not recorded")
        if self.target:
            bits.append(f"at {self.target}")
        return f"{'; '.join(bits)} [{self.assay}] ({self.provenance.describe()})"

    def to_dict(self) -> Dict[str, Any]:
        return {"measure": self.measure.value, "direction": self.direction.value,
                "comparator": self.comparator, "assay": self.assay,
                "fold_change": self.fold_change, "value": self.value, "unit": self.unit,
                "target": self.target, "note": self.note, "tier": self.tier.name,
                "is_quantitative": self.is_quantitative,
                "provenance": self.provenance.describe(),
                "description": self.describe()}


@dataclass(frozen=True)
class Confounding:
    """Whether an outcome on this record can be attributed to one change."""

    is_confounded: bool
    changes: Tuple[str, ...]
    statement: str

    def to_dict(self) -> Dict[str, Any]:
        return {"is_confounded": self.is_confounded, "changes": list(self.changes),
                "statement": self.statement}


@dataclass(frozen=True)
class VariantRecord:
    """
    One modified peptide: what was changed, and what was measured about it.

    A record with modifications and no outcomes is a design, not evidence, and
    reports itself that way. The two are separate fields rather than one list
    because a table of modification names sitting under a column headed
    Evidence acquires the column's meaning.
    """

    name: str
    parent: str                          # the molecule the modifications are relative to
    modifications: Tuple[Modification, ...]
    provenance: Provenance               # for the modification list itself
    outcomes: Tuple[MeasuredOutcome, ...] = ()
    parent_sequence: str = ""
    aliases: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.modifications:
            raise EvidenceSchemaError(
                f"'{self.name}' records no modification, so there is nothing for an "
                f"outcome to be about.")

    @property
    def has_evidence(self) -> bool:
        """Whether anything was measured. A design is not evidence."""
        return bool(self.outcomes)

    @property
    def is_combination(self) -> bool:
        return len(self.modifications) > 1

    @property
    def is_attributable(self) -> bool:
        """
        Whether an outcome here can be attributed to a single change.

        One modification only. This is deliberately strict: it is the property
        that decides whether a measured effect may be carried over to a
        substitution being considered on its own.
        """
        return len(self.modifications) == 1

    @property
    def strongest_tier(self) -> Optional[EvidenceTier]:
        if not self.outcomes:
            return None
        order = list(EvidenceTier)
        return min((o.tier for o in self.outcomes), key=order.index)

    def confounding(self) -> Confounding:
        changes = tuple(m.label for m in self.modifications)
        if not self.is_combination:
            return Confounding(
                is_confounded=False, changes=changes,
                statement=f"One change ({changes[0]}), so a measured effect on this "
                          f"construct is attributable to it.")
        return Confounding(
            is_confounded=True, changes=changes,
            statement=(
                f"{len(changes)} simultaneous changes ({', '.join(changes)}). Any "
                f"measured effect is a property of the whole construct and cannot be "
                f"attributed to any one of them. Attributing it to the substitution is "
                f"the readiest error here, because the substitution is the part that "
                f"looks like the other rows in a substitution table."))

    def outcomes_for(self, measure: OutcomeMeasure) -> Tuple[MeasuredOutcome, ...]:
        return tuple(o for o in self.outcomes if o.measure is measure)

    def substitution_at(self, position: int, wild_type: str,
                        mutant: str) -> Optional[Modification]:
        for modification in self.modifications:
            if modification.matches_substitution(position, wild_type, mutant):
                return modification
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "parent": self.parent, "aliases": list(self.aliases),
                "modifications": [m.to_dict() for m in self.modifications],
                "outcomes": [o.to_dict() for o in self.outcomes],
                "has_evidence": self.has_evidence,
                "is_combination": self.is_combination,
                "is_attributable": self.is_attributable,
                "strongest_tier": self.strongest_tier.name if self.strongest_tier else None,
                "confounding": self.confounding().to_dict(),
                "provenance": self.provenance.describe(),
                "notes": list(self.notes)}


class MatchKind(Enum):
    """
    How closely a stored record bears on the substitution being considered.

    Ordered, and the order is the ranking. The same substitution in the same
    peptide is a different kind of claim from the same substitution somewhere
    else, and a store that returned them in one undifferentiated list would
    leave the distinction to whoever read it.
    """

    SAME_PEPTIDE_SAME_SUBSTITUTION = "SAME_PEPTIDE_SAME_SUBSTITUTION"
    SAME_PEPTIDE_SAME_POSITION = "SAME_PEPTIDE_SAME_POSITION"
    SAME_SUBSTITUTION_OTHER_PEPTIDE = "SAME_SUBSTITUTION_OTHER_PEPTIDE"
    SAME_PEPTIDE_OTHER_CHANGE = "SAME_PEPTIDE_OTHER_CHANGE"

    @property
    def rank(self) -> int:
        return list(MatchKind).index(self)

    @property
    def transfers_directly(self) -> bool:
        """
        Whether a measured effect carries over to the substitution being asked
        about without an argument in between.

        Only the exact change in the exact molecule. The same substitution in a
        different peptide is evidence about that peptide, and carrying it over
        needs the two contexts to be argued equivalent -- which this module
        does not do and does not pretend to.
        """
        return self is MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION

    @property
    def caveat(self) -> str:
        return {
            MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION:
                "Same substitution in the same molecule.",
            MatchKind.SAME_PEPTIDE_SAME_POSITION:
                "Same position in the same molecule, different substituted residue. The "
                "position's tolerance is informative; the specific effect is not.",
            MatchKind.SAME_SUBSTITUTION_OTHER_PEPTIDE:
                "Same substitution in a different peptide. Evidence about that peptide; "
                "carrying it over requires the two contexts to be argued equivalent, "
                "which is not done here.",
            MatchKind.SAME_PEPTIDE_OTHER_CHANGE:
                "Same molecule, a different change. Context only.",
        }[self]


@dataclass(frozen=True)
class EvidenceMatch:
    """One stored record, and why it was returned."""

    record: VariantRecord
    kind: MatchKind

    @property
    def is_usable_as_evidence(self) -> bool:
        """
        Whether this match may support a claim about the proposed substitution.

        Three conditions, all necessary: something was actually measured, the
        measurement is attributable to a single change, and the change is the
        one being asked about. A confounded record is returned and displayed --
        knowing the construct exists is worth something -- but it does not
        support an effect claim.
        """
        return (self.record.has_evidence
                and self.record.is_attributable
                and self.kind.transfers_directly)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "caveat": self.kind.caveat,
                "usable_as_evidence": self.is_usable_as_evidence,
                "record": self.record.to_dict()}


@dataclass
class PrecedentResult:
    """
    What the store knows about a proposed substitution.

    Carries `searched` so that "nothing was found" and "nothing was looked for"
    are distinguishable in the payload. They render identically as an empty
    list, and they mean opposite things.
    """

    query: str
    searched: bool
    matches: List[EvidenceMatch] = field(default_factory=list)
    store_is_empty: bool = False
    statement: str = ""

    @property
    def has_precedent(self) -> bool:
        return any(m.is_usable_as_evidence for m in self.matches)

    @property
    def best_tier(self) -> Optional[EvidenceTier]:
        usable = [m.record.strongest_tier for m in self.matches if m.is_usable_as_evidence]
        usable = [t for t in usable if t is not None]
        if not usable:
            return None
        order = list(EvidenceTier)
        return min(usable, key=order.index)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "searched": self.searched,
                "store_is_empty": self.store_is_empty,
                "has_precedent": self.has_precedent,
                "best_tier": self.best_tier.name if self.best_tier else None,
                "statement": self.statement,
                "matches": [m.to_dict() for m in self.matches]}


EMPTY_STORE_STATEMENT = (
    "No experimental precedent was found: the variant-evidence store holds no records. "
    "Every outcome worth storing needs a citation, and no literature host is reachable "
    "from this deployment, so the store is empty rather than populated from memory. "
    "This is an absence of evidence in the store, not evidence that none exists.")


class VariantEvidenceStore:
    """
    Reads curated variant records and answers questions about precedent.

    Refuses to guess in two places. A record whose outcome carries no citation
    cannot be reported as DIRECT_EXPERIMENTAL, and a record with more than one
    modification cannot support an effect claim about any single one of them.
    Both refusals are properties of the data rather than of the caller.
    """

    def __init__(self, path: Path = DATA_PATH):
        self.path = Path(path)
        self._records: Optional[List[VariantRecord]] = None
        self.load_error = ""

    @property
    def records(self) -> List[VariantRecord]:
        if self._records is None:
            self._records = self._load()
        return self._records

    @property
    def is_empty(self) -> bool:
        return not self.records

    def _load(self) -> List[VariantRecord]:
        if not self.path.exists():
            self.load_error = f"No variant-evidence file at {self.path}."
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Loud rather than silently empty: an unreadable store and an empty
            # store both return no precedent, and only one of them is a bug.
            raise EvidenceSchemaError(
                f"The variant-evidence store at {self.path} could not be read: {exc}. "
                f"Failing here rather than returning no records, because an unreadable "
                f"store and an empty one both look like 'no precedent found'.") from exc
        return [_record_from_dict(entry) for entry in raw.get("variants", [])]

    # -- retrieval --

    def for_substitution(self, peptide: str, position: int, wild_type: str,
                         mutant: str) -> PrecedentResult:
        """
        Everything the store holds bearing on one proposed substitution,
        ranked: exact change in the same molecule first, then the same
        position, then the same change elsewhere, then other changes to the
        same molecule.
        """
        query = f"{peptide}:{wild_type}{position}{mutant}"
        if self.is_empty:
            return PrecedentResult(query=query, searched=True, store_is_empty=True,
                                   statement=EMPTY_STORE_STATEMENT)

        matches: List[EvidenceMatch] = []
        for record in self.records:
            same_peptide = _same_molecule(peptide, record)
            exact = record.substitution_at(position, wild_type, mutant)
            at_position = any(m.kind.is_point_change and m.position == position
                              for m in record.modifications)
            if same_peptide and exact:
                kind = MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION
            elif same_peptide and at_position:
                kind = MatchKind.SAME_PEPTIDE_SAME_POSITION
            elif exact:
                kind = MatchKind.SAME_SUBSTITUTION_OTHER_PEPTIDE
            elif same_peptide:
                kind = MatchKind.SAME_PEPTIDE_OTHER_CHANGE
            else:
                continue
            matches.append(EvidenceMatch(record=record, kind=kind))

        matches.sort(key=lambda m: (m.kind.rank,
                                    not m.record.is_attributable,
                                    not m.record.has_evidence))
        return PrecedentResult(query=query, searched=True, matches=matches,
                               statement=_statement_for(matches))

    def for_peptide(self, peptide: str) -> List[VariantRecord]:
        return [r for r in self.records if _same_molecule(peptide, r)]

    def status(self) -> Dict[str, Any]:
        """What the store holds, for a caller deciding whether to trust an empty answer."""
        with_evidence = [r for r in self.records if r.has_evidence]
        return {
            "path": str(self.path),
            "exists": self.path.exists(),
            "records": len(self.records),
            "records_with_measured_outcomes": len(with_evidence),
            "attributable_records": len([r for r in with_evidence if r.is_attributable]),
            "load_error": self.load_error,
            "statement": EMPTY_STORE_STATEMENT if self.is_empty else
                         f"{len(self.records)} record(s), {len(with_evidence)} with "
                         f"measured outcomes.",
        }


def _same_molecule(peptide: str, record: VariantRecord) -> bool:
    """
    Whole-name comparison, never substring.

    The biological-context layer learned this one the hard way: matching
    "insulin" as a substring selects "Insulin-like growth factor 1", which is a
    different molecule with a different receptor.
    """
    key = _key(peptide)
    if not key:
        return False
    names = [record.parent, record.name, *record.aliases]
    return any(_key(name) == key for name in names)


def _key(name: str) -> str:
    return "".join(ch for ch in (name or "") if ch.isalnum()).upper()


def _statement_for(matches: Sequence[EvidenceMatch]) -> str:
    if not matches:
        return ("No record in the variant-evidence store bears on this substitution. "
                "This is an absence in the store, not evidence that none exists.")
    usable = [m for m in matches if m.is_usable_as_evidence]
    if usable:
        return (f"{len(usable)} record(s) measured this exact change in this molecule. "
                f"{len(matches) - len(usable)} further record(s) are related but do not "
                f"support an effect claim.")
    confounded = [m for m in matches if m.record.is_combination]
    designs = [m for m in matches if not m.record.has_evidence]
    bits = [f"{len(matches)} related record(s), none usable as evidence for this change."]
    if confounded:
        bits.append(f"{len(confounded)} change more than one thing at once, so their "
                    f"measured effects are properties of the whole construct.")
    if designs:
        bits.append(f"{len(designs)} record a modification with no measured outcome, "
                    f"which is a design rather than evidence.")
    return " ".join(bits)


# ---- deserialisation --------------------------------------------------------

def _provenance_from_dict(raw: Dict[str, Any]) -> Provenance:
    return Provenance(
        kind=SourceKind[raw.get("kind", "CURATED_UNVERIFIED")],
        detail=raw.get("detail", ""), accession=raw.get("accession", ""),
        pmid=raw.get("pmid", ""), doi=raw.get("doi", ""), year=raw.get("year"),
        retrieved_utc=raw.get("retrieved_utc", ""),
        needs_verification=raw.get("needs_verification", True))


def _record_from_dict(raw: Dict[str, Any]) -> VariantRecord:
    modifications = tuple(
        Modification(kind=ModificationKind[m["kind"]], description=m["description"],
                     position=m.get("position"), wild_type=m.get("wild_type", ""),
                     mutant=m.get("mutant", ""))
        for m in raw.get("modifications", []))
    outcomes = tuple(
        MeasuredOutcome(
            measure=OutcomeMeasure[o["measure"]], direction=Direction[o["direction"]],
            comparator=o.get("comparator", ""), assay=o.get("assay", ""),
            provenance=_provenance_from_dict(o.get("provenance", {})),
            fold_change=o.get("fold_change"), value=o.get("value"),
            unit=o.get("unit", ""), target=o.get("target", ""), note=o.get("note", ""))
        for o in raw.get("outcomes", []))
    return VariantRecord(
        name=raw["name"], parent=raw.get("parent", ""), modifications=modifications,
        provenance=_provenance_from_dict(raw.get("provenance", {})), outcomes=outcomes,
        parent_sequence=raw.get("parent_sequence", ""),
        aliases=tuple(raw.get("aliases", [])), notes=tuple(raw.get("notes", [])))


_STORE: Optional[VariantEvidenceStore] = None


def store() -> VariantEvidenceStore:
    """The shared store. Loaded once."""
    global _STORE
    if _STORE is None:
        _STORE = VariantEvidenceStore()
    return _STORE


def precedent_for(peptide: str, position: int, wild_type: str,
                  mutant: str) -> PrecedentResult:
    """Convenience entry point used by the workflow and the API."""
    return store().for_substitution(peptide, position, wild_type, mutant)
