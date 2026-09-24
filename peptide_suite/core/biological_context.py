"""
Known biology, retrieved before anything is predicted.

The system could optimise a peptide it could not describe. It would rank
substitutions in IGF-1 against a binding-affinity goal without ever mentioning
IGF1R -- not because it disagreed about the receptor, but because nothing in
the pipeline was responsible for saying what the molecule is. Engineering ran
straight off a sequence.

This module is the missing step. Identity resolution hands it a name; it hands
the engineering engines a structured account of what is already known, and
hands the interface something to show a reader before any score appears.

THE RULE THAT SHAPES EVERYTHING HERE

Every field carries where it came from. Not the object -- the field. A context
record for IGF-1 can hold a sequence verified against UniProt this minute, a
domain layout curated into this repository months ago and never re-checked, and
a receptor relationship with a PMID behind it, and those three are not the same
kind of claim. Collapsing them into one "confidence" number is how a curated
guess ends up being read as a database record.

So a `Claim` wraps a value with its `Provenance`, and the provenance decides
what evidence tier the claim may reach. A curated record with no citation
cannot be DIRECT_EXPERIMENTAL, however certain whoever wrote it felt. That is
enforced in `Provenance.max_tier`, not left to callers.

WHAT THIS MODULE WILL NOT DO

It will not infer a receptor from sequence resemblance. A peptide that looks
like an incretin is not thereby known to bind GLP1R, and reporting one as
established because a motif matched is exactly the failure the rest of this
codebase is built to avoid. An unknown peptide gets an empty context that says
so, and the interface renders that state rather than a biography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import EvidenceTier

_DATA = Path(__file__).parent.parent / "data"


class SourceKind(Enum):
    """
    Where a single field's value came from.

    Ordered loosely by how much a reader should trust it, but the ordering is
    not the point -- the point is that these are different in kind. A live
    database record and a curated note can agree on a value and still warrant
    different treatment downstream.
    """
    LIVE_DATABASE = "LIVE_DATABASE"              # fetched from the source this run
    CACHED_DATABASE = "CACHED_DATABASE"          # fetched earlier, from our cache
    PRIMARY_LITERATURE = "PRIMARY_LITERATURE"    # a specific experimental paper
    EXPERIMENTAL_STRUCTURE = "EXPERIMENTAL_STRUCTURE"   # a PDB entry
    CURATED_UNVERIFIED = "CURATED_UNVERIFIED"    # written into this repo, not re-checked
    COMPUTED = "COMPUTED"                        # derived from the sequence here
    INFERRED = "INFERRED"                        # a model or heuristic said so

    @property
    def is_retrieved(self) -> bool:
        return self in (SourceKind.LIVE_DATABASE, SourceKind.CACHED_DATABASE)


@dataclass(frozen=True)
class Provenance:
    """
    Where a claim came from, and what tier that permits it to reach.

    `max_tier` is the load-bearing method. It is the place the rule "no
    citation, no direct-experimental designation" is actually enforced, rather
    than being a convention each caller is trusted to remember.
    """
    kind: SourceKind
    detail: str = ""
    accession: str = ""          # UniProt, PDB, gene id
    pmid: str = ""
    doi: str = ""
    year: Optional[int] = None
    retrieved_utc: str = ""
    needs_verification: bool = False

    @property
    def has_citation(self) -> bool:
        return bool(self.pmid or self.doi)

    @property
    def max_tier(self) -> EvidenceTier:
        """
        The strongest evidence tier a claim from this source may be reported at.

        A curated record someone typed into this repository is not experimental
        evidence no matter how well-established the fact is, because nothing
        here can check it. It caps at BIOCHEMICAL_PRINCIPLE -- good enough to
        display and to reason from, not good enough to call an observation.
        """
        if self.kind is SourceKind.PRIMARY_LITERATURE and self.has_citation:
            return EvidenceTier.DIRECT_EXPERIMENTAL
        if self.kind is SourceKind.EXPERIMENTAL_STRUCTURE and self.accession:
            return EvidenceTier.DIRECT_EXPERIMENTAL
        if self.kind.is_retrieved:
            return EvidenceTier.HOMOLOG_EXPERIMENTAL
        if self.kind in (SourceKind.CURATED_UNVERIFIED, SourceKind.COMPUTED):
            return EvidenceTier.BIOCHEMICAL_PRINCIPLE
        return EvidenceTier.INFERENCE_ONLY

    def describe(self) -> str:
        bits = [self.kind.value.replace("_", " ").lower()]
        if self.accession:
            bits.append(self.accession)
        if self.pmid:
            bits.append(f"PMID {self.pmid}")
        if self.doi:
            bits.append(f"doi {self.doi}")
        if self.year:
            bits.append(str(self.year))
        if self.detail:
            bits.append(self.detail)
        if self.needs_verification:
            bits.append("NOT VERIFIED against the primary source")
        return " · ".join(bits)


@dataclass(frozen=True)
class Claim:
    """One value and where it came from. The unit this module traffics in."""
    value: Any
    provenance: Provenance

    @property
    def tier(self) -> EvidenceTier:
        return self.provenance.max_tier

    def encode(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "tier": self.provenance.max_tier.name,
            "source_kind": self.provenance.kind.value,
            "source": self.provenance.describe(),
            "accession": self.provenance.accession,
            "pmid": self.provenance.pmid,
            "doi": self.provenance.doi,
            "needs_verification": self.provenance.needs_verification,
        }


class MoleculeForm(Enum):
    """
    What the submitted sequence actually is.

    Kept explicit because these are different biological entities and the
    existing reference data silently conflated them: the IGF1 test fixture held
    a 33-residue sequence under the name of a 70-residue mature peptide. A
    precursor and its mature product have different lengths, different
    termini, different disulfides and different receptor behaviour, and an
    engineering run against the wrong one is wrong from the first step.
    """
    MATURE_PEPTIDE = "MATURE_PEPTIDE"
    PRECURSOR = "PRECURSOR"
    PROPEPTIDE = "PROPEPTIDE"
    ISOLATED_CHAIN = "ISOLATED_CHAIN"
    FRAGMENT = "FRAGMENT"
    SYNTHETIC_CONSTRUCT = "SYNTHETIC_CONSTRUCT"
    ENGINEERED_ANALOG = "ENGINEERED_ANALOG"
    UNKNOWN = "UNKNOWN"

    @property
    def describe(self) -> str:
        return {
            MoleculeForm.MATURE_PEPTIDE: "the mature, processed peptide",
            MoleculeForm.PRECURSOR: "the unprocessed precursor protein",
            MoleculeForm.PROPEPTIDE: "a propeptide, not the final product",
            MoleculeForm.ISOLATED_CHAIN: "one chain of a multi-chain molecule",
            MoleculeForm.FRAGMENT: "a fragment of a larger molecule",
            MoleculeForm.SYNTHETIC_CONSTRUCT: "a synthetic construct, not a natural product",
            MoleculeForm.ENGINEERED_ANALOG: "an engineered analog of a natural peptide",
            MoleculeForm.UNKNOWN: "not established",
        }[self]


class InteractionType(Enum):
    SIGNALLING_RECEPTOR = "SIGNALLING_RECEPTOR"
    CLEARANCE_RECEPTOR = "CLEARANCE_RECEPTOR"
    BINDING_PROTEIN = "BINDING_PROTEIN"
    PROTEASE = "PROTEASE"
    TRANSPORT_PARTNER = "TRANSPORT_PARTNER"
    OTHER = "OTHER"


@dataclass(frozen=True)
class ReceptorRelationship:
    """
    One established interaction between this peptide and a target.

    Affinity is Optional and stays None unless a measured value came with a
    source. A relationship with no number is a real, useful statement -- "IGF-1
    binds IGF1R" -- and inventing a Kd to fill the field would turn it into a
    false one.
    """
    target_name: str
    target_gene: str
    interaction_type: InteractionType
    is_primary: bool
    provenance: Provenance
    target_accession: str = ""
    receptor_class: str = ""
    role: str = ""
    affinity_value: Optional[float] = None
    affinity_unit: str = ""
    affinity_kind: str = ""          # Kd, Ki, IC50, EC50
    assay_context: str = ""
    notes: str = ""

    @property
    def tier(self) -> EvidenceTier:
        return self.provenance.max_tier

    @property
    def has_measured_affinity(self) -> bool:
        return self.affinity_value is not None and bool(self.affinity_unit)

    def encode(self) -> Dict[str, Any]:
        payload = {
            "target_name": self.target_name,
            "target_gene": self.target_gene,
            "target_accession": self.target_accession,
            "interaction_type": self.interaction_type.value,
            "receptor_class": self.receptor_class,
            "is_primary": self.is_primary,
            "role": self.role,
            "assay_context": self.assay_context,
            "notes": self.notes,
            "tier": self.tier.name,
            "source": self.provenance.describe(),
            "source_kind": self.provenance.kind.value,
            "pmid": self.provenance.pmid,
            "needs_verification": self.provenance.needs_verification,
            "affinity_available": self.has_measured_affinity,
        }
        # The key is absent rather than null when nothing was measured: a null
        # affinity is one `?? 0` away from being read as a number.
        if self.has_measured_affinity:
            payload["affinity"] = {
                "kind": self.affinity_kind,
                "value": self.affinity_value,
                "unit": self.affinity_unit,
            }
        return payload


@dataclass(frozen=True)
class StructuralRegion:
    """A named region of the mature sequence, in mature-peptide numbering."""
    name: str
    start: int                   # 1-based, inclusive
    end: int                     # 1-based, inclusive
    provenance: Provenance
    role: str = ""

    def sequence_of(self, sequence: str) -> str:
        return sequence[self.start - 1:self.end]

    def encode(self, sequence: str = "") -> Dict[str, Any]:
        return {
            "name": self.name, "start": self.start, "end": self.end, "role": self.role,
            "residues": self.sequence_of(sequence) if sequence else "",
            "tier": self.provenance.max_tier.name,
            "source": self.provenance.describe(),
            "needs_verification": self.provenance.needs_verification,
        }


@dataclass(frozen=True)
class DisulfideBond:
    """A cysteine pair, in mature-peptide numbering."""
    first: int
    second: int
    provenance: Provenance

    def is_consistent_with(self, sequence: str) -> bool:
        """
        Whether both partners are actually cysteines in the given sequence.

        Checked rather than trusted: a disulfide table that disagrees with the
        sequence beside it means one of them is wrong, and the pair is the only
        place that can be noticed cheaply.
        """
        return (1 <= self.first <= len(sequence)
                and 1 <= self.second <= len(sequence)
                and sequence[self.first - 1] == "C"
                and sequence[self.second - 1] == "C")

    def encode(self, sequence: str = "") -> Dict[str, Any]:
        return {
            "first": self.first, "second": self.second,
            "consistent_with_sequence": self.is_consistent_with(sequence) if sequence else None,
            "tier": self.provenance.max_tier.name,
            "source": self.provenance.describe(),
            "needs_verification": self.provenance.needs_verification,
        }


@dataclass(frozen=True)
class BindingInterface:
    """
    What is established about how this peptide meets one target.

    Empty by default and empty in most records, which is correct: residue-level
    contacts come from structures and mutagenesis, not from reasoning about
    hydrophobicity. A record with no interface annotation says so; it does not
    fall back to generic physicochemistry.
    """
    target_gene: str
    provenance: Provenance
    ligand_regions: Tuple[str, ...] = ()
    receptor_regions: Tuple[str, ...] = ()
    critical_residues: Tuple[str, ...] = ()
    activation_mechanism: str = ""
    structures: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.ligand_regions or self.receptor_regions
                    or self.critical_residues or self.structures)

    def encode(self) -> Dict[str, Any]:
        return {
            "target_gene": self.target_gene,
            "ligand_regions": list(self.ligand_regions),
            "receptor_regions": list(self.receptor_regions),
            "critical_residues": list(self.critical_residues),
            "activation_mechanism": self.activation_mechanism,
            "structures": list(self.structures),
            "notes": self.notes,
            "is_empty": self.is_empty,
            "tier": self.provenance.max_tier.name,
            "source": self.provenance.describe(),
            "needs_verification": self.provenance.needs_verification,
        }


@dataclass
class BiologicalContext:
    """
    What is known about one peptide, assembled before any engineering runs.

    `is_established` is the field the interface branches on. False means the
    peptide was not recognised, and the correct rendering is a statement that
    no identity was found -- not a context object with plausible-looking empty
    sections, which reads as "we looked and it has no receptors".
    """
    query_sequence: str
    is_established: bool
    name: str = ""
    aliases: Tuple[str, ...] = ()
    gene: str = ""
    organism: str = ""
    uniprot: str = ""
    family: str = ""
    form: MoleculeForm = MoleculeForm.UNKNOWN
    form_provenance: Optional[Provenance] = None
    precursor_of: str = ""
    mature_sequence: str = ""
    function: Optional[Claim] = None
    regions: List[StructuralRegion] = field(default_factory=list)
    disulfides: List[DisulfideBond] = field(default_factory=list)
    modifications: List[Claim] = field(default_factory=list)
    receptors: List[ReceptorRelationship] = field(default_factory=list)
    interfaces: List[BindingInterface] = field(default_factory=list)
    references: List[Provenance] = field(default_factory=list)
    retrieval_notes: List[str] = field(default_factory=list)
    unavailable_sources: List[str] = field(default_factory=list)

    @property
    def primary_receptors(self) -> List[ReceptorRelationship]:
        return [r for r in self.receptors if r.is_primary]

    @property
    def secondary_receptors(self) -> List[ReceptorRelationship]:
        return [r for r in self.receptors if not r.is_primary]

    @property
    def sequence_matches_mature(self) -> Optional[bool]:
        if not self.mature_sequence or not self.query_sequence:
            return None
        return self.query_sequence.upper() == self.mature_sequence.upper()

    @property
    def needs_verification(self) -> bool:
        """Whether any displayed claim is unverified against its primary source."""
        sources = ([self.form_provenance] if self.form_provenance else [])
        sources += [c.provenance for c in ([self.function] if self.function else [])]
        sources += [r.provenance for r in self.regions]
        sources += [d.provenance for d in self.disulfides]
        sources += [r.provenance for r in self.receptors]
        sources += [i.provenance for i in self.interfaces]
        return any(p.needs_verification for p in sources)

    def disulfide_inconsistencies(self) -> List[str]:
        """Curated disulfides that do not land on cysteines in the stored sequence."""
        sequence = self.mature_sequence
        if not sequence:
            return []
        return [f"C{d.first}-C{d.second}" for d in self.disulfides
                if not d.is_consistent_with(sequence)]

    def summary(self) -> str:
        if not self.is_established:
            return ("No established peptide identity was found for this sequence. Any "
                    "analysis below rests on sequence-derived properties alone.")
        receptors = ", ".join(r.target_gene for r in self.primary_receptors)
        return (f"{self.name} ({self.organism}, {self.gene or 'gene not recorded'}), "
                f"{self.form.describe}"
                + (f"; established receptor {receptors}" if receptors
                   else "; no established receptor recorded"))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_CURATED_CACHE: Optional[Dict[str, dict]] = None


def _curated() -> Dict[str, dict]:
    global _CURATED_CACHE
    if _CURATED_CACHE is None:
        raw = json.loads((_DATA / "biological_context.json").read_text())
        _CURATED_CACHE = dict(raw["peptides"])
    return _CURATED_CACHE


def _curated_provenance(record: dict, uniprot: str = "") -> Provenance:
    """
    The provenance stamped on every field of a curated record.

    `detail` is left empty on purpose. It used to carry the record's curation
    note, which meant the same 200-word paragraph was rendered under every
    receptor, every domain and every disulfide -- and a caveat repeated eight
    times is a caveat nobody reads. The note belongs once, on the card.
    """
    return Provenance(
        kind=SourceKind.CURATED_UNVERIFIED,
        accession=uniprot or record.get("uniprot", ""),
        needs_verification=True,
    )


def _match_key(name: str) -> Optional[str]:
    """
    Find a curated record by canonical name or alias.

    Whole-name matching, case-insensitive. Deliberately not substring matching:
    "insulin" must not select "Insulin-like growth factor 1", and this module
    is the last place that distinction can be made before an engineering run
    inherits the wrong molecule.
    """
    if not name:
        return None
    wanted = name.strip().lower()
    for key, record in _curated().items():
        candidates = {key.lower(), record.get("name", "").lower()}
        candidates |= {a.lower() for a in record.get("aliases", [])}
        candidates.discard("")
        if wanted in candidates:
            return key
    return None


def _match_sequence(sequence: str) -> Optional[str]:
    """Find a curated record whose mature sequence is exactly this one."""
    if not sequence:
        return None
    wanted = sequence.strip().upper()
    for key, record in _curated().items():
        if record.get("mature_sequence", "").upper() == wanted:
            return key
    return None


def _build(key: str, record: dict, query_sequence: str,
           provenance: Provenance) -> BiologicalContext:
    mature = record.get("mature_sequence", "")

    regions = [
        StructuralRegion(name=r["name"], start=r["start"], end=r["end"],
                         role=r.get("role", ""), provenance=provenance)
        for r in record.get("regions", [])
    ]
    disulfides = [
        DisulfideBond(first=d["first"], second=d["second"], provenance=provenance)
        for d in record.get("disulfides", [])
    ]
    receptors = [
        ReceptorRelationship(
            target_name=r["target_name"],
            target_gene=r["target_gene"],
            target_accession=r.get("target_accession", ""),
            interaction_type=InteractionType(r["interaction_type"]),
            receptor_class=r.get("receptor_class", ""),
            is_primary=bool(r.get("is_primary")),
            role=r.get("role", ""),
            notes=r.get("notes", ""),
            provenance=provenance,
        )
        for r in record.get("receptors", [])
    ]
    interfaces = [
        BindingInterface(
            target_gene=i["target_gene"],
            ligand_regions=tuple(i.get("ligand_regions", [])),
            receptor_regions=tuple(i.get("receptor_regions", [])),
            critical_residues=tuple(i.get("critical_residues", [])),
            activation_mechanism=i.get("activation_mechanism", ""),
            structures=tuple(i.get("structures", [])),
            notes=i.get("notes", ""),
            provenance=provenance,
        )
        for i in record.get("interfaces", [])
    ]

    context = BiologicalContext(
        query_sequence=query_sequence,
        is_established=True,
        name=record.get("name", key),
        aliases=tuple(record.get("aliases", [])),
        gene=record.get("gene", ""),
        organism=record.get("organism", ""),
        uniprot=record.get("uniprot", ""),
        family=record.get("family", ""),
        form=MoleculeForm(record.get("form", "UNKNOWN")),
        form_provenance=provenance,
        precursor_of=record.get("precursor_of", ""),
        mature_sequence=mature,
        function=(Claim(record["function"], provenance) if record.get("function") else None),
        regions=regions,
        disulfides=disulfides,
        modifications=[Claim(m, provenance) for m in record.get("modifications", [])],
        receptors=receptors,
        interfaces=interfaces,
    )

    # The submitted sequence may be a fragment of, or differ from, the mature
    # peptide the record describes. Saying which is the difference between
    # "here is your molecule" and "here is a molecule your input resembles".
    matches = context.sequence_matches_mature
    if matches is False:
        if query_sequence.upper() in mature.upper():
            context.form = MoleculeForm.FRAGMENT
            context.retrieval_notes.append(
                f"The submitted sequence is a {len(query_sequence)}-residue fragment of "
                f"the {len(mature)}-residue mature peptide, not the mature peptide itself. "
                f"Numbering below refers to the mature peptide.")
        else:
            context.retrieval_notes.append(
                f"The submitted sequence differs from the mature {context.name} sequence on "
                f"record. It was matched by name, so the biology below describes "
                f"{context.name} and may not describe the exact molecule submitted.")

    if record.get("curation_note"):
        context.retrieval_notes.append(record["curation_note"])

    broken = context.disulfide_inconsistencies()
    if broken:
        context.retrieval_notes.append(
            f"Curated disulfide pair(s) {', '.join(broken)} do not land on cysteines in the "
            f"stored sequence. One of the two is wrong; neither is used until that is "
            f"resolved.")
    return context


def unknown_context(sequence: str, unavailable: Optional[List[str]] = None
                    ) -> BiologicalContext:
    """
    The context for a peptide nothing is known about.

    A distinct object rather than an empty one, because "we have no record of
    this" and "we looked it up and it has no receptors" render identically if
    the only difference is an empty list.
    """
    return BiologicalContext(
        query_sequence=sequence,
        is_established=False,
        unavailable_sources=list(unavailable or []),
        retrieval_notes=[
            "No established peptide identity was found for this sequence. No receptor, "
            "domain structure or function is asserted, and none is inferred from sequence "
            "resemblance: a peptide that looks like a family member is not thereby known "
            "to bind that family's receptor."
        ],
    )


def retrieve(sequence: str = "", name: str = "",
             uniprot_client: Optional[Any] = None) -> BiologicalContext:
    """
    Assemble what is known about a peptide, preferring live sources.

    The order matters and is the whole design: an exact sequence match against
    a curated record beats a name match, because a name can be attached to the
    wrong molecule and a sequence cannot. A live database lookup is attempted
    on top and records what it found or why it could not.
    """
    sequence = (sequence or "").strip().upper()
    unavailable: List[str] = []

    key = _match_sequence(sequence) or _match_key(name)
    if key is None:
        # Nothing curated. A live lookup is still worth attempting, and its
        # failure is worth reporting: "we could not reach UniProt" and "UniProt
        # has nothing" are different answers and the interface must not merge
        # them.
        if uniprot_client is not None:
            result = uniprot_client.identify(sequence, raw_input=name or sequence)
            if not getattr(result, "reachable", True):
                unavailable.append(f"UniProt: {getattr(result, 'status', 'unreachable')}")
        return unknown_context(sequence, unavailable)

    record = _curated()[key]
    context = _build(key, record, sequence, _curated_provenance(record))

    if uniprot_client is not None:
        result = uniprot_client.identify(sequence, raw_input=name or key)
        if getattr(result, "reachable", False):
            context.retrieval_notes.append(
                "UniProt was reachable and its record was consulted alongside the curated "
                "entry below.")
        else:
            status = getattr(result, "status", "unreachable")
            context.unavailable_sources.append(f"UniProt: {status}")
            context.retrieval_notes.append(
                f"UniProt could not be reached ({status}), so nothing below was verified "
                f"against it this run. The entries are curated records and are marked as "
                f"such; they are not database retrievals.")
    else:
        context.unavailable_sources.append("UniProt: not queried")

    return context
