"""
Native-contact classification and its design consequences.
[Addendum 2 section 7, build step 6e]

Every contact the mature peptide makes in its native context falls into one of
three classes, and the class determines what a proposal is allowed to do to it:

  SCAFFOLD    Holds the peptide in place during biosynthesis, storage or
              trafficking. Removable with no loss of receptor activity, so it is
              a target for deliberate disruption when the goal is faster onset
              or less self-association.

  PROTECTIVE  Extends functional lifetime without contributing to receptor
              engagement. Must be REPLACED, not merely removed: a proposal that
              deletes one with no engineered substitute is rejected.

  ESSENTIAL   Required to reach or hold the active state. Frozen. No
              modification inside the footprint unless the proposal states how
              the essential property is preserved.

SCAFFOLD is the cheapest high-value classification in the system, because its
verification path is usually free: if an isolated synthetic peptide retains
activity in the published literature, the call is confirmed without running
anything.

The failure this module is built around is omission, not misclassification. A
28-residue peptide whose activity depends on one octanoyl group looks, in its
sequence, exactly like a 28-residue peptide that does not. So an unverified
contact is UNRESOLVED and blocks the proposals that depend on it, rather than
being treated as absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

_DATA = Path(__file__).parent.parent / "data"


class ContactClass(Enum):
    SCAFFOLD = "SCAFFOLD"
    PROTECTIVE = "PROTECTIVE"
    ESSENTIAL = "ESSENTIAL"
    UNRESOLVED = "UNRESOLVED"

    @property
    def description(self) -> str:
        return {
            ContactClass.SCAFFOLD:
                "Holds the peptide in place during biosynthesis, storage or trafficking. "
                "Theoretically removable with no loss of receptor activity.",
            ContactClass.PROTECTIVE:
                "Extends functional lifetime without contributing to receptor engagement.",
            ContactClass.ESSENTIAL:
                "Required for the peptide to reach or hold its active state.",
            ContactClass.UNRESOLVED:
                "Not verified. Blocks any proposal that depends on it.",
        }[self]


class EssentialSubclass(Enum):
    """Why an ESSENTIAL contact is essential. Different mechanisms, different substitutes."""
    CONFORMATIONAL = "CONFORMATIONAL"
    ELECTROSTATIC = "ELECTROSTATIC"
    COVALENT_PTM = "COVALENT_PTM"
    PARTNER_DEPENDENT = "PARTNER_DEPENDENT"


class Verdict(Enum):
    """What the classifier permits a proposal to do."""
    PERMITTED = "permitted"
    ENCOURAGED = "encouraged"            # SCAFFOLD disruption, when it serves the goal
    REQUIRES_SUBSTITUTE = "requires_substitute"
    REJECTED = "rejected"
    BLOCKED_UNRESOLVED = "blocked_unresolved"

    @property
    def permits_emission(self) -> bool:
        return self in (Verdict.PERMITTED, Verdict.ENCOURAGED)


class ClassificationError(Exception):
    """Raised when a contact is classified without the evidence the spec requires."""


@dataclass
class ContactClassification:
    """
    One classified contact.

    Citation and confidence are required for anything other than UNRESOLVED.
    The spec is explicit that every classification needs both, and the reason is
    structural: a classification without evidence is an opinion, and an opinion
    that freezes a residue footprint is worse than no classification at all
    because it is indistinguishable from a verified one.
    """
    contact: str
    contact_class: ContactClass
    residues: List[str] = field(default_factory=list)
    subclass: Optional[EssentialSubclass] = None
    claim: str = ""
    citation: str = ""
    citation_precision: str = ""
    confidence: Optional[float] = None
    independently_verified: bool = False
    verification_path: str = ""
    magnitude_source: str = ""
    also_classified: Optional[ContactClass] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.contact_class is ContactClass.UNRESOLVED:
            return
        if not self.citation:
            raise ClassificationError(
                f"Contact '{self.contact}' is classified {self.contact_class.value} with no "
                f"citation. Every classification requires evidence; an unevidenced one is "
                f"indistinguishable from a verified one once it is in the output."
            )
        if self.confidence is None:
            raise ClassificationError(
                f"Contact '{self.contact}' is classified {self.contact_class.value} with no "
                f"confidence. Confidence is a separate axis from the classification itself "
                f"and is not implied by it."
            )
        if self.contact_class is ContactClass.ESSENTIAL and self.subclass is None:
            raise ClassificationError(
                f"Contact '{self.contact}' is ESSENTIAL but names no subclass. "
                f"Conformational, electrostatic, covalent-PTM and partner-dependent "
                f"essentials need different substitutes, so the mechanism is part of the "
                f"classification rather than a detail of it."
            )

    @property
    def classes(self) -> List[ContactClass]:
        """Every class this contact carries. Usually one; pyroglutamate carries two."""
        return [self.contact_class] + ([self.also_classified] if self.also_classified else [])

    def summary(self) -> str:
        label = self.contact_class.value
        if self.subclass:
            label += f" / {self.subclass.value}"
        if self.also_classified:
            label += f" and {self.also_classified.value}"
        return f"{self.contact}: {label}"


@dataclass
class ProposalRuling:
    """What the classifier decided about one proposal, and why."""
    proposal: str
    verdict: Verdict
    reason: str
    contact: Optional[ContactClassification] = None
    required_substitute: str = ""

    @property
    def permits_emission(self) -> bool:
        return self.verdict.permits_emission


# What each class permits, as data rather than branching, so the rules are
# readable next to each other and a new class cannot quietly inherit whichever
# branch happened to fall through.
CONSEQUENCE = {
    ContactClass.SCAFFOLD: (
        Verdict.ENCOURAGED,
        "SCAFFOLD contacts exist for storage and trafficking, not receptor engagement. "
        "Disrupting one is a legitimate design move when the goal is faster onset or "
        "reduced self-association — this is how lispro and aspart were obtained.",
    ),
    ContactClass.PROTECTIVE: (
        Verdict.REQUIRES_SUBSTITUTE,
        "PROTECTIVE contacts extend functional lifetime. Removing one without an "
        "engineered substitute trades away exposure for nothing, so the proposal is "
        "rejected rather than ranked with a caveat.",
    ),
    ContactClass.ESSENTIAL: (
        Verdict.REJECTED,
        "ESSENTIAL contacts are frozen. A modification inside the footprint is refused "
        "unless the proposal states explicitly how the essential property is preserved.",
    ),
    ContactClass.UNRESOLVED: (
        Verdict.BLOCKED_UNRESOLVED,
        "The contact is unverified. An unverified contact blocks the proposals that "
        "depend on it rather than being treated as absent, because absence and "
        "unexamined look identical in a sequence.",
    ),
}


class GoldenCases:
    """
    The seed set from Addendum 2 section 7.

    These are not test fixtures. They are the cases the module is expected to get
    right, and most of them are cases a sequence-only system gets wrong by
    omission: an octanoyl group, a sulfate, an amide, a D-residue, a glycan.
    """

    _cases: Optional[Dict[str, ContactClassification]] = None

    @classmethod
    def all(cls) -> Dict[str, ContactClassification]:
        if cls._cases is None:
            raw = json.loads((_DATA / "native_context_golden.json").read_text())
            cls._cases = {
                key: ContactClassification(
                    contact=entry["contact"],
                    contact_class=ContactClass[entry["classification"]],
                    residues=list(entry.get("residues", [])),
                    subclass=(EssentialSubclass[entry["subclass"]]
                              if entry.get("subclass") else None),
                    claim=entry.get("claim", ""),
                    citation=entry.get("citation", ""),
                    citation_precision=entry.get("citation_precision", ""),
                    # Literature-asserted and unverified here. A high number would
                    # be reporting the literature's confidence as our own.
                    confidence=0.5 if not entry.get("independently_verified") else 0.9,
                    independently_verified=entry.get("independently_verified", False),
                    verification_path=entry.get("verification_path", ""),
                    magnitude_source=entry.get("magnitude_source", ""),
                    also_classified=(ContactClass[entry["also_classified"]]
                                     if entry.get("also_classified") else None),
                    notes=[v for k, v in entry.items()
                           if k.startswith("why_") and isinstance(v, str)],
                )
                for key, entry in raw["cases"].items()
            }
        return cls._cases

    @classmethod
    def for_peptide(cls, name: str) -> List[ContactClassification]:
        """Golden cases naming this peptide. Matched loosely; the names vary in writing."""
        if not name:
            return []
        wanted = _normalise(name)
        raw = json.loads((_DATA / "native_context_golden.json").read_text())
        hits = []
        for key, entry in raw["cases"].items():
            listed = [_normalise(p) for p in entry["peptide"].split(",")]
            if any(wanted and (wanted in p or p in wanted) for p in listed if p):
                hits.append(cls.all()[key])
        return hits


def _normalise(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


# Modifications a proposal can name. Used to tell "installs the essential
# modification" from "removes it" -- both are "a modification within the
# footprint", and treating them alike refuses the one proposal that satisfies
# the rule by construction.
_MODIFICATIONS = {
    "amidation": ("amidat", "amide"),
    "acetylation": ("acetylat", "acetyl"),
    "sulfation": ("sulfat", "sulfo"),
    "octanoylation": ("octanoyl", "acylat"),
    "glycosylation": ("glycosyl", "glycan"),
    "carboxylation": ("carboxylat", "gamma-carboxy"),
    "pyroglutamate": ("pyroglutamate", "pglu"),
}


# Anything that makes a proposal not-a-restoration. Naming the modification is
# not enough to mean installing it: "replace pGlu with Glu" names pyroglutamate
# and removes it, and a substitution arrow means the residue is being changed
# rather than the native state restored. Both are decisive, and both were found
# by tests after an earlier version read them as restorations.
_REMOVAL_MARKERS = (
    "remove", "removal", "delete", "deletion", "strip", "omit", "without",
    "free acid", "des-", "desulfat", "deacyl", "des acyl", "loss of",
    "abolish", "truncat", "unmodified",
    # Substitution and replacement: the residue is being changed, so whatever
    # the proposal names it is not restoring the native form.
    "replace", "substitut", "->", "\u2192",
)


def restores_contact(proposal: str, contact: ContactClassification) -> bool:
    """
    Whether the proposal installs the very modification the contact requires.

    A synthesised fragment of an amidated hormone arrives with a free acid,
    because solid-phase synthesis produces one. Proposing C-terminal amidation
    on that fragment is not a modification of the native molecule; it is
    restoring the native state the fragment never had. Refusing it would block
    the one proposal that satisfies the ESSENTIAL rule by construction.

    Only meaningful for covalent-PTM essentials. A conformational or
    electrostatic contact is not "installed" by naming it.
    """
    if contact.subclass is not EssentialSubclass.COVALENT_PTM:
        return False
    text = (proposal or "").lower()

    # Removal language first, and it is decisive. Keyword overlap alone cannot
    # tell "add the amide" from "remove the amide" -- both name the amide -- and
    # an earlier version of this function read the deletion as a restoration,
    # which encouraged exactly the proposal the freeze exists to refuse.
    if any(marker in text for marker in _REMOVAL_MARKERS):
        return False

    target = f"{contact.contact} {contact.claim}".lower()
    for markers in _MODIFICATIONS.values():
        if any(m in text for m in markers) and any(m in target for m in markers):
            return True
    return False


def rule_on(proposal: str, contact: ContactClassification,
            declares_substitute: bool = False,
            declares_preservation: bool = False) -> ProposalRuling:
    """
    Apply the design-consequence rules to one proposal against one contact.

    `declares_substitute` and `declares_preservation` are the two escape hatches
    the spec allows, and both require the proposal to have said something
    specific. They are parameters rather than inferred from the text because
    inferring them would let a proposal earn an exemption by wording.
    """
    if restores_contact(proposal, contact):
        return ProposalRuling(
            proposal=proposal, verdict=Verdict.ENCOURAGED, contact=contact,
            reason=(f"The proposal installs the modification this contact requires "
                    f"({contact.contact}), rather than modifying the footprint. A "
                    f"synthesised fragment arrives without it, so this restores the "
                    f"native state instead of departing from it."))

    verdict, reason = CONSEQUENCE[contact.contact_class]

    # A contact carrying two classes has to satisfy both. Pyroglutamate is
    # ESSENTIAL and PROTECTIVE, and checking whichever comes first would let a
    # proposal through on the weaker of the two.
    for extra in contact.classes[1:]:
        extra_verdict, extra_reason = CONSEQUENCE[extra]
        if _severity(extra_verdict) > _severity(verdict):
            verdict, reason = extra_verdict, extra_reason
        elif extra_verdict is not verdict:
            reason += f" It is also {extra.value}: {CONSEQUENCE[extra][1]}"

    if verdict is Verdict.REQUIRES_SUBSTITUTE and declares_substitute:
        return ProposalRuling(
            proposal=proposal, verdict=Verdict.PERMITTED, contact=contact,
            reason=("The contact is PROTECTIVE and the proposal supplies an engineered "
                    "substitute, which is what the rule requires."))

    if verdict is Verdict.REJECTED and declares_preservation:
        return ProposalRuling(
            proposal=proposal, verdict=Verdict.PERMITTED, contact=contact,
            reason=("The contact is ESSENTIAL and the proposal states how the essential "
                    "property is preserved, which is the only route through the freeze."))

    required = ""
    if verdict is Verdict.REQUIRES_SUBSTITUTE:
        required = ("An engineered replacement for the protective function — the proposal "
                    "must name it, not imply it.")
    elif verdict is Verdict.REJECTED and contact.subclass:
        required = _preservation_hint(contact.subclass)

    return ProposalRuling(proposal=proposal, verdict=verdict, contact=contact,
                          reason=reason, required_substitute=required)


_SEVERITY = {
    Verdict.ENCOURAGED: 0,
    Verdict.PERMITTED: 1,
    Verdict.REQUIRES_SUBSTITUTE: 2,
    Verdict.BLOCKED_UNRESOLVED: 3,
    Verdict.REJECTED: 4,
}


def _severity(verdict: Verdict) -> int:
    return _SEVERITY[verdict]


def _preservation_hint(subclass: EssentialSubclass) -> str:
    return {
        EssentialSubclass.CONFORMATIONAL:
            "State how the active conformation is held without this contact, with the "
            "pre-organization evidence for it.",
        EssentialSubclass.ELECTROSTATIC:
            "State what supplies the charge or chelation this contact provided.",
        EssentialSubclass.COVALENT_PTM:
            "State what replaces the modification. A synthetic peptide ordered without it "
            "is a different pharmacological molecule, not a variant of this one.",
        EssentialSubclass.PARTNER_DEPENDENT:
            "State how the partner is engaged without this contact, or that the partner is "
            "supplied separately.",
    }[subclass]


def residues_touched(description: str, position: Optional[int]) -> List[str]:
    """
    Residue labels a proposal touches, as written.

    Both forms are produced: a one-based position from the transformation
    itself, and any residue label spelled out in the description ("Ser3",
    "LysB29", "Thr11"). Neither alone is sufficient — whole-molecule moves carry
    no position, and position-carrying moves do not always name the residue.
    """
    import re
    labels = set(re.findall(r"\b([A-Z][a-z]{2}[A-Z]?\d+)\b", description or ""))
    if position is not None:
        labels.add(str(position))
    for term in ("C-terminus", "C-terminal", "N-terminus", "N-terminal"):
        if term.lower() in (description or "").lower():
            labels.add(term.split("-")[0] + "-terminus")
    return sorted(labels)


def _footprint_overlap(contact: "ContactClassification", touched: List[str]) -> bool:
    """
    Whether a proposal lands inside a contact's footprint.

    Matching is deliberately generous: "Ser3" matches the residue "Ser3" and
    also the bare position "3". A false positive routes a proposal to a ruling
    it can argue its way past by declaring preservation; a false negative lets a
    modification into a frozen footprint unnoticed, which is the failure the
    freeze exists to prevent.
    """
    import re
    for residue in contact.residues:
        residue_digits = re.sub(r"\D", "", residue)
        for label in touched:
            if residue.lower() == label.lower():
                return True
            if residue_digits and residue_digits == re.sub(r"\D", "", label) and (
                    label.isdigit() or residue.isdigit()
                    or label.lower().startswith(residue[:3].lower())):
                return True
            if residue.lower().endswith("terminus") and label.lower() == residue.lower():
                return True
    return False


def rule_proposal(transformation, contacts: List["ContactClassification"]):
    """
    Rule on a transformation against every classified contact it touches.

    Returns the strictest ruling, or None when the proposal touches no
    classified contact. Strictest rather than first: a proposal overlapping both
    a SCAFFOLD and an ESSENTIAL contact is governed by the ESSENTIAL one, and
    taking whichever was checked first would let the encouragement win.
    """
    if not contacts:
        return None

    touched = residues_touched(transformation.description, transformation.display_position)
    rulings = [
        rule_on(transformation.description, contact)
        for contact in contacts
        if _footprint_overlap(contact, touched)
    ]
    if not rulings:
        return None
    return max(rulings, key=lambda r: _severity(r.verdict))
