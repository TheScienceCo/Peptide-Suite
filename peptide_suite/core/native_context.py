"""
Native-context analysis.

A peptide excised from a larger protein is not the same object as that segment
in situ. Contacts the segment made with the rest of the parent are gone, and
they were not all doing the same job. Classifying what each lost contact was
doing determines what, if anything, should replace it:

    SCAFFOLDING             held the segment in place and nothing more.
                            Removable. Check for a published truncation or
                            alanine scan that confirms or refutes this.

    PROTECTIVE              shielded from protease, kept it soluble, suppressed
                            aggregation. Removable only with a functional
                            replacement, which must be recommended alongside.

    CONFORMATIONAL          the partner induced or stabilised the bioactive
                            fold. Recommend an INTRAMOLECULAR CONSTRAINT.
                            Do NOT recommend a partner peptide for this class:
                            re-supplying the partner to hold a shape is solving
                            intramolecularly-solvable geometry with a second
                            molecule.

    COMPOSITIONAL/OBLIGATE  the partner is part of the interface or the
                            mechanism. Recommend the PARTNER PEPTIDE.

Covalent PTMs in native context are classified on the same four-way scheme.
They are not exempted for being small: glycans on small peptides are routinely
load-bearing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .contact_classifier import GoldenCases
from .epistemics import Assumption, AssumptionKind, Claim


class ContactClass(Enum):
    SCAFFOLDING = "scaffolding"
    PROTECTIVE = "protective"
    CONFORMATIONAL = "conformational"
    COMPOSITIONAL_OBLIGATE = "compositional_obligate"
    UNCLASSIFIED = "unclassified"


class PartnerClass(Enum):
    """
    Partner recommendations come in two kinds that require different evidence
    and must never be conflated.
    """
    STRUCTURAL_OBLIGATE = "structural_obligate"   # part of the functional unit; structural evidence
    PHARMACOLOGICAL = "pharmacological"           # distinct receptors, converging pathway; pathway evidence


@dataclass
class NativeContact:
    """One interaction the segment makes in its native setting."""
    partner_description: str
    residues_involved: List[int] = field(default_factory=list)
    contact_class: ContactClass = ContactClass.UNCLASSIFIED
    classification_basis: str = ""
    literature_check: str = ""
    recommended_response: str = ""
    claim: Optional[Claim] = None


@dataclass
class PartnerRecommendation:
    """
    A recommendation to supply a second chain.

    Terminology note enforced throughout: this is a PARTNER PEPTIDE forming a
    BINARY COMPLEX. It is never a "dipeptide" — that word means a two-residue
    peptide and using it here would describe a completely different molecule.
    """
    partner_name: str
    partner_class: PartnerClass
    rationale: str
    stoichiometry: str = ""
    engagement_order: str = ""
    contribution_is_geometric: bool = False
    covalent_tether_assessment: str = ""
    evidence_required: str = ""
    claim: Optional[Claim] = None


@dataclass
class ExcisionSiteFinding:
    """
    Result of the automatic excision-site check.

    An internal segment cut out of a parent protein acquires a free alpha-amino
    group and a free alpha-carboxyl that the native segment never had. Those
    non-native terminal charges sit at positions that in the parent were neutral
    amide bonds, and they change local electrostatics, exopeptidase
    susceptibility, and often conformation. This is evaluated before any other
    move because it is an artefact of excision rather than a property of the
    sequence.
    """
    is_internal_fragment: bool
    n_terminal_artifact: str = ""
    c_terminal_artifact: str = ""
    recommended_n_cap: str = ""
    recommended_c_cap: str = ""
    priority_note: str = ""


@dataclass
class NativeContextResult:
    peptide: str
    parent_protein: str = ""
    parent_accession: str = ""
    native_segment_range: str = ""
    retrieval_status: str = ""
    structure_status: str = ""
    contacts: List[NativeContact] = field(default_factory=list)
    ptms: List[NativeContact] = field(default_factory=list)
    partners: List[PartnerRecommendation] = field(default_factory=list)
    excision: Optional[ExcisionSiteFinding] = None
    # Contacts classified under the section 7 taxonomy, with citations. Kept
    # separate from `contacts` because those are structural enumerations and
    # these are literature-asserted classifications with design consequences.
    classified_contacts: List = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    data_notes: List[str] = field(default_factory=list)


class NativeContextAnalyzer:
    """
    Places a peptide in its native context and classifies what excision cost it.

    Parent retrieval and structural contact enumeration require UniProt and PDB
    access, which is not wired up in this build. What runs unconditionally is the
    excision-site check, which is sequence-level, plus the classification and
    recommendation logic, which is exercised over whatever contacts are supplied.
    """

    # Response policy per contact class. Kept as data so the CONFORMATIONAL ->
    # constraint and COMPOSITIONAL -> partner distinction is enforced in one
    # place and cannot drift between call sites.
    RESPONSE_POLICY: Dict[ContactClass, str] = {
        ContactClass.SCAFFOLDING: (
            "Removable. Verify against a published truncation or alanine scan before "
            "relying on this: scaffolding is the class most often assumed rather than shown."
        ),
        ContactClass.PROTECTIVE: (
            "Removable only with a functional replacement. Recommend the replacement "
            "explicitly — removing protection without substituting for it trades a stable "
            "molecule for a degraded one."
        ),
        ContactClass.CONFORMATIONAL: (
            "Recommend an INTRAMOLECULAR CONSTRAINT (staple, lactam bridge, or "
            "macrocyclisation) to hold the bioactive fold. Do NOT recommend a partner "
            "peptide: the partner was supplying geometry the molecule can supply itself."
        ),
        ContactClass.COMPOSITIONAL_OBLIGATE: (
            "Recommend the PARTNER PEPTIDE. The partner participates in the interface or "
            "the mechanism and cannot be replaced by rigidifying this chain."
        ),
    }

    def check_excision_site(
        self, sequence: str, is_internal_fragment: bool = True
    ) -> ExcisionSiteFinding:
        """
        Flag non-native terminal charges introduced by excision.

        Runs automatically and is evaluated before any other move.
        """
        if not is_internal_fragment:
            return ExcisionSiteFinding(
                is_internal_fragment=False,
                priority_note=(
                    "Peptide is not an internal fragment, so its termini are native and "
                    "no excision artefact applies."
                ),
            )

        seq = sequence.upper()
        first, last = (seq[0] if seq else ""), (seq[-1] if seq else "")

        return ExcisionSiteFinding(
            is_internal_fragment=True,
            n_terminal_artifact=(
                f"Free alpha-amino group at {first}1, positively charged at physiological pH. "
                f"In the parent this position was a neutral amide bond. The added positive "
                f"charge alters local electrostatics and creates an aminopeptidase substrate."
            ),
            c_terminal_artifact=(
                f"Free alpha-carboxyl at {last}{len(seq)}, negatively charged at physiological "
                f"pH, likewise non-native, and a carboxypeptidase substrate."
            ),
            recommended_n_cap=(
                "N-terminal acetylation — restores the neutral amide, removes the "
                "aminopeptidase site, and is synthetically trivial."
            ),
            recommended_c_cap=(
                "C-terminal amidation — restores neutrality and removes the "
                "carboxypeptidase site. Note that many native peptide hormones are "
                "amidated in vivo, so this may also restore a native feature rather "
                "than merely masking an artefact."
            ),
            priority_note=(
                "Evaluate terminal capping BEFORE any other move. These charges are "
                "artefacts of excision rather than properties of the sequence, and leaving "
                "them in place means every subsequent optimisation is being performed on a "
                "molecule that differs from the native segment in two places nobody chose."
            ),
        )

    def classify_contact(
        self,
        partner_description: str,
        residues: List[int],
        evidence: str,
        contact_class: ContactClass,
    ) -> NativeContact:
        """Attach the class-appropriate response to an enumerated contact."""
        return NativeContact(
            partner_description=partner_description,
            residues_involved=residues,
            contact_class=contact_class,
            classification_basis=evidence,
            literature_check=(
                "Literature check for an existing truncation or alanine scan is not wired up "
                "in this build; classification rests on the supplied evidence only."
            ),
            recommended_response=self.RESPONSE_POLICY.get(contact_class, "Unclassified."),
            claim=Claim.inferred(
                f"Contact with {partner_description} classified as {contact_class.value}",
                inference_step=evidence,
            ),
        )

    def recommend_partner(
        self,
        partner_name: str,
        partner_class: PartnerClass,
        rationale: str,
        stoichiometry: str = "not established",
        engagement_order: str = "not established",
        contribution_is_geometric: bool = False,
    ) -> PartnerRecommendation:
        """
        Emit a partner-peptide recommendation with its class made explicit.

        A structural/obligate partner and a pharmacological partner are different
        claims resting on different evidence. Conflating them lets structural
        language ("part of the functional unit") attach to what is really a
        pathway-level observation.
        """
        if partner_class is PartnerClass.STRUCTURAL_OBLIGATE:
            evidence_required = (
                "Structural evidence: a complex structure, crosslinking, or mutagenesis "
                "showing the partner participates in the interface or mechanism."
            )
            tether = (
                "Evaluate covalent tethering of the two chains into a single entity as an "
                "alternative to co-formulation. A tether removes the stoichiometry and "
                "co-delivery problem and pre-pays the association entropy; it requires a "
                "linker long enough to span the native geometry without strain, which is "
                "itself an optimisation target."
            )
        else:
            evidence_required = (
                "Pathway-level evidence: the two agents act on distinct receptors converging "
                "on a shared pathway node. Structural evidence is neither available nor "
                "relevant to this claim."
            )
            tether = (
                "Covalent tethering is generally inappropriate for a pharmacological pair: "
                "the two agents act at distinct receptors and typically need independent "
                "distribution and dosing."
            )

        if contribution_is_geometric:
            rationale += (
                " The partner's contribution is docking/geometric, which makes the partner's "
                "own geometry an optimisation target rather than a fixed input."
            )

        return PartnerRecommendation(
            partner_name=partner_name,
            partner_class=partner_class,
            rationale=rationale,
            stoichiometry=stoichiometry,
            engagement_order=engagement_order,
            contribution_is_geometric=contribution_is_geometric,
            covalent_tether_assessment=tether,
            evidence_required=evidence_required,
            claim=Claim.inferred(
                f"Partner peptide recommended: {partner_name} ({partner_class.value})",
                inference_step=rationale,
            ),
        )

    def analyze(self, sequence: str, is_internal_fragment: bool = True,
                peptide_name: str = "") -> NativeContextResult:
        """
        Run the native-context analysis.

        Parent and structure retrieval are unavailable here, so the result
        carries the excision check plus an explicit statement of what could not
        be looked up, rather than an empty contact list that would read as
        "no contacts found".
        """
        # Golden-case contacts for this peptide, if it is one the seed set covers.
        # These come from the literature rather than from a structure lookup, so
        # they arrive even though PDB access is unavailable -- which is the
        # point: the contacts this module exists to catch are the ones that are
        # invisible in a sequence and would otherwise be omitted entirely.
        classified = GoldenCases.for_peptide(peptide_name) if peptide_name else []

        result = NativeContextResult(
            peptide=sequence,
            classified_contacts=classified,
            retrieval_status=(
                "Parent protein NOT retrieved: UniProt search is not wired up in this build. "
                "Whether this peptide is a fragment of a larger protein, and of which, has "
                "not been established."
            ),
            structure_status=(
                "Native structure NOT retrieved: PDB access is not wired up in this build. "
                "No contacts could be enumerated, so the contact list below is empty because "
                "nothing was looked at — not because the segment makes no contacts."
            ),
            excision=self.check_excision_site(sequence, is_internal_fragment),
        )

        result.assumptions.append(Assumption(
            kind=AssumptionKind.NATIVE_CONTEXT,
            statement=(
                f"Treated as an internal fragment of a larger parent protein"
                if is_internal_fragment else
                "Treated as a complete native peptide with native termini"
            ),
            impact_if_wrong=(
                "If this is actually a complete native peptide, the terminal capping "
                "recommendation is unnecessary and may remove a functional free terminus. "
                "If it is a fragment and was treated as complete, the excision artefacts go "
                "uncorrected and confound every downstream measurement."
            ),
        ))

        result.assumptions.append(Assumption(
            kind=AssumptionKind.RECEPTOR_ACCESSORY_SUBUNITS,
            statement=(
                "No receptor accessory subunits assumed. For families where accessory "
                "proteins determine ligand selectivity — RAMPs for the calcitonin receptor "
                "family being the clearest case — the receptor identity alone does not "
                "determine the pharmacology."
            ),
            impact_if_wrong=(
                "Selectivity and potency predictions made against a receptor without its "
                "accessory subunit can be wrong about which ligand the receptor even prefers."
            ),
        ))

        result.data_notes.append(
            "Native-context analysis is structural in nature and this build has no structural "
            "input. The excision-site check below is sequence-level and did run; contact "
            "classification and partner recommendation are available as machinery but have "
            "no retrieved contacts to operate on."
        )

        return result
