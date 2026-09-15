"""
Transformation workflow.

Generates discrete, reviewable transformations from the Tier 0 physics findings
and the excision-site check, scores each against the full objective vector, and
refuses to emit any transformation that violates the output contract.

Objective assessments are deliberately sparse. Most axes cannot be assessed from
sequence alone, and those are recorded as NOT_ASSESSED with a reason rather than
filled with a neutral placeholder — an unexamined axis must not read as a clean
bill of health.
"""

import logging
from typing import Dict, List, Optional

from peptide_suite.core.epistemics import (
    Assumption, AssumptionKind, Claim, check_corpus_leakage,
)
from peptide_suite.core.electrostatics import compute_profile
from peptide_suite.core.native_context import NativeContextAnalyzer
from peptide_suite.core.physics_tiers import LiabilityClass, PhysicsStack
from peptide_suite.core.preorganization import PreOrganizationAnalyzer
from peptide_suite.core.transformations import (
    DEFAULT_WEIGHTS, Direction, MoveType, Objective, ObjectiveDelta,
    Transformation,
)

logger = logging.getLogger(__name__)

NO_STRUCTURE = (
    "Requires a receptor structure or binding data; neither is available, so this axis "
    "was not assessed rather than assumed neutral."
)
NO_ASSAY = (
    "Requires experimental measurement; no assay data is available, so this axis was "
    "not assessed rather than estimated."
)


def _computed(text: str, method: str, tier: int) -> Claim:
    return Claim.computed(text, method=method, tier=tier)


def _inferred(text: str, step: str) -> Claim:
    return Claim.inferred(text, inference_step=step)


def _delta(objective, direction, magnitude, claim, basis="") -> ObjectiveDelta:
    return ObjectiveDelta(objective=objective, direction=direction,
                          magnitude=magnitude, claim=claim, basis=basis)


def _unassessed(*objectives, reason=NO_STRUCTURE) -> List[ObjectiveDelta]:
    return [ObjectiveDelta.not_assessed(o, reason) for o in objectives]


class TransformWorkflow:
    """Produces a ranked transformation list for a peptide."""

    def __init__(self):
        self.physics = PhysicsStack()
        self.native = NativeContextAnalyzer()
        self.preorg = PreOrganizationAnalyzer()

    def run(
        self,
        sequence: str,
        formulation_ph: float = 7.4,
        is_internal_fragment: bool = True,
        weights: Optional[Dict[Objective, float]] = None,
    ) -> Dict:
        seq = sequence.upper()
        weights = weights or DEFAULT_WEIGHTS

        physics = self.physics.run(seq, ph=formulation_ph)
        native = self.native.analyze(seq, is_internal_fragment=is_internal_fragment)
        # Titrated across the compartments the policy names, rather than
        # collapsed to a pI: which residue carries the charge, and whether that
        # changes between compartments, is the part a design decision turns on.
        # Termini are always counted: an uncapped internal fragment really does
        # carry them, which is the whole reason they are worth flagging. Whether
        # they are a property of the peptide or an artefact of excision is a
        # separate statement, made below rather than by dropping the charge.
        electrostatics = compute_profile(seq, include_termini=True)
        if is_internal_fragment:
            electrostatics.notes.insert(0, (
                "The terminal charges counted here are artefacts of excision, not properties "
                "of this sequence in its native setting. Capping changes them; evaluate that "
                "before reading anything else into the charge profile."
            ))

        transformations: List[Transformation] = []

        # The excision-site check is evaluated before any other move, because
        # the terminal charges it finds are artefacts of excision rather than
        # properties of the sequence.
        if native.excision and native.excision.is_internal_fragment:
            transformations.extend(self._terminal_capping(seq, native))

        transformations.extend(self._from_liabilities(seq, physics, formulation_ph))
        transformations.extend(self._conformational_moves(seq, physics))
        transformations.extend(self._exposure_moves(seq))

        emitted, rejected = [], []
        for t in transformations:
            problems = t.validate()
            if problems:
                rejected.append({"description": t.description, "violations": problems})
                logger.warning(f"Refusing to emit '{t.description}': {problems}")
            else:
                t.leakage_flag = check_corpus_leakage(seq, t.description + " " + t.rationale)
                emitted.append(t)

        emitted.sort(key=lambda t: t.scalarize(weights)["score"], reverse=True)

        return {
            "sequence": seq,
            "physics": physics,
            "native_context": native,
            "electrostatics": electrostatics,
            "transformations": emitted,
            "rejected": rejected,
            "weights": {o.value: weights.get(o, 0.0) for o in Objective},
            "scalarization_note": (
                "Scalarised scores below use the weights shown. They are an editorial "
                "choice, not a derived optimum: change them and the ranking changes. The "
                "objective vector on each transformation is the primary output; the scalar "
                "is a convenience for ordering."
            ),
            "comparability_warning": (
                "Scores at different coverage are NOT directly comparable. A move assessed on "
                "two of eight objectives can outscore a move assessed on seven, because the "
                "unassessed axes are excluded from its average rather than counted against it. "
                "A high score at low coverage means 'good on the little we could examine', not "
                "'better overall'. Compare the vectors, and treat coverage as part of the score."
            ),
        }

    # ---- move generators -------------------------------------------------

    def _terminal_capping(self, seq: str, native) -> List[Transformation]:
        ex = native.excision

        deltas = [
            _delta(
                Objective.PROTEOLYTIC_HALF_LIFE, Direction.IMPROVES, 0.55,
                _inferred(
                    "N-terminal acetylation and C-terminal amidation remove the free alpha-amino "
                    "and alpha-carboxyl groups that amino- and carboxypeptidases require",
                    "Exopeptidases recognise free termini. Capping both removes the recognition "
                    "element for the whole exopeptidase class, independent of sequence.",
                ),
                basis="Exopeptidase substrate removal",
            ),
            _delta(
                Objective.SYNTHESIZABILITY, Direction.IMPROVES, 0.10,
                _inferred(
                    "Both caps are routine solid-phase steps",
                    "N-acetylation is a standard capping step and C-amidation follows from "
                    "choosing a Rink amide resin, so neither adds a synthetic route problem.",
                ),
            ),
            _delta(
                Objective.SOLUBILITY, Direction.DEGRADES, 0.15,
                _computed(
                    "Removes one positive and one negative terminal charge, lowering net "
                    "charge magnitude and with it solubility",
                    method="Tier 0 net-charge calculation", tier=0,
                ),
            ),
            *_unassessed(
                Objective.POTENCY, Objective.FUNCTIONAL_SELECTIVITY,
                reason=(
                    "Terminal charges are part of the pharmacophore in several peptide "
                    "families — the free N-terminus of GLP-1 is required for receptor "
                    "activation, for instance. Whether capping costs potency here cannot be "
                    "determined without a receptor structure or an assay. " + NO_STRUCTURE
                ),
            ),
            *_unassessed(Objective.ALBUMIN_FCRN, Objective.IMMUNOGENICITY, reason=NO_ASSAY),
            *_unassessed(Objective.AGGREGATION, reason=NO_STRUCTURE),
        ]

        return [Transformation(
            move=MoveType.TERMINAL_CAPPING,
            position=None,
            description="N-terminal acetylation + C-terminal amidation",
            rationale=(
                f"{ex.priority_note} {ex.recommended_n_cap} {ex.recommended_c_cap}"
            ),
            objective_deltas=deltas,
            evidence_tier="BIOCHEMICAL_PRINCIPLE",
            physics_tier_reached=0,
            assumptions=[
                Assumption(
                    kind=AssumptionKind.NATIVE_CONTEXT,
                    statement="Peptide is an internal fragment, so both termini are excision artefacts",
                    impact_if_wrong=(
                        "If the free N-terminus is native and functionally required, "
                        "acetylation removes activity rather than a liability."
                    ),
                ),
            ],
            notes=[
                "Evaluated first by policy: these charges are artefacts of excision, so "
                "optimising anything else before addressing them optimises the wrong molecule.",
            ],
        )]

    def _from_liabilities(self, seq: str, physics: Dict, ph: float) -> List[Transformation]:
        out: List[Transformation] = []
        liabilities = physics["tier0"].data["liabilities"]

        for lab in liabilities:
            if lab.liability_class is LiabilityClass.PROTEOLYSIS:
                continue  # Handled as a backbone-constraint move below

            severity_magnitude = {"high": 0.6, "medium": 0.35, "low": 0.15}[lab.severity]

            deltas = [
                _delta(
                    Objective.PROTEOLYTIC_HALF_LIFE
                    if lab.liability_class is LiabilityClass.PROTEOLYSIS
                    else Objective.AGGREGATION,
                    Direction.IMPROVES, severity_magnitude,
                    _computed(
                        f"Removes the {lab.motif} liability at position {lab.display_position}: "
                        f"{lab.mechanism}",
                        method="Tier 0 liability motif scan", tier=0,
                    ),
                    basis=lab.mechanism,
                ),
                *_unassessed(
                    Objective.POTENCY, Objective.FUNCTIONAL_SELECTIVITY,
                    reason=(
                        f"Whether position {lab.display_position} contributes to receptor "
                        f"binding is unknown without a structure. " + NO_STRUCTURE
                    ),
                ),
                *_unassessed(
                    Objective.ALBUMIN_FCRN, Objective.IMMUNOGENICITY, reason=NO_ASSAY
                ),
                _delta(
                    Objective.SYNTHESIZABILITY, Direction.NEUTRAL, 0.0,
                    _inferred(
                        "Canonical-for-canonical substitution adds no synthetic difficulty",
                        "The replacement residues named are standard Fmoc building blocks.",
                    ),
                ),
                *_unassessed(Objective.SOLUBILITY, reason=NO_ASSAY),
            ]

            out.append(Transformation(
                move=MoveType.LIABILITY_REMOVAL,
                position=lab.position,
                description=f"{lab.motif} at position {lab.display_position} — {lab.mitigation}",
                rationale=(
                    f"{lab.mechanism}. Severity assessed {lab.severity} from the motif class. "
                    f"Mitigation: {lab.mitigation}"
                ),
                objective_deltas=deltas,
                evidence_tier="DIRECT_EXPERIMENTAL",
                physics_tier_reached=0,
                notes=[
                    "Chemical-degradation liabilities are formulation and shelf-life issues as "
                    "much as in-vivo ones; they show up in stability studies before they show "
                    "up in pharmacokinetics.",
                ],
            ))

        return out

    def _conformational_moves(self, seq: str, physics: Dict) -> List[Transformation]:
        """Backbone-constraint and macrocyclisation moves, each with a pre-organization proxy."""
        out: List[Transformation] = []

        # Aib at the DPP-4 P1 position, when there is one.
        dpp4 = [
            lab for lab in physics["tier0"].data["liabilities"]
            if lab.liability_class is LiabilityClass.PROTEOLYSIS
        ]
        for lab in dpp4:
            pos = lab.position
            proxy = self.preorg.for_substitution(seq, pos, "Aib", constraint_kind="AIB")

            deltas = [
                _delta(
                    Objective.PROTEOLYTIC_HALF_LIFE, Direction.IMPROVES, 0.85,
                    _computed(
                        f"alpha,alpha-disubstitution at position {pos + 1} removes the DPP-4 P1 "
                        f"recognition element; the enzyme cannot accommodate a quaternary "
                        f"alpha-carbon at this position",
                        method="Tier 0 P1 specificity scan + documented Aib behaviour", tier=0,
                    ),
                    basis="DPP-4 P1 pocket cannot accept alpha,alpha-disubstitution",
                ),
                _delta(
                    Objective.SYNTHESIZABILITY, Direction.DEGRADES, 0.30,
                    _inferred(
                        "Aib couples sluggishly; the residue following it is also hindered",
                        "alpha,alpha-disubstituted residues are sterically hindered at both the "
                        "coupling and the subsequent acylation step, commonly requiring stronger "
                        "activation and longer couplings.",
                    ),
                ),
                *_unassessed(
                    Objective.POTENCY, Objective.FUNCTIONAL_SELECTIVITY,
                    reason=(
                        f"Position {pos + 1} sits in the N-terminal region, which in class B "
                        f"GPCR ligands is typically the activation domain rather than the "
                        f"affinity domain. A constraint here could plausibly help or hurt "
                        f"activation and the sign cannot be determined from sequence. "
                        + NO_STRUCTURE
                    ),
                ),
                *_unassessed(
                    Objective.IMMUNOGENICITY,
                    reason=(
                        "Non-canonical residues alter T-cell epitope processing in ways that "
                        "require an epitope-prediction pass over the modified sequence, which "
                        "no canonical-residue predictor covers. " + NO_ASSAY
                    ),
                ),
                *_unassessed(Objective.ALBUMIN_FCRN, Objective.SOLUBILITY,
                             Objective.AGGREGATION, reason=NO_ASSAY),
            ]

            out.append(Transformation(
                move=MoveType.BACKBONE_CONSTRAINT,
                position=pos,
                description=f"{seq[pos]}{pos + 1} -> Aib (alpha-aminoisobutyric acid)",
                rationale=(
                    f"Position {pos + 1} is the DPP-4 P1 residue. Substituting a quaternary "
                    f"alpha-carbon removes the cleavage site and simultaneously rigidifies the "
                    f"backbone. {proxy.render()}"
                ),
                objective_deltas=deltas,
                evidence_tier="DIRECT_EXPERIMENTAL",
                preorganization=proxy,
                physics_tier_reached=0,
                assumptions=[
                    Assumption(
                        kind=AssumptionKind.CONFORMER,
                        statement=(
                            "Helicity delta computed from a sequence-level propensity scale, "
                            "assuming a helical reference state for the affected region"
                        ),
                        impact_if_wrong=(
                            "If the region is not helical in the bioactive conformation, the "
                            "helicity delta is not the relevant pre-organization measure and "
                            "a basin argument should replace it."
                        ),
                    ),
                ],
                notes=[
                    "Aib is non-canonical: Tier 4 QM parameterisation is mandatory before any "
                    "mechanics calculation involving it, and did not run. Predictions here are "
                    "sequence- and literature-level only.",
                ],
            ))

        # Staple, when the sequence shows an amphipathic helical segment worth stabilising.
        wmoment = physics["tier0"].data["windowed_hydrophobic_moment"]
        if wmoment["max_moment"] >= 0.30:
            start = wmoment["start"]
            proxy = self.preorg.for_macrocyclization("STAPLE", span=4)

            deltas = [
                _delta(
                    Objective.PROTEOLYTIC_HALF_LIFE, Direction.IMPROVES, 0.45,
                    _inferred(
                        "Stapled helices resist proteolysis: proteases require an extended "
                        "backbone in the active site and a constrained helix cannot adopt it",
                        "This is a general property of helix constraints rather than a "
                        "sequence-specific prediction.",
                    ),
                ),
                _delta(
                    Objective.POTENCY, Direction.IMPROVES, 0.35,
                    _inferred(
                        f"Pre-organising the amphipathic segment at {wmoment['display_start']}-"
                        f"{wmoment['display_start'] + wmoment['window'] - 1} reduces the "
                        f"conformational entropy paid on binding",
                        "The benefit is entropic. It applies only if this segment is helical in "
                        "the bound state, which is assumed rather than established here.",
                    ),
                ),
                _delta(
                    Objective.SYNTHESIZABILITY, Direction.DEGRADES, 0.45,
                    _inferred(
                        "Requires two non-canonical olefinic residues and an on-resin ring-closing "
                        "metathesis step",
                        "Stapling adds a metal-catalysed step and a purification burden beyond "
                        "routine solid-phase synthesis.",
                    ),
                ),
                _delta(
                    Objective.AGGREGATION, Direction.DEGRADES, 0.30,
                    _inferred(
                        "The hydrocarbon staple adds a hydrophobic surface patch",
                        "Stapled peptides frequently show reduced solubility and increased "
                        "aggregation relative to their linear parents.",
                    ),
                ),
                *_unassessed(Objective.FUNCTIONAL_SELECTIVITY, reason=NO_STRUCTURE),
                *_unassessed(Objective.ALBUMIN_FCRN, Objective.SOLUBILITY,
                             Objective.IMMUNOGENICITY, reason=NO_ASSAY),
            ]

            out.append(Transformation(
                move=MoveType.CYCLIZATION_STAPLING,
                position=start,
                description=(
                    f"i,i+4 hydrocarbon staple across residues "
                    f"{wmoment['display_start']}-{wmoment['display_start'] + 4}"
                ),
                rationale=(
                    f"Windowed hydrophobic moment peaks at {wmoment['max_moment']:.2f} over "
                    f"{wmoment['segment']} (residues {wmoment['display_start']}-"
                    f"{wmoment['display_start'] + wmoment['window'] - 1}), indicating an "
                    f"amphipathic helical segment. {proxy.render()}"
                ),
                objective_deltas=deltas,
                evidence_tier="BIOCHEMICAL_PRINCIPLE",
                preorganization=proxy,
                physics_tier_reached=0,
                assumptions=[
                    Assumption(
                        kind=AssumptionKind.CONFORMER,
                        statement=(
                            "Assumes the identified segment is helical in the bioactive "
                            "conformation, inferred from its hydrophobic moment rather than observed"
                        ),
                        impact_if_wrong=(
                            "Stapling a segment that is not helical when bound locks out the "
                            "bioactive conformation instead of stabilising it, which would "
                            "abolish activity rather than improve it."
                        ),
                    ),
                ],
                notes=[
                    "Staple position and linker length are themselves optimisation variables; "
                    "the i,i+4 placement here is a starting point, not a determined optimum.",
                ],
            ))

        return out

    def _exposure_moves(self, seq: str) -> List[Transformation]:
        """Lipidation for albumin engagement — the canonical potency-for-exposure trade."""
        lys = [i for i, aa in enumerate(seq) if aa == "K"]
        if not lys:
            return []

        site = lys[len(lys) // 2]
        proxy_note = (
            "Acylation is not a conformational constraint, so no pre-organization proxy applies."
        )

        deltas = [
            _delta(
                Objective.ALBUMIN_FCRN, Direction.IMPROVES, 0.85,
                _inferred(
                    "A C18 diacid via a gamma-Glu-2xOEG linker binds serum albumin reversibly, "
                    "and albumin's FcRn-mediated recycling extends the circulating half-life",
                    "This is the established mechanism behind acylated peptide therapeutics; the "
                    "magnitude for this specific peptide is not predicted.",
                ),
            ),
            _delta(
                Objective.PROTEOLYTIC_HALF_LIFE, Direction.IMPROVES, 0.50,
                _inferred(
                    "Albumin association sterically shields the peptide from circulating proteases",
                    "Indirect: the protection comes from the bound state, so it scales with the "
                    "albumin-bound fraction rather than from any change to the sequence.",
                ),
            ),
            _delta(
                Objective.POTENCY, Direction.DEGRADES, 0.45,
                _inferred(
                    "Acylated analogs characteristically lose in-vitro potency relative to their "
                    "unacylated parents, from steric interference and from the fact that the "
                    "albumin-bound fraction is not free to engage the receptor",
                    "In-vitro potency loss of roughly one to two orders of magnitude is typical "
                    "for this chemistry and is accepted in exchange for exposure.",
                ),
            ),
            _delta(
                Objective.SOLUBILITY, Direction.DEGRADES, 0.35,
                _inferred(
                    "A C18 chain is a large hydrophobic addition",
                    "The diacid terminus mitigates this relative to a plain fatty acid, but the "
                    "net effect on aqueous solubility remains unfavourable.",
                ),
            ),
            _delta(
                Objective.AGGREGATION, Direction.DEGRADES, 0.40,
                _inferred(
                    "Acyl chains drive self-association; some acylated peptides form micelles or "
                    "oligomers at formulation concentrations",
                    "For certain products this self-association is engineered deliberately as a "
                    "depot mechanism, so the sign of this effect depends on the intended "
                    "delivery route.",
                ),
            ),
            _delta(
                Objective.SYNTHESIZABILITY, Direction.DEGRADES, 0.35,
                _inferred(
                    "Requires orthogonal Lys side-chain protection, linker assembly, and "
                    "regioselective acylation",
                    "Multiple lysines make regioselectivity the dominant difficulty; directing "
                    "acylation to one site often requires substituting the others.",
                ),
            ),
            *_unassessed(Objective.FUNCTIONAL_SELECTIVITY, reason=NO_STRUCTURE),
            *_unassessed(
                Objective.IMMUNOGENICITY,
                reason=(
                    "Acylation can alter antigen processing and presentation. " + NO_ASSAY
                ),
            ),
        ]

        t = Transformation(
            move=MoveType.LIPIDATION,
            position=site,
            description=(
                f"C18 diacid acylation at Lys{site + 1} via a gamma-Glu-2xOEG linker"
            ),
            rationale=(
                f"Introduces reversible albumin binding for half-life extension. {proxy_note}"
            ),
            objective_deltas=deltas,
            evidence_tier="HOMOLOG_EXPERIMENTAL",
            physics_tier_reached=0,
            tradeoff_label=(
                "EXPOSURE-FOR-POTENCY TRADE. This move degrades in-vitro potency and improves "
                "circulating exposure. It is not a potency improvement and must not be presented "
                "as one: whether it is a net gain depends on whether the dosing interval or the "
                "receptor occupancy is the binding constraint for this programme."
            ),
            assumptions=[
                Assumption(
                    kind=AssumptionKind.FORMULATION_CONDITIONS,
                    statement=(
                        "Assumes systemic administration where albumin binding is available and "
                        "a long dosing interval is desirable"
                    ),
                    impact_if_wrong=(
                        "For a local, topical, or acute-onset indication, extended exposure is "
                        "not a benefit and the potency cost buys nothing."
                    ),
                ),
                Assumption(
                    kind=AssumptionKind.OTHER,
                    statement=f"Assumes Lys{site + 1} is acylatable without blocking the binding interface",
                    impact_if_wrong=(
                        "If this lysine contacts the receptor, acylating it costs far more "
                        "potency than the typical figure above."
                    ),
                ),
            ],
            notes=[
                f"{len(lys)} lysine(s) present. With more than one, regioselective acylation "
                f"usually requires substituting the others (commonly Lys->Arg) so the acyl group "
                f"lands at a single defined site.",
            ],
        )

        return [t]
