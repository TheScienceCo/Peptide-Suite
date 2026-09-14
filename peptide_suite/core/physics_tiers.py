"""
Tiered physics layer.

    Tier 0  always      composition, helical wheel, hydrophobic moment,
                        net-charge-vs-pH curve, liability motif scan
    Tier 1  always      per-residue side-chain pKa. Pocket perturbation requires
                        a structure; without one the free-residue value is
                        reported and labelled as such.
    Tier 2  conditional conformational ensemble via enhanced sampling
    Tier 3  requires 2  FMO/PIEDA per-contact energy decomposition
    Tier 4  required    QM parameterisation for non-canonical residues
            for NCAAs

HARD RULE, enforced structurally rather than by convention: a Tier 3 number is
never produced from a single unvalidated geometry. `Tier3FMO.run` raises unless
it is handed a Tier 2 ensemble. The rule cannot be satisfied by a caller
promising to be careful.

Tiers 2-4 have no engine in this deployment. They return explicit
`TierUnavailable` results naming the missing dependency rather than degrading to
an approximation that would look like a computed value.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .charge_calculator import ChargeCalculator
from .epistemics import Claim

# Eisenberg consensus hydrophobicity scale — the scale the hydrophobic moment
# was defined on. Deliberately not Kyte-Doolittle, which is used elsewhere in
# this codebase for side-chain character and is not interchangeable here.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Pace & Scholtz helix propensity, kcal/mol relative to Ala = 0.
# Lower is more helix-favouring.
HELIX_PROPENSITY = {
    "A": 0.00, "L": 0.21, "R": 0.21, "M": 0.24, "K": 0.26,
    "Q": 0.39, "E": 0.40, "I": 0.41, "W": 0.49, "S": 0.50,
    "Y": 0.53, "F": 0.54, "H": 0.61, "V": 0.61, "N": 0.65,
    "T": 0.66, "C": 0.68, "D": 0.69, "G": 1.00, "P": 3.16,
}

ALPHA_HELIX_ANGLE_DEG = 100.0


class LiabilityClass(Enum):
    CHEMICAL_DEGRADATION = "chemical_degradation"
    OXIDATION = "oxidation"
    PROTEOLYSIS = "proteolysis"
    DISULFIDE_SCRAMBLING = "disulfide_scrambling"


@dataclass
class Liability:
    """A sequence-encoded degradation or cleavage hotspot."""
    motif: str
    position: int              # 0-indexed start
    liability_class: LiabilityClass
    mechanism: str
    severity: str              # high | medium | low
    mitigation: str
    claim: Claim = None

    @property
    def display_position(self) -> int:
        return self.position + 1


@dataclass
class TierResult:
    """Outcome of running one physics tier."""
    tier: int
    available: bool
    data: Dict = field(default_factory=dict)
    unavailable_reason: str = ""
    missing_dependency: str = ""
    illustrative_only: bool = False

    def require_available(self, what: str) -> None:
        if not self.available:
            raise RuntimeError(
                f"{what} requires Tier {self.tier}, which did not run: {self.unavailable_reason}"
            )


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------

class Tier0Sequence:
    """Sequence-computable properties. Always runs; needs nothing external."""

    TIER = 0

    def __init__(self):
        self.charge_calc = ChargeCalculator()

    def hydrophobic_moment(self, sequence: str, angle_deg: float = ALPHA_HELIX_ANGLE_DEG) -> float:
        """
        Eisenberg hydrophobic moment, normalised per residue.

        muH = |sum_i H_i * exp(i * delta * n)| / N

        High muH indicates an amphipathic helix: hydrophobic and hydrophilic
        faces segregate when the backbone is helical.
        """
        seq = sequence.upper()
        if not seq:
            return 0.0

        delta = math.radians(angle_deg)
        sin_sum = sum(EISENBERG.get(aa, 0.0) * math.sin(delta * i) for i, aa in enumerate(seq))
        cos_sum = sum(EISENBERG.get(aa, 0.0) * math.cos(delta * i) for i, aa in enumerate(seq))

        return math.hypot(sin_sum, cos_sum) / len(seq)

    def windowed_hydrophobic_moment(
        self, sequence: str, window: int = 11, angle_deg: float = ALPHA_HELIX_ANGLE_DEG
    ) -> Dict:
        """
        Maximum hydrophobic moment over a sliding window.

        Whole-sequence muH is misleading for anything longer than a single
        helical segment: an amphipathic stretch and a disordered stretch average
        each other out. The convention is an 11-residue window (three helical
        turns), reporting the maximum and where it falls.
        """
        seq = sequence.upper()
        if len(seq) < window:
            return {
                "max_moment": round(self.hydrophobic_moment(seq, angle_deg), 4),
                "window": len(seq),
                "start": 0,
                "display_start": 1,
                "segment": seq,
                "note": f"Sequence shorter than the {window}-residue window; computed whole.",
            }

        best = max(
            (
                (self.hydrophobic_moment(seq[i:i + window], angle_deg), i)
                for i in range(len(seq) - window + 1)
            ),
            key=lambda pair: pair[0],
        )
        moment, start = best

        return {
            "max_moment": round(moment, 4),
            "window": window,
            "start": start,
            "display_start": start + 1,
            "segment": seq[start:start + window],
            "note": (
                f"Maximum over {window}-residue windows. Values above ~0.35 indicate a "
                f"strongly amphipathic helical segment."
            ),
        }

    def mean_hydrophobicity(self, sequence: str) -> float:
        seq = sequence.upper()
        if not seq:
            return 0.0
        return sum(EISENBERG.get(aa, 0.0) for aa in seq) / len(seq)

    def helical_wheel(self, sequence: str, angle_deg: float = ALPHA_HELIX_ANGLE_DEG) -> List[Dict]:
        """Angular position and hydrophobicity of each residue around a helix."""
        return [
            {
                "position": i,
                "display_position": i + 1,
                "residue": aa,
                "angle_deg": (i * angle_deg) % 360.0,
                "hydrophobicity": EISENBERG.get(aa.upper(), 0.0),
            }
            for i, aa in enumerate(sequence.upper())
        ]

    def charge_vs_ph_curve(
        self, sequence: str, lo: float = 1.0, hi: float = 14.0, step: float = 0.25
    ) -> List[Tuple[float, float]]:
        """Net charge across the pH range, from Henderson-Hasselbalch per residue."""
        curve = []
        ph = lo
        while ph <= hi + 1e-9:
            curve.append((round(ph, 2), round(self.charge_calc.net_charge(sequence, ph=ph), 4)))
            ph += step
        return curve

    def isoelectric_point(self, sequence: str) -> float:
        """
        Whole-molecule pI: the pH at which net charge crosses zero.

        pI is a property of the molecule, not of a residue. Residues have pKa
        values; asking for "the pI of residue 7" is a category error, and this
        module deliberately offers no per-residue equivalent.
        """
        lo, hi = 0.0, 14.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if self.charge_calc.net_charge(sequence, ph=mid) > 0:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 2)

    def scan_liabilities(self, sequence: str) -> List[Liability]:
        """Scan for sequence-encoded degradation and cleavage hotspots."""
        seq = sequence.upper()
        found: List[Liability] = []

        def add(motif, pos, cls, mechanism, severity, mitigation):
            found.append(Liability(
                motif=motif, position=pos, liability_class=cls, mechanism=mechanism,
                severity=severity, mitigation=mitigation,
                claim=Claim.computed(
                    f"{motif} at position {pos + 1}: {mechanism}",
                    method="Tier 0 liability motif scan", tier=0,
                ),
            ))

        # DPP-4: cleaves the N-terminal dipeptide when P1 (residue 2) is Pro or Ala
        if len(seq) >= 2 and seq[1] in ("P", "A"):
            add(f"X{seq[1]} (N-terminal)", 1, LiabilityClass.PROTEOLYSIS,
                "Dipeptidyl peptidase-4 removes the N-terminal dipeptide when position 2 is Pro or Ala",
                "high",
                "Aib or other alpha,alpha-disubstituted residue at position 2; N-terminal capping")

        for i in range(len(seq) - 1):
            pair = seq[i:i + 2]

            if pair == "DP":
                add("Asp-Pro", i, LiabilityClass.CHEMICAL_DEGRADATION,
                    "Acid-labile peptide bond; hydrolyses readily at low pH",
                    "high",
                    "Substitute Asp, or avoid low-pH formulation and processing steps")

            elif pair == "DG":
                add("Asp-Gly", i, LiabilityClass.CHEMICAL_DEGRADATION,
                    "Succinimide intermediate leading to isoaspartate formation and backbone isomerisation",
                    "high",
                    "Asp->Glu preserves charge while blocking succinimide formation")

            elif pair == "NG":
                add("Asn-Gly", i, LiabilityClass.CHEMICAL_DEGRADATION,
                    "Fastest deamidation context; Asn -> Asp/isoAsp with charge change",
                    "high",
                    "Asn->Gln or Asn->Ser; or substitute the following Gly to slow the reaction")

            elif pair in ("NS", "NT", "NH", "NA"):
                add(f"Asn-{pair[1]}", i, LiabilityClass.CHEMICAL_DEGRADATION,
                    "Deamidation-prone context, slower than Asn-Gly but still significant",
                    "medium",
                    "Asn->Gln, or substitute the following residue")

        for i, aa in enumerate(seq):
            if aa == "M":
                add("Met", i, LiabilityClass.OXIDATION,
                    "Methionine oxidises to the sulfoxide under oxidative stress and on storage",
                    "medium",
                    "Met->Leu or Nle preserves bulk and hydrophobicity without the thioether")
            elif aa == "W":
                add("Trp", i, LiabilityClass.OXIDATION,
                    "Tryptophan is photo- and oxidation-sensitive",
                    "medium",
                    "Trp->Phe if the indole nitrogen is not required for binding")

        if seq.count("C") % 2 == 1:
            add("unpaired Cys", seq.index("C"), LiabilityClass.DISULFIDE_SCRAMBLING,
                f"Odd cysteine count ({seq.count('C')}): at least one free thiol available for "
                f"intermolecular scrambling and disulfide-mediated aggregation",
                "high",
                "Cap the free thiol, pair it, or substitute Ser/Ala")

        if seq and seq[0] in ("Q", "E"):
            add(f"N-terminal {seq[0]}", 0, LiabilityClass.CHEMICAL_DEGRADATION,
                "N-terminal Gln/Glu cyclises to pyroglutamate, removing the N-terminal charge",
                "medium" if seq[0] == "Q" else "low",
                "N-terminal acetylation, or substitute the first residue")

        return found

    def run(self, sequence: str, formulation_ph: float = 7.4) -> TierResult:
        seq = sequence.upper()
        return TierResult(
            tier=self.TIER,
            available=True,
            data={
                "length": len(seq),
                "hydrophobic_moment": round(self.hydrophobic_moment(seq), 4),
                "windowed_hydrophobic_moment": self.windowed_hydrophobic_moment(seq),
                "mean_hydrophobicity": round(self.mean_hydrophobicity(seq), 4),
                "helical_wheel": self.helical_wheel(seq),
                "charge_vs_ph": self.charge_vs_ph_curve(seq),
                "isoelectric_point": self.isoelectric_point(seq),
                "net_charge_at_formulation_ph": round(
                    self.charge_calc.net_charge(seq, ph=formulation_ph), 3
                ),
                "formulation_ph": formulation_ph,
                "liabilities": self.scan_liabilities(seq),
            },
        )


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------

class Tier1PKa:
    """
    Per-residue side-chain pKa.

    The tier is defined as pKa *with pocket perturbation* — a PROPKA pass over a
    structure, escalating to constant-pH MD or QM/MM for residues in proposed
    ionic interactions. No structure is available here, so what is reported is
    the free-residue value, labelled as unperturbed. A buried or salt-bridged
    residue can sit multiple pH units away from its free-solution pKa, so this
    value must not be used to assert an ionic interaction.
    """

    TIER = 1

    def __init__(self):
        self.charge_calc = ChargeCalculator()

    def run(self, sequence: str, ph: float = 7.4, structure_available: bool = False) -> TierResult:
        seq = sequence.upper()

        residues = []
        for i, aa in enumerate(seq):
            state = self.charge_calc.charge_at_ph(aa, ph=ph, position=i)
            if not state.is_ionizable:
                continue
            residues.append({
                "position": i,
                "display_position": i + 1,
                "residue": aa,
                "free_residue_pka": state.pka,
                "pocket_perturbed_pka": None,
                "charge_at_ph": round(state.effective_charge, 3),
                "note": (
                    "Free-residue pKa. NOT pocket-perturbed: no structure was supplied, "
                    "so burial, local dielectric, and salt-bridge partners are unaccounted for. "
                    "A buried or ion-paired residue can shift several pH units from this value."
                ),
            })

        if structure_available:
            return TierResult(
                tier=self.TIER, available=False,
                unavailable_reason=(
                    "A structure was supplied but no PROPKA pass is wired up in this build."
                ),
                missing_dependency="propka",
                data={"residues": residues},
            )

        return TierResult(
            tier=self.TIER,
            available=True,
            illustrative_only=True,
            data={
                "residues": residues,
                "pocket_perturbation_applied": False,
                "escalation_required_for": (
                    "any residue proposed as part of an ionic interaction — those require "
                    "PROPKA over a structure, escalating to constant-pH MD or QM/MM"
                ),
            },
        )

    def assert_ionic_interaction_supportable(self, position: int) -> None:
        """
        Guard for claims that a specific residue forms an ionic interaction.

        Such a claim depends on the pocket-perturbed pKa, which is exactly what
        is unavailable without a structure. Raising here keeps the claim from
        being made on a free-solution number.
        """
        raise RuntimeError(
            f"An ionic-interaction claim at position {position + 1} requires a pocket-perturbed "
            f"pKa (PROPKA over a structure, escalating to constant-pH MD or QM/MM for the "
            f"residues involved). Only free-residue pKa is available, which cannot support "
            f"the claim. Supply a structure or drop the claim."
        )


# --------------------------------------------------------------------------
# Tiers 2-4
# --------------------------------------------------------------------------

class Tier2Ensemble:
    """Conformational ensemble via enhanced sampling, clustered to representative conformers."""

    TIER = 2

    def run(self, sequence: str) -> TierResult:
        return TierResult(
            tier=self.TIER,
            available=False,
            unavailable_reason=(
                "No molecular dynamics engine is available in this deployment, so no "
                "conformational ensemble can be generated. Without an ensemble there are no "
                "representative conformers and no populations to Boltzmann-weight over."
            ),
            missing_dependency="OpenMM/GROMACS with an enhanced-sampling protocol "
                               "(replica exchange or metadynamics)",
        )


class Tier3FMO:
    """
    FMO/PIEDA per-contact interaction energy decomposition.

    Gated on Tier 2 by construction. The single-geometry failure mode is not a
    style preference: an interaction energy computed on one unvalidated pose
    reports the energy of that pose, not of the molecule, and the difference is
    routinely larger than the effect being measured.
    """

    TIER = 3

    def run(self, ensemble: Optional[TierResult]) -> TierResult:
        if ensemble is None or not ensemble.available:
            reason = (
                "Tier 3 requires a Tier 2 conformational ensemble. "
                + (ensemble.unavailable_reason if ensemble else "Tier 2 was not run.")
                + " An FMO/PIEDA number computed on a single unvalidated geometry would "
                "describe that geometry rather than the molecule, so none is produced."
            )
            raise RuntimeError(reason)

        return TierResult(
            tier=self.TIER,
            available=False,
            unavailable_reason=(
                "No fragment molecular orbital engine is available in this deployment. "
                "Per-contact decomposition into electrostatics, exchange-repulsion, "
                "charge transfer and dispersion cannot be produced."
            ),
            missing_dependency="GAMESS or equivalent with FMO-DFTB for throughput, "
                               "FMO-MP2 for final calls, FMO-PCM where desolvation matters",
        )


class Tier4QMParameterization:
    """
    QM parameterisation of non-canonical residues.

    Non-optional for any NCAA: no force field covers them, so every downstream
    mechanics calculation on an NCAA-containing peptide rests on parameters that
    do not exist until this runs.
    """

    TIER = 4

    def run(self, residue_name: str) -> TierResult:
        return TierResult(
            tier=self.TIER,
            available=False,
            unavailable_reason=(
                f"No quantum chemistry package is available to parameterise '{residue_name}'. "
                f"No published force field covers non-canonical residues, so any mechanics "
                f"result for a peptide containing this residue would rest on parameters that "
                f"have not been derived. Predictions involving it are sequence- and "
                f"literature-level only."
            ),
            missing_dependency="Psi4/Gaussian for RESP charge derivation and torsion fitting",
        )


class PhysicsStack:
    """Runs the tiers in order and reports how far it actually got."""

    def __init__(self):
        self.tier0 = Tier0Sequence()
        self.tier1 = Tier1PKa()
        self.tier2 = Tier2Ensemble()
        self.tier3 = Tier3FMO()
        self.tier4 = Tier4QMParameterization()

    def run(
        self,
        sequence: str,
        ph: float = 7.4,
        structure_available: bool = False,
        contains_noncanonical: bool = False,
    ) -> Dict:
        t0 = self.tier0.run(sequence, formulation_ph=ph)
        t1 = self.tier1.run(sequence, ph=ph, structure_available=structure_available)
        t2 = self.tier2.run(sequence)

        try:
            t3 = self.tier3.run(t2)
            t3_blocked = ""
        except RuntimeError as e:
            t3, t3_blocked = None, str(e)

        t4 = self.tier4.run("non-canonical residue") if contains_noncanonical else None

        reached = max([t.tier for t in (t0, t1) if t.available], default=-1)

        return {
            "tier_reached": reached,
            "tier0": t0,
            "tier1": t1,
            "tier2": t2,
            "tier3": t3,
            "tier3_blocked_reason": t3_blocked,
            "tier4": t4,
            "summary": (
                f"Physics reached Tier {reached}. Tiers 2-4 did not run; results below rest on "
                f"sequence-level and free-residue calculations only, and any conformational or "
                f"interaction-energy statement is correspondingly unsupported."
            ),
        }
