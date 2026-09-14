"""
Pre-organization proxies for conformational-constraint moves.

Interaction-energy features are blind to pre-organization. A staple or an Aib
substitution may not change any contact energy at all and still improve binding
substantially, by paying less conformational entropy on association. A pipeline
that scores only enthalpic terms will rate such a move as inert.

So every constraint move reports a proxy for how much the accessible ensemble
was restricted, and states whether the predicted benefit is entropic or
enthalpic. The honest proxies available without an ensemble are helix propensity
and Ramachandran basin arguments; ensemble RMSF requires Tier 2 and is reported
as unavailable rather than estimated.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .epistemics import Claim
from .physics_tiers import HELIX_PROPENSITY
from .transformations import PreOrganizationProxy

# Helix propensity for residues outside the canonical twenty, in the same
# Pace & Scholtz units (kcal/mol relative to Ala = 0, lower favours helix).
# Aib is strongly helix-inducing: its two alpha-methyl groups restrict phi/psi
# to the helical region almost exclusively.
NONCANONICAL_HELIX_PROPENSITY = {
    "AIB": -0.70,     # alpha-aminoisobutyric acid
    "ACPC": -0.55,    # cyclopropane-constrained
    "HYP": 0.30,      # 4-hydroxyproline, in polyproline-II contexts
    "NLE": 0.15,      # norleucine, close to Leu
    "ORN": 0.26,      # ornithine, close to Lys
    "CIT": 0.40,      # citrulline
}

# Ramachandran consequences of the constraint chemistries.
BASIN_EFFECTS = {
    "AIB": "phi/psi confined to the right- and left-handed helical basins; beta region essentially eliminated",
    "D_AMINO_ACID": "accesses the mirror-image basin, inverting local chirality and disrupting an L-helix",
    "N_METHYL": "removes the backbone amide hydrogen, blocking one i,i+4 hydrogen bond and biasing toward extended/turn conformations",
    "ALPHA_METHYL": "alpha-substitution restricts phi/psi toward helical basins, similar in kind to Aib but weaker",
    "STAPLE": "covalent i,i+4 or i,i+7 crosslink pre-pays the helical folding entropy",
    "LACTAM": "side-chain lactam bridge locks an i,i+4 turn of helix",
    "CYCLIZATION": "head-to-tail macrocyclisation removes terminal fraying and reduces the accessible ensemble globally",
}


@dataclass
class HelicityEstimate:
    """A sequence-level helix propensity calculation, not a folding prediction."""
    wild_type_score: float
    mutant_score: float
    delta_propensity: float       # negative favours helix
    fractional_helicity_delta: float
    method: str


class PreOrganizationAnalyzer:
    """Computes pre-organization proxies from sequence-level propensities."""

    # Empirical conversion from a summed propensity change to an approximate
    # fractional helicity change. This is a coarse linearisation of an
    # Agadir-style relationship, adequate for ranking moves against each other
    # and not adequate for predicting an absolute helicity.
    PROPENSITY_TO_HELICITY = 0.12

    def helix_propensity(self, sequence: str) -> float:
        """Summed Pace & Scholtz propensity. Lower favours helix."""
        return sum(HELIX_PROPENSITY.get(aa.upper(), 0.5) for aa in sequence)

    def substitution_helicity_delta(
        self, sequence: str, position: int, mutant: str
    ) -> HelicityEstimate:
        """
        Predicted helicity change from a single substitution.

        `mutant` may be a one-letter canonical code or a non-canonical name
        such as "Aib".
        """
        seq = sequence.upper()
        wt_aa = seq[position]
        key = mutant.upper()

        wt_prop = HELIX_PROPENSITY.get(wt_aa, 0.5)
        if len(key) == 1:
            mut_prop = HELIX_PROPENSITY.get(key, 0.5)
        else:
            mut_prop = NONCANONICAL_HELIX_PROPENSITY.get(key, 0.5)

        whole_wt = self.helix_propensity(seq)
        whole_mut = whole_wt - wt_prop + mut_prop
        delta = mut_prop - wt_prop

        return HelicityEstimate(
            wild_type_score=round(whole_wt, 3),
            mutant_score=round(whole_mut, 3),
            delta_propensity=round(delta, 3),
            # Negative propensity delta favours helix, hence the sign flip
            fractional_helicity_delta=round(-delta * self.PROPENSITY_TO_HELICITY, 4),
            method="Pace & Scholtz helix propensity scale (sequence-level)",
        )

    def for_substitution(
        self, sequence: str, position: int, mutant: str, constraint_kind: Optional[str] = None
    ) -> PreOrganizationProxy:
        """Build the proxy for a constraint substitution such as Aib or a D-residue."""
        est = self.substitution_helicity_delta(sequence, position, mutant)
        basin = BASIN_EFFECTS.get((constraint_kind or mutant).upper(), "")

        # A move that changes the accessible ensemble far more than it changes
        # any contact is acting entropically.
        if abs(est.fractional_helicity_delta) >= 0.03 or basin:
            mechanism = "entropic (pre-organization), not enthalpic"
        else:
            mechanism = "no meaningful pre-organization change predicted"

        return PreOrganizationProxy(
            helicity_delta=est.fractional_helicity_delta,
            basin_restriction=basin,
            rmsf_change=None,   # Requires a Tier 2 ensemble
            mechanism=mechanism,
            claim=Claim.computed(
                f"Predicted fractional helicity change {est.fractional_helicity_delta:+.3f} "
                f"from {est.method}",
                method=est.method,
                tier=0,
            ),
        )

    def for_macrocyclization(self, kind: str, span: Optional[int] = None) -> PreOrganizationProxy:
        """
        Build the proxy for a staple, lactam bridge, or head-to-tail cyclisation.

        The helicity effect of a crosslink is geometric rather than propensity
        driven, so it is not read off the Pace & Scholtz scale. The values below
        are documented typical magnitudes for the chemistry, reported as
        inferences rather than calculations.
        """
        key = kind.upper()
        basin = BASIN_EFFECTS.get(key, "")

        typical = {
            "STAPLE": 0.25,        # i,i+4 hydrocarbon staples commonly add 20-40% helicity
            "LACTAM": 0.20,
            "CYCLIZATION": 0.05,   # global rigidification, modest helicity effect
        }.get(key)

        span_note = f" over an i,i+{span} span" if span else ""

        return PreOrganizationProxy(
            helicity_delta=typical,
            basin_restriction=basin,
            rmsf_change=None,
            mechanism="entropic (pre-organization), not enthalpic",
            claim=Claim.inferred(
                f"{kind.title()}{span_note}: typical reported helicity gain around "
                f"{typical:+.2f} fractional" if typical else f"{kind.title()}: helicity effect unquantified",
                inference_step=(
                    "Crosslink helicity effects are geometric rather than propensity-driven, so "
                    "they are taken from documented typical magnitudes for the chemistry rather "
                    "than computed from the sequence. The actual value depends on staple "
                    "position and linker length and requires CD measurement to establish."
                ),
            ),
        )

    def rmsf_unavailable_note(self) -> str:
        return (
            "Ensemble RMSF change is not reported: it requires a Tier 2 conformational "
            "ensemble, which did not run. Helicity delta and Ramachandran basin restriction "
            "are the available pre-organization proxies."
        )
