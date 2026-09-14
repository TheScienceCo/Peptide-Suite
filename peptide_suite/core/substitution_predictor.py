"""
Tier 1 substitution effect prediction (sequence-based, no structure).
Predicts primary and off-target effects using charge states, conservation,
hydrophobicity, and known motif impacts.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from . import Effect, EvidenceTier, ConfidenceLevel
from .charge_calculator import ChargeCalculator
from .conservation import ConservationAnalyzer
from .confidence_scoring import ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class HydrophobicityScale:
    """Kyte-Doolittle hydrophobicity scale."""

    SCALE = {
        "A": 1.8,
        "R": -4.5,
        "N": -3.5,
        "D": -3.5,
        "C": 2.5,
        "Q": -3.5,
        "E": -3.5,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "L": 3.8,
        "K": -3.9,
        "M": 1.9,
        "F": 2.8,
        "P": -1.6,
        "S": -0.8,
        "T": -0.7,
        "W": -0.9,
        "Y": -1.3,
        "V": 4.2,
    }

    @classmethod
    def hydrophobicity(cls, aa: str) -> float:
        return cls.SCALE.get(aa.upper(), 0.0)


class SubstitutionPredictor:
    """
    Predicts effects of single-position substitutions.
    Tier 1: sequence-only, no structure.
    """

    def __init__(self):
        self.charge_calc = ChargeCalculator()
        self.conservation = ConservationAnalyzer()
        self.confidence_scorer = ConfidenceScorer()
        self.protease_motifs = self._load_protease_motifs()

    @staticmethod
    def _load_protease_motifs() -> Dict[str, Dict]:
        """Load P1 specificities from the reference data file."""
        path = Path(__file__).parent.parent / "data" / "protease_motifs.json"
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, dict) and "p1_residues" in v}

    def cleavage_liability(self, sequence: str, position: int) -> List[Dict]:
        """
        Report which proteases recognise `position` as a P1 cleavage residue.

        This is motif matching against known specificities, not enzyme kinetics:
        it answers "does this residue match a documented recognition pattern",
        never "what is the kcat/KM".

        Returns:
            List of {protease, full_name, confidence, resistance_strategy} dicts
        """
        residue = sequence[position].upper()
        hits = []

        for name, spec in self.protease_motifs.items():
            fixed = spec.get("fixed_position")
            if fixed is not None and position != fixed:
                continue
            if residue in spec["p1_residues"]:
                hits.append(
                    {
                        "protease": name,
                        "full_name": spec.get("full_name", name),
                        "confidence": spec.get("confidence", 0.5),
                        "resistance_strategy": spec.get("resistance_strategy", ""),
                    }
                )

        return hits

    def predict_substitution_effect(
        self,
        sequence: str,
        position: int,
        wild_type_aa: str,
        mutant_aa: str,
        conservation_profile: Dict[int, float],
        inferred_goal: str = "generic_improvement",
        ph: float = 7.4,
        conservation_available: bool = False,
    ) -> Tuple[Effect, List[Effect]]:
        """
        Predict primary and off-target effects of a single substitution.

        Args:
            sequence: Full peptide sequence
            position: 0-indexed position to mutate
            wild_type_aa: Original AA (1-letter)
            mutant_aa: Proposed AA (1-letter)
            conservation_profile: Dict from ConservationAnalyzer
            inferred_goal: "protease_resistance", "binding_affinity", etc.
            ph: pH for charge calculations
            conservation_available: False when too few homologs were retrieved for
                entropy to carry information; suppresses the conservation term
                rather than reporting a value with no computation path behind it.

        Returns:
            Tuple of (primary_effect, off_target_effects_list)
        """
        wild_type_aa = wild_type_aa.upper()
        mutant_aa = mutant_aa.upper()

        # Build mutant sequence
        mutant_seq = sequence[:position] + mutant_aa + sequence[position + 1 :]

        # Primary effect depends on inferred goal
        primary_effect = self._predict_primary_effect(
            sequence,
            mutant_seq,
            position,
            wild_type_aa,
            mutant_aa,
            conservation_profile,
            inferred_goal,
            ph,
        )

        # Off-target effects: always check these
        off_targets = self._predict_off_target_effects(
            sequence,
            mutant_seq,
            position,
            wild_type_aa,
            mutant_aa,
            conservation_profile,
            ph,
            conservation_available,
        )

        return primary_effect, off_targets

    def _predict_primary_effect(
        self,
        wild_seq: str,
        mutant_seq: str,
        position: int,
        wt_aa: str,
        mut_aa: str,
        conservation_profile: Dict[int, float],
        goal: str,
        ph: float,
    ) -> Effect:
        """Predict intended effect based on inferred goal."""

        if goal == "protease_resistance":
            return self._predict_protease_effect(wild_seq, mutant_seq, position, wt_aa, mut_aa)
        elif goal == "binding_affinity":
            return self._predict_binding_effect(wild_seq, mutant_seq, position, wt_aa, mut_aa, ph)
        else:
            # Generic: try to infer from charge/hydrophobic changes
            return self._predict_generic_effect(wild_seq, mutant_seq, position, wt_aa, mut_aa, ph)

    def _predict_protease_effect(
        self, wild_seq: str, mutant_seq: str, position: int, wt_aa: str, mut_aa: str
    ) -> Effect:
        """Predict protease resistance benefit."""

        # Which proteases recognise the wild-type residue here, and which would
        # still recognise the mutant? A substitution only helps if it removes a
        # liability without introducing a new one.
        removed = self.cleavage_liability(wild_seq, position)
        introduced = self.cleavage_liability(mutant_seq, position)

        removed_names = {h["protease"] for h in removed}
        introduced_names = {h["protease"] for h in introduced}
        net_removed = removed_names - introduced_names
        net_added = introduced_names - removed_names

        if net_removed and not net_added:
            best = max((h for h in removed if h["protease"] in net_removed), key=lambda h: h["confidence"])
            score = self.confidence_scorer.score_effect(
                description=(
                    f"Predicted cleavage liability reduced: removes {wt_aa} P1 site for "
                    f"{', '.join(sorted(net_removed))}"
                ),
                evidence_tier=EvidenceTier.DIRECT_EXPERIMENTAL,
                reasoning=(
                    f"{wt_aa} at position {position + 1} matches the documented P1 specificity of "
                    f"{best['full_name']}; {mut_aa} does not. Predicted cleavage liability: "
                    f"high → low at this site. Motif match against known recognition patterns "
                    f"(Michaelis-Menten framing only — no kcat/KM is computed or implied). "
                    f"Documented strategy: {best['resistance_strategy']}"
                ),
                modifier=best["confidence"],
                magnitude=0.75,
                equation_refs=[22, 23],
            )
        elif net_added:
            score = self.confidence_scorer.score_effect(
                description=(
                    f"Predicted cleavage liability INCREASED: introduces P1 site for "
                    f"{', '.join(sorted(net_added))}"
                ),
                evidence_tier=EvidenceTier.DIRECT_EXPERIMENTAL,
                reasoning=(
                    f"{mut_aa} at position {position + 1} matches the P1 specificity of "
                    f"{', '.join(sorted(net_added))}, creating a cleavage site that the wild-type "
                    f"residue {wt_aa} did not present. This works against the stated goal."
                ),
                modifier=0.85,
                magnitude=0.0,  # No benefit; the cost is carried as a negative below
                equation_refs=[22, 23],
            )
        else:
            score = self.confidence_scorer.score_effect(
                description="No change in predicted cleavage liability at this position",
                evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
                reasoning=(
                    f"Neither {wt_aa} nor {mut_aa} at position {position + 1} matches a P1 "
                    f"specificity in the reference set ({', '.join(sorted(self.protease_motifs))}). "
                    "Backbone-level strategies (D-amino acids, N-methylation) would be required "
                    "here instead; those are non-canonical and out of scope for this scan."
                ),
                modifier=0.7,
                magnitude=0.05,
                equation_refs=[],
            )

        score.category = "primary"
        return score

    def _predict_binding_effect(
        self, wild_seq: str, mutant_seq: str, position: int, wt_aa: str, mut_aa: str, ph: float
    ) -> Effect:
        """Predict impact on receptor/ligand binding affinity."""

        # Charge change analysis (Coulomb-style)
        wt_charge = self.charge_calc.charge_at_ph(wt_aa, ph=ph, position=position)
        mut_charge = self.charge_calc.charge_at_ph(mut_aa, ph=ph, position=position)

        charge_delta = mut_charge.effective_charge - wt_charge.effective_charge

        # Hydrophobicity change
        hydro_delta = HydrophobicityScale.hydrophobicity(mut_aa) - HydrophobicityScale.hydrophobicity(wt_aa)

        description = f"Charge change: {charge_delta:+.2f}e at pH {ph}"
        if abs(hydro_delta) > 1.0:
            description += f"; hydrophobicity shift: {hydro_delta:+.1f}"

        # Magnitude is derived from the size of the physicochemical change actually
        # computed from the sequence, not asserted. A substitution that changes
        # neither charge nor hydrophobicity cannot plausibly change binding much.
        magnitude = min(1.0, (abs(charge_delta) * 0.6) + (abs(hydro_delta) / 9.0) * 0.4)

        score = self.confidence_scorer.score_effect(
            description=description,
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            reasoning=(
                f"Charge state computed via Henderson-Hasselbalch at pH {ph}; hydrophobicity "
                f"from the Kyte-Doolittle scale. Direction of the effect on binding is NOT "
                f"predicted: without a receptor structure there is no way to know whether this "
                f"change is complementary or antagonistic to the binding interface. Magnitude "
                f"reflects only how large a physicochemical perturbation this is "
                f"(Δq = {charge_delta:+.2f}e, Δhydrophobicity = {hydro_delta:+.1f}). "
                f"No binding energy is computed — that requires Tier 2 structure prediction."
            ),
            modifier=0.5,  # Moderate confidence without structure
            magnitude=magnitude,
            equation_refs=[1, 11],  # Coulomb, Henderson-Hasselbalch
        )

        score.category = "primary"
        return score

    def _predict_generic_effect(
        self, wild_seq: str, mutant_seq: str, position: int, wt_aa: str, mut_aa: str, ph: float
    ) -> Effect:
        """Generic improvement prediction (neutral goal)."""

        score = self.confidence_scorer.score_effect(
            description="No functional goal specified — no benefit can be predicted",
            evidence_tier=EvidenceTier.INFERENCE_ONLY,
            reasoning=(
                "Benefit is defined relative to a goal. With no goal supplied there is nothing "
                "to score against, so no primary benefit is claimed and the ranking below "
                "reflects off-target cost only. Specify a target function "
                "(protease_resistance, binding_affinity) to activate primary-effect scoring."
            ),
            modifier=1.0,
            magnitude=0.0,
            equation_refs=[],
        )

        score.category = "primary"
        return score

    def _predict_off_target_effects(
        self,
        wild_seq: str,
        mutant_seq: str,
        position: int,
        wt_aa: str,
        mut_aa: str,
        conservation_profile: Dict[int, float],
        ph: float,
        conservation_available: bool = False,
    ) -> List[Effect]:
        """Predict secondary/off-target effects of substitution."""

        off_targets = []

        # Off-target 1: Loss of specific residue property
        off_targets.append(self._residue_property_loss_effect(wt_aa, mut_aa, position))

        # Off-target 2: Conservation penalty
        conservation_entropy = conservation_profile.get(position, 2.0)
        off_targets.append(
            self._conservation_penalty_effect(
                conservation_entropy, position, wt_aa, mut_aa, conservation_available
            )
        )

        # Off-target 3: Structural/backbone effects
        off_targets.append(self._backbone_effect(wt_aa, mut_aa, position))

        # Off-target 4: Charge redistribution
        off_targets.append(self._charge_redistribution_effect(wild_seq, mutant_seq, position, ph))

        return off_targets

    def _residue_property_loss_effect(self, wt_aa: str, mut_aa: str, position: int) -> Effect:
        """Warn about loss of special properties (Pro flexibility loss, etc.)."""

        reason = ""
        if wt_aa == "P":
            reason = (
                f"Pro→{mut_aa} removes ring constraint on backbone φ dihedral. "
                "Increases backbone flexibility, may disrupt tertiary structure packing."
            )
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.7
            magnitude = 0.7
        elif wt_aa == "C":
            reason = f"Cys→{mut_aa} removes disulfide-bonding capability (if applicable)."
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.6
            magnitude = 0.65
        elif wt_aa in ["W", "Y", "F"] and mut_aa not in ["W", "Y", "F"]:
            reason = (
                f"{wt_aa}→{mut_aa} removes aromatic ring. "
                "May disrupt π-π stacking or hydrophobic pocket interactions."
            )
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.7
            magnitude = 0.55
        else:
            # Scale the cost by how far apart the two residues actually are on the
            # hydrophobicity scale, rather than calling every swap "significant".
            hydro_delta = abs(
                HydrophobicityScale.hydrophobicity(mut_aa)
                - HydrophobicityScale.hydrophobicity(wt_aa)
            )
            magnitude = min(1.0, hydro_delta / 9.0)  # Scale spans -4.5..4.5
            reason = (
                f"{wt_aa}→{mut_aa}: Kyte-Doolittle hydrophobicity differs by {hydro_delta:.1f} "
                f"(scale range 9.0), a {'substantial' if magnitude > 0.4 else 'modest'} "
                "change in side-chain character."
            )
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.6

        effect = self.confidence_scorer.score_effect(
            description=f"Loss of {wt_aa} biochemical properties",
            evidence_tier=evidence,
            reasoning=reason,
            modifier=modifier,
            magnitude=magnitude,
        )
        effect.category = "off_target"
        return effect

    def _conservation_penalty_effect(
        self,
        entropy: float,
        position: int,
        wt_aa: str,
        mut_aa: str,
        conservation_available: bool = False,
    ) -> Effect:
        """Warn if mutating a highly conserved position."""

        # Without enough homologs there is no conservation signal to report.
        # Entropy over a single sequence is 0 at every position by construction,
        # which would otherwise flag the whole peptide as "highly conserved".
        if not conservation_available:
            effect = self.confidence_scorer.score_effect(
                description="Conservation risk: NOT COMPUTED (insufficient homolog data)",
                evidence_tier=EvidenceTier.INFERENCE_ONLY,
                reasoning=(
                    "Too few homologous sequences were retrieved to compute per-position "
                    "Shannon entropy. No conservation claim is made for this position, and "
                    "no conservation penalty is applied to the net score. Supply homologs "
                    "or enable NCBI retrieval to activate this term."
                ),
                modifier=1.0,
                magnitude=0.0,  # Contributes nothing to the net score
                equation_refs=[],
            )
            effect.category = "off_target"
            return effect

        if entropy < 0.5:
            reason = (
                f"Position {position} is highly conserved (entropy {entropy:.2f}). "
                "Suggests strong functional constraint. Mutation may disable critical interactions."
            )
            modifier = 0.9  # High confidence in the risk
            magnitude = 0.8  # Breaking a conserved position is a large cost
            evidence = EvidenceTier.DIRECT_EXPERIMENTAL  # MSA is direct data
        elif entropy < 2.0:
            reason = (
                f"Position {position} is moderately conserved (entropy {entropy:.2f}). "
                "May be important but tolerate some variation."
            )
            modifier = 0.6
            magnitude = 0.4
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
        else:
            reason = f"Position {position} is variable (entropy {entropy:.2f}). No conservation penalty."
            modifier = 0.9
            magnitude = 0.05
            evidence = EvidenceTier.DIRECT_EXPERIMENTAL

        effect = self.confidence_scorer.score_effect(
            description="Conservation-based risk penalty",
            evidence_tier=evidence,
            reasoning=reason,
            modifier=modifier,
            magnitude=magnitude,
            equation_refs=[43],  # Shannon entropy
        )
        effect.category = "off_target"
        return effect

    def _backbone_effect(self, wt_aa: str, mut_aa: str, position: int) -> Effect:
        """Predict backbone conformational changes."""

        # Proline is special: cyclic, restricts φ. Glycine is the other outlier:
        # no side chain, so it samples backbone conformations nothing else can.
        if wt_aa == "P":
            reason = f"Pro removal at {position} increases backbone flexibility (loss of ring constraint)."
            modifier = 0.8
            magnitude = 0.6
        elif mut_aa == "P":
            reason = f"Introduction of Pro at {position} restricts backbone flexibility (new ring)."
            modifier = 0.8
            magnitude = 0.6
        elif wt_aa == "G":
            reason = (
                f"Gly removal at {position} restricts backbone φ/ψ sampling; Gly is the only "
                "residue that readily occupies left-handed conformations."
            )
            modifier = 0.7
            magnitude = 0.45
        else:
            reason = "Standard backbone dynamics, minimal effect expected."
            modifier = 0.6
            magnitude = 0.1

        effect = self.confidence_scorer.score_effect(
            description="Backbone conformational impact",
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            reasoning=reason,
            modifier=modifier,
            magnitude=magnitude,
        )
        effect.category = "off_target"
        return effect

    def _charge_redistribution_effect(
        self, wild_seq: str, mutant_seq: str, position: int, ph: float
    ) -> Effect:
        """Check for unexpected charge cluster effects."""

        # Compare the substituted residue's own charge against its neighbours.
        # (Comparing the flanking windows alone is a no-op: they are identical by
        # construction, since the sequences differ only at `position`.)
        window = 3
        start = max(0, position - window)
        end = min(len(wild_seq), position + window + 1)

        wt_charge = self.charge_calc.charge_at_ph(wild_seq[position], ph=ph, position=position)
        mut_charge = self.charge_calc.charge_at_ph(mutant_seq[position], ph=ph, position=position)
        charge_delta = mut_charge.effective_charge - wt_charge.effective_charge

        neighbour_charge = sum(
            self.charge_calc.charge_at_ph(wild_seq[i], ph=ph, position=i).effective_charge
            for i in range(start, end)
            if i != position
        )

        if abs(charge_delta) < 0.2:
            reason = (
                f"Substitution does not materially change charge at this position "
                f"(Δq = {charge_delta:+.2f}e at pH {ph})."
            )
            magnitude = 0.05
        elif charge_delta * neighbour_charge > 0.2:
            # New charge has the same sign as the local environment.
            reason = (
                f"Introduces {charge_delta:+.2f}e into a local environment already carrying "
                f"{neighbour_charge:+.2f}e within ±{window} residues. Like charges in proximity "
                "may repel and destabilise local packing. Heuristic flag (Coulomb-style "
                "reasoning), not a computed interaction energy: no structure or solvent model."
            )
            magnitude = 0.5
        else:
            reason = (
                f"Introduces {charge_delta:+.2f}e against a local environment of "
                f"{neighbour_charge:+.2f}e within ±{window} residues. Opposite charges may be "
                "stabilising, but without structure the geometry is unknown. Heuristic flag, "
                "not a computed interaction energy."
            )
            magnitude = 0.2

        effect = self.confidence_scorer.score_effect(
            description="Local charge redistribution",
            evidence_tier=EvidenceTier.INFERENCE_ONLY,
            reasoning=reason,
            modifier=0.8,
            magnitude=magnitude,
            equation_refs=[1, 11],  # Coulomb heuristic, Henderson-Hasselbalch
        )
        effect.category = "off_target"
        return effect
