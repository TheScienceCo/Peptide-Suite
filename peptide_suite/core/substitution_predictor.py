"""
Tier 1 substitution effect prediction (sequence-based, no structure).
Predicts primary and off-target effects using charge states, conservation,
hydrophobicity, and known motif impacts.
"""

import logging
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

        # Known protease cleavage patterns (simplified)
        # Real version would use Prosite or Merops databases
        self.protease_motifs = {
            "DPP4": ["[AEP]P"],  # Dipeptidyl peptidase IV
            "neprilysin": ["AX", "GX"],  # Neprilysin (NEP) preferred sites
            "elastase": ["VX", "AX", "LX"],  # Serine elastase sites
            "trypsin": ["KR"],  # Trypsin cleavage after K or R
        }

    def predict_substitution_effect(
        self,
        sequence: str,
        position: int,
        wild_type_aa: str,
        mutant_aa: str,
        conservation_profile: Dict[int, float],
        inferred_goal: str = "generic_improvement",
        ph: float = 7.4,
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

        # Check if we're modifying a known protease site
        window_size = 2
        disrupt_motif = False

        for motif_aa_set in ["[AEP]", "AX", "GX", "KR", "VX", "LX"]:
            for i in range(max(0, position - window_size), min(len(wild_seq), position + window_size)):
                if i < position and wt_aa in motif_aa_set:
                    disrupt_motif = True
                    break

        if disrupt_motif:
            score = self.confidence_scorer.score_effect(
                description=f"Disrupts protease recognition motif (substituting {wt_aa} with {mut_aa})",
                evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
                reasoning=(
                    "Known protease sites (DPP4, neprilysin, elastase) have specific "
                    "recognition preferences. Removing or altering these residues prevents cleavage."
                ),
                modifier=0.8,  # Good evidence but context-dependent
                equation_refs=[22, 23],  # Michaelis-Menten framing
            )
        else:
            # Generic proteolysis resistance (harder to predict without structure)
            score = self.confidence_scorer.score_effect(
                description=f"Potential protease resistance (position not in known site)",
                evidence_tier=EvidenceTier.INFERENCE_ONLY,
                reasoning=(
                    f"D-amino acids and N-methylated positions generally resist proteolysis. "
                    f"However, without structure, cannot assess local accessibility."
                ),
                modifier=0.4,
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

        description = f"Charge change: {charge_delta:.2f}e"
        if abs(hydro_delta) > 1.0:
            description += f"; hydrophobicity shift: {hydro_delta:+.1f}"

        # Heuristic: if charge match to known interacting residues improves, likely beneficial
        # But we have no structure, so this is weak inference
        score = self.confidence_scorer.score_effect(
            description=description,
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            reasoning=(
                f"Charge-based prediction (Coulomb-style heuristic). "
                f"Without structural data, cannot compute binding energies. "
                f"Assumes electrostatic interactions dominate locally. "
                f"Hydrophobicity change of {hydro_delta:+.1f} may affect water solvation layer."
            ),
            modifier=0.5,  # Moderate confidence without structure
            equation_refs=[1, 11],  # Coulomb, Henderson-Hasselbalch
        )

        score.category = "primary"
        return score

    def _predict_generic_effect(
        self, wild_seq: str, mutant_seq: str, position: int, wt_aa: str, mut_aa: str, ph: float
    ) -> Effect:
        """Generic improvement prediction (neutral goal)."""

        wt_charge = self.charge_calc.charge_at_ph(wt_aa, ph=ph, position=position)
        mut_charge = self.charge_calc.charge_at_ph(mut_aa, ph=ph, position=position)

        score = self.confidence_scorer.score_effect(
            description="Generic substitution (no specific functional goal specified)",
            evidence_tier=EvidenceTier.INFERENCE_ONLY,
            reasoning="Without a specific goal, difficult to predict benefit. Recommend specifying target function.",
            modifier=0.3,
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
    ) -> List[Effect]:
        """Predict secondary/off-target effects of substitution."""

        off_targets = []

        # Off-target 1: Loss of specific residue property
        off_targets.append(self._residue_property_loss_effect(wt_aa, mut_aa, position))

        # Off-target 2: Conservation penalty
        conservation_entropy = conservation_profile.get(position, 2.0)
        off_targets.append(
            self._conservation_penalty_effect(conservation_entropy, position, wt_aa, mut_aa)
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
        elif wt_aa == "C":
            reason = f"Cys→{mut_aa} removes disulfide-bonding capability (if applicable)."
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.6
        elif wt_aa in ["W", "Y", "F"]:
            reason = (
                f"{wt_aa}→{mut_aa} removes aromatic ring. "
                "May disrupt π-π stacking or hydrophobic pocket interactions."
            )
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
            modifier = 0.7
        else:
            reason = f"{wt_aa}→{mut_aa} is a significant property change."
            evidence = EvidenceTier.INFERENCE_ONLY
            modifier = 0.4

        effect = self.confidence_scorer.score_effect(
            description=f"Loss of {wt_aa} biochemical properties",
            evidence_tier=evidence,
            reasoning=reason,
            modifier=modifier,
        )
        effect.category = "off_target"
        return effect

    def _conservation_penalty_effect(
        self, entropy: float, position: int, wt_aa: str, mut_aa: str
    ) -> Effect:
        """Warn if mutating a highly conserved position."""

        if entropy < 0.5:
            reason = (
                f"Position {position} is highly conserved (entropy {entropy:.2f}). "
                "Suggests strong functional constraint. Mutation may disable critical interactions."
            )
            modifier = 0.9  # High confidence in the risk
            evidence = EvidenceTier.DIRECT_EXPERIMENTAL  # MSA is direct data
        elif entropy < 2.0:
            reason = (
                f"Position {position} is moderately conserved (entropy {entropy:.2f}). "
                "May be important but tolerate some variation."
            )
            modifier = 0.6
            evidence = EvidenceTier.BIOCHEMICAL_PRINCIPLE
        else:
            reason = f"Position {position} is variable (entropy {entropy:.2f}). No conservation penalty."
            modifier = 0.1
            evidence = EvidenceTier.DIRECT_EXPERIMENTAL

        effect = self.confidence_scorer.score_effect(
            description="Conservation-based risk penalty",
            evidence_tier=evidence,
            reasoning=reason,
            modifier=modifier,
            equation_refs=[43],  # Shannon entropy
        )
        effect.category = "off_target"
        return effect

    def _backbone_effect(self, wt_aa: str, mut_aa: str, position: int) -> Effect:
        """Predict backbone conformational changes."""

        # Proline is special: cyclic, restricts φ
        if wt_aa == "P":
            reason = f"Pro removal at {position} increases backbone flexibility (loss of ring constraint)."
            modifier = 0.8
        elif mut_aa == "P":
            reason = f"Introduction of Pro at {position} restricts backbone flexibility (new ring)."
            modifier = 0.8
        else:
            # Generic: hydrophobic -> charged might shift local hydration
            reason = "Standard backbone dynamics, minimal effect expected."
            modifier = 0.3

        effect = self.confidence_scorer.score_effect(
            description="Backbone conformational impact",
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            reasoning=reason,
            modifier=modifier,
        )
        effect.category = "off_target"
        return effect

    def _charge_redistribution_effect(
        self, wild_seq: str, mutant_seq: str, position: int, ph: float
    ) -> Effect:
        """Check for unexpected charge cluster effects."""

        # Simple: look for charge neighbors
        window = 3
        start = max(0, position - window)
        end = min(len(wild_seq), position + window + 1)

        wt_charged = sum(
            1
            for i in range(start, end)
            if i != position
            and wild_seq[i] in ["D", "E", "K", "R", "H"]
        )
        mut_charged = sum(
            1
            for i in range(start, end)
            if i != position
            and mutant_seq[i] in ["D", "E", "K", "R", "H"]
        )

        reason = (
            f"Local charge environment unchanged (charged neighbors: {wt_charged} → {mut_charged})."
        )
        modifier = 0.5 if wt_charged == mut_charged else 0.7

        effect = self.confidence_scorer.score_effect(
            description="Local charge redistribution",
            evidence_tier=EvidenceTier.INFERENCE_ONLY,
            reasoning=reason,
            modifier=modifier,
        )
        effect.category = "off_target"
        return effect
