"""
Sequence conservation analysis via homolog MSA and Shannon entropy.
Equation #43 reference: Shannon entropy H = -sum(f_i * log2(f_i))
where f_i is the frequency of amino acid i at a position.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class ConservationAnalyzer:
    """
    Computes per-position conservation entropy from a set of homologous sequences.
    Used to determine if a position is tolerant to substitution.
    """

    def __init__(self):
        self.standard_aas = set("ACDEFGHIKLMNPQRSTVWY")

    def shannon_entropy(self, position_aas: List[str]) -> float:
        """
        Compute Shannon entropy at a single position.

        Args:
            position_aas: List of amino acids at this position across homologs

        Returns:
            Shannon entropy (0 = fully conserved, 4.3 = maximum for 20 AAs)
        """
        if not position_aas:
            return 0.0

        # Count frequencies
        counts = Counter(position_aas)
        total = len(position_aas)

        # Filter out gaps (if present as '-')
        if "-" in counts:
            total_no_gap = total - counts["-"]
            if total_no_gap == 0:
                return 0.0
        else:
            total_no_gap = total

        entropy = 0.0
        for aa, count in counts.items():
            if aa == "-":  # Skip gap
                continue
            freq = count / total_no_gap
            if freq > 0:
                entropy -= freq * math.log2(freq)

        return entropy

    def conservation_profile(self, msa: List[str]) -> Dict[int, float]:
        """
        Compute Shannon entropy for every position in an MSA.

        Args:
            msa: List of aligned sequences (same length)

        Returns:
            Dict mapping position (0-indexed) -> entropy value
        """
        if not msa or len(msa) == 0:
            return {}

        alignment_length = len(msa[0])
        entropy_profile = {}

        for pos in range(alignment_length):
            position_aas = [seq[pos] for seq in msa if pos < len(seq)]
            entropy = self.shannon_entropy(position_aas)
            entropy_profile[pos] = entropy

        return entropy_profile

    def conservation_flag(self, entropy: float) -> Tuple[str, str]:
        """
        Classify position by conservation level and flag if risky to mutate.

        Args:
            entropy: Shannon entropy at position

        Returns:
            Tuple of (classification, flag_message)
        """
        # Thresholds (empirical)
        # 0.0 - 0.5 = highly conserved (red flag for substitution)
        # 0.5 - 2.0 = moderately conserved (orange flag)
        # 2.0+ = variable (green, tolerable to substitute)

        if entropy < 0.5:
            return "highly_conserved", (
                "⚠️  RED FLAG: This position is highly conserved across homologs. "
                "Substitution may disrupt critical function. Proceed with caution."
            )
        elif entropy < 2.0:
            return "moderately_conserved", (
                "⚠️  ORANGE FLAG: Moderately conserved position. "
                "Substitutions possible but not without risk. Consider alternatives."
            )
        else:
            return "variable", (
                "✓ GREEN: Position is variable across homologs, "
                "suggesting tolerance to substitution."
            )

    def position_tolerance_score(self, entropy: float) -> float:
        """
        Convert entropy to a 0-1 'tolerance' score for substitution.

        Args:
            entropy: Shannon entropy at position

        Returns:
            Score 0-1 (0 = intolerant, 1 = highly tolerant)
        """
        # Sigmoidal mapping: entropy 0->0, entropy 3->1
        # Using: tolerance = 1 / (1 + exp(-k * (entropy - inflection)))
        k = 1.5  # Steepness
        inflection = 1.5
        import math as m

        tolerance = 1.0 / (1.0 + m.exp(-k * (entropy - inflection)))
        return tolerance

    def build_msa_from_sequences(self, sequences: List[str]) -> List[str]:
        """
        Simple MSA: pad all sequences to max length.
        In production, would use Clustal/MAFFT via subprocess, but
        this naive method works for testing.

        Args:
            sequences: List of unaligned sequences

        Returns:
            List of aligned sequences (padded to same length)
        """
        if not sequences:
            return []

        max_len = max(len(s) for s in sequences)
        aligned = [s.ljust(max_len, "-") for s in sequences]
        return aligned

    def get_position_conservation_summary(
        self, conservation_profile: Dict[int, float], position: int
    ) -> Dict:
        """Return human-readable summary for a specific position."""
        entropy = conservation_profile.get(position, 0.0)
        classification, flag = self.conservation_flag(entropy)
        tolerance = self.position_tolerance_score(entropy)

        return {
            "position": position,
            "entropy": round(entropy, 3),
            "classification": classification,
            "flag": flag,
            "tolerance_score": round(tolerance, 2),
        }
