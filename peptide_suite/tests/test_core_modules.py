"""
Unit tests for core modules.
Tests confidence scoring, charge calculations, conservation, and peptide management.
"""

import unittest
import math

from peptide_suite.core import EvidenceTier, ConfidenceLevel, Effect
from peptide_suite.core.confidence_scoring import ConfidenceScorer
from peptide_suite.core.charge_calculator import ChargeCalculator
from peptide_suite.core.conservation import ConservationAnalyzer
from peptide_suite.core.peptide_manager import PeptideManager


class TestConfidenceScoring(unittest.TestCase):
    """Test confidence scoring framework."""

    def setUp(self):
        self.scorer = ConfidenceScorer()

    # These used to assert specific scores and band names. Both are policy now,
    # so the assertions are the orderings the engine actually guarantees: a
    # better-evidenced claim scores higher and never lands in a worse band.
    ORDER = [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]

    def _score(self, tier):
        return self.scorer.score_effect(
            description="Test effect", evidence_tier=tier, reasoning="fixture",
        )

    def test_evidence_tier_is_preserved(self):
        effect = self._score(EvidenceTier.DIRECT_EXPERIMENTAL)
        self.assertEqual(effect.evidence_tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_better_evidence_scores_strictly_higher(self):
        tiers = [EvidenceTier.DIRECT_EXPERIMENTAL, EvidenceTier.HOMOLOG_EXPERIMENTAL,
                 EvidenceTier.BIOCHEMICAL_PRINCIPLE, EvidenceTier.INFERENCE_ONLY]
        scores = [self._score(t).score for t in tiers]
        for upper, lower in zip(scores, scores[1:]):
            self.assertGreater(upper, lower)

    def test_better_evidence_never_lands_in_a_worse_band(self):
        direct = self._score(EvidenceTier.DIRECT_EXPERIMENTAL)
        inference = self._score(EvidenceTier.INFERENCE_ONLY)
        self.assertGreaterEqual(self.ORDER.index(direct.confidence),
                                self.ORDER.index(inference.confidence))

    def test_combine_effect_scores(self):
        """Combining effects should weight them appropriately."""
        primary = self.scorer.score_effect(
            description="Primary benefit",
            evidence_tier=EvidenceTier.DIRECT_EXPERIMENTAL,
            reasoning="Good data",
        )

        off_target = self.scorer.score_effect(
            description="Off-target cost",
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            reasoning="Plausible risk",
        )

        net_score, combined_conf = self.scorer.combine_effect_scores([primary, off_target])

        # Net should be positive (benefit outweighs cost)
        self.assertGreater(net_score, 0)
        # Combined confidence is no better than the worse input. Asserting a
        # specific label here would bake in a set of cutoffs, which now come
        # from the policy: the invariant is the ordering, not the band name.
        order = [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
        worse = min(primary.confidence, off_target.confidence, key=order.index)
        self.assertLessEqual(order.index(combined_conf), order.index(worse))


class TestChargeCalculator(unittest.TestCase):
    """Test Henderson-Hasselbalch charge calculations."""

    def setUp(self):
        self.calc = ChargeCalculator()

    def test_charge_acidic_residue_low_ph(self):
        """Aspartic acid at low pH should be mostly protonated (neutral)."""
        charge = self.calc.charge_at_ph("D", ph=2.0)

        self.assertTrue(charge.is_ionizable)
        # At pH 2 < pKa 3.9, mostly protonated (effective charge ~ 0)
        self.assertAlmostEqual(charge.effective_charge, 0.0, places=1)

    def test_charge_acidic_residue_high_ph(self):
        """Aspartic acid at high pH should be negative."""
        charge = self.calc.charge_at_ph("D", ph=7.4)

        self.assertTrue(charge.is_ionizable)
        # At pH 7.4 > pKa 3.9, mostly deprotonated (negative)
        self.assertLess(charge.effective_charge, -0.5)

    def test_charge_basic_residue_high_ph(self):
        """Lysine at high pH should be neutral."""
        charge = self.calc.charge_at_ph("K", ph=12.0)

        self.assertTrue(charge.is_ionizable)
        # At pH 12 > pKa 10.5, mostly deprotonated (neutral)
        self.assertLess(charge.effective_charge, 0.3)

    def test_sequence_charge_profile(self):
        """Compute charge profile for a sequence."""
        sequence = "DEKR"
        charges = self.calc.sequence_charge_profile(sequence, ph=7.4)

        self.assertEqual(len(charges), 4)
        # D and E should be negative, K and R should be positive
        self.assertLess(charges[0].effective_charge, 0)  # D
        self.assertLess(charges[1].effective_charge, 0)  # E
        self.assertGreater(charges[2].effective_charge, 0)  # K
        self.assertGreater(charges[3].effective_charge, 0)  # R

    def test_net_charge_calculation(self):
        """Net charge calculation should sum contributions."""
        sequence = "K" * 3 + "D" * 2
        net = self.calc.net_charge(sequence, ph=7.4)

        # 3 Lys (~+3) and 2 Asp (~-2) at pH 7.4 -> net ~+1
        self.assertGreater(net, 0)


class TestConservationAnalyzer(unittest.TestCase):
    """Test conservation entropy calculations."""

    def setUp(self):
        self.analyzer = ConservationAnalyzer()

    def test_shannon_entropy_conserved(self):
        """Fully conserved position should have entropy ~0."""
        position_aas = ["A", "A", "A", "A", "A"]
        entropy = self.analyzer.shannon_entropy(position_aas)

        self.assertAlmostEqual(entropy, 0.0, places=2)

    def test_shannon_entropy_maximum(self):
        """Maximum entropy with 20 AAs equally frequent."""
        position_aas = list("ACDEFGHIKLMNPQRSTVWY") * 5
        entropy = self.analyzer.shannon_entropy(position_aas)

        max_entropy = math.log2(20)  # ~4.32
        self.assertGreater(entropy, 4.0)

    def test_shannon_entropy_partial(self):
        """Partial conservation should yield intermediate entropy."""
        position_aas = ["A"] * 8 + ["G"] * 2
        entropy = self.analyzer.shannon_entropy(position_aas)

        # Mixed but skewed towards A
        self.assertGreater(entropy, 0.5)
        self.assertLess(entropy, 1.0)

    def test_conservation_flag_red(self):
        """Low entropy should produce red flag."""
        flag_class, flag_msg = self.analyzer.conservation_flag(0.3)

        self.assertEqual(flag_class, "highly_conserved")
        self.assertIn("RED FLAG", flag_msg)

    def test_conservation_flag_green(self):
        """High entropy should produce green flag."""
        flag_class, flag_msg = self.analyzer.conservation_flag(3.5)

        self.assertEqual(flag_class, "variable")
        self.assertIn("GREEN", flag_msg)


class TestPeptideManager(unittest.TestCase):
    """Test peptide sequence management."""

    def setUp(self):
        self.manager = PeptideManager()

    def test_validate_sequence_valid(self):
        """Valid sequence should pass validation."""
        is_valid, error = self.manager.validate_sequence("MGFPGLQPRRVSCGQAK")

        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_validate_sequence_invalid_chars(self):
        """Sequence with invalid characters should fail."""
        is_valid, error = self.manager.validate_sequence("MGFPGLQP1RVSCGQAK")

        self.assertFalse(is_valid)
        self.assertIn("Invalid", error)

    def test_validate_sequence_too_short(self):
        """Very short sequence should fail."""
        is_valid, error = self.manager.validate_sequence("AKL")

        self.assertFalse(is_valid)
        self.assertIn("short", error)

    def test_load_sequence_raw(self):
        """Load raw amino acid sequence."""
        sequence, name = self.manager.load_sequence("MGFPGLQPRR")

        self.assertEqual(sequence, "MGFPGLQPRR")
        self.assertEqual(name, "unnamed_peptide")

    def test_load_sequence_fasta(self):
        """Load FASTA format sequence."""
        fasta = ">IGF-1\nMGFPGLQPRR"
        sequence, name = self.manager.load_sequence(fasta)

        self.assertEqual(sequence, "MGFPGLQPRR")
        self.assertEqual(name, "IGF-1")

    def test_basic_properties(self):
        """Compute basic sequence properties."""
        props = self.manager.basic_properties("DEKRKL")

        self.assertEqual(props["length"], 6)
        self.assertGreater(props["charged_residues"], 0)
        self.assertIn("composition", props)

    def test_motif_finding(self):
        """Find motifs in sequence."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        # A=0, C=1, D=2, E=3, F=4 -> DEF at position 2
        positions = self.manager.get_motif_positions(sequence, "DEF")

        self.assertEqual(positions, [2])

    def test_motif_with_wildcard(self):
        """Find motifs with wildcard."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        # DEF at position 2 matches DXF (X is wildcard)
        positions = self.manager.get_motif_positions(sequence, "DXF")

        self.assertEqual(positions, [2])  # DEF matches DXF


if __name__ == "__main__":
    unittest.main()
