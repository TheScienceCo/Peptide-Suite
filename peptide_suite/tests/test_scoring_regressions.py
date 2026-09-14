"""
Regression tests for the three scoring defects that made v1 ranking meaningless.

Each test here pins a property that was silently violated before, and each
failure mode was invisible in the output: the scan still produced confident
looking numbers, they just did not mean anything.
"""

import unittest

from peptide_suite.core import Effect, EvidenceTier, ConfidenceLevel
from peptide_suite.core.confidence_scoring import ConfidenceScorer
from peptide_suite.core.substitution_predictor import SubstitutionPredictor

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"


class TestMagnitudeIsNotConfidence(unittest.TestCase):
    """Confidence answers 'is it real'; magnitude answers 'does it matter'."""

    def setUp(self):
        self.scorer = ConfidenceScorer()

    def _effect(self, score, magnitude, confidence=ConfidenceLevel.MEDIUM):
        return Effect(
            category="effect",
            description="test",
            evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
            confidence=confidence,
            score=score,
            magnitude=magnitude,
            reasoning="test",
        )

    def test_certain_trivial_off_target_does_not_sink_a_large_benefit(self):
        """
        The original bug: an off-target we were 95% sure about but which barely
        mattered outweighed a substantial predicted benefit, because confidence
        was subtracted as though it were effect size.
        """
        benefit = self._effect(score=0.6, magnitude=0.9)
        trivial_but_certain = self._effect(score=0.95, magnitude=0.02)

        net, _ = self.scorer.combine_effect_scores([benefit, trivial_but_certain])
        self.assertGreater(net, 0.4, "a large benefit must survive a trivial off-target")

    def test_severe_off_target_does_outweigh_a_small_benefit(self):
        small_benefit = self._effect(score=0.9, magnitude=0.1)
        severe_cost = self._effect(score=0.9, magnitude=0.9)

        net, _ = self.scorer.combine_effect_scores([small_benefit, severe_cost])
        self.assertLess(net, 0.0, "a severe off-target must outweigh a marginal benefit")

    def test_off_target_count_does_not_dominate_the_score(self):
        """
        Costs combine as independent risks, so adding more near-zero checks must
        not push a good substitution negative purely by arithmetic.
        """
        benefit = self._effect(score=0.9, magnitude=0.8)
        noise = [self._effect(score=0.3, magnitude=0.02) for _ in range(12)]

        net, _ = self.scorer.combine_effect_scores([benefit] + noise)
        self.assertGreater(net, 0.5, "twelve negligible checks must not sink the score")

    def test_zero_magnitude_contributes_nothing(self):
        benefit = self._effect(score=0.8, magnitude=0.5)
        inert = self._effect(score=1.0, magnitude=0.0)

        with_inert, _ = self.scorer.combine_effect_scores([benefit, inert])
        alone, _ = self.scorer.combine_effect_scores([benefit])
        self.assertAlmostEqual(with_inert, alone, places=6)

    def test_breakdown_reconciles_with_reported_score(self):
        """A displayed score must be re-derivable from the terms shown beside it."""
        effects = [self._effect(0.9, 0.8), self._effect(0.5, 0.4), self._effect(0.3, 0.2)]

        net, _ = self.scorer.combine_effect_scores(effects)
        bd = self.scorer.explain_net_score(effects)

        self.assertAlmostEqual(bd["net"], net, places=4)
        self.assertAlmostEqual(
            bd["expected_benefit"] - bd["combined_cost"], bd["net"], places=3
        )

    def test_overall_confidence_ignores_immaterial_effects(self):
        benefit = self._effect(0.9, 0.8, ConfidenceLevel.HIGH)
        immaterial = self._effect(0.05, 0.05, ConfidenceLevel.LOW)

        _, conf = self.scorer.combine_effect_scores([benefit, immaterial])
        self.assertEqual(conf, ConfidenceLevel.HIGH)

    def test_limiting_effect_is_reported(self):
        benefit = self._effect(0.9, 0.8, ConfidenceLevel.HIGH)
        material_low = self._effect(0.5, 0.5, ConfidenceLevel.LOW)
        material_low.description = "the weak link"

        limiting = self.scorer.limiting_effect([benefit, material_low])
        self.assertEqual(limiting.description, "the weak link")


class TestConservationGating(unittest.TestCase):
    """Entropy over too few sequences must not be reported as a finding."""

    def setUp(self):
        self.predictor = SubstitutionPredictor()

    def test_no_conservation_claim_without_enough_homologs(self):
        """
        The original bug: with a single sequence, entropy is 0 everywhere by
        construction, and every position was flagged 'highly conserved' at
        DIRECT_EXPERIMENTAL tier — a fabricated quantity.
        """
        effect = self.predictor._conservation_penalty_effect(
            entropy=0.0, position=0, wt_aa="H", mut_aa="A", conservation_available=False
        )

        self.assertEqual(effect.magnitude, 0.0, "must not penalise on absent data")
        self.assertEqual(effect.evidence_tier, EvidenceTier.INFERENCE_ONLY)
        self.assertIn("NOT COMPUTED", effect.description)

    def test_conservation_claim_made_when_data_supports_it(self):
        effect = self.predictor._conservation_penalty_effect(
            entropy=0.1, position=0, wt_aa="H", mut_aa="A", conservation_available=True
        )

        self.assertGreater(effect.magnitude, 0.5)
        self.assertEqual(effect.evidence_tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_positions_are_reported_one_indexed(self):
        """Display text must match the mutation labels, which are 1-indexed."""
        effect = self.predictor._conservation_penalty_effect(
            entropy=0.8, position=1, wt_aa="A", mut_aa="C", conservation_available=True
        )
        self.assertIn("Position 2", effect.reasoning)


class TestCleavageLiability(unittest.TestCase):
    """P1 specificity matching must be real, not accidental substring matching."""

    def setUp(self):
        self.predictor = SubstitutionPredictor()

    def test_glp1_position_2_is_the_dpp4_site(self):
        """Ala at GLP-1 position 2 is the cleavage site clinical analogs engineer around."""
        hits = self.predictor.cleavage_liability(GLP1, 1)
        self.assertIn("dpp4", {h["protease"] for h in hits})

    def test_dpp4_is_positional_not_sequence_wide(self):
        """DPP4 acts at the N-terminus; an internal Ala is not a DPP4 site."""
        internal_ala = GLP1.index("A", 5)
        hits = self.predictor.cleavage_liability(GLP1, internal_ala)
        self.assertNotIn("dpp4", {h["protease"] for h in hits})

    def test_trypsin_matches_basic_residues_only(self):
        lys = GLP1.index("K")
        self.assertIn("trypsin", {h["protease"] for h in self.predictor.cleavage_liability(GLP1, lys)})

        ser = GLP1.index("S")
        self.assertNotIn("trypsin", {h["protease"] for h in self.predictor.cleavage_liability(GLP1, ser)})

    def test_substitution_introducing_a_site_is_not_scored_as_a_benefit(self):
        """Swapping in a cleavable residue works against protease resistance."""
        ser = GLP1.index("S")
        primary, _ = self.predictor.predict_substitution_effect(
            sequence=GLP1,
            position=ser,
            wild_type_aa="S",
            mutant_aa="K",  # introduces a trypsin site
            conservation_profile={},
            inferred_goal="protease_resistance",
        )

        self.assertEqual(primary.magnitude, 0.0)
        self.assertIn("INCREASED", primary.description)


class TestChargeRedistribution(unittest.TestCase):
    """The check compared identical windows and could never fire."""

    def setUp(self):
        self.predictor = SubstitutionPredictor()

    def test_charge_change_is_detected(self):
        seq = "AAAAKAAAA"
        mutant = "AAAADAAAA"  # +1 -> -1 at the centre
        effect = self.predictor._charge_redistribution_effect(seq, mutant, 4, ph=7.4)

        self.assertGreater(effect.magnitude, 0.05, "a charge reversal must register")
        self.assertIn("e", effect.reasoning)

    def test_neutral_swap_registers_as_immaterial(self):
        seq = "AAAALAAAA"
        mutant = "AAAAIAAAA"  # both uncharged
        effect = self.predictor._charge_redistribution_effect(seq, mutant, 4, ph=7.4)

        self.assertLessEqual(effect.magnitude, 0.05)


if __name__ == "__main__":
    unittest.main()
