"""
Holdout protocols and rank-based reporting.  [Addendum 2 section 11, step 6i]

The specific failure: excluding every document naming semaglutide leaves
liraglutide (Arg34, gamma-Glu-linked acylation at Lys26) and taspoglutide
([Aib8, Aib35]-GLP-1) intact, and the union of those two is the complete
answer. The experiment demonstrates correct retrieval and recombination of
established motifs -- a real capability -- and not de novo discovery.

The reporting contract is the deliverable: never a binary hit. "2nd of 47 under
motif ablation, 0.11 false-positive rate on decoys" is a claim. "Predicted two
of three modifications" is not.
"""

import unittest

from peptide_suite.core.holdout import (
    DecoyOutcome, GoldenSet, HoldoutReport, Protocol, RankedOutcome, ReportingViolation,
)


class TestNamedEntityExclusionIsInsufficient(unittest.TestCase):
    """The argument, made from the data rather than asserted."""

    def test_every_semaglutide_motif_survives_a_name_based_exclusion(self):
        leakage = GoldenSet.named_entity_leakage("semaglutide")
        uncovered = [m for m in GoldenSet.motifs_of("semaglutide") if m not in leakage]
        self.assertEqual(uncovered, [],
                         f"these motifs are genuinely removed by name exclusion: {uncovered}")

    def test_liraglutide_and_taspoglutide_are_the_named_leak(self):
        leakage = GoldenSet.named_entity_leakage("semaglutide")
        self.assertIn("liraglutide", leakage["gamma_glu_linker"])
        self.assertIn("taspoglutide", leakage["aib"])

    def test_the_union_of_two_other_drugs_covers_everything(self):
        """Two drugs between them carry every motif semaglutide has."""
        covered = set(GoldenSet.motifs_of("liraglutide")) | set(GoldenSet.motifs_of("taspoglutide"))
        self.assertTrue(set(GoldenSet.motifs_of("semaglutide")).issubset(covered))

    def test_named_entity_is_not_a_clean_holdout(self):
        self.assertFalse(Protocol.NAMED_ENTITY.is_clean_holdout)

    def test_the_replacement_protocols_are(self):
        for protocol in (Protocol.MOTIF_ABLATION, Protocol.TEMPORAL, Protocol.DECOY):
            with self.subTest(protocol=protocol.value):
                self.assertTrue(protocol.is_clean_holdout)


class TestProtocolsRequireTheirEvidence(unittest.TestCase):

    def test_motif_ablation_must_name_the_ablated_motifs(self):
        """
        Without them the run cannot be told apart from a named-entity
        exclusion, which is the protocol it replaces.
        """
        with self.assertRaises(ReportingViolation) as ctx:
            RankedOutcome("Aib8", Protocol.MOTIF_ABLATION, rank=1, total_proposals=10)
        self.assertIn("name the motifs", str(ctx.exception))

    def test_a_temporal_holdout_must_state_its_cutoff(self):
        with self.assertRaises(ReportingViolation) as ctx:
            RankedOutcome("Aib8", Protocol.TEMPORAL, rank=1, total_proposals=10)
        self.assertIn("cutoff year", str(ctx.exception))

    def test_a_complete_motif_ablation_is_accepted(self):
        outcome = RankedOutcome("Aib8", Protocol.MOTIF_ABLATION, rank=2,
                                total_proposals=47, ablated_motifs=["aib"])
        self.assertEqual(outcome.proposals_ranked_above, 1)


class TestRankBasedReporting(unittest.TestCase):

    def _make(self, **over):
        kwargs = dict(modification="Aib8", protocol=Protocol.MOTIF_ABLATION,
                      rank=2, total_proposals=47, ablated_motifs=["aib"])
        kwargs.update(over)
        return RankedOutcome(**kwargs)

    def test_the_report_states_rank_and_total(self):
        """2nd of 3 and 2nd of 47 are different results; a rank alone hides that."""
        report = self._make().report()
        self.assertIn("2nd of 47", report)
        self.assertIn("1 ranked above", report)

    def test_the_report_names_the_protocol_and_the_ablation(self):
        report = self._make().report()
        self.assertIn("motif_ablation", report)
        self.assertIn("aib", report)

    def test_a_named_entity_result_is_labelled_as_not_a_clean_holdout(self):
        report = self._make(protocol=Protocol.NAMED_ENTITY, ablated_motifs=[]).report()
        self.assertIn("NOT a clean holdout", report)

    def test_a_modification_that_was_never_proposed_says_so(self):
        report = self._make(rank=None).report()
        self.assertIn("not proposed at all", report)

    def test_there_is_no_binary_hit_field(self):
        """
        A system whose output can be reduced to yes-or-no will be, and 2nd of
        47 gets written up alongside 2nd of 3.
        """
        outcome = self._make()
        for attribute in ("hit", "success", "correct", "passed"):
            self.assertFalse(hasattr(outcome, attribute))

    def test_ordinals_are_written_correctly(self):
        for rank, expected in ((1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"),
                               (11, "11th"), (12, "12th"), (13, "13th"),
                               (21, "21st"), (22, "22nd"), (101, "101st")):
            with self.subTest(rank=rank):
                self.assertIn(expected, self._make(rank=rank,
                                                      total_proposals=200).report())


class TestDecoyControls(unittest.TestCase):

    def test_the_false_positive_rate_is_reported(self):
        decoy = DecoyOutcome("aib", trials=18, times_proposed=2)
        self.assertAlmostEqual(decoy.false_positive_rate, 2 / 18)
        self.assertIn("0.11", decoy.report())

    def test_no_trials_means_no_rate_rather_than_zero(self):
        """An unmeasured false-positive rate is not a false-positive rate of zero."""
        decoy = DecoyOutcome("aib", trials=0, times_proposed=0)
        self.assertIsNone(decoy.false_positive_rate)
        self.assertIn("none is assumed", decoy.report())

    def test_the_seeded_decoy_list_is_empty_and_says_why(self):
        """
        A decoy asserted without evidence manufactures a false-positive rate,
        which is worse than having none.
        """
        self.assertEqual(GoldenSet.decoys(), [])


class TestClaimGuidance(unittest.TestCase):

    def _clean(self):
        return HoldoutReport(
            outcomes=[RankedOutcome("Aib8", Protocol.MOTIF_ABLATION, rank=2,
                                    total_proposals=47, ablated_motifs=["aib"])],
            decoys=[DecoyOutcome("aib", trials=18, times_proposed=2)])

    def test_a_clean_protocol_with_decoys_permits_the_stronger_claim(self):
        self.assertTrue(self._clean().permits_discovery_claim())

    def test_missing_decoys_withhold_it(self):
        report = self._clean()
        report.decoys = []
        self.assertFalse(report.permits_discovery_claim())
        self.assertIn("no decoy controls", report.claim_guidance())

    def test_a_named_entity_result_withholds_it(self):
        report = HoldoutReport(
            outcomes=[RankedOutcome("Aib8", Protocol.NAMED_ENTITY, rank=2,
                                    total_proposals=47)],
            decoys=[DecoyOutcome("aib", trials=18, times_proposed=2)])
        self.assertFalse(report.permits_discovery_claim())

    def test_the_weaker_claim_is_stated_as_real_rather_than_as_a_failure(self):
        """
        Correct retrieval and recombination of established motifs is a genuine
        capability, and the guidance says so rather than only saying what the
        results are not.
        """
        report = self._clean()
        report.decoys = []
        guidance = report.claim_guidance()
        self.assertIn("real and useful", guidance)
        self.assertIn("recombination", guidance)

    def test_an_empty_report_permits_nothing(self):
        self.assertFalse(HoldoutReport().permits_discovery_claim())
        self.assertIn("No holdout runs", HoldoutReport().summary())


class TestGoldenSet(unittest.TestCase):

    def test_the_seed_drugs_are_present(self):
        drugs = GoldenSet.drugs()
        for name in ("semaglutide", "liraglutide", "tirzepatide", "exenatide",
                     "pramlintide", "teriparatide", "degarelix", "plecanatide",
                     "setmelanotide", "insulin_lispro", "insulin_glargine",
                     "insulin_degludec"):
            self.assertIn(name, drugs)

    def test_every_drug_declares_its_motifs(self):
        for name, entry in GoldenSet.drugs().items():
            with self.subTest(drug=name):
                self.assertTrue(entry.get("motifs"))

    def test_motif_lookup_is_symmetric(self):
        for motif in ("aib", "lipidation"):
            for drug in GoldenSet.drugs_carrying(motif):
                with self.subTest(motif=motif, drug=drug):
                    self.assertIn(motif, GoldenSet.motifs_of(drug))


if __name__ == "__main__":
    unittest.main()
