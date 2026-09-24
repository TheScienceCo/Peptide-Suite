"""
Blind recovery: does the scan propose what was actually made?

The tests are mostly about the ways this benchmark could report a number that
means nothing -- a blinding that did not hold, a position placed in the wrong
numbering scheme, a modification the method could never have proposed counted
as a miss, and a rank with no null to read it against.
"""

import unittest

from peptide_suite.core.holdout import GoldenSet, Protocol
from peptide_suite.core.recovery import (
    Blinding, CandidateSpace, NullBaseline, NumberingError, RecoveryResult,
    blind_by_date, blind_by_motif, leakage_after_blinding, map_to_sequence,
    run_benchmark, summarise,
)

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"


class TestNamedEntityBlindingIsNotEnough(unittest.TestCase):
    """
    The argument the whole protocol choice rests on, made concrete rather than
    asserted: removing semaglutide leaves the answer in the corpus twice over.
    """

    def test_every_semaglutide_motif_survives_a_named_exclusion(self):
        leakage = leakage_after_blinding("semaglutide")
        for motif in GoldenSet.motifs_of("semaglutide"):
            with self.subTest(motif=motif):
                self.assertIn(motif, leakage)
                self.assertTrue(leakage[motif])

    def test_the_specific_molecules_are_named(self):
        leakage = leakage_after_blinding("semaglutide")
        self.assertIn("taspoglutide", leakage["aib"])
        self.assertIn("liraglutide", leakage["gamma_glu_linker"])

    def test_named_entity_is_not_a_clean_protocol(self):
        self.assertFalse(Protocol.NAMED_ENTITY.is_clean_holdout)
        self.assertTrue(Protocol.MOTIF_ABLATION.is_clean_holdout)
        self.assertTrue(Protocol.TEMPORAL.is_clean_holdout)


class TestMotifAblationRemovesTheChemistry(unittest.TestCase):

    def test_it_removes_more_than_the_named_drug(self):
        blinding = blind_by_motif("semaglutide")
        self.assertIn("semaglutide", blinding.removed_drugs)
        self.assertIn("taspoglutide", blinding.removed_drugs)
        self.assertIn("liraglutide", blinding.removed_drugs)
        self.assertGreater(len(blinding.removed_drugs), 1)

    def test_it_leaves_no_residual_leakage(self):
        blinding = blind_by_motif("semaglutide")
        self.assertEqual(blinding.residual_leakage, {})
        self.assertTrue(blinding.is_clean)

    def test_a_drug_with_no_motifs_cannot_be_ablated(self):
        blinding = blind_by_motif("not-a-drug")
        self.assertFalse(blinding.is_clean)
        self.assertIn("nothing to ablate", blinding.note)

    def test_an_unverified_blinding_forbids_a_rank(self):
        blinding = Blinding(protocol=Protocol.MOTIF_ABLATION, verified=False)
        self.assertFalse(blinding.is_clean)
        self.assertIn("NOT VERIFIED", blinding.describe())


class TestTemporalRefusesWithoutAYear(unittest.TestCase):
    """
    A temporal holdout's entire claim is that the modification was published
    after the cutoff. With no year there is no claim, and a plausible invented
    one produces a benchmark that looks rigorous and tests nothing.
    """

    def test_no_year_means_no_run(self):
        blinding = blind_by_date("semaglutide")
        self.assertFalse(blinding.is_clean)
        self.assertIn("No first-publication year", blinding.note)
        self.assertIn("first_published", blinding.note)

    def test_a_cutoff_after_publication_is_not_a_holdout(self):
        drugs = GoldenSet.drugs()
        drugs["__test__"] = {"parent": "x", "modifications": [], "motifs": [],
                             "first_published": 2012}
        try:
            self.assertFalse(blind_by_date("__test__", cutoff_year=2015).is_clean)
            self.assertTrue(blind_by_date("__test__", cutoff_year=2010).is_clean)
        finally:
            drugs.pop("__test__")

    def test_a_temporal_outcome_must_state_its_cutoff(self):
        from peptide_suite.core.holdout import RankedOutcome, ReportingViolation
        with self.assertRaises(ReportingViolation):
            RankedOutcome(modification="x", protocol=Protocol.TEMPORAL,
                          rank=1, total_proposals=10)


class TestTheCandidateSpaceIsNotAMiss(unittest.TestCase):
    """
    The scan enumerates single canonical substitutions. A non-canonical residue
    or an attached chain could not have been proposed by any ranking, so
    scoring it as a miss understates the method and dropping it overstates it.
    """

    def test_a_non_canonical_residue_is_out_of_scope(self):
        verdict, position, residue = CandidateSpace.classify("Aib8")
        self.assertEqual(verdict, CandidateSpace.NON_CANONICAL)
        self.assertEqual(position, 8)

    def test_an_attached_chain_is_not_a_substitution(self):
        for text in ("C18 diacid at Lys26 via gamma-Glu-2xOEG",
                     "C16 palmitoyl at Lys26 via gamma-Glu",
                     "hexadecanedioic acid at LysB29 via gamma-Glu"):
            with self.subTest(modification=text):
                self.assertEqual(CandidateSpace.classify(text)[0],
                                 CandidateSpace.NOT_A_SUBSTITUTION)

    def test_a_canonical_substitution_is_in_scope(self):
        verdict, position, residue = CandidateSpace.classify("Arg34")
        self.assertEqual(verdict, CandidateSpace.IN)
        self.assertEqual((position, residue), (34, "R"))

    def test_an_unreadable_modification_is_not_guessed_at(self):
        self.assertEqual(CandidateSpace.classify("multiple D-residues")[0],
                         CandidateSpace.UNPARSEABLE)


class TestNumberingIsCheckedNotAssumed(unittest.TestCase):
    """
    Semaglutide's Arg34 is GLP-1 (7-37) numbering: position 34 is index 28 of
    the 31-residue stored chain. Applied without the offset it lands six
    residues away, the scan is asked about the wrong substitution, and the run
    still produces a presentable number.
    """

    def test_the_offset_is_applied(self):
        mapped = map_to_sequence(GLP1, 34, "R", offset=6)
        self.assertEqual(mapped.position, 28)
        self.assertEqual(mapped.wild_type, "K")
        self.assertEqual(mapped.stated_position, 34)

    def test_a_missing_offset_refuses_rather_than_assuming_zero(self):
        with self.assertRaises(NumberingError) as caught:
            map_to_sequence(GLP1, 34, "R", offset=None)
        self.assertIn("numbering offset", str(caught.exception))

    def test_an_offset_that_runs_off_the_sequence_is_refused(self):
        with self.assertRaises(NumberingError):
            map_to_sequence(GLP1, 34, "R", offset=0)

    def test_a_position_already_holding_the_mutant_is_refused(self):
        """Either the offset is wrong or this is not the right parent."""
        with self.assertRaises(NumberingError) as caught:
            map_to_sequence(GLP1, 34, "K", offset=6)
        self.assertIn("already", str(caught.exception))

    def test_the_golden_set_records_offsets_for_the_glp1_drugs(self):
        for drug in ("semaglutide", "liraglutide", "taspoglutide"):
            with self.subTest(drug=drug):
                self.assertEqual(GoldenSet.drugs()[drug]["numbering_offset"], 6)


class TestARankIsReadAgainstANull(unittest.TestCase):

    def test_the_null_reports_a_median_and_a_p_value(self):
        null = NullBaseline(trials=1000, median_rank=294.0, better_or_equal=320,
                            total_candidates=589)
        self.assertAlmostEqual(null.p_value, 321 / 1001, places=6)
        self.assertIn("median rank", null.describe())

    def test_no_trials_means_no_p_value_and_says_so(self):
        null = NullBaseline(trials=0, median_rank=None, better_or_equal=0,
                            total_candidates=589)
        self.assertIsNone(null.p_value)
        self.assertIn("nothing to be read against", null.describe())


class TestTheBenchmarkRunsAndRefusesHonestly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = run_benchmark(drugs=["semaglutide"])

    def test_every_modification_produces_a_result(self):
        """Including the unscoreable ones. A benchmark that silently dropped
        them would report the subset it could score and call that the result."""
        self.assertEqual(len(self.results),
                         len(GoldenSet.drugs()["semaglutide"]["modifications"]))

    def test_the_canonical_substitution_is_scored(self):
        scored = [r for r in self.results if r.is_evaluable]
        self.assertEqual(len(scored), 1)
        self.assertTrue(scored[0].outcome.was_proposed)
        self.assertIsNotNone(scored[0].null)

    def test_the_others_are_out_of_scope_not_missed(self):
        out = [r for r in self.results if r.candidate_space != CandidateSpace.IN]
        self.assertEqual(len(out), 2)
        for result in out:
            with self.subTest(modification=result.modification):
                self.assertIn("NOT RECOVERABLE BY THIS METHOD", result.report())
                self.assertIn("not as a miss", result.report())

    def test_the_report_names_the_ablated_motifs(self):
        scored = [r for r in self.results if r.is_evaluable][0]
        self.assertIn("ablated:", scored.report())
        self.assertIn("aib", scored.report())

    def test_the_summary_refuses_the_hit_rate_framing(self):
        text = summarise(self.results)
        self.assertIn("ranks", text)
        self.assertIn("this contract exists to refuse", text)

    def test_the_summary_separates_out_of_scope_from_failure(self):
        self.assertIn("are not misses", summarise(self.results))

    def test_an_empty_run_says_so(self):
        self.assertIn("No recovery cases", summarise([]))


if __name__ == "__main__":
    unittest.main()
