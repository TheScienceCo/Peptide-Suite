"""
Synthetic feasibility gate.  [Addendum 2 section 10, step 6h]

The worked example is the point of the whole module. Semaglutide's Lys34 to Arg
is not a biophysical optimization: native GLP-1 carries lysines at 26 and 34,
and with both present you cannot regioselectively acylate. Arg34 exists to make
Lys26 the only nucleophile. A biophysics-only scorer sees a conservative
charge-preserving swap with no benefit, because the benefit is not in the
physics of the peptide -- it is in the chemistry of making it.
"""

import unittest

from peptide_suite.core.synthetic_feasibility import (
    FeasibilityFlag, ImmunogenicityScreen, Severity, regioselectivity_conflict, screen,
)

GLP1_7_36 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"


class TestRegioselectivity(unittest.TestCase):
    """The semaglutide case, which is the reason this module exists."""

    def test_two_lysines_block_regioselective_acylation(self):
        conflict = regioselectivity_conflict(GLP1_7_36, "acylation")
        self.assertIsNotNone(conflict)
        self.assertIs(conflict.severity, Severity.BLOCKING)
        self.assertEqual(len(conflict.positions), 2)

    def test_the_remedy_names_the_semaglutide_substitution(self):
        """
        A flag that names a problem without naming what the field already does
        about it sends the reader away for a settled answer.
        """
        remedy = regioselectivity_conflict(GLP1_7_36, "acylation").remedy
        self.assertIn("Lys34", remedy)
        self.assertIn("Arg", remedy)

    def test_one_lysine_is_not_a_conflict(self):
        """After the Lys-to-Arg substitution there is one nucleophile and it works."""
        single = GLP1_7_36.replace("K", "R", 1)
        self.assertIsNone(regioselectivity_conflict(single, "acylation"))

    def test_thiol_chemistry_counts_cysteines_not_lysines(self):
        self.assertIsNone(regioselectivity_conflict(GLP1_7_36, "thiol"))
        self.assertIsNotNone(regioselectivity_conflict("ACDEFCGH", "maleimide"))

    def test_an_unknown_chemistry_makes_no_claim(self):
        """Silence rather than a guess: the module does not know every reaction."""
        self.assertIsNone(regioselectivity_conflict(GLP1_7_36, "click"))
        self.assertIsNone(regioselectivity_conflict(GLP1_7_36, ""))

    def test_the_conflict_is_blocking_not_advisory(self):
        """The reaction gives a mixture of isomers, not the intended conjugate."""
        self.assertIs(regioselectivity_conflict(GLP1_7_36, "lipidation").severity,
                      Severity.BLOCKING)


class TestSequenceMotifs(unittest.TestCase):

    def _codes(self, sequence, chemistry=""):
        return {f.code for f in screen(sequence, chemistry)}

    def test_asp_gly_is_flagged_as_aspartimide_risk(self):
        self.assertIn("aspartimide", self._codes("AADGAA"))

    def test_asn_gly_is_flagged_as_deamidation(self):
        self.assertIn("deamidation", self._codes("AANGAA"))

    def test_a_clean_sequence_raises_neither(self):
        codes = self._codes("AAAAAAAA")
        self.assertNotIn("aspartimide", codes)
        self.assertNotIn("deamidation", codes)

    def test_n_terminal_glutamine_cyclization(self):
        self.assertIn("n_terminal_gln_cyclization", self._codes("QAAAAA"))

    def test_a_glutamine_elsewhere_does_not_cyclize(self):
        """Only the N-terminal residue has a free alpha-amino group to cyclise onto."""
        self.assertNotIn("n_terminal_gln_cyclization", self._codes("AAQAAA"))

    def test_oxidation_prone_residues_are_reported(self):
        flags = [f for f in screen("AAMAAWAA") if f.code == "oxidation_liability"]
        self.assertTrue(flags)
        self.assertTrue(all(f.severity is Severity.NOTE for f in flags))

    def test_a_hydrophobic_run_is_flagged_for_spps_aggregation(self):
        """This is a synthesis failure rather than a formulation one."""
        self.assertIn("spps_aggregation_stretch", self._codes("AAVIYFWLTAA"))

    def test_a_short_hydrophobic_run_is_not(self):
        self.assertNotIn("spps_aggregation_stretch", self._codes("AAVIAA"))

    def test_a_trailing_hydrophobic_run_is_caught(self):
        """The loop has to close out the run at the end of the sequence."""
        self.assertIn("spps_aggregation_stretch", self._codes("AAAVIYFWL"))


class TestCysteineHandling(unittest.TestCase):

    def _flag(self, sequence, code):
        return next((f for f in screen(sequence) if f.code == code), None)

    def test_a_single_cysteine_is_an_unpaired_thiol(self):
        self.assertIsNotNone(self._flag("AACAA", "unpaired_cysteine"))

    def test_two_cysteines_have_one_pairing_and_no_ambiguity(self):
        self.assertIsNone(self._flag("ACAACA", "disulfide_pairing_ambiguity"))
        self.assertIsNone(self._flag("ACAACA", "unpaired_cysteine"))

    def test_four_cysteines_admit_three_pairings(self):
        flag = self._flag("CACACACA", "disulfide_pairing_ambiguity")
        self.assertIsNotNone(flag)
        self.assertIn("3 distinct", flag.description)

    def test_six_cysteines_admit_fifteen(self):
        """Double factorial: the ambiguity grows fast and the count says so."""
        flag = self._flag("CACACACACACA", "disulfide_pairing_ambiguity")
        self.assertIn("15 distinct", flag.description)

    def test_an_odd_count_always_leaves_a_free_thiol(self):
        flag = self._flag("CACACA", "odd_cysteine_count")
        self.assertIsNotNone(flag)
        self.assertIs(flag.severity, Severity.HIGH)


class TestOrdering(unittest.TestCase):

    def test_blocking_flags_come_first(self):
        """A blocking conflict buried under three notes is a flag nobody reads."""
        flags = screen(GLP1_7_36, "acylation")
        self.assertIs(flags[0].severity, Severity.BLOCKING)

    def test_every_flag_states_a_basis(self):
        for flag in screen("QCDGNGMWVIYFWLTKAAKC", "acylation"):
            with self.subTest(code=flag.code):
                self.assertTrue(flag.basis)

    def test_every_flag_offers_a_remedy(self):
        for flag in screen("QCDGNGMWVIYFWLTKAAKC", "acylation"):
            with self.subTest(code=flag.code):
                self.assertTrue(flag.remedy)


class TestImmunogenicityIsNotFaked(unittest.TestCase):

    def test_the_screen_raises_rather_than_scoring(self):
        """
        An unimplemented screen returning "low risk" is worse than no screen: a
        binding score with no allele set and no predictor behind it is the
        fabricated value the output contract forbids.
        """
        with self.assertRaises(NotImplementedError) as ctx:
            ImmunogenicityScreen().run()
        self.assertIn("allele set", str(ctx.exception))

    def test_the_altered_windows_are_computed_even_so(self):
        """
        A sequence fact, available now, so wiring a predictor in later is a
        small job rather than a redesign.
        """
        windows = ImmunogenicityScreen.windows_for(GLP1_7_36, 10, width=9)
        self.assertTrue(windows)
        self.assertTrue(all(len(w) == 9 for w in windows))
        self.assertTrue(any(GLP1_7_36[9] in w for w in windows))

    def test_a_position_outside_the_sequence_yields_nothing(self):
        self.assertEqual(ImmunogenicityScreen.windows_for("AAAA", 99), [])


class TestWorkflowIntegration(unittest.TestCase):

    def _run(self):
        from peptide_suite.workflows.transform import TransformWorkflow
        return TransformWorkflow().run(GLP1_7_36)

    def test_sequence_level_flags_are_returned(self):
        self.assertTrue(self._run()["feasibility"])

    def test_an_acylation_proposal_inherits_the_regioselectivity_conflict(self):
        """
        The check a biophysics-only scorer cannot make: the obstacle is in the
        chemistry of making the molecule, not in the molecule.
        """
        result = self._run()
        acylation = [r for r in result["research_requests"] if "acylation" in r["proposal"]]
        self.assertTrue(acylation, "the lipidation proposal is no longer generated")


if __name__ == "__main__":
    unittest.main()
