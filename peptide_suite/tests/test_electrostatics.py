"""
Multi-pH protonation.  [Addendum 2 section 6b]

The claims this module makes are checkable against the Henderson-Hasselbalch
equation directly, so these tests check the chemistry rather than the plumbing:
a group at its own pKa is half protonated, acids and bases carry opposite sign,
and the only residues reported as pH-switchable are the ones whose pKa actually
sits inside the window examined.
"""

import unittest

from peptide_suite.core.charge_calculator import ChargeCalculator
from peptide_suite.core.electrostatics import compute_profile, _protonated_fraction

# GLP-1(7-36): one histidine at the N-terminus, several carboxylates, no
# cysteine. The histidine is the only group whose pKa lies inside 5.5-7.4.
GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"


class TestHendersonHasselbalch(unittest.TestCase):

    def test_group_at_its_own_pka_is_half_protonated(self):
        self.assertAlmostEqual(_protonated_fraction(6.0, 6.0), 0.5, places=9)

    def test_one_unit_below_pka_is_ninety_one_percent(self):
        """A decade of ratio per pH unit: 10:1 protonated one unit below."""
        self.assertAlmostEqual(_protonated_fraction(6.0, 5.0), 10 / 11, places=9)

    def test_one_unit_above_pka_is_nine_percent(self):
        self.assertAlmostEqual(_protonated_fraction(6.0, 7.0), 1 / 11, places=9)

    def test_protonation_decreases_monotonically_with_ph(self):
        previous = 1.0
        for ph in [x / 2 for x in range(0, 29)]:
            current = _protonated_fraction(7.0, ph)
            self.assertLessEqual(current, previous)
            previous = current


class TestSwitchableResidues(unittest.TestCase):

    def setUp(self):
        self.profile = compute_profile(GLP1)

    def test_histidine_is_the_only_switchable_residue(self):
        """
        The chemistry: only a group whose pKa falls inside the pH window can
        change state across it. Among the standard side chains that is
        histidine alone for 5.5-7.4.
        """
        self.assertEqual([t.residue for t in self.profile.switchable], ["H"])
        self.assertEqual(self.profile.switchable[0].display_position, 1)

    def test_histidine_is_mostly_neutral_at_physiological_ph(self):
        his = self.profile.switchable[0]
        physiological = next(p for p in his.points if p.compartment == "physiological")
        self.assertLess(physiological.protonated_fraction, 0.15)

    def test_histidine_is_mostly_protonated_in_the_endosome(self):
        his = self.profile.switchable[0]
        endosomal = next(p for p in his.points if p.compartment == "endosomal")
        self.assertGreater(endosomal.protonated_fraction, 0.6)

    def test_lysine_and_arginine_do_not_move(self):
        """Their pKa sits four units above the window; nothing changes."""
        for titration in self.profile.titrations:
            if titration.residue in ("K", "R"):
                with self.subTest(residue=f"{titration.residue}{titration.display_position}"):
                    self.assertLess(titration.protonation_swing, 0.01)

    def test_net_charge_rises_as_ph_falls(self):
        """Protonation adds positive charge, so acidification can only raise it."""
        by_ph = sorted(
            ((self.profile.compartments[label], charge)
             for label, charge in self.profile.net_charge.items()),
            reverse=True,
        )
        charges = [charge for _ph, charge in by_ph]
        for higher_ph_charge, lower_ph_charge in zip(charges, charges[1:]):
            self.assertLessEqual(higher_ph_charge, lower_ph_charge)


class TestNoIonizableSideChains(unittest.TestCase):

    def test_peptide_of_neutral_residues_reports_no_titrations(self):
        profile = compute_profile("AAGGLLVV")
        self.assertEqual(profile.titrations, [])
        self.assertIn("comes entirely from its termini", profile.summary())

    def test_termini_still_carry_charge(self):
        """
        The case that made pI report 0.0 before the termini were counted: a
        peptide with no ionizable side chain is not an uncharged peptide.
        """
        profile = compute_profile("AAGGLLVV")
        physiological = profile.terminal_charges["physiological"]
        self.assertGreater(physiological["n_terminus"], 0.5)
        self.assertLess(physiological["c_terminus"], -0.5)


class TestUncertaintyIsReportedWhereItMatters(unittest.TestCase):

    def test_histidine_spread_is_flagged(self):
        """Its cited range spans the window, so it changes the answer."""
        profile = compute_profile(GLP1)
        his = profile.switchable[0]
        self.assertIn("Sources disagree", his.uncertainty_note(profile.switch_threshold))
        self.assertGreater(his.source_disagreement(), 0.3)

    def test_arginine_spread_is_not_flagged(self):
        """Five units away from the window; it moves nothing."""
        profile = compute_profile(GLP1)
        arg = next(t for t in profile.titrations if t.residue == "R")
        self.assertEqual(arg.uncertainty_note(profile.switch_threshold), "")

    def test_glutamate_spread_is_not_flagged(self):
        """
        The regression this replaced: a proximity rule flagged every glutamate
        in the sequence. Their cited range is 4.1-4.5, and at pH 5.5 and above
        both ends give a protonated fraction near zero, so the disagreement
        changes nothing. Three identical warnings that change nothing are how a
        reader learns to skip warnings.
        """
        profile = compute_profile(GLP1)
        for titration in profile.titrations:
            if titration.residue == "E":
                with self.subTest(position=titration.display_position):
                    self.assertLess(titration.source_disagreement(), 0.1)
                    self.assertEqual(titration.uncertainty_note(profile.switch_threshold), "")

    def test_absent_structure_is_stated(self):
        profile = compute_profile(GLP1)
        self.assertTrue(any("No structure was supplied" in n for n in profile.notes))


class TestTyrosineHasNoDiscontinuity(unittest.TestCase):
    """
    The charge curve used to snap tyrosine to zero below a fixed ionized
    fraction, which put a step in a titration curve that is smooth by
    construction.
    """

    def test_charge_is_continuous_across_the_old_cutoff(self):
        calculator = ChargeCalculator()
        charges = [calculator.charge_at_ph("Y", ph=ph / 100).effective_charge
                   for ph in range(900, 1000)]
        for previous, current in zip(charges, charges[1:]):
            self.assertLess(abs(current - previous), 0.02)

    def test_tyrosine_is_negative_when_ionized(self):
        calculator = ChargeCalculator()
        self.assertLess(calculator.charge_at_ph("Y", ph=12.0).effective_charge, -0.9)


class TestReferenceDataIsWellFormed(unittest.TestCase):

    def test_every_pka_lies_inside_its_own_cited_range(self):
        reference = ChargeCalculator.reference()
        for block in ("sidechain", "terminal"):
            for name, entry in reference[block].items():
                lo, hi = entry["cited_range"]
                with self.subTest(group=name):
                    self.assertLessEqual(lo, entry["pka"])
                    self.assertLessEqual(entry["pka"], hi)

    def test_context_dependent_groups_are_the_ones_with_notes_or_wide_spread(self):
        """A group marked context-dependent should say why."""
        reference = ChargeCalculator.reference()
        for block in ("sidechain", "terminal"):
            for name, entry in reference[block].items():
                if not entry.get("context_dependent"):
                    continue
                lo, hi = entry["cited_range"]
                with self.subTest(group=name):
                    self.assertTrue(entry.get("note") or (hi - lo) >= 0.4)


if __name__ == "__main__":
    unittest.main()
