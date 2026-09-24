"""
Binding scores that know which residues touch the receptor.

The bug: `_predict_binding_effect` computed the charge and hydrophobicity
change and scored them identically whether the residue sat in the binding
interface or pointed into solvent, because nothing in it ever consulted a
contact map. A 30-residue peptide has 570 candidate substitutions, and the
ranking was, with respect to binding, arbitrary -- a charge swap on a
solvent-facing loop outranked a conservative change inside the contact helix
whenever the loop change was physicochemically larger.
"""

import unittest

from peptide_suite.core import EvidenceTier
from peptide_suite.core.biological_context import (
    ContactContext, InterfaceSpan, Provenance, SourceKind, contact_context,
    contact_spans, contact_spans_at, has_contact_map, retrieve,
)
from peptide_suite.core.substitution_predictor import SubstitutionPredictor

CURATED = Provenance(kind=SourceKind.CURATED_UNVERIFIED, needs_verification=True)


def span(start, end, gene="RCPT", role="a contact region"):
    return InterfaceSpan(start=start, end=end, role=role, target_gene=gene,
                         provenance=CURATED)


def effect(position, contact, sequence="HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG", mutant="K"):
    return SubstitutionPredictor().predict_substitution_effect(
        sequence=sequence, position=position, wild_type_aa=sequence[position],
        mutant_aa=mutant, conservation_profile={},
        inferred_goal="binding_affinity", contact=contact)[0]


class TestASpanIsPositional(unittest.TestCase):

    def test_covers_is_inclusive_and_one_indexed(self):
        s = span(3, 5)
        self.assertEqual([s.covers(i) for i in range(1, 7)],
                         [False, False, True, True, True, False])

    def test_a_backwards_or_zero_span_is_refused(self):
        """A span that cannot be checked against a sequence is not storable."""
        for start, end in ((0, 5), (5, 3), (-1, 2)):
            with self.subTest(span=(start, end)):
                with self.assertRaises(ValueError):
                    span(start, end)

    def test_a_span_running_off_the_sequence_does_not_fit(self):
        self.assertFalse(span(1, 40).fits("A" * 30))
        self.assertTrue(span(1, 30).fits("A" * 30))


class TestTheScorerUsesTheContactMap(unittest.TestCase):

    CONTACT = ContactContext(spans=(span(1, 8, "GLP1R"), span(20, 31, "GLP1R")))

    def test_a_contact_position_keeps_its_magnitude(self):
        e = effect(2, self.CONTACT)          # position 3, inside 1-8
        self.assertGreater(e.magnitude, 0.0)
        self.assertIn("contact region", e.description)
        self.assertIn("GLP1R", e.reasoning)

    def test_a_non_contact_position_claims_no_binding_benefit(self):
        """
        The fix. Not a claim that the substitution does nothing -- a refusal to
        say it does anything to BINDING, which is what the goal asked about.
        """
        e = effect(11, self.CONTACT)         # position 12, between the spans
        self.assertEqual(e.magnitude, 0.0)
        self.assertIn("OUTSIDE every curated receptor-contact region", e.reasoning)
        self.assertIn("not because the substitution does nothing", e.reasoning)

    def test_the_non_contact_reason_names_the_regions_it_is_outside_of(self):
        e = effect(11, self.CONTACT)
        self.assertIn("1-8", e.reasoning)
        self.assertIn("20-31", e.reasoning)

    def test_a_contact_position_outranks_an_equal_perturbation_outside_one(self):
        """The behaviour the whole change exists for."""
        inside = effect(2, self.CONTACT, mutant="K")
        outside = effect(11, self.CONTACT, mutant="K")
        self.assertGreater(inside.magnitude, outside.magnitude)

    def test_no_map_is_reported_as_unknown_not_as_no_contact(self):
        """
        "Nothing binds here" and "nobody recorded what binds anywhere" are
        different answers, and only the first licenses a zero.
        """
        e = effect(11, ContactContext())
        self.assertGreater(e.magnitude, 0.0)
        self.assertIn("No receptor-contact map is available", e.reasoning)
        self.assertIn("carries no information", e.reasoning)

    def test_no_contact_map_is_the_same_as_no_context_at_all(self):
        self.assertEqual(effect(11, None).magnitude, effect(11, ContactContext()).magnitude)

    def test_the_tier_never_rises_above_biochemical_principle(self):
        """A contact region is curated, not measured. It does not buy a tier."""
        for contact in (self.CONTACT, ContactContext(), None):
            with self.subTest(contact=type(contact).__name__):
                self.assertIs(effect(2, contact).evidence_tier,
                              EvidenceTier.BIOCHEMICAL_PRINCIPLE)


class TestTheCuratedRecordsCarryContactMaps(unittest.TestCase):
    """
    The spans are the records' OWN curated domain regions restated as
    positions. No new structural claim is introduced by making them
    machine-readable, and nothing residue-level is asserted.
    """

    def test_glp1_and_igf1_both_have_maps(self):
        for name in ("GLP-1 (7-37)", "IGF-1"):
            with self.subTest(peptide=name):
                self.assertTrue(has_contact_map(retrieve(name=name)))

    def test_every_span_fits_the_sequence_it_describes(self):
        """A span running off the end means the record disagrees with itself."""
        for name in ("GLP-1 (7-37)", "IGF-1"):
            context = retrieve(name=name)
            for s in contact_spans(context):
                with self.subTest(peptide=name, span=(s.start, s.end)):
                    self.assertTrue(s.fits(context.mature_sequence))

    def test_spans_match_the_curated_domain_boundaries(self):
        """
        They are the same coordinates, which is the point: this is existing
        curated knowledge made usable, not new knowledge.
        """
        context = retrieve(name="IGF-1")
        boundaries = {(r.start, r.end) for r in context.regions}
        for s in contact_spans(context):
            with self.subTest(span=(s.start, s.end)):
                self.assertIn((s.start, s.end), boundaries)

    def test_the_igf1_d_domain_is_not_marked_as_contact(self):
        """
        Its curated role is "C-terminal extension with no insulin counterpart"
        and says nothing about receptor contact, so it is not claimed as one.
        Marking every domain a contact region would make the map meaningless.
        """
        context = retrieve(name="IGF-1")
        self.assertEqual(contact_spans_at(context, 66), [])
        self.assertTrue(contact_spans_at(context, 20))

    def test_no_span_claims_a_tier_above_biochemical_principle(self):
        for name in ("GLP-1 (7-37)", "IGF-1"):
            for s in contact_spans(retrieve(name=name)):
                with self.subTest(peptide=name, span=(s.start, s.end)):
                    self.assertIs(s.provenance.max_tier,
                                  EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_no_structure_identifier_is_claimed(self):
        """No PDB ID is verifiable from this environment, so none is recorded."""
        for name in ("GLP-1 (7-37)", "IGF-1"):
            for interface in retrieve(name=name).interfaces:
                with self.subTest(peptide=name):
                    self.assertEqual(interface.structures, ())


class TestTheScanUsesTheMapEndToEnd(unittest.TestCase):

    def test_a_named_peptide_gets_its_contact_map(self):
        from peptide_suite.workflows.optimize import _contact_context_for

        class Ctx:
            name = "IGF-1"
            sequence = retrieve(name="IGF-1").mature_sequence
        self.assertTrue(_contact_context_for(Ctx()).has_map)

    def test_an_unnamed_peptide_gets_an_empty_map_not_a_crash(self):
        from peptide_suite.workflows.optimize import _contact_context_for

        class Ctx:
            name = "unnamed_peptide"
            sequence = "AAAAAAAA"
        self.assertFalse(_contact_context_for(Ctx()).has_map)

    def test_binding_recommendations_say_which_region_they_are_in(self):
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        _ctx, recs = OptimizeWorkflow().run(
            "GPETLCGAELVDALQFVCGDRGFYFNKPTGYGSSSRRAPQTGIVDECCFRSCDLRRLEMYCAPLKPAKSA",
            confirmed_goal="binding_affinity", auto_confirm=True)
        self.assertTrue(recs)
        self.assertTrue(
            any("contact region" in r.primary_effect.description for r in recs),
            "no recommendation named the contact region it sits in")


if __name__ == "__main__":
    unittest.main()
