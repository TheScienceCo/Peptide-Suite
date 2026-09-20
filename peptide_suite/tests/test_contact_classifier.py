"""
Native-contact classification and design consequences.  [Addendum 2 section 7, step 6e]

The failure this module exists to catch is omission. A 28-residue peptide whose
activity depends on one octanoyl group looks, in its sequence, exactly like a
28-residue peptide that does not. Most of the golden cases are invisible in a
one-letter sequence: an acyl group, a sulfate, an amide, a D-residue, a glycan.

So these tests spend most of their effort on the cases the seed set exists to
stop the system getting wrong, and on the two directions of the rules — a freeze
that refuses everything is as useless as one that refuses nothing.
"""

import unittest

from peptide_suite.core.contact_classifier import (
    ClassificationError, ContactClass, ContactClassification, EssentialSubclass,
    GoldenCases, Verdict, residues_touched, restores_contact, rule_on, rule_proposal,
)


class TestEveryClassificationCarriesEvidence(unittest.TestCase):
    """
    The spec requires a citation and a confidence on every classification. An
    unevidenced one is an opinion, and an opinion that freezes a residue
    footprint is worse than no classification because it is indistinguishable
    from a verified one once it reaches the output.
    """

    def test_a_classification_without_a_citation_is_refused(self):
        with self.assertRaises(ClassificationError) as ctx:
            ContactClassification(contact="x", contact_class=ContactClass.SCAFFOLD,
                                  confidence=0.9)
        self.assertIn("no citation", str(ctx.exception))

    def test_a_classification_without_a_confidence_is_refused(self):
        with self.assertRaises(ClassificationError) as ctx:
            ContactClassification(contact="x", contact_class=ContactClass.SCAFFOLD,
                                  citation="somewhere")
        self.assertIn("no confidence", str(ctx.exception))

    def test_an_essential_contact_must_name_its_subclass(self):
        """
        Conformational, electrostatic, covalent-PTM and partner-dependent
        essentials need different substitutes, so the mechanism is part of the
        classification rather than a detail of it.
        """
        with self.assertRaises(ClassificationError) as ctx:
            ContactClassification(contact="x", contact_class=ContactClass.ESSENTIAL,
                                  citation="somewhere", confidence=0.9)
        self.assertIn("subclass", str(ctx.exception))

    def test_unresolved_needs_no_evidence(self):
        """It is the absence of evidence, so requiring evidence would be circular."""
        ContactClassification(contact="x", contact_class=ContactClass.UNRESOLVED)

    def test_every_golden_case_carries_a_citation_and_confidence(self):
        for key, case in GoldenCases.all().items():
            with self.subTest(case=key):
                self.assertTrue(case.citation)
                self.assertIsNotNone(case.confidence)

    def test_no_golden_case_claims_independent_verification(self):
        """
        Nothing here was checked against its source in this build, and saying
        otherwise would be the same fabrication as an invented PMID.
        """
        for key, case in GoldenCases.all().items():
            with self.subTest(case=key):
                self.assertFalse(case.independently_verified)
                self.assertLessEqual(case.confidence, 0.5)

    def test_quantitative_claims_attribute_their_magnitude(self):
        """
        The fold-changes came from the project addendum, not from a paper this
        system read. Attributing them to a paper would be an invented citation.
        """
        for key in ("cck8_tyr_sulfation", "c_terminal_amidation"):
            with self.subTest(case=key):
                self.assertTrue(GoldenCases.all()[key].magnitude_source)


class TestGoldenCases(unittest.TestCase):

    def test_the_seed_set_is_present(self):
        cases = GoldenCases.all()
        for key in ("insulin_b28_b29", "ghrelin_ser3_octanoylation", "cck8_tyr_sulfation",
                    "c_terminal_amidation", "pyroglutamate_n_terminus",
                    "gamma_carboxyglutamate", "dermorphin_d_ala2",
                    "drosocin_thr11_glycosylation"):
            self.assertIn(key, cases)

    def test_insulin_interface_is_scaffold_not_essential(self):
        """
        The proof case. It exists for zinc-hexamer storage, not receptor
        binding, and getting that class right is what produced lispro, aspart
        and degludec — one classification, opposite design directions.
        """
        self.assertIs(GoldenCases.all()["insulin_b28_b29"].contact_class,
                      ContactClass.SCAFFOLD)

    def test_ghrelin_acylation_is_an_essential_ptm(self):
        case = GoldenCases.all()["ghrelin_ser3_octanoylation"]
        self.assertIs(case.contact_class, ContactClass.ESSENTIAL)
        self.assertIs(case.subclass, EssentialSubclass.COVALENT_PTM)

    def test_gla_is_electrostatic_not_conformational(self):
        """Gla residues chelate calcium; the mechanism is the classification."""
        self.assertIs(GoldenCases.all()["gamma_carboxyglutamate"].subclass,
                      EssentialSubclass.ELECTROSTATIC)

    def test_dermorphin_d_ala_is_conformational(self):
        self.assertIs(GoldenCases.all()["dermorphin_d_ala2"].subclass,
                      EssentialSubclass.CONFORMATIONAL)

    def test_drosocin_is_present_to_break_a_wrong_prior(self):
        """
        Instinct says glycosylation is a protein phenomenon and can be ignored
        on a 19-residue peptide. That is right for mammalian hormones and wrong
        in general, and a prior that is right most of the time goes unexamined.
        """
        case = GoldenCases.all()["drosocin_thr11_glycosylation"]
        self.assertIs(case.contact_class, ContactClass.ESSENTIAL)
        self.assertTrue(any("prior" in note for note in case.notes))

    def test_pyroglutamate_carries_two_classes(self):
        case = GoldenCases.all()["pyroglutamate_n_terminus"]
        self.assertEqual([c.value for c in case.classes], ["ESSENTIAL", "PROTECTIVE"])

    def test_lookup_by_peptide_name(self):
        for name, expected in (("ghrelin", "Ser3"), ("insulin", "B28"),
                               ("drosocin", "Thr11"), ("oxytocin", "amidation")):
            with self.subTest(peptide=name):
                hits = GoldenCases.for_peptide(name)
                self.assertTrue(hits)
                self.assertTrue(any(expected in h.contact for h in hits))

    def test_an_unlisted_peptide_returns_nothing(self):
        self.assertEqual(GoldenCases.for_peptide("GLP-1"), [])


class TestDesignConsequences(unittest.TestCase):

    def setUp(self):
        self.cases = GoldenCases.all()

    def test_scaffold_disruption_is_encouraged(self):
        ruling = rule_on("ProB28Asp", self.cases["insulin_b28_b29"])
        self.assertIs(ruling.verdict, Verdict.ENCOURAGED)
        self.assertTrue(ruling.permits_emission)

    def test_essential_contacts_are_frozen(self):
        ruling = rule_on("Ser3 -> Ala", self.cases["ghrelin_ser3_octanoylation"])
        self.assertIs(ruling.verdict, Verdict.REJECTED)
        self.assertFalse(ruling.permits_emission)

    def test_a_frozen_contact_says_what_would_unfreeze_it(self):
        """A refusal with no route forward is a dead end rather than a rule."""
        ruling = rule_on("Ser3 -> Ala", self.cases["ghrelin_ser3_octanoylation"])
        self.assertIn("replaces the modification", ruling.required_substitute)

    def test_declaring_preservation_unfreezes_an_essential_contact(self):
        ruling = rule_on("Ser3 -> Cys with a thioester octanoyl surrogate",
                         self.cases["ghrelin_ser3_octanoylation"],
                         declares_preservation=True)
        self.assertIs(ruling.verdict, Verdict.PERMITTED)

    def test_a_dual_classified_contact_takes_the_stricter_rule(self):
        """
        Pyroglutamate is ESSENTIAL and PROTECTIVE. Checking whichever comes
        first would let a proposal through on the weaker of the two.
        """
        ruling = rule_on("replace pGlu with Glu", self.cases["pyroglutamate_n_terminus"])
        self.assertIs(ruling.verdict, Verdict.REJECTED)

    def test_a_substitute_does_not_satisfy_the_essential_half(self):
        ruling = rule_on("replace pGlu", self.cases["pyroglutamate_n_terminus"],
                         declares_substitute=True)
        self.assertIs(ruling.verdict, Verdict.REJECTED)

    def test_protective_removal_requires_a_substitute(self):
        protective = ContactClassification(
            contact="albumin-binding surface", contact_class=ContactClass.PROTECTIVE,
            citation="fixture", confidence=0.6)
        self.assertIs(rule_on("delete it", protective).verdict, Verdict.REQUIRES_SUBSTITUTE)
        self.assertIs(rule_on("delete it", protective, declares_substitute=True).verdict,
                      Verdict.PERMITTED)

    def test_unresolved_contacts_block_rather_than_being_ignored(self):
        """
        Absence and unexamined look identical in a sequence, so an unverified
        contact blocks instead of being treated as absent.
        """
        unresolved = ContactClassification(
            contact="unknown interface", contact_class=ContactClass.UNRESOLVED)
        ruling = rule_on("anything", unresolved)
        self.assertIs(ruling.verdict, Verdict.BLOCKED_UNRESOLVED)
        self.assertFalse(ruling.permits_emission)


class TestInstallingIsNotRemoving(unittest.TestCase):
    """
    A synthesised fragment of an amidated hormone arrives with a free acid,
    because solid-phase synthesis produces one. Proposing C-terminal amidation
    restores the native state rather than departing from it, and refusing it
    would block the one proposal that satisfies the freeze by construction.
    """

    def setUp(self):
        self.amide = GoldenCases.all()["c_terminal_amidation"]
        self.ghrelin = GoldenCases.all()["ghrelin_ser3_octanoylation"]

    def test_installing_the_required_modification_is_encouraged(self):
        self.assertIs(
            rule_on("N-terminal acetylation + C-terminal amidation", self.amide).verdict,
            Verdict.ENCOURAGED)

    def test_removal_language_is_decisive(self):
        """
        Regression. Keyword overlap alone cannot tell "add the amide" from
        "remove the amide" -- both name the amide -- and an earlier version read
        the deletion as a restoration, encouraging exactly what the freeze
        exists to refuse.
        """
        for proposal in ("remove the C-terminal amide to give the free acid",
                         "omit the amide", "truncate the C-terminal amide",
                         "C-terminus -> free acid", "unmodified C-terminus"):
            with self.subTest(proposal=proposal):
                self.assertIs(rule_on(proposal, self.amide).verdict, Verdict.REJECTED)

    def test_des_acyl_is_not_a_restoration(self):
        self.assertIs(rule_on("des-acyl form at Ser3", self.ghrelin).verdict,
                      Verdict.REJECTED)
        self.assertFalse(restores_contact("des-acyl form at Ser3", self.ghrelin))

    def test_restoration_applies_only_to_covalent_ptms(self):
        """A conformational contact is not 'installed' by naming it."""
        self.assertFalse(restores_contact(
            "D-Ala2 conformational constraint", GoldenCases.all()["dermorphin_d_ala2"]))


class TestFootprintMatching(unittest.TestCase):

    def test_residue_labels_and_positions_are_both_extracted(self):
        self.assertEqual(residues_touched("Ser3 -> Ala", 3), ["3", "Ser3"])

    def test_terminal_moves_are_recognised(self):
        touched = residues_touched("N-terminal acetylation + C-terminal amidation", None)
        self.assertIn("C-terminus", touched)
        self.assertIn("N-terminus", touched)

    def test_a_proposal_touching_no_classified_contact_is_not_ruled_on(self):
        class FakeMove:
            description = "Trp25 -> Phe"
            display_position = 25
        self.assertIsNone(rule_proposal(FakeMove(), GoldenCases.for_peptide("ghrelin")))

    def test_no_contacts_means_no_ruling(self):
        class FakeMove:
            description = "anything"
            display_position = 1
        self.assertIsNone(rule_proposal(FakeMove(), []))

    def test_the_strictest_ruling_wins_across_contacts(self):
        """
        A proposal overlapping both a SCAFFOLD and an ESSENTIAL contact is
        governed by the ESSENTIAL one; taking whichever was checked first would
        let the encouragement win.
        """
        class FakeMove:
            description = "Ser3 modification"
            display_position = 3
        scaffold = ContactClassification(
            contact="storage interface", contact_class=ContactClass.SCAFFOLD,
            residues=["Ser3"], citation="fixture", confidence=0.6)
        essential = GoldenCases.all()["ghrelin_ser3_octanoylation"]
        ruling = rule_proposal(FakeMove(), [scaffold, essential])
        self.assertIs(ruling.verdict, Verdict.REJECTED)


class TestWorkflowHonoursTheRules(unittest.TestCase):

    def _run(self, sequence, name):
        from peptide_suite.core.structure_template import TemplateCandidate
        from peptide_suite.workflows.transform import TransformWorkflow
        template = TemplateCandidate(
            identifier="fixture", is_experimental=True, is_complex=True,
            same_peptide=True, same_receptor=True)
        return TransformWorkflow().run(sequence, structure_candidates=[template],
                                       peptide_name=name)

    def test_a_listed_peptide_gets_its_classified_contacts(self):
        result = self._run("CYIQNCPLG", "oxytocin")
        self.assertTrue(result["classified_contacts"])

    def test_an_unlisted_peptide_gets_none(self):
        result = self._run("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", "GLP-1 (7-36) amide")
        self.assertEqual(result["classified_contacts"], [])

    def test_scaffold_contacts_are_surfaced_as_opportunities(self):
        result = self._run("GIVEQCCTSICSLYQLENYCN", "insulin")
        self.assertTrue(result["scaffold_opportunities"])

    def test_a_blocked_proposal_names_the_contact_and_its_citation(self):
        """A refusal that cannot say what it rests on is an assertion."""
        result = self._run("CYIQNCPLG", "oxytocin")
        blocked = [r for r in result["rejected"]
                   if str(r.get("blocked_by", "")).startswith("native_contact_")]
        for entry in blocked:
            self.assertTrue(entry["contact"])
            self.assertTrue(entry["citation"])


if __name__ == "__main__":
    unittest.main()
