"""
Partner-peptide module and the RAMP schema requirement.  [Addendum 2 section 8, step 6f]

Two things are being tested. The partner modes, which must be required rather
than optional because the four of them ship as different products and fail in
different ways. And the RAMP rule, which is the one that silently corrupts a
training set: CLR+RAMP1 is the CGRP receptor and CLR+RAMP2 is AM1, so affinities
against different accessory complexes are not the same quantity.
"""

import unittest

from peptide_suite.core.partner_module import (
    AccessoryStatus, PartnerCases, PartnerMode, PartnerProposal, ReceptorRecord,
    SchemaError, pool_measurements, triggers_partner_analysis,
)


class TestRampSchema(unittest.TestCase):

    def test_same_gene_different_accessory_is_a_different_complex(self):
        cgrp = ReceptorRecord("CLR", ("RAMP1",), "human")
        am1 = ReceptorRecord("CLR", ("RAMP2",), "human")
        self.assertNotEqual(cgrp, am1)
        self.assertFalse(cgrp.poolable_with(am1))

    def test_an_unrecorded_accessory_field_is_not_the_same_as_none(self):
        """
        The distinction the original type did not make. An accessory list empty
        because the receptor has none is a claim; one empty because nobody
        looked is an absence, and defaulting the field collapses them.
        """
        unspecified = ReceptorRecord("CLR", accessory=None)
        checked_none = ReceptorRecord("CLR", accessory=())
        self.assertIs(unspecified.status, AccessoryStatus.UNSPECIFIED)
        self.assertIs(checked_none.status, AccessoryStatus.DECLARED_NONE)
        self.assertFalse(unspecified.is_complete)
        self.assertTrue(checked_none.is_complete)
        self.assertFalse(unspecified.poolable_with(checked_none))

    def test_two_unspecified_records_still_do_not_pool(self):
        """
        Two records that both failed to say are not thereby known to agree.
        Pooling them is how three pharmacologies become one number.
        """
        a = ReceptorRecord("CLR", accessory=None)
        b = ReceptorRecord("CLR", accessory=None)
        self.assertFalse(a.poolable_with(b))

    def test_accessory_order_does_not_matter(self):
        self.assertTrue(ReceptorRecord("CTR", ("RAMP1", "RAMP3"))
                        .poolable_with(ReceptorRecord("CTR", ("RAMP3", "RAMP1"))))

    def test_species_is_part_of_the_identity(self):
        self.assertFalse(ReceptorRecord("CLR", ("RAMP1",), "human")
                         .poolable_with(ReceptorRecord("CLR", ("RAMP1",), "rat")))

    def test_the_identifier_shows_an_unspecified_field(self):
        """Invisible incompleteness is the failure; the identifier says it."""
        self.assertIn("UNSPECIFIED", ReceptorRecord("CLR", accessory=None).identifier)

    def test_require_complete_raises_with_the_reason(self):
        with self.assertRaises(SchemaError) as ctx:
            ReceptorRecord("CLR", accessory=None).require_complete()
        message = str(ctx.exception)
        self.assertIn("RAMP1", message)
        self.assertIn("CGRP", message)

    def test_pooling_groups_by_complex(self):
        cgrp = ReceptorRecord("CLR", ("RAMP1",))
        am1 = ReceptorRecord("CLR", ("RAMP2",))
        pools = pool_measurements([(cgrp, 1.0), (cgrp, 1.5), (am1, 9.0)])
        self.assertEqual(sorted(len(v) for v in pools.values()), [1, 2])

    def test_pooling_raises_rather_than_dropping_an_incomplete_record(self):
        """
        A silently smaller pool looks like a normal result, and the parameter
        budget downstream would be computed against a count that was never true.
        """
        with self.assertRaises(SchemaError):
            pool_measurements([(ReceptorRecord("CLR", ("RAMP1",)), 1.0),
                               (ReceptorRecord("CLR", accessory=None), 2.0)])

    def test_the_seeded_ramp_complexes_are_all_distinct(self):
        complexes = list(PartnerCases.ramp_complexes().values())
        for i, a in enumerate(complexes):
            for b in complexes[i + 1:]:
                with self.subTest(pair=(a.identifier, b.identifier)):
                    self.assertFalse(a.poolable_with(b))

    def test_clr_ramp1_is_cgrp_and_clr_ramp2_is_am1(self):
        ramps = PartnerCases.ramp_complexes()
        self.assertIn("CGRP", ramps["CLR+RAMP1"].identity)
        self.assertIn("AM1", ramps["CLR+RAMP2"].identity)


class TestPartnerModes(unittest.TestCase):

    def test_the_four_modes_exist(self):
        self.assertEqual(
            {m.value for m in PartnerMode},
            {"OBLIGATE_HETERODIMER", "SYNERGISTIC", "ACCESSORY_DEPENDENT",
             "CHIMERIC_CANDIDATE"})

    def test_every_mode_states_a_shipping_requirement(self):
        """The mode is not a label; it determines what the product is."""
        for mode in PartnerMode:
            with self.subTest(mode=mode.value):
                self.assertTrue(mode.shipping_requirement)

    def test_mode_is_required_on_a_proposal(self):
        with self.assertRaises(TypeError):
            PartnerProposal(primary="a", partner="b", rationale="c")

    def test_an_obligate_pair_needs_a_stoichiometry(self):
        proposal = PartnerProposal(primary="LtnA1", partner="LtnA2",
                                   mode=PartnerMode.OBLIGATE_HETERODIMER,
                                   rationale="neither is active alone")
        self.assertFalse(proposal.is_actionable)
        self.assertTrue(any("Stoichiometry" in u for u in proposal.unmet_requirements()))

    def test_a_synergistic_pair_needs_a_measured_index_not_an_assertion(self):
        """Superadditivity is the claim, so asserting it asserts the conclusion."""
        proposal = PartnerProposal(primary="magainin 2", partner="PGLa",
                                   mode=PartnerMode.SYNERGISTIC,
                                   rationale="they work well together",
                                   stoichiometry="1:1")
        self.assertTrue(any("synergy index" in u for u in proposal.unmet_requirements()))

    def test_an_accessory_dependent_proposal_must_name_the_accessory(self):
        proposal = PartnerProposal(primary="CGRP", partner="CLR",
                                   mode=PartnerMode.ACCESSORY_DEPENDENT,
                                   rationale="receptor pharmacology is RAMP-determined")
        self.assertTrue(any("Accessory protein" in u for u in proposal.unmet_requirements()))
        proposal.accessory_protein = "RAMP1"
        self.assertTrue(proposal.is_actionable)

    def test_a_chimeric_candidate_must_argue_fusability(self):
        proposal = PartnerProposal(primary="GLP-1", partner="GIP",
                                   mode=PartnerMode.CHIMERIC_CANDIDATE,
                                   rationale="both are useful in diabetes")
        self.assertTrue(any("fus" in u for u in proposal.unmet_requirements()))

    def test_requirements_are_per_mode_not_uniform(self):
        """A stoichiometry is meaningless for a chimera and mandatory for a pair."""
        chimera = PartnerProposal(primary="a", partner="b",
                                  mode=PartnerMode.CHIMERIC_CANDIDATE,
                                  rationale="fusion topology: N-terminal GLP-1 segment")
        self.assertTrue(chimera.is_actionable)
        self.assertEqual(chimera.stoichiometry, "")

    def test_a_partner_proposal_carries_no_score(self):
        proposal = PartnerProposal(primary="a", partner="b",
                                   mode=PartnerMode.SYNERGISTIC, rationale="x")
        self.assertFalse(hasattr(proposal, "score"))


class TestSeedCases(unittest.TestCase):

    def test_the_seed_set_is_present(self):
        cases = PartnerCases.all()
        for key in ("lacticin_3147", "magainin2_pgla", "conotoxin_lightning_strike_cabal",
                    "distinctin", "hnp1_homodimer", "incretin_dual_agonists"):
            self.assertIn(key, cases)

    def test_lacticin_is_obligate_not_synergistic(self):
        """Neither component kills alone, which is the whole distinction."""
        self.assertIs(PartnerCases.all()["lacticin_3147"].mode,
                      PartnerMode.OBLIGATE_HETERODIMER)

    def test_magainin_pgla_is_synergistic_not_obligate(self):
        self.assertIs(PartnerCases.all()["magainin2_pgla"].mode, PartnerMode.SYNERGISTIC)

    def test_the_homodimer_case_is_present(self):
        """
        A module keyed on 'find the other peptide' misses HNP1 entirely: the
        partner requirement is identical in kind and the partner is a second
        copy of the same molecule.
        """
        case = PartnerCases.all()["hnp1_homodimer"]
        self.assertIs(case.mode, PartnerMode.OBLIGATE_HETERODIMER)
        self.assertEqual(case.primary, case.partner)

    def test_every_case_carries_evidence_and_says_it_is_unverified(self):
        for key, case in PartnerCases.all().items():
            with self.subTest(case=key):
                self.assertTrue(case.citation)
                self.assertFalse(case.independently_verified)
                self.assertLessEqual(case.confidence, 0.5)

    def test_the_synergy_magnitude_is_attributed(self):
        notes = " ".join(PartnerCases.all()["magainin2_pgla"].notes)
        self.assertIn("addendum", notes)

    def test_lookup_by_component_name(self):
        self.assertTrue(PartnerCases.for_peptide("magainin 2"))
        self.assertTrue(PartnerCases.for_peptide("distinctin"))

    def test_an_unrelated_peptide_matches_nothing(self):
        self.assertEqual(PartnerCases.for_peptide("oxytocin"), [])


class TestTriggers(unittest.TestCase):

    def test_synergy_language_opens_the_module(self):
        for text in ("reported synergy with PGLa", "potentiates the response",
                     "co-secreted with amylin", "a two-component lantibiotic",
                     "dual agonist activity"):
            with self.subTest(text=text):
                self.assertTrue(triggers_partner_analysis(text))

    def test_ordinary_text_does_not(self):
        for text in ("increases protease resistance", "binds the receptor", ""):
            with self.subTest(text=text):
                self.assertFalse(triggers_partner_analysis(text))


class TestWorkflowIntegration(unittest.TestCase):

    def _run(self, sequence, name):
        from peptide_suite.workflows.transform import TransformWorkflow
        return TransformWorkflow().run(sequence, peptide_name=name)

    def test_a_seeded_peptide_gets_its_partner_proposal(self):
        result = self._run("GIGKFLHSAKKFGKAFVGEIMNS", "magainin 2")
        self.assertTrue(result["partner_proposals"])
        self.assertIs(result["partner_proposals"][0].mode, PartnerMode.SYNERGISTIC)

    def test_partner_proposals_stay_out_of_the_ranked_list(self):
        result = self._run("GIGKFLHSAKKFGKAFVGEIMNS", "magainin 2")
        descriptions = " ".join(t.description for t in result["transformations"])
        self.assertNotIn("PGLa", descriptions)

    def test_an_unseeded_peptide_gets_none(self):
        self.assertEqual(self._run("CYIQNCPLG", "oxytocin")["partner_proposals"], [])


if __name__ == "__main__":
    unittest.main()
