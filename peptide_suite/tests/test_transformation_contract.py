"""
Tests for the transformation output contract.

These pin the properties that keep the output honest: tiers that cannot run must
refuse rather than approximate, unassessed objectives must stay distinguishable
from neutral ones, constraint moves must report pre-organization, and a trade
must be labelled as a trade.
"""

import unittest

from peptide_suite.core.epistemics import Claim, ClaimType, check_corpus_leakage
from peptide_suite.core.native_context import (
    ContactClass, NativeContextAnalyzer, PartnerClass,
)
from peptide_suite.core.physics_tiers import (
    PhysicsStack, Tier0Sequence, Tier1PKa, Tier2Ensemble, Tier3FMO,
)
from peptide_suite.core.preorganization import PreOrganizationAnalyzer
from peptide_suite.core.transformations import (
    Direction, MoveType, Objective, ObjectiveDelta, Transformation,
)
from peptide_suite.workflows.transform import TransformWorkflow

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"


class TestEpistemicContract(unittest.TestCase):

    def test_retrieved_claim_requires_a_citation(self):
        with self.assertRaises(ValueError):
            Claim.retrieved("albumin binding extends half-life", citations=[])

    def test_computed_claim_requires_a_named_method(self):
        with self.assertRaises(ValueError):
            Claim.computed("delta G = -4.2 kcal/mol", method="", tier=3)

    def test_inferred_claim_requires_the_inference_stated(self):
        with self.assertRaises(ValueError):
            Claim.inferred("this will improve potency", inference_step="")

    def test_computed_claims_render_as_untested(self):
        c = Claim.computed("muH = 0.41", method="Eisenberg moment", tier=0)
        self.assertIn("untested", c.render())
        self.assertFalse(c.experimentally_tested)

    def test_corpus_leakage_flagged_on_known_analog_modification(self):
        flag = check_corpus_leakage(GLP1, "Ala2 -> Aib for DPP-4 evasion")
        self.assertIsNotNone(flag)
        self.assertIn("semaglutide", flag.marketed_analogs)

    def test_no_leakage_flag_for_an_unrelated_move(self):
        self.assertIsNone(check_corpus_leakage(GLP1, "Trp25 -> Phe to remove an oxidation site"))


class TestPhysicsTierGating(unittest.TestCase):

    def test_tier3_refuses_without_a_tier2_ensemble(self):
        """The hard rule: no FMO number from a single unvalidated geometry."""
        with self.assertRaises(RuntimeError) as ctx:
            Tier3FMO().run(None)
        self.assertIn("Tier 2", str(ctx.exception))

    def test_tier3_refuses_when_tier2_is_unavailable(self):
        ensemble = Tier2Ensemble().run(GLP1)
        self.assertFalse(ensemble.available)
        with self.assertRaises(RuntimeError):
            Tier3FMO().run(ensemble)

    def test_tier2_names_its_missing_dependency(self):
        result = Tier2Ensemble().run(GLP1)
        self.assertFalse(result.available)
        self.assertTrue(result.missing_dependency)
        self.assertTrue(result.unavailable_reason)

    def test_tier1_labels_pka_as_not_pocket_perturbed(self):
        result = Tier1PKa().run(GLP1)
        self.assertFalse(result.data["pocket_perturbation_applied"])
        for residue in result.data["residues"]:
            self.assertIsNone(residue["pocket_perturbed_pka"])
            self.assertIn("NOT pocket-perturbed", residue["note"])

    def test_ionic_interaction_claim_is_blocked_without_structure(self):
        with self.assertRaises(RuntimeError) as ctx:
            Tier1PKa().assert_ionic_interaction_supportable(12)
        self.assertIn("pocket-perturbed", str(ctx.exception))

    def test_stack_reports_the_tier_it_actually_reached(self):
        result = PhysicsStack().run(GLP1)
        self.assertEqual(result["tier_reached"], 1)
        self.assertTrue(result["tier3_blocked_reason"])


class TestTier0Computations(unittest.TestCase):

    def setUp(self):
        self.t0 = Tier0Sequence()

    def test_hydrophobic_moment_is_high_for_an_amphipathic_helix(self):
        """Alternating polar/apolar on a 100-degree periodicity is the amphipathic case."""
        amphipathic = "LKKLLKLLKKLLKLLKKLLKL"
        uniform = "LLLLLLLLLLLLLLLLLLLLL"
        self.assertGreater(
            self.t0.hydrophobic_moment(amphipathic),
            self.t0.hydrophobic_moment(uniform),
        )

    def test_isoelectric_point_is_acidic_for_an_acidic_peptide(self):
        self.assertLess(self.t0.isoelectric_point("DDDEEE"), 5.0)
        self.assertGreater(self.t0.isoelectric_point("KKKRRR"), 9.0)

    def test_charge_curve_decreases_monotonically_with_ph(self):
        curve = self.t0.charge_vs_ph_curve(GLP1)
        charges = [c for _, c in curve]
        self.assertTrue(all(a >= b - 1e-6 for a, b in zip(charges, charges[1:])))

    def test_dpp4_site_detected_at_position_two(self):
        hits = [l for l in self.t0.scan_liabilities(GLP1) if "N-terminal" in l.motif]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].display_position, 2)
        self.assertEqual(hits[0].severity, "high")

    def test_deamidation_and_isoaspartate_motifs_detected(self):
        found = {l.motif for l in self.t0.scan_liabilities("AANGAADGAADPAA")}
        self.assertIn("Asn-Gly", found)
        self.assertIn("Asp-Gly", found)
        self.assertIn("Asp-Pro", found)

    def test_unpaired_cysteine_detected(self):
        odd = {l.motif for l in self.t0.scan_liabilities("AACAA")}
        even = {l.motif for l in self.t0.scan_liabilities("AACAACAA")}
        self.assertIn("unpaired Cys", odd)
        self.assertNotIn("unpaired Cys", even)


class TestObjectiveVector(unittest.TestCase):

    def _t(self, deltas, move=MoveType.SIDE_CHAIN_SUBSTITUTION, **kw):
        return Transformation(move=move, position=0, description="test",
                              rationale="test", objective_deltas=deltas, **kw)

    def test_unassessed_is_not_neutral(self):
        """The distinction that matters most: 'not looked at' is not 'no effect'."""
        unassessed = ObjectiveDelta.not_assessed(Objective.POTENCY, "no structure")
        self.assertFalse(unassessed.assessed)
        self.assertIsNone(unassessed.magnitude)
        self.assertIsNone(unassessed.signed)

    def test_unassessed_objectives_are_excluded_not_counted_as_zero(self):
        good = ObjectiveDelta(
            objective=Objective.PROTEOLYTIC_HALF_LIFE, direction=Direction.IMPROVES,
            magnitude=0.8, claim=Claim.inferred("x", "y"))
        unassessed = ObjectiveDelta.not_assessed(Objective.POTENCY, "no structure")

        with_unassessed = self._t([good, unassessed]).scalarize()
        alone = self._t([good]).scalarize()

        self.assertAlmostEqual(with_unassessed["score"], alone["score"], places=6)
        self.assertIn("potency", with_unassessed["unassessed_objectives"])

    def test_coverage_reflects_how_much_was_assessed(self):
        good = ObjectiveDelta(
            objective=Objective.PROTEOLYTIC_HALF_LIFE, direction=Direction.IMPROVES,
            magnitude=0.8, claim=Claim.inferred("x", "y"))
        result = self._t([good]).scalarize()
        self.assertLess(result["coverage"], 0.5)
        self.assertGreater(result["coverage"], 0.0)

    def test_constraint_move_without_preorganization_is_rejected(self):
        t = self._t([], move=MoveType.BACKBONE_CONSTRAINT)
        problems = t.validate()
        self.assertTrue(any("pre-organization" in p for p in problems))

    def test_constraint_move_with_preorganization_passes(self):
        proxy = PreOrganizationAnalyzer().for_substitution(GLP1, 1, "Aib", constraint_kind="AIB")
        t = self._t([], move=MoveType.BACKBONE_CONSTRAINT, preorganization=proxy)
        self.assertEqual(t.validate(), [])

    def test_potency_for_exposure_trade_must_be_labelled(self):
        deltas = [
            ObjectiveDelta(objective=Objective.POTENCY, direction=Direction.DEGRADES,
                           magnitude=0.4, claim=Claim.inferred("x", "y")),
            ObjectiveDelta(objective=Objective.ALBUMIN_FCRN, direction=Direction.IMPROVES,
                           magnitude=0.8, claim=Claim.inferred("x", "y")),
        ]
        unlabelled = self._t(deltas, move=MoveType.LIPIDATION)
        self.assertTrue(any("tradeoff" in p for p in unlabelled.validate()))

        labelled = self._t(deltas, move=MoveType.LIPIDATION,
                           tradeoff_label="EXPOSURE-FOR-POTENCY TRADE")
        self.assertEqual(labelled.validate(), [])


class TestNativeContext(unittest.TestCase):

    def setUp(self):
        self.na = NativeContextAnalyzer()

    def test_excision_check_flags_both_non_native_termini(self):
        ex = self.na.check_excision_site(GLP1, is_internal_fragment=True)
        self.assertTrue(ex.is_internal_fragment)
        self.assertIn("acetylation", ex.recommended_n_cap)
        self.assertIn("amidation", ex.recommended_c_cap)

    def test_complete_peptide_gets_no_excision_artifact(self):
        ex = self.na.check_excision_site(GLP1, is_internal_fragment=False)
        self.assertFalse(ex.is_internal_fragment)
        self.assertEqual(ex.recommended_n_cap, "")

    def test_conformational_contact_recommends_a_constraint_not_a_partner(self):
        """The class distinction the spec is most explicit about."""
        c = self.na.classify_contact("parent helix", [3], "evidence", ContactClass.CONFORMATIONAL)
        self.assertIn("INTRAMOLECULAR CONSTRAINT", c.recommended_response)
        self.assertIn("Do NOT recommend a partner peptide", c.recommended_response)

    def test_compositional_contact_recommends_a_partner(self):
        c = self.na.classify_contact("interface chain", [3],
                                     "evidence", ContactClass.COMPOSITIONAL_OBLIGATE)
        self.assertIn("PARTNER PEPTIDE", c.recommended_response)

    def test_protective_contact_requires_a_functional_replacement(self):
        c = self.na.classify_contact("shielding loop", [3], "evidence", ContactClass.PROTECTIVE)
        self.assertIn("functional replacement", c.recommended_response)

    def test_partner_classes_require_different_evidence(self):
        structural = self.na.recommend_partner("chain B", PartnerClass.STRUCTURAL_OBLIGATE, "r")
        pharmacological = self.na.recommend_partner("agent B", PartnerClass.PHARMACOLOGICAL, "r")

        self.assertIn("Structural evidence", structural.evidence_required)
        self.assertIn("Pathway-level evidence", pharmacological.evidence_required)
        self.assertIn("covalent tethering", structural.covalent_tether_assessment.lower())

    def test_partner_terminology_never_says_dipeptide(self):
        """'Dipeptide' means a two-residue peptide and would describe a different molecule."""
        p = self.na.recommend_partner("chain B", PartnerClass.STRUCTURAL_OBLIGATE, "rationale")
        blob = " ".join([p.rationale, p.covalent_tether_assessment, p.evidence_required]).lower()
        self.assertNotIn("dipeptide", blob)

    def test_analysis_states_what_was_not_retrieved(self):
        result = self.na.analyze(GLP1)
        self.assertIn("NOT retrieved", result.retrieval_status)
        self.assertIn("NOT retrieved", result.structure_status)
        self.assertTrue(any(a.kind.value == "receptor_accessory_subunits" for a in result.assumptions))


class TestTransformWorkflow(unittest.TestCase):

    def setUp(self):
        self.result = TransformWorkflow().run(GLP1)

    def test_emits_transformations_and_rejects_none(self):
        self.assertGreater(len(self.result["transformations"]), 0)
        self.assertEqual(self.result["rejected"], [])

    def test_every_emitted_transformation_satisfies_the_contract(self):
        for t in self.result["transformations"]:
            self.assertEqual(t.validate(), [], f"{t.description} violates the output contract")

    def test_move_types_come_from_the_controlled_vocabulary(self):
        for t in self.result["transformations"]:
            self.assertIsInstance(t.move, MoveType)

    def test_lipidation_is_labelled_as_a_trade(self):
        lipid = [t for t in self.result["transformations"] if t.move is MoveType.LIPIDATION]
        self.assertTrue(lipid)
        self.assertIn("TRADE", lipid[0].tradeoff_label)

    def test_aib_move_carries_preorganization_and_leakage_flag(self):
        aib = [t for t in self.result["transformations"] if "Aib" in t.description]
        self.assertTrue(aib)
        self.assertIsNotNone(aib[0].preorganization)
        self.assertIsNotNone(aib[0].leakage_flag)

    def test_weights_are_reported_with_the_output(self):
        self.assertEqual(len(self.result["weights"]), len(Objective))
        self.assertIn("comparability_warning", self.result)


if __name__ == "__main__":
    unittest.main()
