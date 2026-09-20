"""
Class B1 restricted zone and the three-field contract.  [Addendum 2 section 9, step 6g]

Two rules with different scopes, and the difference matters.

The placement rule applies inside the N-terminal restricted zone: those residues
insert into the transmembrane core and dominate efficacy and bias, so a
modification there must say what it does to them.

The reporting rule applies everywhere: an affinity gain may never be reported as
an improvement on its own. A peptide agonist that binds better and signals worse
is a worse drug, and one collapsed "improvement" number cannot express that.
"""

import unittest

from peptide_suite import runtime
from peptide_suite.core.class_b1 import (
    CLASS_B1_RECEPTORS, Effect, ThreeFieldPrediction, conjugation_site_guidance,
    evaluate, is_class_b1, normalise_receptor, restricted_zone_length,
)


class TestReceptorIdentification(unittest.TestCase):

    def test_the_nine_class_b1_receptors_are_listed(self):
        self.assertEqual(
            CLASS_B1_RECEPTORS,
            {"GLP1R", "GIPR", "GCGR", "PTH1R", "CTR", "CRF1R", "SCTR", "VIPR", "PAC1"})

    def test_common_spellings_resolve(self):
        """
        A caller writing "GLP-1R" must not be silently treated as non-B1 and
        exempted from the rule.
        """
        for written in ("GLP-1R", "GLP1R", "glp-1 receptor", "GLP1-R"):
            with self.subTest(spelling=written):
                self.assertEqual(normalise_receptor(written), "GLP1R")
                self.assertTrue(is_class_b1(written))

    def test_other_aliases_resolve(self):
        self.assertEqual(normalise_receptor("PTHR1"), "PTH1R")
        self.assertEqual(normalise_receptor("CALCR"), "CTR")
        self.assertEqual(normalise_receptor("CRHR1"), "CRF1R")
        self.assertTrue(is_class_b1("VPAC1"))

    def test_a_non_b1_receptor_is_not_matched(self):
        for other in ("MOR", "GHSR1a", "CCK1", "", "insulin receptor"):
            with self.subTest(receptor=other):
                self.assertFalse(is_class_b1(other))


class TestRestrictedZone(unittest.TestCase):

    def _prediction(self, **over):
        return ThreeFieldPrediction(**over)

    def test_the_zone_length_comes_from_the_policy(self):
        """The two-domain model is engine; where the line falls is policy."""
        self.assertEqual(restricted_zone_length(),
                         runtime.int_threshold("class_b1.restricted_zone_residues"))

    def test_the_zone_is_unavailable_without_a_policy(self):
        saved = runtime._active
        runtime.clear_active_policy()
        try:
            with self.assertRaises(runtime.PolicyNotLoaded):
                restricted_zone_length()
        finally:
            runtime.set_active_policy(saved)

    def test_a_modification_in_the_zone_must_report_efficacy_or_bias(self):
        ruling = evaluate("Ala2 -> Aib", 2, "GLP-1R", self._prediction())
        self.assertTrue(ruling.in_restricted_zone)
        self.assertFalse(ruling.permits_emission)
        self.assertTrue(any("transmembrane core" in v for v in ruling.violations))

    def test_reporting_efficacy_satisfies_the_placement_rule(self):
        ruling = evaluate("Ala2 -> Aib", 2, "GLP-1R",
                          self._prediction(efficacy=Effect.DEGRADES))
        self.assertTrue(ruling.in_restricted_zone)
        self.assertTrue(ruling.permits_emission)

    def test_reporting_bias_alone_also_satisfies_it(self):
        ruling = evaluate("Ala2 -> Aib", 2, "GLP-1R",
                          self._prediction(bias=Effect.NEUTRAL))
        self.assertTrue(ruling.permits_emission)

    def test_a_position_outside_the_zone_is_not_placement_restricted(self):
        """Half-life chemistry belongs here; semaglutide's lipid is at Lys26."""
        ruling = evaluate("Lys26 acylation", 26, "GLP-1R", self._prediction())
        self.assertFalse(ruling.in_restricted_zone)
        self.assertTrue(ruling.permits_emission)
        self.assertTrue(any("Lys26" in n for n in ruling.notes))

    def test_the_boundary_residue_is_inside(self):
        zone = restricted_zone_length()
        self.assertTrue(evaluate("x", zone, "GLP-1R", self._prediction()).in_restricted_zone)
        self.assertFalse(
            evaluate("x", zone + 1, "GLP-1R", self._prediction()).in_restricted_zone)

    def test_the_placement_rule_does_not_apply_to_a_non_b1_receptor(self):
        ruling = evaluate("Ala2 -> Aib", 2, "MOR", self._prediction())
        self.assertFalse(ruling.in_restricted_zone)
        self.assertTrue(any("not in the class B1 set" in n for n in ruling.notes))

    def test_a_whole_molecule_move_has_no_position_and_no_zone(self):
        self.assertFalse(evaluate("C-terminal amidation", None, "GLP-1R",
                                  self._prediction()).in_restricted_zone)


class TestThreeFieldReporting(unittest.TestCase):

    def test_an_affinity_gain_alone_may_not_be_called_an_improvement(self):
        prediction = ThreeFieldPrediction(affinity=Effect.IMPROVES)
        self.assertTrue(prediction.is_affinity_only)
        self.assertFalse(prediction.improvement_claim_permitted())

    def test_an_affinity_gain_with_efficacy_reported_is_permitted(self):
        prediction = ThreeFieldPrediction(affinity=Effect.IMPROVES,
                                          efficacy=Effect.DEGRADES)
        self.assertTrue(prediction.improvement_claim_permitted())

    def test_the_rule_applies_outside_the_zone_too(self):
        """
        The placement rule is zone-scoped; the reporting rule is not. An
        affinity-only improvement at position 20 gets its affinity gain
        discounted just as one at position 2 would.
        """
        ruling = evaluate("staple 19-23", 20, "GLP-1R",
                          ThreeFieldPrediction(affinity=Effect.IMPROVES))
        self.assertFalse(ruling.in_restricted_zone)
        self.assertTrue(ruling.affinity_claim_discounted)

    def test_the_reporting_rule_discounts_rather_than_blocks(self):
        """
        UNKNOWN is an expected and acceptable value, so a proposal with unknown
        efficacy and bias is still a legitimate proposal. What it may not do is
        have its affinity gain counted as an improvement -- and in a ranked
        system, counting it is what reporting it would mean.
        """
        ruling = evaluate("staple 19-23", 20, "GLP-1R",
                          ThreeFieldPrediction(affinity=Effect.IMPROVES))
        self.assertTrue(ruling.permits_emission)
        self.assertTrue(any("not counted toward" in n for n in ruling.notes))

    def test_the_rule_applies_to_a_non_b1_receptor_too(self):
        ruling = evaluate("x", 20, "MOR", ThreeFieldPrediction(affinity=Effect.IMPROVES))
        self.assertTrue(ruling.affinity_claim_discounted)

    def test_a_degraded_affinity_needs_no_companion(self):
        """The rule is about claiming improvement, not about reporting at all."""
        prediction = ThreeFieldPrediction(affinity=Effect.DEGRADES)
        self.assertTrue(prediction.improvement_claim_permitted())

    def test_unknown_is_a_first_class_value(self):
        """
        Expected and acceptable per the section. A fabricated number is not, so
        the enum offers no neutral member an unexamined axis could take.
        """
        prediction = ThreeFieldPrediction()
        self.assertIs(prediction.affinity, Effect.UNKNOWN)
        self.assertEqual(prediction.known_axes, [])
        self.assertFalse(Effect.UNKNOWN.is_known)

    def test_the_three_axes_are_never_collapsed(self):
        prediction = ThreeFieldPrediction(affinity=Effect.IMPROVES,
                                          efficacy=Effect.DEGRADES,
                                          bias=Effect.UNKNOWN)
        summary = prediction.summary()
        self.assertIn("affinity improves", summary)
        self.assertIn("efficacy degrades", summary)
        self.assertIn("bias UNKNOWN", summary)


class TestConjugationGuidance(unittest.TestCase):

    def test_restricted_and_permitted_spans_are_returned(self):
        guidance = conjugation_site_guidance("A" * 30, "GLP-1R")
        self.assertTrue(guidance["applies"])
        self.assertEqual(guidance["restricted_positions"],
                         list(range(1, restricted_zone_length() + 1)))
        self.assertIn(26, guidance["permitted_positions"])

    def test_the_worked_example_is_carried(self):
        self.assertIn("Lys26", conjugation_site_guidance("A" * 30, "GLP-1R")["worked_example"])

    def test_a_short_peptide_does_not_get_positions_past_its_length(self):
        guidance = conjugation_site_guidance("AAAA", "GLP-1R")
        self.assertEqual(guidance["restricted_positions"], [1, 2, 3, 4])

    def test_guidance_does_not_apply_off_class(self):
        self.assertFalse(conjugation_site_guidance("A" * 30, "MOR")["applies"])


class TestWorkflowIntegration(unittest.TestCase):

    def _run(self, receptor=""):
        from peptide_suite.core.structure_template import TemplateCandidate
        from peptide_suite.workflows.transform import TransformWorkflow
        template = TemplateCandidate(identifier="fixture", is_experimental=True,
                                     is_complex=True, same_peptide=True, same_receptor=True)
        return TransformWorkflow().run("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                                       structure_candidates=[template], receptor=receptor)

    def test_a_modification_in_the_zone_is_blocked_on_a_b1_receptor(self):
        result = self._run("GLP-1R")
        blocked = [r for r in result["rejected"]
                   if r.get("blocked_by") == "class_b1_restricted_zone"]
        self.assertTrue(blocked)

    def test_each_block_reports_the_three_fields(self):
        """The prediction travels with the refusal, so the reason is checkable."""
        for entry in self._run("GLP-1R")["rejected"]:
            if entry.get("blocked_by") == "class_b1_restricted_zone":
                self.assertIn("affinity", entry["prediction"])
                self.assertIn("efficacy", entry["prediction"])
                self.assertIn("bias", entry["prediction"])

    def test_no_ranked_proposal_makes_an_undiscounted_affinity_only_claim(self):
        """
        The guarantee, stated as an invariant over the whole ranked list rather
        than over one proposal.

        As of this build nothing in the ranked list claims a potency gain at
        all: every proposal that does is already a research request, because
        its chemistry is unparameterized. So this passes vacuously today and
        earns its place the moment a generator introduces such a claim, which
        is exactly when it would otherwise slip through.
        """
        from peptide_suite.core.transformations import Direction, Objective
        for t in self._run("GLP-1R")["transformations"]:
            potency = t.delta(Objective.POTENCY)
            if potency is None or not potency.assessed:
                continue
            if potency.direction is not Direction.IMPROVES:
                continue
            with self.subTest(proposal=t.description):
                self.assertIn(Objective.POTENCY, t.discount_objectives,
                              "an affinity-only improvement reached the ranking "
                              "with its potency still counted")

    def test_the_discount_removes_the_objective_from_the_scalar_only(self):
        """
        Removed from the rank, not from the output. The mechanism is tested
        here directly because the workflow path to it is currently unreachable.
        """
        from peptide_suite.core.transformations import Objective
        from peptide_suite.workflows.transform import TransformWorkflow

        workflow = TransformWorkflow()
        physics = workflow.physics.run("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", ph=7.4)
        staple = next(t for t in workflow._conformational_moves(
            "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", physics) if "staple" in t.description)

        before = staple.scalarize()
        self.assertEqual(before["discounted_objectives"], [])

        staple.discount_objectives = {Objective.POTENCY}
        after = staple.scalarize()
        self.assertEqual(after["discounted_objectives"], ["potency"])
        self.assertNotEqual(after["score"], before["score"])
        # Still present in the vector, and still assessed.
        self.assertTrue(staple.delta(Objective.POTENCY).assessed)

    def test_guidance_is_returned_when_a_receptor_is_named(self):
        self.assertTrue(self._run("GLP-1R")["class_b1"]["applies"])

    def test_guidance_says_why_it_is_absent_otherwise(self):
        self.assertIn("No receptor", self._run("")["class_b1"]["reason"])


if __name__ == "__main__":
    unittest.main()
