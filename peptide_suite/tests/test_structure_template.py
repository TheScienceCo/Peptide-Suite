"""
Structure template hierarchy and the refusal path.  [Addendum 2 sections 4, 6c]

The hierarchy is strictly ordered and the last tier is refusal. These tests are
mostly about tier 4, because that is the tier a system gets wrong: degrading
gracefully into a guess is the failure being designed against, and a
conformational proposal made without a conformation reads in the output exactly
like one made with it.
"""

import unittest

from peptide_suite import runtime
from peptide_suite.core.structure_template import (
    CONFORMATION_DEPENDENT_MOVES, RejectedTemplateKind, TemplateCandidate,
    TemplateTier, blocked_moves, evaluate_confidence_gate, select_template,
)

BOUND_EXPERIMENTAL = TemplateCandidate(
    identifier="6X18", is_experimental=True, is_complex=True,
    same_peptide=True, same_receptor=True)

HOMOLOG_EXPERIMENTAL = TemplateCandidate(
    identifier="6B3J", is_experimental=True, is_complex=True,
    same_peptide=False, same_receptor=True, homolog_identity=0.92)

GOOD_PREDICTION = TemplateCandidate(
    identifier="AF3-model", is_experimental=False, is_complex=True,
    same_receptor=True,
    metadata={"plddt": 95.0, "pae": 1.0, "iptm": 0.95, "seed_count": 100})


class TestHierarchyIsStrictlyOrdered(unittest.TestCase):

    def test_no_candidates_refuses(self):
        self.assertIs(select_template().tier, TemplateTier.REFUSED)

    def test_experimental_this_complex_is_tier_one(self):
        self.assertIs(select_template([BOUND_EXPERIMENTAL]).tier,
                      TemplateTier.EXPERIMENTAL_THIS_COMPLEX)

    def test_homolog_complex_is_tier_two(self):
        self.assertIs(select_template([HOMOLOG_EXPERIMENTAL]).tier,
                      TemplateTier.EXPERIMENTAL_HOMOLOG_COMPLEX)

    def test_gated_prediction_is_tier_three(self):
        self.assertIs(select_template([GOOD_PREDICTION]).tier,
                      TemplateTier.PREDICTED_COMPLEX_GATED)

    def test_a_confident_prediction_never_outranks_a_real_structure(self):
        """
        Ordering, not scoring. Picking the best-scoring candidate across tiers
        would let a 99-pLDDT model beat a crystal structure, which inverts the
        hierarchy the section exists to impose.
        """
        decision = select_template([GOOD_PREDICTION, BOUND_EXPERIMENTAL])
        self.assertIs(decision.tier, TemplateTier.EXPERIMENTAL_THIS_COMPLEX)
        self.assertEqual(decision.template.identifier, "6X18")

    def test_a_homolog_never_outranks_the_peptide_itself(self):
        decision = select_template([HOMOLOG_EXPERIMENTAL, BOUND_EXPERIMENTAL])
        self.assertIs(decision.tier, TemplateTier.EXPERIMENTAL_THIS_COMPLEX)

    def test_a_prediction_never_outranks_a_homolog_structure(self):
        decision = select_template([GOOD_PREDICTION, HOMOLOG_EXPERIMENTAL])
        self.assertIs(decision.tier, TemplateTier.EXPERIMENTAL_HOMOLOG_COMPLEX)


class TestTemplatesRejectedOnIdentityNotScore(unittest.TestCase):
    """
    Two templates are wrong in kind rather than in quality. Rejecting them on a
    confidence metric would let a well-resolved structure of the wrong state
    through, and resolution is not the issue.
    """

    def test_free_state_structure_is_refused(self):
        free = TemplateCandidate(
            identifier="2N0I", is_experimental=True, is_complex=False,
            same_peptide=True, kind=RejectedTemplateKind.FREE_STATE)
        decision = select_template([free])
        self.assertTrue(decision.is_refusal)
        self.assertTrue(any("fold on binding" in r for r in decision.rejections))

    def test_precursor_conformation_is_refused(self):
        precursor = TemplateCandidate(
            identifier="proglucagon", is_experimental=True, is_complex=False,
            same_peptide=True, kind=RejectedTemplateKind.PRECURSOR)
        decision = select_template([precursor])
        self.assertTrue(decision.is_refusal)
        self.assertTrue(any("about the parent" in r for r in decision.rejections))

    def test_a_perfect_free_state_structure_is_still_refused(self):
        """Identity is checked before any metric, so quality cannot rescue it."""
        free = TemplateCandidate(
            identifier="perfect", is_experimental=True, is_complex=True,
            same_peptide=True, same_receptor=True,
            kind=RejectedTemplateKind.FREE_STATE,
            metadata={"plddt": 100, "pae": 0.0, "iptm": 1.0, "seed_count": 1000})
        self.assertTrue(select_template([free]).is_refusal)

    def test_apo_receptor_is_refused(self):
        apo = TemplateCandidate(
            identifier="apo-GLP-1R", is_experimental=True, is_complex=False,
            same_receptor=True, kind=RejectedTemplateKind.UNBOUND_RECEPTOR)
        self.assertTrue(select_template([apo]).is_refusal)

    def test_complex_with_a_different_receptor_is_refused(self):
        other = TemplateCandidate(
            identifier="GIPR complex", is_experimental=True, is_complex=True,
            same_peptide=True, same_receptor=False)
        decision = select_template([other])
        self.assertTrue(decision.is_refusal)
        self.assertTrue(any("different receptor" in r.lower() for r in decision.rejections))


class TestConfidenceGate(unittest.TestCase):

    def _metadata(self, **over):
        base = {"plddt": 95.0, "pae": 1.0, "iptm": 0.95, "seed_count": 100}
        base.update(over)
        return base

    def test_a_complete_confident_model_passes(self):
        self.assertEqual(evaluate_confidence_gate(self._metadata()), [])

    def test_a_missing_metric_is_a_failure_not_a_pass(self):
        """
        An unreported pLDDT is not evidence of a good structure. Defaulting it
        to acceptable is how an ungated structure enters looking gated.
        """
        for metric in ("plddt", "pae", "iptm", "seed_count"):
            metadata = self._metadata()
            del metadata[metric]
            with self.subTest(missing=metric):
                failures = evaluate_confidence_gate(metadata)
                self.assertTrue(any("not reported" in f for f in failures))

    def test_every_failure_is_reported_not_just_the_first(self):
        failures = evaluate_confidence_gate(
            {"plddt": 1.0, "pae": 999.0, "iptm": 0.0, "seed_count": 0})
        self.assertEqual(len(failures), 4)

    def test_a_failing_prediction_refuses_rather_than_downgrading(self):
        weak = TemplateCandidate(
            identifier="weak-model", is_experimental=False, is_complex=True,
            same_receptor=True, metadata={"plddt": 30.0, "pae": 25.0,
                                          "iptm": 0.1, "seed_count": 1})
        decision = select_template([weak])
        self.assertTrue(decision.is_refusal)
        self.assertTrue(any("confidence gate" in f for f in decision.gate_failures))

    def test_gate_thresholds_come_from_the_policy(self):
        saved = runtime._active
        runtime.clear_active_policy()
        try:
            with self.assertRaises(runtime.PolicyNotLoaded):
                evaluate_confidence_gate({"plddt": 99, "pae": 1, "iptm": 1, "seed_count": 99})
        finally:
            runtime.set_active_policy(saved)


class TestRefusalBlocksProposals(unittest.TestCase):

    def test_refusal_blocks_the_conformation_dependent_moves(self):
        self.assertEqual(blocked_moves(select_template()), CONFORMATION_DEPENDENT_MOVES)

    def test_an_admitted_template_blocks_nothing(self):
        self.assertEqual(blocked_moves(select_template([BOUND_EXPERIMENTAL])), frozenset())

    def test_sequence_level_moves_are_never_blocked(self):
        """
        Side-chain substitution, capping and charge engineering argue from the
        sequence. Blocking them on a missing structure would refuse work the
        system can actually do.
        """
        for move in ("side_chain_substitution", "terminal_capping",
                     "charge_engineering", "liability_removal", "lipidation"):
            with self.subTest(move=move):
                self.assertNotIn(move, CONFORMATION_DEPENDENT_MOVES)


class TestWorkflowHonoursTheRefusal(unittest.TestCase):
    SEQUENCE = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"

    def _run(self, candidates=None):
        from peptide_suite.workflows.transform import TransformWorkflow
        return TransformWorkflow().run(self.SEQUENCE, structure_candidates=candidates)

    def test_without_a_structure_conformational_moves_do_not_reach_the_output(self):
        result = self._run()
        self.assertIs(result["structure_template"].tier, TemplateTier.REFUSED)
        emitted = {t.move.value for t in result["transformations"]}
        self.assertFalse(emitted & CONFORMATION_DEPENDENT_MOVES,
                         f"conformational moves emitted without a structure: {emitted}")

    def test_blocked_moves_are_reported_not_silently_dropped(self):
        """
        A refusal the user cannot see is indistinguishable from the system never
        having considered the move.
        """
        result = self._run()
        blocked = [r for r in result["rejected"]
                   if r.get("blocked_by") == "structure_template_refusal"]
        self.assertTrue(blocked, "conformational moves vanished without explanation")
        for entry in blocked:
            self.assertTrue(entry["violations"][0])

    def test_with_a_bound_structure_the_same_moves_are_emitted(self):
        """The block is about the missing template, not about the move itself."""
        result = self._run([BOUND_EXPERIMENTAL])
        self.assertIs(result["structure_template"].tier,
                      TemplateTier.EXPERIMENTAL_THIS_COMPLEX)
        emitted = {t.move.value for t in result["transformations"]}
        self.assertTrue(emitted & CONFORMATION_DEPENDENT_MOVES)
        self.assertFalse([r for r in result["rejected"]
                          if r.get("blocked_by") == "structure_template_refusal"])

    def test_sequence_level_proposals_survive_a_refusal(self):
        emitted = {t.move.value for t in self._run()["transformations"]}
        self.assertTrue(emitted, "a refusal silenced the entire output")


class TestRefusalExplainsItself(unittest.TestCase):

    def test_an_empty_refusal_says_no_structure_was_supplied(self):
        self.assertIn("No structure was supplied", select_template().summary())

    def test_a_refusal_after_review_says_what_it_reviewed(self):
        free = TemplateCandidate(
            identifier="2N0I", is_experimental=True, is_complex=False,
            kind=RejectedTemplateKind.FREE_STATE)
        decision = select_template([free])
        self.assertIn("1 candidate", decision.summary())
        self.assertTrue(decision.detail_lines())

    def test_refusal_is_described_as_designed_behaviour(self):
        """It is not a gap to be filled in later; it is the specified outcome."""
        summary = select_template().summary()
        self.assertTrue("blocked" in summary and "estimated" in summary)


if __name__ == "__main__":
    unittest.main()
