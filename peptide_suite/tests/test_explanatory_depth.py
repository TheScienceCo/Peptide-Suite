"""
Conformer ensembles, n->pi* analysis, and SAPT decomposition.
[Addendum 2 sections 4 and 5, after the 6a-6i build order]

None of the three can run here: no conformational search engine, no NBO
implementation, no QM package. So what is tested is what each module does
INSTEAD of producing a number, which is the part that has to be right when the
engines do arrive.
"""

import unittest

from peptide_suite import runtime
from peptide_suite.core.backbone_nbo import (
    NBOAnalysisNotAvailable, NBOTrigger, analyse, requirement_for,
)
from peptide_suite.core.conformer_ensemble import (
    ConstraintKind, EnsembleNotAvailable, EnsembleStatus, assess, detect_constraint,
    ensemble_length_ceiling, generate,
)
from peptide_suite.core.contact_decomposition import (
    AffinitySummationError, ComponentClaim, ContactDecomposition,
    DecompositionNotAvailable, SAPTComponent, decompose, read_component_claim,
)

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"


def a_decomposition(**over):
    kwargs = dict(residue_a="Trp25", residue_b="Phe12", electrostatics=-2.1,
                  exchange_repulsion=8.4, induction=-1.2, dispersion=-9.8,
                  method="SAPT0", basis_set="jun-cc-pVDZ",
                  truncation_scheme="sidechain+Cb")
    kwargs.update(over)
    return ContactDecomposition(**kwargs)


class TestEnsembleEligibility(unittest.TestCase):

    def test_constraint_beats_length(self):
        """
        A stapled 30-mer is tractable and a linear 30-mer is not, because the
        constraint is what shrinks the accessible space.
        """
        stapled = assess(GLP1, "i,i+4 hydrocarbon staple")
        linear = assess(GLP1, "")
        self.assertIs(stapled.constraint, ConstraintKind.STAPLED)
        self.assertIsNot(stapled.status, EnsembleStatus.UNAVAILABLE_TOO_LONG)
        self.assertIs(linear.status, EnsembleStatus.UNAVAILABLE_TOO_LONG)

    def test_a_long_unconstrained_peptide_is_refused_on_length(self):
        assessment = assess("A" * (ensemble_length_ceiling() + 1))
        self.assertIs(assessment.status, EnsembleStatus.UNAVAILABLE_TOO_LONG)
        self.assertIn("different object", assessment.reason)

    def test_the_length_reason_names_the_actual_hazard(self):
        """
        Undersampling is not a smaller ensemble. Boltzmann-weighting over
        conformers that were never found gives a distribution's shape with none
        of its content, and nothing downstream can tell.
        """
        reason = assess("A" * 80).reason
        self.assertIn("Boltzmann", reason)

    def test_the_boundary_is_the_policy_threshold(self):
        ceiling = ensemble_length_ceiling()
        self.assertIsNot(assess("A" * ceiling).status, EnsembleStatus.UNAVAILABLE_TOO_LONG)
        self.assertIs(assess("A" * (ceiling + 1)).status, EnsembleStatus.UNAVAILABLE_TOO_LONG)

    def test_the_ceiling_comes_from_the_policy(self):
        saved = runtime._active
        runtime.clear_active_policy()
        try:
            with self.assertRaises(runtime.PolicyNotLoaded):
                ensemble_length_ceiling()
        finally:
            runtime.set_active_policy(saved)

    def test_two_cysteines_are_possible_not_proven(self):
        """
        A disulfide makes the space small; an unformed one does not. The
        assessment says which it is assuming.
        """
        assessment = assess("CYIQNCPLG")
        self.assertIs(assessment.constraint, ConstraintKind.DISULFIDE)
        self.assertTrue(any("possible rather than certain" in n for n in assessment.notes))

    def test_cyclisation_language_is_recognised(self):
        for text in ("head-to-tail cyclisation", "lactam bridge", "macrocyclisation"):
            with self.subTest(text=text):
                self.assertIs(detect_constraint("AAAAAAAA", text), ConstraintKind.CYCLIC)

    def test_nothing_is_eligible_without_an_engine(self):
        """Eligible still needs a search engine, and there is not one here."""
        for sequence, text in ((GLP1, "staple"), ("CYIQNCPLG", ""), ("AGLVKF", "")):
            with self.subTest(sequence=sequence):
                self.assertIs(assess(sequence, text).status,
                              EnsembleStatus.UNAVAILABLE_NO_ENGINE)

    def test_generate_raises_rather_than_returning_one_conformer(self):
        """
        A single conformer has the type of an ensemble, so everything
        downstream would weight a population of one and call it an average.
        """
        with self.assertRaises(EnsembleNotAvailable):
            generate(GLP1)


class TestNBOIsRequiredNotSuggested(unittest.TestCase):

    def test_aib_requires_an_account(self):
        requirement = requirement_for("A2 -> Aib (alpha-aminoisobutyric acid)")
        self.assertTrue(requirement.is_required)
        self.assertIn(NBOTrigger.ALPHA_ALPHA_DISUBSTITUTED, requirement.triggers)

    def test_n_methylation_requires_one(self):
        self.assertIn(NBOTrigger.N_METHYLATION,
                      requirement_for("N-methylation of the backbone amide").triggers)

    def test_proline_richness_is_read_from_the_sequence(self):
        """
        A proposal modifying a proline-rich segment need not mention proline,
        so the trigger cannot come from the proposal's words alone.
        """
        requirement = requirement_for("Gly substitution", "GEPPPGKPADDAGLV")
        self.assertIn(NBOTrigger.PROLINE_RICH, requirement.triggers)

    def test_an_ordinary_substitution_requires_nothing(self):
        self.assertFalse(requirement_for("Trp25 -> Phe", GLP1).is_required)

    def test_the_obligation_is_outstanding_not_satisfied(self):
        requirement = requirement_for("A2 -> Aib")
        self.assertTrue(requirement.is_outstanding)
        self.assertFalse(requirement.satisfied)

    def test_the_statement_distinguishes_mechanism_from_observation(self):
        """
        "Aib rigidifies the backbone" is the observation, and it was already
        known. The n->pi* donation is the mechanism.
        """
        statement = requirement_for("A2 -> Aib").statement()
        self.assertIn("mechanistic account", statement)
        self.assertIn("observation rather than the mechanism", statement)

    def test_no_statement_when_nothing_is_owed(self):
        self.assertEqual(requirement_for("Trp25 -> Phe", GLP1).statement(), "")

    def test_analysis_raises_rather_than_inventing_a_stabilisation_energy(self):
        with self.assertRaises(NBOAnalysisNotAvailable):
            analyse("A2 -> Aib")


class TestSAPTComponentsAreNeverSummedIntoAffinity(unittest.TestCase):
    """
    The prohibition is enforced in code because the sum is one line to write
    and looks like the obvious thing to do with four numbers sharing a unit.
    """

    def test_as_binding_free_energy_raises(self):
        with self.assertRaises(AffinitySummationError) as ctx:
            a_decomposition().as_binding_free_energy()
        message = str(ctx.exception)
        self.assertIn("omits desolvation and entropy", message)
        self.assertIn("FEP/TI", message)

    def test_there_is_no_predicted_affinity_attribute(self):
        decomposition = a_decomposition()
        for attribute in ("predicted_affinity", "binding_free_energy", "delta_g",
                          "affinity", "kd"):
            self.assertFalse(hasattr(decomposition, attribute),
                             f"{attribute} exists and invites the forbidden reading")

    def test_the_legitimate_sum_is_available_and_named_for_what_it_is(self):
        """
        Interaction energy is a real quantity. Removing the sum entirely would
        push someone to recompute it by hand with no label at all.
        """
        self.assertAlmostEqual(a_decomposition().interaction_energy(), -4.7, places=6)

    def test_dominance_is_by_magnitude_not_sign(self):
        """
        Exchange-repulsion is positive and the attractive terms are negative, so
        a signed comparison always returns the repulsion.
        """
        self.assertIs(a_decomposition().dominant, SAPTComponent.DISPERSION)

    def test_an_electrostatics_dominated_contact_is_identified(self):
        decomposition = a_decomposition(electrostatics=-15.0, dispersion=-2.0)
        self.assertIs(decomposition.dominant, SAPTComponent.ELECTROSTATICS)

    def test_every_component_says_what_it_responds_to(self):
        """A component that does not change a design decision is decoration."""
        for component in SAPTComponent:
            with self.subTest(component=component.value):
                self.assertTrue(component.responds_to)

    def test_induction_dominance_is_reported_as_a_warning(self):
        self.assertIn("fixed-charge force field",
                      SAPTComponent.INDUCTION.responds_to)

    def test_decompose_raises_rather_than_returning_plausible_components(self):
        """
        A fabricated dispersion-dominated result sends someone to enlarge a
        hydrophobic surface on a contact that is actually electrostatic.
        """
        with self.assertRaises(DecompositionNotAvailable):
            decompose("Trp25", "Phe12")


class TestProposalsMustNameTheComponent(unittest.TestCase):

    def test_a_named_component_is_read_from_the_proposal(self):
        for proposal, expected in (
                ("enlarge the buried hydrophobic surface", SAPTComponent.DISPERSION),
                ("introduce a salt bridge", SAPTComponent.ELECTROSTATICS),
                ("relieve steric clash", SAPTComponent.EXCHANGE_REPULSION),
                ("exploits induction at the interface", SAPTComponent.INDUCTION)):
            with self.subTest(proposal=proposal):
                self.assertIs(read_component_claim(proposal).targeted, expected)

    def test_a_vague_proposal_names_nothing(self):
        """"Improves the interaction" is the conclusion restated, not a mechanism."""
        claim = read_component_claim("improves the interaction")
        self.assertFalse(claim.is_stated)
        self.assertIn("conclusion in place of its mechanism", claim.violation())

    def test_an_unrecognised_proposal_is_reported_rather_than_guessed(self):
        self.assertIsNone(read_component_claim("does something novel").targeted)

    def test_a_stated_claim_has_no_violation(self):
        self.assertEqual(read_component_claim("increases buried hydrophobic area").violation(), "")


class TestWorkflowIntegration(unittest.TestCase):

    def _run(self, sequence):
        from peptide_suite.workflows.transform import TransformWorkflow
        return TransformWorkflow().run(sequence)

    def test_the_ensemble_assessment_is_returned(self):
        self.assertIn("conformer_ensemble", self._run(GLP1))

    def test_a_long_linear_peptide_reports_the_length_refusal(self):
        self.assertIs(self._run(GLP1)["conformer_ensemble"].status,
                      EnsembleStatus.UNAVAILABLE_TOO_LONG)

    def test_proposals_carry_their_nbo_obligation(self):
        for transformation in self._run("GEPPPGKPADDAGLV")["transformations"]:
            with self.subTest(proposal=transformation.description):
                self.assertIsNotNone(transformation.nbo_requirement)

    def test_a_proline_rich_peptide_owes_an_account(self):
        outstanding = [t for t in self._run("GEPPPGKPADDAGLV")["transformations"]
                       if t.nbo_requirement and t.nbo_requirement.is_outstanding]
        self.assertTrue(outstanding)


if __name__ == "__main__":
    unittest.main()
