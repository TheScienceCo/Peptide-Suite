"""
Non-canonical residue parameterization gate.  [Addendum 2 section 6, step 6d]

The registry is empty and these tests assert that this has teeth. An empty
registry that still lets a proposal rank is worse than no registry at all: it
looks like a control and is not one.

The gate is independent of the structure-template gate. A proposal can have a
perfect bound experimental structure and still be unrankable because its
chemistry has no parameters, and the two refusals have different remedies.
"""

import unittest

from peptide_suite.core.ncaa_registry import (
    NCAARegistry, PIPELINE, ParameterizationError, ParameterizationStage,
    PipelineNotImplemented, ResidueStatus, detect_noncanonical, gate_proposal,
    pipeline_specification, run_stage,
)
from peptide_suite.core.structure_template import TemplateCandidate

BOUND_STRUCTURE = TemplateCandidate(
    identifier="6X18", is_experimental=True, is_complex=True,
    same_peptide=True, same_receptor=True)

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"


class TestRegistryIsEmpty(unittest.TestCase):

    def test_registry_is_empty(self):
        """
        The correct current state. This project has parameterized nothing, and
        an entry here is a claim that the work was done.
        """
        self.assertTrue(NCAARegistry.is_empty())
        self.assertEqual(NCAARegistry.registry(), {})

    def test_catalogue_is_not_the_registry(self):
        """
        Recognising a residue is not being ready to score it. Merging the two
        files would let recognition be mistaken for readiness.
        """
        self.assertTrue(NCAARegistry.catalogue())
        for code in NCAARegistry.catalogue():
            self.assertNotIn(code, NCAARegistry.registry())


class TestStatus(unittest.TestCase):

    def test_catalogued_residues_are_unparameterized(self):
        for code in ("Aib", "Nle", "Cha", "staple", "OEG", "gamma-Glu"):
            with self.subTest(residue=code):
                self.assertIs(NCAARegistry.status(code), ResidueStatus.UNPARAMETERIZED)

    def test_unknown_chemistry_is_unrecognised_not_assumed_fine(self):
        self.assertIs(NCAARegistry.status("Xyz"), ResidueStatus.UNRECOGNISED)
        self.assertFalse(NCAARegistry.status("Xyz").permits_scoring)

    def test_d_residues_need_no_new_parameters(self):
        """
        The one case in the catalogue that transfers. Standard force-field
        functional forms are achiral: the bonded and non-bonded terms are the L
        values and the inversion rides on the C-alpha improper dihedral.
        """
        self.assertIs(NCAARegistry.status("D-residue"), ResidueStatus.CANONICAL)
        self.assertTrue(NCAARegistry.status("D-residue").permits_scoring)

    def test_the_d_residue_entry_states_what_must_still_be_checked(self):
        """Transferable parameters are not the same as nothing to verify."""
        note = NCAARegistry.catalogue()["D-residue"].note
        self.assertIn("improper", note)

    def test_only_canonical_and_parameterized_permit_scoring(self):
        self.assertTrue(ResidueStatus.CANONICAL.permits_scoring)
        self.assertTrue(ResidueStatus.PARAMETERIZED.permits_scoring)
        self.assertFalse(ResidueStatus.UNPARAMETERIZED.permits_scoring)
        self.assertFalse(ResidueStatus.UNRECOGNISED.permits_scoring)


class TestDetection(unittest.TestCase):

    def test_detects_the_engines_own_proposals(self):
        cases = {
            "A2 -> Aib (alpha-aminoisobutyric acid)": "Aib",
            "i,i+4 hydrocarbon staple across residues 19-23": "staple",
            "N-methylation of the backbone amide": "N-methyl",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIn(expected, detect_noncanonical(text))

    def test_a_linker_naming_several_chemistries_reports_all_of_them(self):
        """
        Matching only the first would understate what needs parameterizing, and
        the cost of the proposal would look like one residue's work.
        """
        found = detect_noncanonical(
            "C18 diacid acylation at Lys28 via a gamma-Glu-2xOEG linker")
        for expected in ("gamma-Glu", "OEG", "fatty-acid"):
            self.assertIn(expected, found)

    def test_oeg_is_found_inside_a_token(self):
        """
        Regression: the pattern required a word boundary before OEG, which never
        matches in "2xOEG" because 'x' and 'O' are both word characters. Found
        on the engine's own lipidation proposal.
        """
        self.assertIn("OEG", detect_noncanonical("gamma-Glu-2xOEG linker"))
        self.assertIn("OEG", detect_noncanonical("a 2xAEEA spacer"))

    def test_canonical_substitutions_are_not_flagged(self):
        for text in ("Trp25 -> Phe", "Ala2 -> Ser", "charge engineering at Lys20"):
            with self.subTest(text=text):
                self.assertEqual(detect_noncanonical(text), [])

    def test_terminal_capping_is_not_mistaken_for_acylation(self):
        """
        "acetylation" and "acylation" differ by one letter, and terminal capping
        is genuinely parameterized (ACE and NHE caps ship with ff19SB). A
        regression here would route a good proposal to a research request.
        """
        self.assertEqual(
            detect_noncanonical("N-terminal acetylation + C-terminal amidation"), [])


class TestGate(unittest.TestCase):

    def test_an_unparameterized_proposal_yields_a_research_request(self):
        requests = gate_proposal("A2 -> Aib")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].residue, "Aib")

    def test_a_canonical_proposal_passes_the_gate(self):
        self.assertEqual(gate_proposal("Trp25 -> Phe"), [])

    def test_a_d_residue_proposal_passes_the_gate(self):
        self.assertEqual(gate_proposal("D-Ala substitution at position 2"), [])

    def test_require_parameterized_raises_for_unparameterized_chemistry(self):
        with self.assertRaises(ParameterizationError) as ctx:
            NCAARegistry.require_parameterized("Aib")
        self.assertIn("may not be scored, simulated or ranked", str(ctx.exception))

    def test_require_parameterized_passes_for_a_d_residue(self):
        NCAARegistry.require_parameterized("D-residue")

    def test_a_research_request_carries_no_score(self):
        """
        A research request with a rank beside it reads as a recommendation with
        a caveat, which is the same failure wearing a different label.
        """
        request = gate_proposal("A2 -> Aib")[0]
        self.assertFalse(hasattr(request, "score"))
        self.assertFalse(hasattr(request, "rank"))


class TestCostIsStatedNotInvented(unittest.TestCase):

    def test_all_six_stages_are_outstanding_for_everything(self):
        """Nothing has been parameterized, so nothing has stages behind it."""
        for code in ("Aib", "staple", "OEG"):
            with self.subTest(residue=code):
                self.assertEqual(len(NCAARegistry.cost(code).stages_outstanding), len(PIPELINE))

    def test_cost_reports_no_hours_or_currency(self):
        """
        Those depend on hardware, level of theory and how much is automated, so
        a figure would be invented and would then travel as though estimated.
        """
        for code in ("Aib", "staple", "OEG", "fatty-acid"):
            summary = NCAARegistry.cost(code).summary().lower()
            with self.subTest(residue=code):
                for tell in ("hour", "day", "week", "$", "usd", "cpu-h", "core-h"):
                    self.assertNotIn(tell, summary)

    def test_a_known_dihedral_count_is_used_when_it_is_a_plain_fact(self):
        """Aib's two alpha-methyls leave no side-chain dihedral to scan."""
        cost = NCAARegistry.cost("Aib")
        self.assertEqual(cost.known_dihedral_count, 0)
        self.assertIn("phi/psi", cost.summary())

    def test_an_unknown_count_says_what_it_scales_with_instead(self):
        cost = NCAARegistry.cost("staple")
        self.assertIsNone(cost.known_dihedral_count)
        self.assertTrue(cost.is_estimable)
        self.assertIn("scales with", cost.summary().lower())

    def test_the_summary_is_not_doubled_when_the_driver_is_a_clause(self):
        """Regression: drivers written as clauses produced 'scales with scales with'."""
        for code in NCAARegistry.catalogue():
            summary = NCAARegistry.cost(code).summary().lower()
            with self.subTest(residue=code):
                self.assertNotIn("scales with scales with", summary)

    def test_a_parameterized_residue_would_have_nothing_outstanding(self):
        self.assertEqual(NCAARegistry.cost("D-residue").stages_outstanding, [])


class TestPipelineIsStubbedNotFaked(unittest.TestCase):

    def test_every_stage_raises(self):
        """
        A stage returning something plausible would let the registry fill with
        parameters nobody computed, which is the failure this module prevents.
        """
        for stage in PIPELINE:
            with self.subTest(stage=stage.value):
                with self.assertRaises(PipelineNotImplemented):
                    run_stage(stage, "Aib")

    def test_the_stub_says_what_the_stage_would_need(self):
        with self.assertRaises(PipelineNotImplemented) as ctx:
            run_stage(ParameterizationStage.RESP_CHARGES, "Aib")
        message = str(ctx.exception)
        self.assertIn("ESP grid", message)
        self.assertIn("restraint weights", message)

    def test_the_pipeline_has_six_ordered_stages(self):
        self.assertEqual(len(PIPELINE), 6)
        self.assertIs(PIPELINE[0], ParameterizationStage.GEOMETRY_OPTIMIZATION)
        self.assertIs(PIPELINE[-1], ParameterizationStage.EXPERIMENTAL_VALIDATION)

    def test_validation_is_a_required_stage_not_an_optional_one(self):
        """
        Parameters that reproduce the QM are not the same as parameters that
        reproduce the molecule.
        """
        self.assertIn(ParameterizationStage.EXPERIMENTAL_VALIDATION, PIPELINE)
        self.assertIn("not reproduced",
                      ParameterizationStage.EXPERIMENTAL_VALIDATION.description
                      + " " + " ".join(pipeline_specification()))


class TestWorkflowRoutesProposals(unittest.TestCase):

    def _run(self):
        from peptide_suite.workflows.transform import TransformWorkflow
        return TransformWorkflow().run(GLP1, structure_candidates=[BOUND_STRUCTURE])

    def test_unparameterized_proposals_do_not_reach_the_ranked_list(self):
        result = self._run()
        for t in result["transformations"]:
            with self.subTest(proposal=t.description):
                self.assertEqual(gate_proposal(f"{t.description} {t.rationale}"), [])

    def test_they_appear_as_research_requests_instead(self):
        requests = self._run()["research_requests"]
        self.assertTrue(requests)
        proposals = " ".join(r["proposal"] for r in requests)
        self.assertIn("Aib", proposals)
        self.assertIn("staple", proposals)

    def test_the_gate_is_independent_of_the_structure_gate(self):
        """
        A perfect bound structure does not make unparameterized chemistry
        rankable. The two refusals have different remedies.
        """
        result = self._run()
        self.assertFalse(result["structure_template"].is_refusal)
        self.assertTrue(result["research_requests"])

    def test_canonical_proposals_still_rank(self):
        """A gate that silenced everything would be indistinguishable from a bug."""
        self.assertTrue(self._run()["transformations"])

    def test_the_empty_registry_is_reported_with_the_result(self):
        self.assertTrue(self._run()["registry_is_empty"])


if __name__ == "__main__":
    unittest.main()
