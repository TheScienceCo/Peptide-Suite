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


class TestNothingIsUsableYet(unittest.TestCase):
    """
    Parameterization work has now genuinely started -- Aib has three of six
    stages recorded, with real HF/6-31G* artifacts behind them. None of that
    makes a residue usable, and these tests are the difference between
    recording progress and claiming a result.
    """

    def test_nothing_may_be_scored(self):
        self.assertEqual(NCAARegistry.usable(), {})
        self.assertTrue(NCAARegistry.is_empty())

    def test_a_record_exists_and_licenses_nothing(self):
        # The registry file is no longer empty. That is the point: progress is
        # checkable. What must not change is what it permits.
        self.assertTrue(NCAARegistry.registry(),
                        "expected at least one partial record to guard")
        for code, record in NCAARegistry.registry().items():
            with self.subTest(residue=code):
                self.assertFalse(NCAARegistry.status(code).permits_scoring)
                self.assertLess(len(record["stages_completed"]), len(PIPELINE),
                                "a complete record would have to be validated, not just run")

    def test_catalogue_is_not_the_registry(self):
        """
        Recognising a residue is not being ready to score it. Merging the two
        files would let recognition be mistaken for readiness.
        """
        self.assertTrue(NCAARegistry.catalogue())
        for code in NCAARegistry.catalogue():
            self.assertNotIn(code, NCAARegistry.usable())

    def test_a_partial_record_cannot_be_promoted_by_adding_stages_it_did_not_run(self):
        # The mechanism must work in both directions, or it is not a mechanism.
        # A record listing every stage does make a residue usable -- which is
        # why writing one is a deliberate act and not a side effect.
        import peptide_suite.core.ncaa_registry as module
        saved = module.NCAARegistry._registry
        try:
            module.NCAARegistry._registry = {
                "Aib": {"stages_completed": [s.value for s in PIPELINE]},
                "Nle": {"stages_completed": [PIPELINE[0].value]},
            }
            self.assertIs(NCAARegistry.status("Aib"), ResidueStatus.PARAMETERIZED)
            self.assertIs(NCAARegistry.status("Nle"), ResidueStatus.IN_PROGRESS)
            self.assertFalse(NCAARegistry.status("Nle").permits_scoring)
        finally:
            module.NCAARegistry._registry = saved


class TestStatus(unittest.TestCase):

    def test_catalogued_residues_are_not_scorable(self):
        # Aib has work behind it now and the others do not; neither state
        # permits scoring, and that is the only thing the gate cares about.
        for code in ("Aib", "Nle", "Cha", "staple", "OEG", "gamma-Glu"):
            with self.subTest(residue=code):
                self.assertFalse(NCAARegistry.status(code).permits_scoring)

    def test_a_residue_with_partial_work_is_in_progress_not_unparameterized(self):
        # Reporting Aib as untouched would understate what is on disk; the
        # research request would then quote a cost that has already been paid.
        self.assertIs(NCAARegistry.status("Aib"), ResidueStatus.IN_PROGRESS)
        self.assertIs(NCAARegistry.status("Nle"), ResidueStatus.UNPARAMETERIZED)

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

    def test_a_residue_with_no_work_behind_it_has_every_stage_outstanding(self):
        for code in ("staple", "OEG", "Nle"):
            with self.subTest(residue=code):
                self.assertEqual(len(NCAARegistry.cost(code).stages_outstanding), len(PIPELINE))

    def test_completed_stages_are_subtracted_from_the_cost(self):
        # The cost has to fall as work is done, or the research request keeps
        # quoting for work already paid for.
        done = NCAARegistry.completed_stages("Aib")
        self.assertTrue(done, "expected Aib to have recorded stages")
        outstanding = NCAARegistry.cost("Aib").stages_outstanding
        self.assertEqual(len(outstanding), len(PIPELINE) - len(done))
        for stage in done:
            self.assertNotIn(stage, outstanding)

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

    def test_every_stage_without_an_engine_still_raises(self):
        """
        A stage returning something plausible would let the registry fill with
        parameters nobody computed, which is the failure this module prevents.

        The geometry stage has an engine now and is exercised separately; the
        rest have none in this process and must still refuse rather than
        return.
        """
        for stage in PIPELINE:
            if stage is ParameterizationStage.GEOMETRY_OPTIMIZATION:
                continue
            with self.subTest(stage=stage.value):
                with self.assertRaises(PipelineNotImplemented):
                    run_stage(stage, "Aib")

    def test_the_geometry_stage_will_not_guess_a_structure(self):
        # Guessing a capped structure for a residue whose chemistry is the
        # thing in question would be the whole error.
        with self.assertRaises(ValueError) as ctx:
            run_stage(ParameterizationStage.GEOMETRY_OPTIMIZATION, "Aib")
        self.assertIn("SMILES", str(ctx.exception))

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


class TestTheRecordedArtifactIsReal(unittest.TestCase):
    """
    Aib's record is backed by an HF/6-31G* calculation that actually ran. These
    check the artifact against things that must be true of a real RESP fit, so
    a hand-edited or invented record fails rather than passing for a real one.
    """

    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cls.artifact = json.loads((root / "data" / "qm_artifacts" / "Aib.json").read_text())
        cls.record = NCAARegistry.registry()["Aib"]

    def test_charges_sum_to_the_formal_charge(self):
        # A RESP fit is constrained to this. A set that misses it was not
        # produced by one.
        total = sum(self.record["charges"])
        self.assertAlmostEqual(total, self.artifact["formal_charge"], places=6)

    def test_there_is_one_charge_per_atom(self):
        self.assertEqual(len(self.record["charges"]), len(self.record["symbols"]))
        self.assertEqual(len(self.record["symbols"]), 25)       # ACE-Aib-NME

    def test_methyl_hydrogens_came_out_equivalenced(self):
        # Stage 2 constrains each methyl's three hydrogens to one charge. Four
        # methyls here, so four groups of three identical values.
        hydrogens = [round(q, 9) for s, q in zip(self.record["symbols"], self.record["charges"])
                     if s == "H"]
        triples = sum(1 for value in set(hydrogens) if hydrogens.count(value) == 3)
        self.assertEqual(triples, 4, f"expected four equivalenced methyls, got {hydrogens}")

    def test_the_fit_residual_is_a_residual_and_not_a_charge(self):
        # The first version of the QM script reported an atomic charge under
        # the name esp_rrms. It read as a plausible residual (0.86) and was a
        # carbonyl carbon. A relative RMS is non-negative and well under 1 for
        # any fit worth recording.
        rrms = self.record["esp_rrms"]
        self.assertGreater(rrms, 0.0)
        self.assertLess(rrms, 0.5, "an RRMS this large is not a usable RESP fit")
        self.assertGreater(self.record["n_grid_points"], 100)

    def test_carbonyl_and_amide_charges_have_the_right_signs(self):
        # Cheap chemistry check: both carbonyl oxygens negative, both amide
        # nitrogens negative, both carbonyl carbons positive.
        by_element = {}
        for index, (symbol, charge) in enumerate(
                zip(self.record["symbols"], self.record["charges"])):
            by_element.setdefault(symbol, []).append(charge)
        self.assertTrue(all(q < -0.4 for q in by_element["O"]), by_element["O"])
        self.assertTrue(all(q < -0.3 for q in by_element["N"]), by_element["N"])
        self.assertGreaterEqual(sum(1 for q in by_element["C"] if q > 0.4), 2)

    def test_the_level_of_theory_is_the_one_the_stage_requires(self):
        from peptide_suite.core.qm_engine import LevelOfTheory
        self.assertEqual(self.record["geometry_method"], LevelOfTheory.HF_631Gd.value)
        self.assertTrue(LevelOfTheory.HF_631Gd.licenses_resp_charges)
        self.assertIn("GFN2-xTB", self.record["geometry_started_from"])

    def test_the_record_states_what_it_does_not_reproduce(self):
        # A record whose validation section is blank reads as a validated one.
        self.assertIn("Nothing", self.record["validated_against"])
        self.assertTrue(self.record["known_deficiencies"],
                        "a single-conformer RESP fit has known deficiencies; a record "
                        "claiming none is a record that did not look")

    def test_the_unfinished_stages_are_left_blank_not_filled_in(self):
        for field in ("torsion_scans", "experimental_validation"):
            self.assertEqual(self.record[field], "",
                             f"{field} was not run, so it must not carry prose")
