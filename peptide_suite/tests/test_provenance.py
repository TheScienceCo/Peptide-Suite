"""
Tests for the provenance tier schema and licensing enforcement. [Addendum 2, 6a]

These exist because the failure mode being prevented is silent. A QM descriptor
reaching a learner, or a feature computed on a structure whose gate failed,
produces output that looks entirely normal. Nothing downstream reveals it, so
the enforcement has to be here and it has to raise.
"""

import unittest

from peptide_suite.core.provenance import (
    UNRESOLVED, AggregateTerm, AggregateTermRegistry, Capability, ClaimTrace,
    LicenseViolation, ParameterBudget, PolicyPack, ProvenanceTier, Quantity,
    StructureGate, UnresolvedValue, licensing_table_markdown,
)


def a_measured(**over):
    kwargs = dict(name="Kd_GLP1R", value=2.4, assay_type="radioligand competition",
                  readout="Kd", construct="GLP-1R(1-463) full length", species="human",
                  temperature_c=25.0, buffer="20 mM HEPES pH 7.4, 150 mM NaCl",
                  citation="doi:10.0000/example", units="nM")
    kwargs.update(over)
    return Quantity.measured(**kwargs)


def a_qm(**over):
    kwargs = dict(name="sapt_electrostatics", value=-8.2,
                  level_of_theory="SAPT0/jun-cc-pVDZ", solvation_model="none (gas phase)",
                  truncation_scheme="residue pair, capped with methyl", units="kcal/mol")
    kwargs.update(over)
    return Quantity.qm(**kwargs)


class TestMetadataContract(unittest.TestCase):
    """A number that cannot be reproduced should fail to exist, not exist unlabelled."""

    def test_measured_requires_full_assay_metadata(self):
        with self.assertRaises(ValueError) as ctx:
            Quantity(name="Kd", value=2.4, tier=ProvenanceTier.MEASURED,
                     metadata={"citation": "doi:10.0000/example"})
        msg = str(ctx.exception)
        for required in ("assay_type", "readout", "construct", "species", "buffer"):
            self.assertIn(required, msg)

    def test_predicted_structure_requires_confidence_metadata(self):
        with self.assertRaises(ValueError):
            Quantity(name="contact_distance", value=3.1,
                     tier=ProvenanceTier.PREDICTED_STRUCTURE,
                     metadata={"model": "AF3"})

    def test_qm_requires_level_of_theory(self):
        with self.assertRaises(ValueError):
            Quantity(name="e_elst", value=-8.2, tier=ProvenanceTier.QM,
                     metadata={"solvation_model": "none"})

    def test_every_tier_declares_required_metadata(self):
        from peptide_suite.core.provenance import REQUIRED_METADATA
        for tier in ProvenanceTier:
            self.assertIn(tier, REQUIRED_METADATA, f"{tier} has no metadata contract")
            self.assertTrue(REQUIRED_METADATA[tier], f"{tier} requires nothing")

    def test_measured_rejects_an_unrecognised_readout(self):
        """Kd, IC50 and EC50 are not one quantity and must not be pooled."""
        with self.assertRaises(ValueError) as ctx:
            a_measured(readout="percent inhibition")
        self.assertIn("readout", str(ctx.exception))

    def test_valid_construction_succeeds(self):
        q = a_measured()
        self.assertEqual(q.tier, ProvenanceTier.MEASURED)
        self.assertTrue(q.resolved)
        self.assertEqual(q.as_float(), 2.4)


class TestLicensingTable(unittest.TestCase):

    def test_only_measured_may_be_a_regression_target(self):
        permitted = [t for t in ProvenanceTier
                     if Capability.REGRESSION_TARGET in
                     __import__("peptide_suite.core.provenance", fromlist=["LICENSING"]).LICENSING[t]]
        self.assertEqual(permitted, [ProvenanceTier.MEASURED])

    def test_qm_cannot_be_a_regression_target(self):
        with self.assertRaises(LicenseViolation) as ctx:
            a_qm().require(Capability.REGRESSION_TARGET)
        self.assertIn("Only MEASURED", str(ctx.exception))

    def test_qm_cannot_be_directly_weighted(self):
        with self.assertRaises(LicenseViolation) as ctx:
            a_qm().require(Capability.SCORE_WEIGHTED)
        self.assertIn("aggregate", str(ctx.exception).lower())

    def test_raw_qm_may_not_reach_a_learner(self):
        with self.assertRaises(LicenseViolation) as ctx:
            a_qm().require(Capability.LEARNER_RAW_INPUT)
        self.assertIn("learner", str(ctx.exception).lower())

    def test_qm_may_gate_and_explain(self):
        q = a_qm()
        q.require(Capability.GATE)
        q.require(Capability.EXPLAIN)
        self.assertTrue(q.permits(Capability.SCORE_AGGREGATE_ONLY))

    def test_semiempirical_has_the_same_restrictions_as_qm(self):
        q = Quantity.semiempirical(name="conformer_dG", value=1.4, method="GFN2-xTB",
                                   solvation_model="ALPB(water)", ensemble_size=48,
                                   energy_window_kcal=6.0)
        self.assertFalse(q.permits(Capability.SCORE_WEIGHTED))
        self.assertFalse(q.permits(Capability.LEARNER_RAW_INPUT))
        self.assertTrue(q.permits(Capability.SCORE_AGGREGATE_ONLY))

    def test_literature_asserted_may_never_be_weighted(self):
        q = Quantity.literature_asserted(
            name="ser3_octanoylation_required",
            claim_string="Des-acyl ghrelin does not activate GHSR1a",
            citation="doi:10.0000/ghrelin")
        self.assertTrue(q.permits(Capability.GATE))
        self.assertTrue(q.permits(Capability.EXPLAIN))
        with self.assertRaises(LicenseViolation):
            q.require(Capability.SCORE_WEIGHTED)
        with self.assertRaises(LicenseViolation):
            q.require(Capability.REGRESSION_TARGET)

    def test_literature_asserted_carries_no_number(self):
        q = Quantity.literature_asserted(name="x", claim_string="y", citation="z")
        self.assertFalse(q.resolved)

    def test_measured_structure_is_not_a_regression_target(self):
        q = Quantity.measured_structure(
            name="contact_distance", value=3.1, pdb_id="6X18", method="cryo-EM",
            resolution_angstrom=3.3, peptide_resolved=True, peptide_b_factor=88.0)
        self.assertTrue(q.permits(Capability.SCORE_WEIGHTED))
        with self.assertRaises(LicenseViolation):
            q.require(Capability.REGRESSION_TARGET)

    def test_violation_messages_explain_the_reason(self):
        """An enforcement that does not say why gets worked around."""
        with self.assertRaises(LicenseViolation) as ctx:
            a_qm().require(Capability.REGRESSION_TARGET)
        self.assertGreater(len(str(ctx.exception)), 80)

    def test_licensing_table_renders(self):
        table = licensing_table_markdown()
        self.assertIn("MEASURED", table)
        self.assertIn("regression_target", table)
        self.assertIn("confidence gate", table)


class TestStructureConfidenceGate(unittest.TestCase):
    """A feature on a failed structure is not a low-confidence number; it is not a number."""

    def _predicted(self, **over):
        kwargs = dict(name="interface_contact", value=3.4, model="AF3",
                      model_version="2024.1", plddt=88.0, pae=3.1, iptm=0.81, seed_count=5)
        kwargs.update(over)
        return Quantity.predicted_structure(**kwargs)

    def test_passing_gate_grants_capabilities(self):
        q = self._predicted()
        self.assertTrue(q.gate.passed)
        self.assertTrue(q.resolved)
        q.require(Capability.SCORE_WEIGHTED)

    def test_failing_plddt_yields_unresolved(self):
        q = self._predicted(plddt=42.0)
        self.assertFalse(q.gate.passed)
        self.assertFalse(q.resolved)
        self.assertIs(q.value, UNRESOLVED)

    def test_failing_gate_grants_no_capability_at_all(self):
        q = self._predicted(iptm=0.2)
        self.assertEqual(q.capabilities, frozenset())
        for capability in Capability:
            with self.assertRaises(LicenseViolation):
                q.require(capability)

    def test_gate_reports_every_failure(self):
        q = self._predicted(plddt=40.0, pae=12.0, iptm=0.1, seed_count=1)
        self.assertEqual(len(q.gate.failures), 4)

    def test_insufficient_seeds_fail_the_gate(self):
        self.assertFalse(self._predicted(seed_count=1).gate.passed)

    def test_gate_thresholds_are_configurable_but_default_conservative(self):
        strict = StructureGate(min_plddt=95.0, max_pae=1.0, min_iptm=0.95, min_seed_count=10)
        q = Quantity.predicted_structure(
            name="x", value=1.0, model="AF3", model_version="1", plddt=88.0,
            pae=3.1, iptm=0.81, seed_count=5, gate=strict)
        self.assertFalse(q.gate.passed)


class TestUnresolved(unittest.TestCase):
    """UNRESOLVED must not silently behave like a number."""

    def test_is_not_none_and_not_zero(self):
        self.assertIsNotNone(UNRESOLVED)
        self.assertNotEqual(UNRESOLVED, 0)

    def test_arithmetic_raises(self):
        for op in (lambda: UNRESOLVED + 1, lambda: UNRESOLVED * 2,
                   lambda: float(UNRESOLVED), lambda: UNRESOLVED < 1):
            with self.assertRaises(UnresolvedValue):
                op()

    def test_as_float_raises_with_the_gate_reason(self):
        q = Quantity.predicted_structure(name="x", value=3.0, model="AF3",
                                         model_version="1", plddt=30.0, pae=9.0,
                                         iptm=0.2, seed_count=1)
        with self.assertRaises(UnresolvedValue) as ctx:
            q.as_float()
        self.assertIn("pLDDT", str(ctx.exception))

    def test_is_falsy_for_presence_checks(self):
        self.assertFalse(bool(UNRESOLVED))


class TestAggregateTerms(unittest.TestCase):
    """The only route by which QM reaches scoring, and it is bounded."""

    def setUp(self):
        self.registry = AggregateTermRegistry()
        self.term = self.registry.declare(AggregateTerm(
            name="interface_dispersion_fraction",
            source_tiers=frozenset({ProvenanceTier.QM}),
            reduction="mean",
            max_inputs=12,
            description="Mean dispersion share across interface contact pairs",
        ))

    def test_undeclared_term_is_refused(self):
        with self.assertRaises(LicenseViolation) as ctx:
            self.registry.get("per_atom_charge_transfer")
        self.assertIn("declared", str(ctx.exception))

    def test_declared_term_accepts_its_source_tier(self):
        self.term.validate_inputs([a_qm(), a_qm()])

    def test_input_bound_is_enforced(self):
        with self.assertRaises(LicenseViolation) as ctx:
            self.term.validate_inputs([a_qm() for _ in range(50)])
        self.assertIn("at most 12", str(ctx.exception))

    def test_wrong_source_tier_is_refused(self):
        with self.assertRaises(LicenseViolation):
            self.term.validate_inputs([a_measured()])

    def test_duplicate_declaration_is_refused(self):
        with self.assertRaises(ValueError):
            self.registry.declare(AggregateTerm(
                name="interface_dispersion_fraction",
                source_tiers=frozenset({ProvenanceTier.QM}),
                reduction="sum", max_inputs=4))

    def test_registry_exposes_names_publicly(self):
        self.assertEqual(self.registry.names, ["interface_dispersion_fraction"])


class TestParameterBudget(unittest.TestCase):

    def test_within_budget(self):
        b = ParameterBudget(target_class="GLP-1R", n_measured=300, free_parameters=25)
        self.assertTrue(b.within_budget)
        self.assertEqual(b.budget, 30.0)
        b.require()

    def test_over_budget_raises(self):
        b = ParameterBudget(target_class="GLP-1R", n_measured=300, free_parameters=64)
        self.assertFalse(b.within_budget)
        with self.assertRaises(LicenseViolation) as ctx:
            b.require()
        self.assertIn("removing one or acquiring data", str(ctx.exception))

    def test_report_line_states_the_ratio(self):
        line = ParameterBudget(target_class="GLP-1R", n_measured=300,
                               free_parameters=25).report_line()
        self.assertIn("GLP-1R", line)
        self.assertIn("25", line)
        self.assertIn("300", line)
        self.assertIn("WITHIN BUDGET", line)

    def test_zero_measured_data_is_infinite_utilisation(self):
        b = ParameterBudget(target_class="novel", n_measured=0, free_parameters=1)
        self.assertFalse(b.within_budget)


class TestClaimTraceability(unittest.TestCase):
    """Any output claim must be traceable to the tier of every number behind it."""

    def test_weakest_tier_limits_the_claim(self):
        trace = ClaimTrace(claim="Contact is dispersion-dominated",
                           quantities=[a_measured(), a_qm()])
        self.assertEqual(trace.weakest_tier, ProvenanceTier.QM)

    def test_unresolved_inputs_are_surfaced(self):
        failed = Quantity.predicted_structure(
            name="x", value=3.0, model="AF3", model_version="1",
            plddt=30.0, pae=9.0, iptm=0.2, seed_count=1)
        trace = ClaimTrace(claim="Salt bridge present", quantities=[failed])
        self.assertTrue(trace.has_unresolved)
        self.assertIn("UNRESOLVED", trace.render())

    def test_render_names_every_tier(self):
        trace = ClaimTrace(claim="c", quantities=[a_measured(), a_qm()])
        rendered = trace.render()
        self.assertIn("MEASURED", rendered)
        self.assertIn("QM", rendered)


class TestPolicyPackSeparation(unittest.TestCase):
    """Weights are private policy; the engine must not score without one loaded."""

    def test_unloaded_policy_raises_rather_than_scoring_with_zeros(self):
        with self.assertRaises(NotImplementedError) as ctx:
            PolicyPack().weight_for("interface_electrostatics")
        self.assertIn("private", str(ctx.exception).lower())

    def test_engine_module_contains_no_weights(self):
        """A weight literal in the engine is a boundary violation."""
        import inspect
        from peptide_suite.core import provenance
        source = inspect.getsource(provenance)
        self.assertNotIn("WEIGHTS", source)
        self.assertNotIn("weight =", source)


if __name__ == "__main__":
    unittest.main()
