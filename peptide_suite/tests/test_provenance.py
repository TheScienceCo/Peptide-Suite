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
    LicenseViolation, ParameterBudget, ProvenanceTier, Quantity,
    ReceptorComplex,
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
    """Budget is computed over the pooled measurement set, not per class."""

    def _pooled(self, free_parameters):
        return (ParameterBudget(divisor=10.0, free_parameters=free_parameters)
                .add_pool(ReceptorComplex("GLP1R", species="human"), 220)
                .add_pool(ReceptorComplex("GIPR", species="human"), 160))

    def test_within_budget(self):
        b = self._pooled(25)
        self.assertEqual(b.n_measured, 380)
        self.assertEqual(b.budget, 38.0)
        self.assertTrue(b.within_budget)
        b.require()

    def test_over_budget_raises(self):
        b = self._pooled(64)
        self.assertFalse(b.within_budget)
        with self.assertRaises(LicenseViolation) as ctx:
            b.require()
        self.assertIn("removing one or acquiring data", str(ctx.exception))

    def test_pooling_sums_across_complexes(self):
        """Pooling is the point: one artifact, one parameter count, all the data."""
        single = ParameterBudget(divisor=10.0).add_pool(
            ReceptorComplex("GLP1R", species="human"), 220)
        self.assertEqual(single.budget, 22.0)
        self.assertEqual(self._pooled(0).budget, 38.0)

    def test_report_line_names_the_pools(self):
        line = self._pooled(25).report_line()
        self.assertIn("GLP1R(human)=220", line)
        self.assertIn("GIPR(human)=160", line)
        self.assertIn("WITHIN BUDGET", line)

    def test_no_measured_data_is_infinite_utilisation(self):
        b = ParameterBudget(divisor=10.0, free_parameters=1)
        self.assertFalse(b.within_budget)


class TestReceptorComplexPooling(unittest.TestCase):
    """
    A receptor gene alone is underspecified where accessory proteins determine
    pharmacology. CLR+RAMP1 is the CGRP receptor; CLR+RAMP2 is AM1.
    """

    def test_accessory_subunits_distinguish_complexes(self):
        cgrp = ReceptorComplex("CLR", ("RAMP1",), "human")
        am1 = ReceptorComplex("CLR", ("RAMP2",), "human")
        self.assertNotEqual(cgrp, am1)
        self.assertFalse(cgrp.poolable_with(am1))

    def test_identifier_includes_accessory(self):
        self.assertIn("RAMP1", ReceptorComplex("CLR", ("RAMP1",)).identifier)

    def test_accessory_order_does_not_matter(self):
        a = ReceptorComplex("CTR", ("RAMP1", "RAMP3"))
        b = ReceptorComplex("CTR", ("RAMP3", "RAMP1"))
        self.assertTrue(a.poolable_with(b))

    def test_pooling_different_accessory_complexes_is_refused(self):
        """Silently pooling these corrupts the training set."""
        budget = ParameterBudget(divisor=10.0)
        budget.add_pool(ReceptorComplex("CLR", ("RAMP1",)), 40)
        with self.assertRaises(LicenseViolation) as ctx:
            budget.add_pool(ReceptorComplex("CLR", ("RAMP2",)), 30)
        self.assertIn("different pharmacology", str(ctx.exception))

    def test_distinct_receptors_pool_freely(self):
        budget = ParameterBudget(divisor=10.0)
        budget.add_pool(ReceptorComplex("GLP1R"), 100)
        budget.add_pool(ReceptorComplex("GCGR"), 50)
        self.assertEqual(budget.n_measured, 150)


class TestPhysicalConstants(unittest.TestCase):
    """Exempt from the boundary because they are exact by SI definition."""

    def test_si_defining_constants_are_exact(self):
        from peptide_suite.core import constants as c
        self.assertEqual(c.SPEED_OF_LIGHT_M_PER_S, 299_792_458)
        self.assertEqual(c.BOLTZMANN_J_PER_K, 1.380649e-23)
        self.assertEqual(c.AVOGADRO_PER_MOL, 6.02214076e23)

    def test_gas_constant_derives_correctly(self):
        from peptide_suite.core import constants as c
        self.assertAlmostEqual(c.GAS_CONSTANT_J_PER_MOL_K, 8.31446261815324, places=8)

    def test_rt_matches_the_textbook_value(self):
        from peptide_suite.core import constants as c
        self.assertAlmostEqual(c.rt_kcal_per_mol(25.0), 0.5925, places=3)

    def test_ph_is_not_treated_as_a_constant(self):
        """pH 7.4 is a modelling convention, not a physical constant."""
        from peptide_suite.core import constants as c
        names = [n for n in dir(c) if not n.startswith("_")]
        self.assertFalse([n for n in names if "PH" in n.upper() and "PLANCK" not in n.upper()],
                         "pH must not live in the constants module")

    def test_pka_values_are_not_treated_as_constants(self):
        from peptide_suite.core import constants as c
        self.assertFalse([n for n in dir(c) if "PKA" in n.upper()],
                         "pKa values are measured reference data, not constants")


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
    """
    Weights are private policy; the engine must not score without one loaded.

    This module used to carry its own PolicyPack sketch, written before
    Addendum 1 existed. The real one now lives in peptide_suite.policy and the
    sketch is gone -- two names for the source of coefficients is how a caller
    ends up holding the wrong one.
    """

    def test_unloaded_policy_raises_rather_than_scoring_with_zeros(self):
        from peptide_suite import runtime
        saved = runtime._active
        runtime.clear_active_policy()
        try:
            with self.assertRaises(runtime.PolicyNotLoaded) as ctx:
                runtime.weight("evidence.tier_direct_experimental")
        finally:
            runtime.set_active_policy(saved)
        self.assertIn("will not substitute defaults", str(ctx.exception))

    def test_provenance_module_exposes_no_policy_sketch(self):
        """A second policy interface is a second place to forget the boundary."""
        from peptide_suite.core import provenance
        self.assertFalse(hasattr(provenance, "PolicyPack"))

    def test_engine_module_contains_no_weights(self):
        """A weight literal in the engine is a boundary violation."""
        import inspect
        from peptide_suite.core import provenance
        source = inspect.getsource(provenance)
        self.assertNotIn("WEIGHTS", source)
        self.assertNotIn("weight =", source)


if __name__ == "__main__":
    unittest.main()
