"""
The policy boundary as behaviour, not architecture.  [Addendum 1 section 1]

A fail-loud loader that nothing calls is documentation. These tests check the
part that matters: the engine genuinely cannot produce a number without a
policy, and the migrated coefficients really come from the artifact rather than
from a constant that happens to agree with it.

Each test clears the process policy and restores it, so the rest of the suite
(which loads the demo pack at import) is unaffected.
"""

import unittest
from contextlib import contextmanager

from peptide_suite import runtime
from peptide_suite.core import min_homologs_for_conservation
from peptide_suite.core.function_inference import FunctionInferencer
from peptide_suite.core.peptide_manager import PeptideManager, peptide_length_ceiling
from peptide_suite.policy import load_policy
from peptide_suite.tests import DEMO_PACK


@contextmanager
def no_policy():
    saved = runtime._active
    runtime.clear_active_policy()
    try:
        yield
    finally:
        runtime.set_active_policy(saved)


@contextmanager
def policy_with(threshold_id, value):
    """Run with one threshold overridden, to prove the value is really read."""
    saved = runtime._active
    document = _load_document()
    document["thresholds"][threshold_id]["value"] = value
    runtime.set_active_policy(_policy_from(document))
    try:
        yield
    finally:
        runtime.set_active_policy(saved)


def _load_document():
    import json
    return json.loads(DEMO_PACK.read_text())


def _policy_from(document):
    """Build a Policy from an in-memory document, re-signing it first."""
    import json
    import tempfile
    from pathlib import Path
    from peptide_suite.policy import compute_digest
    document["integrity"]["digest"] = compute_digest(document)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(document, handle)
    handle.close()
    return load_policy(Path(handle.name))


class TestEngineRefusesToRunWithoutPolicy(unittest.TestCase):

    def test_length_classification_raises(self):
        with no_policy():
            with self.assertRaises(runtime.PolicyNotLoaded):
                PeptideManager().classify_length("A" * 50)

    def test_conservation_gate_raises(self):
        with no_policy():
            with self.assertRaises(runtime.PolicyNotLoaded):
                min_homologs_for_conservation()

    def test_identification_raises(self):
        with no_policy():
            with self.assertRaises(runtime.PolicyNotLoaded):
                FunctionInferencer().infer("GEPPPGKPADDAGLV")

    def test_error_names_a_remedy(self):
        """A fail-loud error the operator cannot act on is just a crash."""
        with no_policy():
            with self.assertRaises(runtime.PolicyNotLoaded) as ctx:
                runtime.threshold("chemistry.reference_ph")
        message = str(ctx.exception)
        self.assertIn("PEPTIDE_SUITE_POLICY", message)
        self.assertIn("load_active_policy", message)
        self.assertIn("will not substitute defaults", message)


class TestValuesActuallyComeFromThePolicy(unittest.TestCase):
    """
    The failure this guards against: a migration that reads the policy but is
    shadowed by a constant that happens to hold the same number. Changing the
    pack must change the behaviour.
    """

    def test_length_ceiling_follows_the_pack(self):
        self.assertEqual(peptide_length_ceiling(), 100)
        with policy_with("identification.peptide_length_ceiling", 40.0):
            self.assertEqual(peptide_length_ceiling(), 40)
            self.assertTrue(PeptideManager().classify_length("A" * 50)["is_protein"])
        self.assertFalse(PeptideManager().classify_length("A" * 50)["is_protein"])

    def test_conservation_gate_follows_the_pack(self):
        self.assertEqual(min_homologs_for_conservation(), 3)
        with policy_with("conservation.min_distinct_sequences", 7.0):
            self.assertEqual(min_homologs_for_conservation(), 7)

    def test_containment_length_follows_the_pack(self):
        """
        A raised containment floor must suppress the fragment-of path that the
        default floor accepts. The sequence may still be identified by another
        route — the assertion is about which route, not about whether anything
        matched, because a weaker route firing is exactly what a raised floor is
        supposed to leave behind.
        """
        fragment = "GEPPPGKP"
        baseline = FunctionInferencer().infer(fragment)
        self.assertIn("Fragment of", baseline.basis,
                      "fixture no longer takes the containment path; pick another")

        with policy_with("identification.min_containment_length", 50.0):
            raised = FunctionInferencer().infer(fragment)
        self.assertNotIn("Fragment of", raised.basis)


class TestCountsMustBeWholeNumbers(unittest.TestCase):

    def test_fractional_count_is_an_error_not_a_truncation(self):
        """
        Truncating silently would turn a malformed pack into a working one, and
        the operator would never learn the pack was wrong.
        """
        with policy_with("identification.peptide_length_ceiling", 40.5):
            with self.assertRaises(ValueError) as ctx:
                peptide_length_ceiling()
        self.assertIn("not a whole number", str(ctx.exception))


class TestRuntimeReportsPlaceholders(unittest.TestCase):

    def test_demo_pack_is_flagged_as_running_on_placeholders(self):
        self.assertTrue(runtime.running_on_placeholders())

    def test_is_loaded_tracks_state(self):
        self.assertTrue(runtime.is_loaded())
        with no_policy():
            self.assertFalse(runtime.is_loaded())
        self.assertTrue(runtime.is_loaded())


if __name__ == "__main__":
    unittest.main()
