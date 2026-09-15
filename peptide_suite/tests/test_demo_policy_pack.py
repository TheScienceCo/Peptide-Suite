"""
Tests for the committed demonstration pack.  [Addendum 1 step 3]

The risk a demonstration pack carries is not that it is wrong. It is that it
looks right: an artifact with a plausible spread of numbers is indistinguishable
at a glance from a fitted one, and whoever copies it into production ships
values nobody measured.

So these tests pin the two properties that make the pack visibly unfitted —
uniform weights, and placeholders that are exactly the midpoint of their
declared range — and they fail if anyone tunes it.
"""

import json
import unittest
from pathlib import Path

from peptide_suite.policy import (
    FEATURE_FAMILIES, THRESHOLDS, PolicyValidationError, compute_digest,
    is_opaque_term_id, load_policy, validate_document,
)
from peptide_suite.policy.loader import (
    EVIDENCE_TIER_ORDER, ORDERED_THRESHOLD_GROUPS, SCHEMA_VERSION,
)

ORDERED_MEMBERS = {k for group in ORDERED_THRESHOLD_GROUPS for k in group}

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "policy" / "demo.v1.json"


def load_document():
    return json.loads(PACK.read_text())


class TestPackExistsAndLoads(unittest.TestCase):

    def test_pack_is_committed(self):
        self.assertTrue(PACK.exists(), f"{PACK} is missing; run tools/make_demo_policy.py")

    def test_pack_loads_with_integrity_verification(self):
        policy = load_policy(PACK)
        self.assertEqual(policy.policy_version, "1.0.0")

    def test_digest_matches_content(self):
        """A stale digest means the file was hand-edited after generation."""
        document = load_document()
        self.assertEqual(compute_digest(document), document["integrity"]["digest"])

    def test_schema_version_matches_engine(self):
        self.assertEqual(load_document()["schema_version"], SCHEMA_VERSION)

    def test_pack_supplies_every_key_the_engine_declares(self):
        document = load_document()
        self.assertEqual(set(document["weights"]), set(FEATURE_FAMILIES))
        self.assertEqual(set(document["thresholds"]), set(THRESHOLDS))


class TestPackIsVisiblyUnfitted(unittest.TestCase):

    def test_marked_as_a_demonstration_pack(self):
        self.assertTrue(load_policy(PACK).is_demonstration)

    def test_banner_says_so(self):
        banner = load_policy(PACK).banner()
        self.assertIn("DEMONSTRATION PACK", banner)
        self.assertIn("placeholder", banner)

    def test_non_tier_weights_are_uniform(self):
        """
        No weight here was fitted. Equal weights state that; a spread of numbers
        would imply someone had measured which family mattered more.

        The evidence tiers are the one exception, and only because the engine
        requires them ordered — see below.
        """
        weights = load_document()["weights"]
        values = {v for k, v in weights.items() if k not in EVIDENCE_TIER_ORDER}
        self.assertEqual(len(values), 1, f"demo weights are not uniform: {sorted(values)}")

    def test_evidence_tiers_are_strictly_decreasing(self):
        """
        The hierarchy is ordinal by construction. A pack that flattens it has
        not reweighted the engine, it has removed the meaning of evidence.
        """
        weights = load_document()["weights"]
        ordered = [weights[k] for k in EVIDENCE_TIER_ORDER]
        for upper, lower in zip(ordered, ordered[1:]):
            self.assertGreater(upper, lower)

    def test_evidence_tier_spacing_is_mechanical(self):
        """
        Evenly spaced at (n-i)/n. The spacing satisfies the ordering constraint
        and is otherwise chosen by nothing — it is not a claim about how much
        more a measurement is worth than an inference.
        """
        weights = load_document()["weights"]
        n = len(EVIDENCE_TIER_ORDER)
        for i, key in enumerate(EVIDENCE_TIER_ORDER):
            with self.subTest(tier=key):
                self.assertAlmostEqual(weights[key], (n - i) / n, places=5)

    def test_every_placeholder_is_exactly_the_declared_midpoint(self):
        """
        The midpoint is chosen by nothing. If one drifts off it, someone tuned
        the demo pack, and a tuned demo pack is the thing this file exists to
        prevent.
        """
        for tid, entry in load_document()["thresholds"].items():
            if entry["basis"] != "placeholder_midpoint" or tid in ORDERED_MEMBERS:
                continue
            lo, hi = THRESHOLDS[tid].valid_range
            with self.subTest(threshold=tid):
                self.assertAlmostEqual(entry["value"], (lo + hi) / 2.0, places=5)

    def test_ordered_threshold_bands_are_not_collapsed(self):
        """
        Members of an ordered group cannot all sit on the midpoint: equal
        cutoffs delete the band between them, so a label like MEDIUM silently
        stops existing rather than becoming rare.
        """
        entries = load_document()["thresholds"]
        for group in ORDERED_THRESHOLD_GROUPS:
            values = [entries[k]["value"] for k in group]
            for (upper_id, upper), (lower_id, lower) in zip(
                    zip(group, values), zip(group[1:], values[1:])):
                with self.subTest(pair=(upper_id, lower_id)):
                    self.assertGreater(upper, lower)

    def test_every_threshold_declares_a_basis(self):
        for tid, entry in load_document()["thresholds"].items():
            with self.subTest(threshold=tid):
                self.assertIn(entry.get("basis"), {"derived", "placeholder_midpoint"})

    def test_pack_claims_no_measured_data(self):
        self.assertEqual(load_document()["provenance"]["n_measured"], 0)

    def test_placeholders_are_reported_by_the_loaded_policy(self):
        policy = load_policy(PACK)
        self.assertGreater(len(policy.placeholder_thresholds), 0)
        for tid in policy.placeholder_thresholds:
            self.assertEqual(policy.threshold_basis(tid), "placeholder_midpoint")


class TestDerivedValuesHaveRealSources(unittest.TestCase):
    """
    The derived thresholds are the ones a production pack would plausibly carry
    unchanged, so they are worth pinning against accidental edits.
    """

    def test_reference_ph_is_physiological(self):
        self.assertEqual(load_policy(PACK).threshold("chemistry.reference_ph"), 7.4)

    def test_multi_ph_set_matches_the_specification(self):
        policy = load_policy(PACK)
        self.assertEqual(policy.threshold("chemistry.interstitial_ph"), 6.5)
        self.assertEqual(policy.threshold("chemistry.endosomal_ph"), 5.5)

    def test_conservation_requires_three_distinct_sequences(self):
        """Below three, positional entropy is zero everywhere by construction."""
        self.assertEqual(load_policy(PACK).threshold("conservation.min_distinct_sequences"), 3.0)

    def test_parameter_budget_divisor_matches_the_specification(self):
        self.assertEqual(load_policy(PACK).threshold("parameter_budget.measured_ratio_divisor"), 10.0)

    def test_derived_values_are_not_midpoints(self):
        """A 'derived' value sitting on its midpoint is a mislabelled placeholder."""
        for tid, entry in load_document()["thresholds"].items():
            if entry["basis"] != "derived":
                continue
            lo, hi = THRESHOLDS[tid].valid_range
            with self.subTest(threshold=tid):
                self.assertNotAlmostEqual(entry["value"], (lo + hi) / 2.0, places=5)


class TestPackCannotBeRelabelled(unittest.TestCase):

    def test_relabelling_as_production_is_rejected(self):
        """
        The failure mode this blocks: someone flips pack_kind to ship a demo
        pack, and the placeholders become production coefficients silently.
        """
        document = load_document()
        document["pack_kind"] = "production"
        problems = validate_document(document, verify_integrity=False)
        self.assertTrue(
            any("placeholder" in p and "production pack" in p for p in problems),
            f"relabelling was accepted; problems were {problems}",
        )

    def test_relabelling_breaks_the_digest_too(self):
        document = load_document()
        document["pack_kind"] = "production"
        self.assertNotEqual(compute_digest(document), document["integrity"]["digest"])


class TestPackUsesOpaqueTerms(unittest.TestCase):

    def test_aggregate_term_keys_are_opaque(self):
        for term_id in load_document()["aggregate_terms"]:
            with self.subTest(term=term_id):
                self.assertTrue(is_opaque_term_id(term_id))

    def test_pack_declares_terms(self):
        self.assertGreater(len(load_policy(PACK).declared_aggregate_terms), 0)


class TestCommittedPolicyFilesAreActuallyTracked(unittest.TestCase):
    """
    The .gitignore rules for policy/ are deny-by-default, which is right for an
    artifact directory and has one failure mode: a file that is supposed to be
    committed is silently never added. Nothing surfaces it — the file exists on
    disk, the tools read it, and it is simply absent from every clone.

    That already happened once to schema.json and BOUNDARY_DEBT.txt. This test
    is the check that would have caught it.
    """

    def _tracked(self):
        import subprocess
        out = subprocess.run(["git", "ls-files", "policy/"],
                             cwd=REPO, capture_output=True, text=True).stdout
        return set(out.split())

    def test_files_the_boundary_check_allowlists_are_committed(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_policy_boundary", REPO / "tools" / "check_policy_boundary.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        tracked = self._tracked()
        for path in sorted(checker.ALLOWED_POLICY_FILES):
            with self.subTest(path=path):
                self.assertIn(path, tracked,
                              f"{path} is allowlisted by the boundary check but is not "
                              f"tracked by git; a .gitignore rule is swallowing it")

    def test_demo_pack_is_committed(self):
        self.assertIn("policy/demo.v1.json", self._tracked())

    def test_no_unexpected_policy_file_is_committed(self):
        """The other direction: nothing in policy/ is tracked that should not be."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_policy_boundary", REPO / "tools" / "check_policy_boundary.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        for path in sorted(self._tracked()):
            allowed = (path in checker.ALLOWED_POLICY_FILES
                       or checker.ALLOWED_POLICY_GLOB.match(path))
            with self.subTest(path=path):
                self.assertTrue(allowed, f"{path} is committed but is not an allowed policy file")


class TestGeneratorIsReproducible(unittest.TestCase):

    def test_regenerating_reproduces_the_committed_pack(self):
        """
        The pack is generated, not authored. If regeneration diverges, the
        committed file has been edited by hand and its provenance is a lie.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_demo_policy", REPO / "tools" / "make_demo_policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.build(), load_document())


if __name__ == "__main__":
    unittest.main()
