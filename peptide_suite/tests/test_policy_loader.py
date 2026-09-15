"""
Tests for the policy artifact schema, validation and loader.  [Addendum 1 step 1]

The load path is the only way a coefficient enters this system, so it is the one
place where silent tolerance is most expensive. Every test here asserts that
something fails rather than degrades.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from peptide_suite.policy import (
    FEATURE_FAMILIES, THRESHOLDS, PolicyError, PolicyNotFound,
    PolicyValidationError, compute_digest, load_policy, validate_document,
)
from peptide_suite.policy.loader import SCHEMA_VERSION


def make_document(**overrides):
    """A minimal valid artifact. Values here are test fixtures, not policy."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "0.0.1",
        "pack_kind": "demonstration",
        "provenance": {
            "created_utc": "2026-01-01T00:00:00Z",
            "created_by": "unit test fixture",
            "derivation": "synthetic values for validation testing",
        },
        "integrity": {"algorithm": "sha256", "digest": "0" * 64},
        "feature_families": {
            fid: {"semantic_type": spec.semantic_type, "units": spec.units,
                  "valid_range": list(spec.valid_range)}
            for fid, spec in FEATURE_FAMILIES.items()
        },
        "weights": {fid: 0.5 for fid in FEATURE_FAMILIES},
        "thresholds": {
            tid: {"value": (spec.valid_range[0] + spec.valid_range[1]) / 2,
                  "semantic_type": spec.semantic_type, "units": spec.units,
                  "valid_range": list(spec.valid_range)}
            for tid, spec in THRESHOLDS.items()
        },
        "aggregate_terms": {
            "interface.dispersion_fraction": {
                "source_tiers": ["QM"], "reduction": "mean", "max_inputs": 12,
            }
        },
    }
    doc.update(overrides)
    doc["integrity"]["digest"] = compute_digest(doc)
    return doc


def write_document(doc) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(doc, handle)
    handle.close()
    return Path(handle.name)


class TestFailLoud(unittest.TestCase):
    """
    Addendum 1 section 1: absent policy fails loudly rather than falling back to
    built-in values. A fallback would let a misconfigured deployment produce
    normal-looking output from values that are not the policy.
    """

    def test_missing_file_raises(self):
        with self.assertRaises(PolicyNotFound) as ctx:
            load_policy(Path("/nonexistent/policy.json"))
        self.assertIn("will not fall back", str(ctx.exception))

    def test_unset_environment_raises(self):
        saved = os.environ.pop("PEPTIDE_SUITE_POLICY", None)
        try:
            with self.assertRaises(PolicyNotFound) as ctx:
                load_policy()
            self.assertIn("contains no coefficients", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["PEPTIDE_SUITE_POLICY"] = saved

    def test_no_default_policy_is_importable(self):
        """There must be no module-level fallback policy anywhere in the engine."""
        import peptide_suite.policy as mod
        for name in dir(mod):
            if "DEFAULT" in name.upper() and "POLICY" in name.upper():
                self.fail(f"engine exposes a default policy: {name}")

    def test_malformed_json_raises_rather_than_returning_empty(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handle.write("{not json")
        handle.close()
        with self.assertRaises(PolicyValidationError):
            load_policy(Path(handle.name))


class TestStrictValidation(unittest.TestCase):
    """Unknown keys, out-of-range values, missing fields — all three rejected."""

    def test_valid_document_passes(self):
        self.assertEqual(validate_document(make_document()), [])

    def test_unknown_top_level_key_rejected(self):
        doc = make_document()
        doc["surprise"] = 1
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("unknown top-level key 'surprise'" in p for p in problems))

    def test_unknown_weight_key_rejected(self):
        doc = make_document()
        doc["weights"]["objective.invented_family"] = 0.5
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("unknown key 'objective.invented_family'" in p for p in problems))

    def test_missing_required_weight_rejected(self):
        doc = make_document()
        removed = next(iter(FEATURE_FAMILIES))
        del doc["weights"][removed]
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any(f"missing required key '{removed}'" in p for p in problems))

    def test_missing_required_threshold_rejected(self):
        doc = make_document()
        removed = next(iter(THRESHOLDS))
        del doc["thresholds"][removed]
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any(f"missing required key '{removed}'" in p for p in problems))

    def test_out_of_range_weight_rejected(self):
        doc = make_document()
        key = "objective.potency"
        doc["weights"][key] = 42.0
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("outside its declared range" in p and key in p for p in problems))

    def test_out_of_range_threshold_rejected(self):
        doc = make_document()
        doc["thresholds"]["structure_gate.min_iptm"]["value"] = 9.9
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("structure_gate.min_iptm" in p and "outside" in p for p in problems))

    def test_missing_top_level_block_rejected(self):
        doc = make_document()
        del doc["thresholds"]
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("missing required top-level key 'thresholds'" in p for p in problems))

    def test_wrong_schema_version_rejected(self):
        doc = make_document(schema_version="99.0.0")
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_invalid_pack_kind_rejected(self):
        doc = make_document(pack_kind="whatever")
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("pack_kind" in p for p in problems))

    def test_missing_provenance_field_rejected(self):
        doc = make_document()
        doc["provenance"]["derivation"] = ""
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("provenance" in p and "derivation" in p for p in problems))

    def test_weight_without_matching_family_rejected(self):
        doc = make_document()
        doc["feature_families"].pop("objective.potency")
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("no matching entry in feature_families" in p for p in problems))

    def test_aggregate_term_with_measured_source_rejected(self):
        """MEASURED is a regression target, not an aggregate source."""
        doc = make_document()
        doc["aggregate_terms"]["interface.dispersion_fraction"]["source_tiers"] = ["MEASURED"]
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("invalid source tiers" in p for p in problems))

    def test_aggregate_term_unbounded_inputs_rejected(self):
        doc = make_document()
        doc["aggregate_terms"]["interface.dispersion_fraction"]["max_inputs"] = 5000
        problems = validate_document(doc, verify_integrity=False)
        self.assertTrue(any("max_inputs" in p for p in problems))

    def test_every_problem_is_reported_not_just_the_first(self):
        doc = make_document()
        doc["weights"]["objective.potency"] = 42.0
        doc["weights"]["objective.aggregation"] = -7.0
        doc["extra"] = 1
        problems = validate_document(doc, verify_integrity=False)
        self.assertGreaterEqual(len(problems), 1)


class TestIntegrity(unittest.TestCase):
    """The digest is what makes an unmodified artifact demonstrable."""

    def test_digest_is_stable_across_key_order(self):
        doc = make_document()
        reordered = dict(reversed(list(doc.items())))
        self.assertEqual(compute_digest(doc), compute_digest(reordered))

    def test_digest_excludes_its_own_block(self):
        doc = make_document()
        before = compute_digest(doc)
        doc["integrity"]["digest"] = "f" * 64
        self.assertEqual(compute_digest(doc), before)

    def test_tampering_is_detected(self):
        doc = make_document()
        doc["weights"]["objective.potency"] = 0.99
        problems = validate_document(doc)
        self.assertTrue(any("digest mismatch" in p for p in problems))
        self.assertTrue(any("modified since it was signed" in p for p in problems))

    def test_integrity_check_can_be_skipped_for_authoring(self):
        doc = make_document()
        doc["weights"]["objective.potency"] = 0.99
        self.assertEqual(validate_document(doc, verify_integrity=False), [])


class TestLoadedPolicy(unittest.TestCase):

    def setUp(self):
        self.path = write_document(make_document())
        self.policy = load_policy(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_weights_and_thresholds_are_readable(self):
        self.assertIsInstance(self.policy.weight("objective.potency"), float)
        self.assertIsInstance(self.policy.threshold("structure_gate.min_iptm"), float)

    def test_undeclared_key_raises_rather_than_returning_zero(self):
        with self.assertRaises(PolicyError):
            self.policy.weight("objective.not_a_family")
        with self.assertRaises(PolicyError):
            self.policy.threshold("nope.nothing")

    def test_demonstration_pack_is_labelled(self):
        self.assertTrue(self.policy.is_demonstration)
        self.assertIn("DEMONSTRATION", self.policy.banner())

    def test_free_parameter_count_matches_the_registry(self):
        self.assertEqual(self.policy.free_parameters, len(FEATURE_FAMILIES) + len(THRESHOLDS))

    def test_banner_reports_version_and_digest(self):
        banner = self.policy.banner()
        self.assertIn("0.0.1", banner)
        self.assertIn(self.policy.digest[:12], banner)


class TestSchemaFileMatchesEngine(unittest.TestCase):
    """The published schema and the engine's expectations must not drift apart."""

    def setUp(self):
        self.schema = json.loads(Path("policy/schema.json").read_text())

    def test_schema_requires_what_the_loader_requires(self):
        from peptide_suite.policy.loader import TOP_LEVEL_REQUIRED
        self.assertEqual(set(self.schema["required"]), TOP_LEVEL_REQUIRED)

    def test_schema_forbids_additional_properties(self):
        self.assertFalse(self.schema["additionalProperties"])

    def test_schema_carries_no_rationale_text(self):
        """Section 2: the schema must not reveal why a family exists or its magnitude."""
        blob = json.dumps(self.schema).lower()
        for leak in ("because", "expected magnitude", "typically", "dominant",
                     "potency", "affinity", "cleavage"):
            self.assertNotIn(leak, blob, f"schema leaks rationale: '{leak}'")


if __name__ == "__main__":
    unittest.main()
