"""
Tests for opaque aggregate-term identifiers.  [Addendum 1 section 2]

The property under test is not "the function returns a string". It is that a
reader holding the public artifact cannot recover the term label, and that a
deployment cannot accidentally slip back into legible names or a shared salt.
"""

import os
import tempfile
import unittest
from pathlib import Path

from peptide_suite.policy.opaque import (
    OpaqueIdError, TermLegend, is_opaque_term_id, load_salt, mint_term_id,
    require_opaque,
)

SALT_A = b"salt-a"
SALT_B = b"salt-b"


class TestMinting(unittest.TestCase):

    def test_deterministic_under_fixed_salt(self):
        """The owner must regenerate the same id without storing a lookup table."""
        self.assertEqual(
            mint_term_id("buried polar desolvation", SALT_A),
            mint_term_id("buried polar desolvation", SALT_A),
        )

    def test_label_is_normalised(self):
        """Case and surrounding whitespace are not part of the term's identity."""
        self.assertEqual(
            mint_term_id("Buried Polar Desolvation", SALT_A),
            mint_term_id("  buried polar desolvation  ", SALT_A),
        )

    def test_different_salts_give_different_ids(self):
        """
        This is the property that makes the id one-way. Without it the id is a
        plain hash and a reader recovers the label by hashing a wordlist of
        plausible chemistry terms.
        """
        self.assertNotEqual(
            mint_term_id("buried polar desolvation", SALT_A),
            mint_term_id("buried polar desolvation", SALT_B),
        )

    def test_distinct_labels_give_distinct_ids(self):
        labels = [
            "charge transfer at contact", "buried polar desolvation",
            "interface dispersion fraction", "backbone preorganization",
        ]
        ids = {mint_term_id(label, SALT_A) for label in labels}
        self.assertEqual(len(ids), len(labels))

    def test_minted_ids_match_the_declared_shape(self):
        term_id = mint_term_id("charge transfer at contact", SALT_A)
        self.assertTrue(is_opaque_term_id(term_id))
        self.assertEqual(require_opaque(term_id), term_id)

    def test_id_does_not_contain_the_label(self):
        """A weak but direct check that nothing legible survives minting."""
        term_id = mint_term_id("dispersion", SALT_A)
        self.assertNotIn("dispersion", term_id)

    def test_empty_label_refused(self):
        with self.assertRaises(OpaqueIdError):
            mint_term_id("   ", SALT_A)

    def test_empty_salt_refused(self):
        """An empty salt reduces the HMAC to a dictionary-attackable hash."""
        with self.assertRaises(OpaqueIdError) as ctx:
            mint_term_id("buried polar desolvation", b"")
        self.assertIn("dictionary attack", str(ctx.exception))


class TestShapeChecking(unittest.TestCase):

    def test_legible_name_refused(self):
        with self.assertRaises(OpaqueIdError) as ctx:
            require_opaque("interface.dispersion_fraction")
        self.assertIn("not an opaque aggregate-term identifier", str(ctx.exception))

    def test_near_misses_refused(self):
        for bad in ("agg.0123456789", "agg.0123456789abc", "agg.0123456789AB",
                    "aggregate.0123456789ab", "agg.0123456789zz", "agg.", "", "0123456789ab"):
            with self.subTest(candidate=bad):
                self.assertFalse(is_opaque_term_id(bad))

    def test_trailing_content_refused(self):
        """Anchoring matters: a valid prefix with a suffix must not pass."""
        self.assertFalse(is_opaque_term_id("agg.0123456789ab.potency"))
        self.assertFalse(is_opaque_term_id("agg.0123456789ab\n"))


class TestSaltLoading(unittest.TestCase):

    def test_absent_salt_is_fatal_not_defaulted(self):
        """
        A built-in default salt would be identical across every deployment and
        therefore public, which is the same as having no salt at all.
        """
        saved = os.environ.pop("PEPTIDE_SUITE_TERM_SALT", None)
        try:
            with self.assertRaises(OpaqueIdError) as ctx:
                load_salt()
            self.assertIn("no default", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["PEPTIDE_SUITE_TERM_SALT"] = saved

    def test_empty_salt_in_environment_is_fatal(self):
        saved = os.environ.get("PEPTIDE_SUITE_TERM_SALT")
        os.environ["PEPTIDE_SUITE_TERM_SALT"] = ""
        try:
            with self.assertRaises(OpaqueIdError):
                load_salt()
        finally:
            if saved is None:
                os.environ.pop("PEPTIDE_SUITE_TERM_SALT", None)
            else:
                os.environ["PEPTIDE_SUITE_TERM_SALT"] = saved

    def test_set_salt_is_returned(self):
        saved = os.environ.get("PEPTIDE_SUITE_TERM_SALT")
        os.environ["PEPTIDE_SUITE_TERM_SALT"] = "a-private-salt"
        try:
            self.assertEqual(load_salt(), b"a-private-salt")
        finally:
            if saved is None:
                os.environ.pop("PEPTIDE_SUITE_TERM_SALT", None)
            else:
                os.environ["PEPTIDE_SUITE_TERM_SALT"] = saved


class TestLegend(unittest.TestCase):

    def test_absent_legend_degrades_to_showing_the_id(self):
        """
        A debug surface without the legend must still render. Raising here would
        make the presence of the legend a runtime requirement, which would push
        someone to commit it.
        """
        legend = TermLegend()
        self.assertEqual(legend.describe("agg.0123456789ab"), "agg.0123456789ab")
        self.assertIsNone(legend.label_for("agg.0123456789ab"))

    def test_present_legend_resolves_the_label(self):
        legend = TermLegend.from_labels(["buried polar desolvation"], SALT_A)
        term_id = mint_term_id("buried polar desolvation", SALT_A)
        self.assertEqual(legend.describe(term_id), "buried polar desolvation")

    def test_legend_from_wrong_salt_does_not_resolve(self):
        legend = TermLegend.from_labels(["buried polar desolvation"], SALT_A)
        foreign = mint_term_id("buried polar desolvation", SALT_B)
        self.assertEqual(legend.describe(foreign), foreign)

    def test_missing_legend_file_loads_empty_rather_than_raising(self):
        legend = TermLegend.load(Path("/nonexistent/legend.json"))
        self.assertEqual(len(legend), 0)

    def test_legend_round_trips_through_a_file(self):
        labels = ["charge transfer at contact", "backbone preorganization"]
        legend = TermLegend.from_labels(labels, SALT_A)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legend.json"
            legend.save(path)
            reloaded = TermLegend.load(path)
        self.assertEqual(len(reloaded), 2)
        for label in labels:
            self.assertEqual(reloaded.describe(mint_term_id(label, SALT_A)), label)


class TestLegendIsNeverCommitted(unittest.TestCase):
    """
    The legend undoes the opacity entirely, so .gitignore has to block it by
    name pattern. This test fails if someone removes those rules.
    """

    def test_gitignore_blocks_legend_and_salt(self):
        import subprocess
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            "term_legend.json", "policy/term_legend.json", "my.legend.json",
            "policy/legend.json", ".term_salt", "minting.salt",
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", *candidates],
            cwd=repo_root, capture_output=True, text=True,
        )
        ignored = set(result.stdout.split())
        for candidate in candidates:
            with self.subTest(path=candidate):
                self.assertIn(candidate, ignored,
                              f"{candidate} is not blocked by .gitignore")


if __name__ == "__main__":
    unittest.main()
