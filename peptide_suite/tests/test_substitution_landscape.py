"""
The substitution landscape grid.  [Addendum 3 section 9]

The grid is where the project's organising rule is easiest to break. A heatmap
wants a number in every cell; the pipeline does not always have one. The tests
that matter here are the ones that check a missing value stays missing all the
way to the wire -- not as zero, not as the midpoint of a scale, not as a key
the browser can read with `?? 0`.
"""

import json
import re
import unittest
from pathlib import Path

from peptide_suite.core import PeptideContext
from peptide_suite.core.substitution_landscape import (
    AA_ROWS,
    DEFAULT_METRIC,
    MAX_LANDSCAPE_LENGTH,
    METRICS,
    METRICS_BY_KEY,
    CellState,
    LandscapeError,
    _EXTRACTORS,
    build_landscape,
)
from peptide_suite.core.substitution_predictor import TERM_CONSERVATION
from peptide_suite.workflows.optimize import OptimizeWorkflow

SEQ = "YGGFMTSEKSQ"
GOAL = "protease_resistance"
HOMOLOGS = ["YGGFMTSEKSA", "YGGFLTSEKSQ", "YGGFMTSDKSQ"]

_scans = {}


def scan(homologs=None):
    """One scan per distinct input, reused across tests -- 11 x 19 predictions."""
    key = tuple(homologs or ())
    if key not in _scans:
        _scans[key] = OptimizeWorkflow().scan(
            SEQ, confirmed_goal=GOAL, ph=7.4, homologs=list(homologs) if homologs else None
        )
    ctx, recs = _scans[key]
    return ctx, recs


class TestGridShape(unittest.TestCase):

    def test_every_position_meets_every_residue(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs)
        self.assertEqual(len(ls.cells), len(SEQ) * len(AA_ROWS))

    def test_rows_are_the_twenty_canonical_residues_once_each(self):
        self.assertEqual(len(AA_ROWS), 20)
        self.assertEqual(len(set(AA_ROWS)), 20)
        self.assertEqual(set(AA_ROWS), set("ACDEFGHIKLMNPQRSTVWY"))

    def test_the_wild_type_residue_is_not_a_substitution(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs)
        wt = [c for c in ls.cells if c.state is CellState.WILD_TYPE]
        self.assertEqual(len(wt), len(SEQ))
        for cell in wt:
            self.assertEqual(cell.mutant_aa, SEQ[cell.position])
            self.assertIsNone(cell.value,
                              "a wild-type cell holds no value: it is not a substitution with "
                              "no effect, it is not a substitution")

    def test_the_scan_is_the_whole_grid_not_the_top_five(self):
        ctx, recs = scan()
        self.assertEqual(len(recs), len(SEQ) * 19)
        _, ranked = OptimizeWorkflow().run(SEQ, confirmed_goal=GOAL, ph=7.4)
        self.assertLessEqual(len(ranked), 5)


class TestValuesComeFromTheScan(unittest.TestCase):
    """The grid re-projects a computed scan; it does not recompute or invent."""

    def test_net_score_cells_equal_the_recommendation_they_came_from(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "net_score")
        by_cell = {(r.position, r.mutant_aa): r for r in recs}
        checked = 0
        for cell in ls.cells:
            if cell.state is not CellState.COMPUTED:
                continue
            self.assertEqual(cell.value, by_cell[(cell.position, cell.mutant_aa)].net_score)
            checked += 1
        self.assertEqual(checked, len(SEQ) * 19)

    def test_every_declared_metric_has_an_extractor(self):
        self.assertEqual(set(METRICS_BY_KEY), set(_EXTRACTORS))
        self.assertIn(DEFAULT_METRIC, METRICS_BY_KEY)

    def test_every_metric_builds_without_the_scan_being_rerun(self):
        ctx, recs = scan()
        for metric in METRICS:
            ls = build_landscape(ctx, recs, metric.key)
            self.assertEqual(len(ls.cells), len(SEQ) * len(AA_ROWS), metric.key)


class TestNotComputedIsNotZero(unittest.TestCase):
    """The rule the whole project turns on, at the one place a chart erases it."""

    def test_conservation_without_homologs_computes_nothing_at_all(self):
        ctx, recs = scan()
        self.assertFalse(ctx.conservation_available)
        ls = build_landscape(ctx, recs, "conservation_cost")
        self.assertEqual(ls.n_computed, 0)
        self.assertEqual(ls.n_not_computed, len(SEQ) * 19)
        self.assertIsNone(ls.scale_bound,
                          "with nothing computed there is no range, so there is no scale")

    def test_an_uncomputed_cell_carries_no_number_anywhere(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "conservation_cost")
        for cell in ls.cells:
            if cell.state is CellState.NOT_COMPUTED:
                self.assertIsNone(cell.value)
                self.assertTrue(cell.detail.strip(), "an absent value owes a reason")

    def test_the_reason_travels_with_the_grid(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "conservation_cost")
        self.assertIn("homolog", ls.not_computed_reason.lower())

    def test_conservation_computes_once_there_are_enough_homologs(self):
        ctx, recs = scan(HOMOLOGS)
        self.assertTrue(ctx.conservation_available)
        ls = build_landscape(ctx, recs, "conservation_cost")
        self.assertEqual(ls.n_not_computed, 0)
        self.assertEqual(ls.n_computed, len(SEQ) * 19)
        self.assertIsNotNone(ls.scale_bound)

    def test_the_conservation_term_is_found_by_key_not_by_its_wording(self):
        _, recs = scan()
        term = [e for e in recs[0].off_target_effects if e.term_key == TERM_CONSERVATION]
        self.assertEqual(len(term), 1)
        self.assertFalse(term[0].computed)
        self.assertEqual(term[0].magnitude, 0.0,
                         "an uncomputed term contributes nothing to the net score")

    def test_the_breakdown_marks_which_terms_were_computed(self):
        _, recs = scan()
        terms = recs[0].score_breakdown["terms"]
        conservation = [t for t in terms if t["term_key"] == TERM_CONSERVATION]
        self.assertEqual(len(conservation), 1)
        self.assertFalse(conservation[0]["computed"])


class TestScale(unittest.TestCase):

    def test_a_diverging_scale_is_symmetric_about_zero(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "net_score")
        values = [c.value for c in ls.cells if c.state is CellState.COMPUTED]
        self.assertAlmostEqual(ls.scale_bound, max(abs(v) for v in values))

    def test_a_one_signed_quantity_does_not_get_a_diverging_scale(self):
        for metric in METRICS:
            if metric.encoding == "sequential":
                self.assertEqual(metric.midpoint_meaning, "",
                                 f"{metric.key} declares a midpoint meaning but has no midpoint")
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "off_target_cost")
        values = [c.value for c in ls.cells if c.state is CellState.COMPUTED]
        self.assertTrue(all(v >= 0 for v in values))
        self.assertAlmostEqual(ls.scale_bound, max(values))

    def test_the_scale_says_it_is_derived_from_this_grid(self):
        ctx, recs = scan()
        ls = build_landscape(ctx, recs, "net_score")
        self.assertIn("not a fixed range", ls.scale_basis)


class TestRefusals(unittest.TestCase):

    def test_an_unknown_metric_is_refused_by_name(self):
        ctx, recs = scan()
        with self.assertRaises(LandscapeError) as cm:
            build_landscape(ctx, recs, "vibes")
        self.assertIn("vibes", str(cm.exception))

    def test_a_peptide_too_long_to_render_is_refused_not_truncated(self):
        long_ctx = PeptideContext(sequence="A" * (MAX_LANDSCAPE_LENGTH + 1), name="long")
        long_ctx.confirmed_goal = GOAL
        with self.assertRaises(LandscapeError) as cm:
            build_landscape(long_ctx, [])
        self.assertIn("readability limit", str(cm.exception))

    def test_no_sequence_is_refused(self):
        with self.assertRaises(LandscapeError):
            build_landscape(PeptideContext(sequence="", name="x"), [])


class TestTheWireFormat(unittest.TestCase):
    """
    A missing value must not survive serialisation as a key the client can
    coerce. `cell.value ?? 0` is one keystroke away in any consumer, so the key
    is absent rather than null.
    """

    def test_an_uncomputed_cell_has_no_value_key_on_the_wire(self):
        from peptide_suite.api import encode_landscape
        ctx, recs = scan()
        payload = encode_landscape(build_landscape(ctx, recs, "conservation_cost"))
        blob = json.dumps(payload)
        self.assertNotIn('"value"', blob)
        for cell in payload["cells"]:
            self.assertNotIn("value", cell)
            self.assertIn(cell["state"], ("NOT_COMPUTED", "WILD_TYPE"))

    def test_a_computed_cell_does_carry_its_value(self):
        from peptide_suite.api import encode_landscape
        ctx, recs = scan()
        payload = encode_landscape(build_landscape(ctx, recs, "net_score"))
        computed = [c for c in payload["cells"] if c["state"] == "COMPUTED"]
        self.assertEqual(len(computed), len(SEQ) * 19)
        self.assertTrue(all("value" in c for c in computed))


class TestTheFrontendReferencesTokensThatExist(unittest.TestCase):
    """
    The renderer names colour tokens as strings. A typo in one would silently
    paint a cell with nothing -- which, on this chart, is indistinguishable
    from a cell that was never computed. That is exactly the confusion the
    whole module exists to prevent, so the token names are checked.
    """

    STATIC = Path(__file__).resolve().parents[1] / "static"

    def test_every_ramp_token_the_renderer_uses_is_defined_in_both_modes(self):
        app = (self.STATIC / "app.js").read_text()
        css = (self.STATIC / "styles.css").read_text()

        used = set(re.findall(r'(--(?:div|seq)-[a-z0-9-]+)', app))
        self.assertTrue(used, "expected the renderer to name ramp tokens")
        self.assertIn("--div-mid", used)

        # Light scope is :root; dark is declared twice, under the media query
        # and under the explicit theme stamp, and must match.
        dark_blocks = re.findall(
            r'(?:@media \(prefers-color-scheme: dark\)|:root\[data-theme="dark"\])(.*?)\n\}',
            css, re.S)
        self.assertEqual(len(dark_blocks), 2, "expected both dark scopes")

        for token in sorted(used):
            self.assertIn(f"{token}:", css, f"{token} is used by the renderer but never defined")
            for i, block in enumerate(dark_blocks):
                self.assertIn(f"{token}:", block,
                              f"{token} has no dark-mode value in scope {i}: dark mode would "
                              f"inherit the light step rather than being selected for its surface")

    def test_the_two_dark_scopes_agree(self):
        css = (self.STATIC / "styles.css").read_text()
        blocks = re.findall(
            r'(?:@media \(prefers-color-scheme: dark\)|:root\[data-theme="dark"\])(.*?)\n\}',
            css, re.S)
        decls = [dict(re.findall(r'(--[a-z0-9-]+):\s*([^;]+);', b)) for b in blocks]
        shared = set(decls[0]) & set(decls[1])
        self.assertTrue(shared)
        for key in sorted(shared):
            self.assertEqual(decls[0][key].strip(), decls[1][key].strip(),
                             f"{key} disagrees between the OS dark scope and the theme stamp")


if __name__ == "__main__":
    unittest.main()
