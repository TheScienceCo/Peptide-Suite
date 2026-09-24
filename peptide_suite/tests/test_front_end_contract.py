"""
The front end, asserted from Python because there is no JS test runner here.

These are contract tests, not rendering tests: a browser check confirms the
page draws, and nothing in a browser check survives into the next change. What
is guarded here is the small set of facts about the interface that were arrived
at by getting them wrong first.
"""

import re
import unittest
from pathlib import Path

STATIC = Path("peptide_suite/static")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


class TestTheTwoStepFlowIsWiredUp(unittest.TestCase):
    """
    Identify answers "what is this". Analyze answers "what should change about
    it". They were one button, so the goal selector appeared beside the
    identification and a reader chose an engineering objective while still
    reading what the molecule was.
    """

    def test_both_buttons_exist(self):
        self.assertIn('id="btn-identify"', INDEX)
        self.assertIn('id="btn-analyze"', INDEX)

    def test_both_buttons_are_wired(self):
        self.assertIn('$("#btn-identify").addEventListener', APP_JS)
        self.assertIn('$("#btn-analyze").addEventListener', APP_JS)

    def test_the_old_single_button_is_gone_everywhere(self):
        """A rename that leaves one reference behind is a dead listener."""
        self.assertNotIn("btn-infer", APP_JS)
        self.assertNotIn("btn-infer", INDEX)

    def test_editing_the_sequence_clears_the_stale_result(self):
        """
        Resetting the buttons while leaving one peptide's receptors on screen
        above another peptide's scan is the confusion the split exists to
        prevent. The first version did exactly that.
        """
        self.assertIn("resetSteps({ clear: true })", APP_JS)
        self.assertRegex(APP_JS, r"function resetSteps\(\{\s*clear = false")

    def test_identify_does_not_render_the_goal_gate(self):
        """
        `renderIdentity` must not open the objective selector. If it does, the
        two steps have silently become one again.
        """
        body = _function_body("renderIdentity")
        self.assertNotIn("renderGoalGate", body)
        self.assertIn("identityCard", body)

    def test_the_goal_gate_refuses_to_stack(self):
        body = _function_body("renderGoalGate")
        self.assertIn('if ($("#goal-select")) return;', body)


class TestTheContextCardCannotContradictTheIdentityCard(unittest.TestCase):
    """
    The context card opened "No established peptide identity was found for this
    sequence" directly above a card reading "matched a known peptide — exact
    sequence match to GLP-1 (7-36) amide". Both were true: the inferencer
    recognised the sequence and the curated context layer, keyed on GLP-1
    (7-37), had no record for it. Printed together they read as the system
    disagreeing with itself, and a reader cannot tell which half to believe.
    """

    def test_the_context_renderer_is_told_what_was_identified(self):
        self.assertRegex(APP_JS, r"function renderBiologicalContext\(bc,\s*identity\)")
        self.assertRegex(APP_JS,
                         r"renderBiologicalContext\(\s*info\.biological_context,")

    def test_the_empty_state_distinguishes_the_two_failures(self):
        # The source is concatenated across lines, so the literals are joined
        # before matching: asserting on the raw text would test where the
        # string breaks rather than what it says.
        body = _joined_literals(_function_body("renderBiologicalContext"))
        self.assertIn("no curated biological record for it was retrieved", body)
        self.assertIn("Identification and biological context are separate lookups", body)
        self.assertIn("No established peptide identity was found", body)

    def test_identity_is_rendered_before_context(self):
        body = _function_body("renderIdentity")
        self.assertLess(body.index("identityCard"),
                        body.index("renderBiologicalContext"),
                        "the context card reporting an empty lookup would sit above the "
                        "card that just identified the molecule")


class TestNoFunctionIsDefinedTwice(unittest.TestCase):
    """
    Splitting `onAnalyze` left two top-level definitions of it. JavaScript
    hoists both and the later one silently wins, so the code ran correctly
    while containing a function nothing could ever call -- which is the kind of
    thing that is fixed once and reintroduced by the next split.
    """

    def test_every_top_level_function_name_is_unique(self):
        names = re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
                           APP_JS, re.M)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(duplicates, [],
                         f"defined more than once; the later definition wins and the "
                         f"earlier one is unreachable: {duplicates}")


def _joined_literals(source: str) -> str:
    """Collapse `"a " + "b"` and `` `a ` + "b" `` so a phrase can be matched whole."""
    return re.sub(r'["`]\s*\+\s*["`]', "", source)


def _function_body(name: str) -> str:
    """
    The source of a top-level function, by brace matching.

    Brace counting rather than a regex because the bodies contain both, and a
    test that reads the wrong span silently asserts about the wrong function.
    """
    match = re.search(rf"^(?:async\s+)?function\s+{re.escape(name)}\s*\(", APP_JS, re.M)
    if not match:
        raise AssertionError(f"no top-level function named {name}")

    # Walk the parameter list to its closing paren before looking for the body
    # brace. Taking the first `{` after the name grabs a DESTRUCTURED PARAMETER
    # instead -- `function resetSteps({ clear = false })` returned the
    # parameter object as the body, so every assertion about that function
    # tested its own signature.
    depth = 0
    for i in range(match.end() - 1, len(APP_JS)):
        if APP_JS[i] == "(":
            depth += 1
        elif APP_JS[i] == ")":
            depth -= 1
            if depth == 0:
                start = APP_JS.index("{", i)
                break
    else:
        raise AssertionError(f"unbalanced parameter list in {name}")
    depth = 0
    for i in range(start, len(APP_JS)):
        if APP_JS[i] == "{":
            depth += 1
        elif APP_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return APP_JS[start:i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


if __name__ == "__main__":
    unittest.main()


class TestIdentificationDoesNotWaitToBeAskedFor(unittest.TestCase):
    """
    A button is the wrong affordance for the first question. Pasting a sequence
    and then pressing "Identify" to be told what you pasted is a step that buys
    nothing: the answer is already determined by the paste.
    """

    def test_a_paste_triggers_identification(self):
        body = _function_body("wireAutoIdentify")
        self.assertIn('addEventListener("paste"', body)

    def test_a_paste_does_not_wait_out_the_typing_delay(self):
        """A paste is complete by definition; there is no more of it coming."""
        body = _function_body("wireAutoIdentify")
        self.assertRegex(body, r'addEventListener\("paste",\s*\(\)\s*=>\s*schedule\(0\)\)')

    def test_typing_is_debounced_rather_than_firing_per_keystroke(self):
        body = _function_body("wireAutoIdentify")
        self.assertIn("AUTO_IDENTIFY_DELAY_MS", body)
        self.assertIn("clearTimeout", _joined_literals(APP_JS))

    def test_a_chip_click_is_picked_up(self):
        """Example chips set the value directly, which fires no input event."""
        self.assertIn('addEventListener("change"', _function_body("wireAutoIdentify"))

    def test_too_short_an_input_does_not_spend_a_request(self):
        body = _function_body("maybeAutoIdentify")
        self.assertIn("AUTO_IDENTIFY_MIN_CHARS", body)

    def test_the_same_input_is_not_identified_twice(self):
        body = _function_body("maybeAutoIdentify")
        self.assertIn("autoIdentifyLast", body)
        self.assertIn("IDENTIFIED.input === input", body)

    def test_clearing_forgets_what_was_auto_identified(self):
        """
        Otherwise clearing the box and pasting the same sequence back does
        nothing at all, because the memo still says it was already answered.
        """
        self.assertIn("autoIdentifyLast = \"\"", _function_body("resetSteps"))

    def test_an_automatic_run_does_not_scold_an_empty_box(self):
        """
        The empty-input notice is for someone who pressed the button. Firing it
        at a user who is still filling the box in is worse than staying quiet.
        """
        body = _function_body("runIdentify")
        self.assertIn("if (!auto)", body)

    def test_the_button_survives_as_the_way_to_insist(self):
        """
        Auto-identification is deliberately conservative, so there has to be a
        way to ask for it anyway.
        """
        self.assertIn('id="btn-identify"', INDEX)
        self.assertIn('$("#btn-identify").addEventListener', APP_JS)


class TestTheBrowserCannotServeAStaleInterface(unittest.TestCase):
    """
    `index.html` gains a button, the browser reuses a cached `app.js` that has
    no handler for it, and the page is broken in a way reloading does not
    reliably fix. The markup and the script have to version together.
    """

    def test_the_assets_are_version_stamped(self):
        api = Path("peptide_suite/api.py").read_text(encoding="utf-8")
        self.assertIn("/static/app.js?v=", api)
        self.assertIn("/static/styles.css?v=", api)

    def test_the_version_is_a_content_hash_not_a_timestamp(self):
        """
        A checkout, a rebuild or a rebase changes mtimes without changing a
        byte, and a version that churns defeats caching without buying
        correctness.
        """
        api = Path("peptide_suite/api.py").read_text(encoding="utf-8")
        self.assertIn("hashlib.sha256", api)
        self.assertNotIn("st_mtime", api)

    def test_the_page_itself_is_not_cached(self):
        api = Path("peptide_suite/api.py").read_text(encoding="utf-8")
        self.assertIn('"Cache-Control": "no-cache"', api)
