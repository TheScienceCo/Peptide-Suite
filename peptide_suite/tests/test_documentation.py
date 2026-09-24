"""
The documentation's figures and links.  [Addendum 3, section 12]

A broken image in a README is invisible to the person who wrote it and the
first thing a reader sees. These are cheap checks against the two ways the
figures rot: a file renamed without its reference, and a file committed but
never pointed at by anything.
"""

import re
import unittest
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

# Markdown link and image, which differ only by the leading bang. Both count as
# a reference: a document is referenced by being linked, a figure by being
# shown, and neither should be judged by the other's rule.
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def markdown_files():
    return sorted(list(ROOT.glob("*.md")) + list(DOCS.rglob("*.md")))


def references():
    """Every local target any of this repository's markdown points at."""
    found = set()
    for source in markdown_files():
        for target in LINK.findall(source.read_text()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            # Relative to the file doing the pointing, not to the repository
            # root: docs/DELIVERABLES.md says architecture.svg and means its
            # own neighbour.
            found.add((source.parent / target.split("#")[0]).resolve())
    return found


class TestDocumentationLinks(unittest.TestCase):

    def test_every_referenced_file_exists(self):
        missing = sorted(str(path) for path in references() if not path.exists())
        self.assertEqual(missing, [], f"markdown points at files that are not here: {missing}")

    def test_every_committed_document_is_pointed_at(self):
        if not DOCS.exists():
            self.skipTest("no docs directory")
        referenced = references()
        orphans = sorted(
            str(path.relative_to(ROOT))
            for path in DOCS.rglob("*")
            if path.is_file() and path.resolve() not in referenced
        )
        self.assertEqual(orphans, [],
                         f"committed but pointed at by nothing, so nobody will notice when "
                         f"they go stale: {orphans}")

    def test_the_architecture_diagram_is_well_formed(self):
        path = DOCS / "architecture.svg"
        self.assertTrue(path.exists())
        xml.dom.minidom.parse(str(path))     # raises on malformed markup

    def test_the_diagram_selects_its_dark_mode(self):
        # Rendered as an <img> on a dark README, a diagram with no dark scope
        # is a white rectangle. The media query is how it follows the reader.
        text = (DOCS / "architecture.svg").read_text()
        self.assertIn("prefers-color-scheme: dark", text)
        self.assertIn('role="img"', text)
        self.assertIn("aria-label", text)


if __name__ == "__main__":
    unittest.main()


class TestTheSuiteNeedsNoThirdPartyPackages(unittest.TestCase):
    """
    CI runs the whole suite with nothing installed, on purpose, and this guard
    is what tells a developer about it before the runner does.

    The failure mode is quiet and has happened twice: a test reaches for
    `peptide_suite.api` to read a label or a helper, that module imports
    FastAPI at its top, and the test passes on a machine that has FastAPI and
    fails in the `boundary` job that deliberately has none. The remedy both
    times was the same -- the thing being read was engine data, so it moved
    into `peptide_suite/core/` and the HTTP layer kept only the route.

    Stated as a rule the suite enforces on itself rather than as a comment in
    the workflow file, because the workflow file is not what someone reads
    while writing a test.
    """

    #: Modules that import a third-party package at module scope. Add to this
    #: only if the dependency is genuinely unavoidable -- and then the test
    #: importing it needs a skip guard, not an entry here.
    NEEDS_INSTALL = {
        "peptide_suite.api": "fastapi",
    }

    def test_no_test_imports_a_module_that_needs_an_install(self):
        offenders = []
        for path in sorted(Path("peptide_suite/tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            for module, package in self.NEEDS_INSTALL.items():
                pattern = rf"(?m)^\s*(?:from {re.escape(module)} import|import {re.escape(module)})"
                for match in re.finditer(pattern, text):
                    line = text[:match.start()].count("\n") + 1
                    offenders.append(f"{path}:{line} imports {module} (needs {package})")
        self.assertEqual(
            offenders, [],
            "These imports pass locally and fail in the boundary CI job, which installs "
            "nothing. Move what is being read into peptide_suite/core/ and import it "
            f"from there: {offenders}")

    def test_the_boundary_job_still_installs_nothing(self):
        """
        The guard above is only worth having while the job it protects stays
        bare. If someone adds a `pip install` to the boundary job, the
        hermetic-suite guarantee is gone and this should say so loudly rather
        than the rule quietly becoming decorative.
        """
        workflow = Path(".github/workflows/policy-boundary.yml").read_text(encoding="utf-8")
        boundary = workflow.split("  ml:")[0]
        # Comments stripped first: the job's own comment says "Deliberately no
        # `pip install`", and matching that would make the guard fail on the
        # sentence explaining why it should pass.
        steps = "\n".join(line for line in boundary.splitlines()
                          if not line.lstrip().startswith("#"))
        self.assertNotIn(
            "pip install", steps,
            "The boundary job installs a package now. That job existing without one is "
            "what proves the analysis pipeline is standard-library-only and the suite is "
            "hermetic; if the install is genuinely needed, this guarantee needs replacing "
            "rather than deleting.")
