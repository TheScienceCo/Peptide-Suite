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
