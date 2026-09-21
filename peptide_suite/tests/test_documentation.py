"""
The README's figures.  [Addendum 3, section 12]

A broken image in a README is invisible to the person who wrote it and the
first thing a reader sees. These are cheap checks against the two ways the
figures rot: a file renamed without its reference, and a file committed but
never referenced.
"""

import re
import unittest
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def referenced():
    return {m for m in IMAGE.findall(README.read_text()) if not m.startswith("http")}


class TestReadmeFigures(unittest.TestCase):

    def test_every_referenced_figure_exists(self):
        missing = [ref for ref in referenced() if not (ROOT / ref).exists()]
        self.assertEqual(missing, [], f"README points at files that are not here: {missing}")

    def test_every_committed_figure_is_referenced(self):
        if not DOCS.exists():
            self.skipTest("no docs directory")
        refs = {str(Path(ref)) for ref in referenced()}
        orphans = [
            str(path.relative_to(ROOT))
            for path in DOCS.rglob("*")
            if path.is_file() and str(path.relative_to(ROOT)) not in refs
        ]
        self.assertEqual(orphans, [],
                         f"committed but shown nowhere, so nobody will notice when they go "
                         f"stale: {orphans}")

    def test_the_architecture_diagram_is_well_formed(self):
        path = ROOT / "docs" / "architecture.svg"
        self.assertTrue(path.exists())
        xml.dom.minidom.parse(str(path))     # raises on malformed markup

    def test_the_diagram_selects_its_dark_mode(self):
        # Rendered as an <img> on a dark README, a diagram with no dark scope
        # is a white rectangle. The media query is how it follows the reader.
        text = (ROOT / "docs" / "architecture.svg").read_text()
        self.assertIn("prefers-color-scheme: dark", text)
        self.assertIn("role=\"img\"", text)
        self.assertIn("aria-label", text)


if __name__ == "__main__":
    unittest.main()
