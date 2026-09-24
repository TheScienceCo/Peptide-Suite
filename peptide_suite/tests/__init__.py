"""
Tests for the peptide optimization pipeline.

The engine holds no coefficients, so the suite loads the demonstration pack
before anything imports. This is not scaffolding to work around the boundary —
it is the boundary being exercised: if the demo pack stops supplying a key the
engine needs, the suite fails at import rather than at some later assertion.

test_policy_boundary.py deliberately clears the policy to check that the engine
refuses to run without one, and restores it afterwards.
"""

import os
from pathlib import Path

from peptide_suite.core.uniprot_client import OFFLINE_ENV_VAR
from peptide_suite.runtime import load_active_policy

DEMO_PACK = Path(__file__).resolve().parents[2] / "policy" / "demo.v1.json"

# The suite is hermetic: no test reaches a live host, so no assertion depends on
# whether one happens to be reachable. Three tests were doing exactly that --
# they took the reference-set path in a sandbox with no route to uniprot.org and
# the live path in CI, which has one, and passed or failed accordingly. The
# difference was invisible from either side, which is the worst property a test
# can have. Set before the policy loads, so it is in place before any module
# builds a client at import time.
os.environ.setdefault(OFFLINE_ENV_VAR, "1")

load_active_policy(DEMO_PACK)
