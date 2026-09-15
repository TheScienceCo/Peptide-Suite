"""
Tests for the peptide optimization pipeline.

The engine holds no coefficients, so the suite loads the demonstration pack
before anything imports. This is not scaffolding to work around the boundary —
it is the boundary being exercised: if the demo pack stops supplying a key the
engine needs, the suite fails at import rather than at some later assertion.

test_policy_boundary.py deliberately clears the policy to check that the engine
refuses to run without one, and restores it afterwards.
"""

from pathlib import Path

from peptide_suite.runtime import load_active_policy

DEMO_PACK = Path(__file__).resolve().parents[2] / "policy" / "demo.v1.json"

load_active_policy(DEMO_PACK)
