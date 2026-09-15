"""
The active policy.  [Addendum 1 section 1]

A fail-loud loader that nothing calls is not enforcement, it is documentation.
This module is where the engine actually reaches for a coefficient, so it is
where the absence of a policy becomes an error rather than a silent default.

The policy is process-global and loaded once. Global state is usually the wrong
shape, but the alternative — threading a Policy object through every call
signature in the engine — has a failure mode this does not: a parameter with a
default. One `policy=None` default anywhere reintroduces exactly the fallback
the whole boundary exists to prevent, and it would be invisible in review.

There is no default policy, no built-in pack and no lazy fallback. Reading a
coefficient before a policy is loaded raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .policy import Policy, PolicyNotFound, load_policy

_active: Optional[Policy] = None


class PolicyNotLoaded(PolicyNotFound):
    """Raised when the engine reaches for a coefficient and none is loaded."""


def load_active_policy(path: Optional[Path] = None, *, verify_integrity: bool = True) -> Policy:
    """
    Load the process policy. Call once, at startup, before any analysis runs.

    Deliberately explicit rather than lazy: a lazy load would mean a
    misconfigured deployment fails on its first request instead of at boot, and
    the failure would look like a request problem.
    """
    global _active
    _active = load_policy(path, verify_integrity=verify_integrity)
    return _active


def set_active_policy(policy: Policy) -> None:
    """Install an already-validated policy. For tests and for embedding."""
    global _active
    _active = policy


def clear_active_policy() -> None:
    global _active
    _active = None


def active_policy() -> Policy:
    if _active is None:
        raise PolicyNotLoaded(
            "No policy is loaded. The engine holds no coefficients of its own and will "
            "not substitute defaults.\n"
            "  at startup:  peptide_suite.runtime.load_active_policy(path)\n"
            "  from env:    export PEPTIDE_SUITE_POLICY=policy/demo.v1.json, then "
            "load_active_policy()\n"
            "  for tests:   python -m unittest discover -t . -s peptide_suite/tests\n"
            "               (-t . matters: it lets the test package load the demo pack "
            "before any test imports)"
        )
    return _active


def is_loaded() -> bool:
    return _active is not None


def threshold(key: str) -> float:
    return active_policy().threshold(key)


def weight(key: str) -> float:
    return active_policy().weight(key)


def int_threshold(key: str) -> int:
    """
    A threshold used as a count.

    Counts are validated as numbers like everything else, so a pack can supply
    3.5 sequences. Truncating silently would turn a malformed pack into a
    working one, so a non-integral value is an error.
    """
    value = threshold(key)
    if value != int(value):
        raise ValueError(
            f"Policy threshold '{key}' is used as a count but the pack supplies {value}, "
            f"which is not a whole number."
        )
    return int(value)


def running_on_placeholders() -> bool:
    """
    Whether any threshold in the active policy is a stand-in rather than a
    derived value. Callers that emit numbers are expected to say so.
    """
    return bool(active_policy().placeholder_thresholds)
