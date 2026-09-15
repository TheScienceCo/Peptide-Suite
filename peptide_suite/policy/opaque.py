"""
Opaque aggregate-term identifiers.  [Addendum 1 section 2, confirmed]

Feature-family key names are public by specification. Aggregate terms are the
exception: a term named `interface.dispersion_fraction` in a public artifact
tells a reader which physics was found predictive, which is the finding itself
rather than its magnitude. So the artifact carries opaque identifiers and the
mapping back to a human label is held privately.

The identifier is an HMAC of the label under a private salt, truncated. That
makes it deterministic — the owner regenerates the same id from the same label
without storing a lookup table — while remaining one-way to a reader who does
not hold the salt. A plain hash would be reversible by dictionary attack over
the small space of plausible chemistry terms; the salt is what prevents that.

The engine never sees a label. It resolves, weights and reports opaque ids
throughout, so a debug surface that lacks the legend degrades to showing ids
rather than leaking meaning.
"""

from __future__ import annotations

import hmac
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

# \Z rather than $, and fullmatch rather than match: Python's $ also matches
# immediately before a trailing newline, so "agg.<hex>\n" would otherwise pass
# the shape check and reach the artifact as a distinct key.
TERM_ID_PATTERN = re.compile(r"\Aagg\.[0-9a-f]{12}\Z")
_ID_LENGTH = 12


class OpaqueIdError(Exception):
    pass


def mint_term_id(label: str, salt: bytes) -> str:
    """
    Derive the opaque identifier for a term label.

    Deterministic under a fixed salt, so the same label always yields the same
    id and no lookup table needs to be persisted on the public side.
    """
    if not label.strip():
        raise OpaqueIdError("Cannot mint an identifier for an empty label")
    if not salt:
        raise OpaqueIdError(
            "A salt is required. Without one the identifier is a plain hash and is "
            "recoverable by dictionary attack over the space of plausible chemistry terms."
        )
    digest = hmac.new(salt, label.strip().lower().encode("utf-8"), sha256).hexdigest()
    return f"agg.{digest[:_ID_LENGTH]}"


def is_opaque_term_id(value: str) -> bool:
    return isinstance(value, str) and bool(TERM_ID_PATTERN.fullmatch(value))


def require_opaque(value: str) -> str:
    if not is_opaque_term_id(value):
        raise OpaqueIdError(
            f"'{value}' is not an opaque aggregate-term identifier. Aggregate terms must be "
            f"declared as agg.<12 hex> so the artifact does not disclose which physics is "
            f"weighted. Mint one with mint_term_id(label, salt)."
        )
    return value


def load_salt(env_var: str = "PEPTIDE_SUITE_TERM_SALT") -> bytes:
    """
    Read the minting salt from the environment.

    Absent salt is fatal rather than defaulted: a default salt is equivalent to
    no salt, since it would be identical across every deployment and therefore
    public.
    """
    raw = os.environ.get(env_var)
    if not raw:
        raise OpaqueIdError(
            f"{env_var} is not set. Minting requires a private salt and there is no default, "
            f"because a shared default salt is a public salt."
        )
    return raw.encode("utf-8")


class TermLegend:
    """
    Private mapping from opaque identifier back to a human label.

    Never committed. Used only by an authoring tool or a local debugging
    surface; the engine neither requires nor consults it. Absence is normal, and
    lookups then return None so a caller displays the opaque id rather than
    failing.
    """

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self._mapping = dict(mapping or {})

    @classmethod
    def load(cls, path: Path) -> "TermLegend":
        if not Path(path).exists():
            return cls()
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_labels(cls, labels, salt: bytes) -> "TermLegend":
        return cls({mint_term_id(label, salt): label for label in labels})

    def label_for(self, term_id: str) -> Optional[str]:
        return self._mapping.get(term_id)

    def describe(self, term_id: str) -> str:
        """Human label when the legend is present, the opaque id when it is not."""
        return self._mapping.get(term_id, term_id)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self._mapping, indent=2, sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self._mapping)
