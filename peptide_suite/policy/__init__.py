"""Policy artifact loading. The engine's only source of coefficients."""

from .loader import (
    Policy, PolicyError, PolicyIntegrityError, PolicyNotFound,
    PolicyValidationError, compute_digest, load_policy, validate_document,
)
from .opaque import (
    OpaqueIdError, TERM_ID_PATTERN, TermLegend, is_opaque_term_id, load_salt,
    mint_term_id, require_opaque,
)
from .registry import FEATURE_FAMILIES, THRESHOLDS, free_parameter_count, required_keys

__all__ = [
    "Policy", "PolicyError", "PolicyIntegrityError", "PolicyNotFound",
    "PolicyValidationError", "compute_digest", "load_policy", "validate_document",
    "FEATURE_FAMILIES", "THRESHOLDS", "free_parameter_count", "required_keys",
    "OpaqueIdError", "TERM_ID_PATTERN", "TermLegend", "is_opaque_term_id",
    "load_salt", "mint_term_id", "require_opaque",
]
