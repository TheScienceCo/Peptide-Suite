# Policy artifacts

The scoring policy is proprietary and is **not** in this repository.

The engine contains no coefficients. Every weight, threshold and cutoff is
supplied by a versioned policy artifact loaded at runtime. If no artifact is
supplied the system fails at startup rather than falling back to built-in
values — a fallback would let a misconfigured deployment produce normal-looking
output from values that are not the policy, with nothing to indicate it.

## Files

| File | Committed | Purpose |
|---|---|---|
| `schema.json` | yes | The artifact schema. Key names are public; values are not. |
| `demo.v1.json` | yes, once created | Demonstration pack. Every example, test and walkthrough runs against it. |
| `BOUNDARY_DEBT.txt` | yes | Outstanding engine/policy violations, ratcheted downward. |
| anything else | **no** | Blocked by `.gitignore` and by `tools/check_policy_boundary.py`. |

## Loading

```bash
export PEPTIDE_SUITE_POLICY=policy/demo.v1.json
```

```python
from peptide_suite.policy import load_policy
policy = load_policy()          # reads PEPTIDE_SUITE_POLICY
print(policy.banner())
```

Validation is strict in three directions: unknown keys are rejected, values
outside their declared range are rejected, and missing required fields are
rejected. The artifact also carries a SHA-256 digest computed over the document
with the integrity block removed, so modification after signing is detectable.

## Boundary enforcement

`tools/check_policy_boundary.py` runs as a pre-commit hook (`--scope staged`)
and in CI (`--scope all`). It blocks three things:

1. Any policy artifact that is not the demonstration pack. Absolute, always.
2. A coefficient hardcoded or defaulted in engine code.
3. Prose that justifies a weighting choice biochemically.

Classes 2 and 3 are ratcheted against `BOUNDARY_DEBT.txt` rather than allowlisted.
The recorded counts are violations from code written before the boundary
existed; the check fails if a count rises, so the retrofit can only move one
way. Lower the numbers as modules migrate to the policy artifact. Never raise
them.

To re-baseline after a deliberate reduction:

```bash
python tools/check_policy_boundary.py --scope all --write-debt
```
