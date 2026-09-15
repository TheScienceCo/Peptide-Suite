#!/usr/bin/env python3
"""
Engine/policy boundary enforcement.  [Addendum 1 section 4 — never-commit list]

Blocks three classes of violation:

  1. A policy file that is not the demonstration pack
  2. A coefficient hardcoded, defaulted, or inlined in engine code
  3. Prose that justifies a weighting choice biochemically

Class 1 is absolute and always blocks. Classes 2 and 3 are ratcheted: the
current count is recorded in policy/BOUNDARY_DEBT.txt and the check fails if the
count rises. That makes the outstanding retrofit visible and monotonically
decreasing, rather than hidden behind an allowlist that would quietly permit the
exact thing the rule exists to prevent.

Usage:
    check_policy_boundary.py --scope staged    # pre-commit hook
    check_policy_boundary.py --scope all       # CI
    check_policy_boundary.py --scope all --write-debt
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEBT_FILE = REPO / "policy" / "BOUNDARY_DEBT.txt"

ALLOWED_POLICY_FILES = {"policy/README.md", "policy/BOUNDARY_DEBT.txt", "policy/schema.json"}
ALLOWED_POLICY_GLOB = re.compile(r"^policy/demo\.[\w.]+\.json$")

BLOCKED_POLICY_PATTERNS = [
    re.compile(r"^policy/(?!demo\.|README|BOUNDARY_DEBT|schema\.json)"),
    re.compile(r"\.policy\.json$"),
    re.compile(r"policy\.(prod|production)"),
    re.compile(r"\.private\.json$"),
    re.compile(r"_policy_weights"),
]

# A named numeric constant in engine code. Structural values (array indices,
# string lengths, HTTP codes) are not what this targets; a name carrying
# WEIGHT/THRESHOLD/COEFF/CUTOFF semantics is.
COEFFICIENT_NAME = re.compile(
    r"^\s*([A-Z_][A-Z0-9_]*)?\s*\b(\w*(?:WEIGHT|THRESHOLD|COEFF|CUTOFF|CEILING|FLOOR|SCALE|FACTOR|MODIFIER)\w*)\s*[:=]"
    r"\s*(?:float\s*=\s*)?(-?\d+\.?\d*)",
    re.IGNORECASE,
)
# A bare float assigned to a module-level or class-level constant.
BARE_CONSTANT = re.compile(r"^\s{0,8}([A-Z_][A-Z0-9_]{2,})\s*(?::\s*\w+\s*)?=\s*(-?\d+\.\d+)\s*(?:#|$)")
# A float inside a dict literal, which is how a weight vector is usually written.
DICT_FLOAT = re.compile(r"^\s+[\w.\[\]\"']+\s*:\s*(-?\d+\.\d+)\s*,\s*(?:#|$)")

# Prose that explains WHY a weight is what it is, or what magnitude to expect.
#
# The shape being caught is a hedged or quantified generalisation sitting next to
# a biophysical property: that combination is what turns a comment into a
# justification for a number. A bare description of what a function computes is
# not a rationale and must not fire, or the check gets disabled within a week.
_PROPERTY = (r"potenc\w*|affinit\w*|half.?li\w*|helicit\w*|aggregat\w*|solubilit\w*|"
             r"immunogenic\w*|bind\w*|cleav\w*|proteolytic|bioavailab\w*|exposure")
_HEDGE = (r"typically|characteristically|usually|commonly|generally|roughly|"
          r"approximately|on average|tends? to|in practice")
_QUANTIFIED = (r"\d+\s*%|\d+\s*-\s*\d+\s*%|\d+\s*-?fold|orders?\s+of\s+magnitude|"
               r"\d+\s*x\b|\d+\s*to\s+\d+\s*(?:fold|orders)")

RATIONALE_PATTERNS = [
    # Explicit: a weighting term justified by a reason
    re.compile(r"(weight|magnitude|coefficient|threshold)\w*\b.{0,40}\bbecause\b", re.I),
    re.compile(r"expected\s+(magnitude|weight|value|range|contribution)", re.I),
    re.compile(r"(set|chose|picked|tuned)\s+\w{0,12}\s*(to|at)\s+-?\d+\.\d+\s+.{0,30}"
               r"(because|since|as it)", re.I),
    # A hedged generalisation about a biophysical property
    re.compile(rf"({_HEDGE})\b.{{0,80}}({_PROPERTY})", re.I),
    re.compile(rf"({_PROPERTY}).{{0,80}}\b({_HEDGE})\b", re.I),
    # A quantified magnitude claim about a biophysical property
    re.compile(rf"({_QUANTIFIED}).{{0,80}}({_PROPERTY})", re.I),
    re.compile(rf"({_PROPERTY}).{{0,80}}({_QUANTIFIED})", re.I),
    # A numeric literal on the same line as a property claim
    re.compile(rf"-?\d+\.\d+.{{0,20}}#.{{0,60}}({_PROPERTY}|{_QUANTIFIED})", re.I),
]

ENGINE_ROOTS = ("peptide_suite/",)
EXEMPT_PATH_PARTS = ("/tests/", "/static/", "/data/")

# Exact-by-SI-definition constants. Exempt because a value that cannot vary is
# not a coefficient: putting it behind the policy artifact would invite someone
# to change something that is true by definition. The exemption is one file and
# that file is the whole argument for it.
EXEMPT_FILES = ("peptide_suite/core/constants.py",)


def staged_files():
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         cwd=REPO, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def all_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def staged_added_lines():
    """
    Line numbers this commit adds or changes, per file.

    The staged check has to be scoped to these rather than to whole files.
    Flagging every pre-existing violation in a file the commit happens to touch
    makes unrelated work impossible -- migrating one constant out of a module
    would be blocked by the other eighty still in it -- and a check that blocks
    honest work gets deleted, which costs more than the violations it caught.
    The ratchet on --scope all is what holds the totals down.
    """
    out = subprocess.run(["git", "diff", "--cached", "-U0", "--diff-filter=ACM"],
                         cwd=REPO, capture_output=True, text=True).stdout
    added = {}
    current = None
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            added.setdefault(current, set())
        elif line.startswith("@@") and current is not None:
            # @@ -old,n +new,m @@
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2) or 1)
                added[current].update(range(start, start + count))
    return added


def is_engine_source(path: str) -> bool:
    return (path.endswith(".py")
            and any(path.startswith(r) for r in ENGINE_ROOTS)
            and path not in EXEMPT_FILES
            and not any(part in f"/{path}" for part in EXEMPT_PATH_PARTS))


def check_policy_files(files):
    violations = []
    for f in files:
        if f in ALLOWED_POLICY_FILES or ALLOWED_POLICY_GLOB.match(f):
            continue
        for pattern in BLOCKED_POLICY_PATTERNS:
            if pattern.search(f):
                violations.append((f, 0, "policy artifact that is not the demo pack", f))
                break
    return violations


def check_coefficients(files):
    violations = []
    for f in files:
        if not is_engine_source(f):
            continue
        path = REPO / f
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if COEFFICIENT_NAME.search(line) or BARE_CONSTANT.match(line) or DICT_FLOAT.match(line):
                violations.append((f, i, "coefficient inlined in engine", line.strip()[:90]))
    return violations


def check_rationale(files):
    violations = []
    for f in files:
        if not (f.endswith((".py", ".md")) and not any(p in f"/{f}" for p in ("/tests/",))):
            continue
        path = REPO / f
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for pattern in RATIONALE_PATTERNS:
                if pattern.search(line):
                    violations.append((f, i, "weighting rationale in prose", line.strip()[:90]))
                    break
    return violations


def read_debt():
    if not DEBT_FILE.exists():
        return None
    try:
        return json.loads(DEBT_FILE.read_text().split("---")[-1].strip())
    except Exception:
        return None


def write_debt(counts):
    DEBT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEBT_FILE.write_text(
        "Engine/policy boundary debt.  [Addendum 1 section 4]\n"
        "\n"
        "Violations outstanding from code written before the boundary was enforced.\n"
        "The check fails if any count rises above these numbers, so the retrofit can\n"
        "only move one way. Lower these as modules migrate to the policy artifact;\n"
        "never raise them.\n"
        "\n"
        "---\n" + json.dumps(counts, indent=2) + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["staged", "all"], default="staged")
    ap.add_argument("--write-debt", action="store_true")
    args = ap.parse_args()

    files = staged_files() if args.scope == "staged" else all_files()

    policy_v = check_policy_files(files)
    coeff_v = check_coefficients(files)
    rationale_v = check_rationale(files)

    counts = {"coefficients_in_engine": len(coeff_v), "weighting_rationale_in_prose": len(rationale_v)}

    if args.write_debt:
        write_debt(counts)
        print(f"Recorded boundary debt: {json.dumps(counts)}")
        return 0

    failed = False

    # Class 1 is absolute.
    if policy_v:
        failed = True
        print("BLOCKED — policy artifact that is not the demonstration pack:\n")
        for f, _, _, _ in policy_v:
            print(f"    {f}")
        print("\n  The scoring policy is proprietary and must not be committed.")
        print("  Only policy/demo.*.json may enter the repository.\n")

    if args.scope == "staged":
        # A violation on a line this commit introduces blocks outright. A
        # pre-existing one in the same file does not: it is already counted in
        # the debt, and the --scope all ratchet is what stops it growing.
        added = staged_added_lines()

        def introduced(vlist):
            return [v for v in vlist if v[1] in added.get(v[0], set())]

        for label, vlist in (("coefficient inlined in engine", introduced(coeff_v)),
                             ("weighting rationale in prose", introduced(rationale_v))):
            if vlist:
                failed = True
                print(f"BLOCKED — {label} introduced by this commit ({len(vlist)}):\n")
                for f, ln, _, text in vlist[:12]:
                    print(f"    {f}:{ln}: {text}")
                if len(vlist) > 12:
                    print(f"    ... {len(vlist) - 12} more")
                print()
    else:
        debt = read_debt()
        if debt is None:
            print("No boundary debt recorded. Run with --write-debt to establish the baseline.")
            print(json.dumps(counts, indent=2))
            return 1
        print("Engine/policy boundary check (scope: all)\n")
        for key, current in counts.items():
            allowed = debt.get(key, 0)
            status = "OK" if current <= allowed else "REGRESSION"
            if current > allowed:
                failed = True
            arrow = "" if current == allowed else f"  ({current - allowed:+d})"
            print(f"  {status:<11} {key:<32} {current:>4} / {allowed} allowed{arrow}")
        print()
        if failed:
            print("  A count rose above the recorded debt. The retrofit may only move one way.")
        else:
            print("  No regression. Lower the numbers in policy/BOUNDARY_DEBT.txt as modules migrate.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
