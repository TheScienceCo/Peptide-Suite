# Peptide-Suite
Peptide optimizer based on quantum biochemical principles and robust relevant research retrieval

## Running

The engine holds no coefficients. Every weight, threshold and cutoff comes from
a policy artifact loaded at startup, and the system refuses to start without
one rather than falling back to built-in values. A demonstration pack is
committed so this repository runs out of the box.

```bash
pip install -r requirements.txt

export PEPTIDE_SUITE_POLICY=policy/demo.v1.json
python -m uvicorn peptide_suite.api:app --reload
```

Then open http://127.0.0.1:8000.

`policy/demo.v1.json` is a demonstration pack: no weight in it was fitted to
data. Every scored response says so, in the API payload and in the UI, so a
number produced under it is never mistaken for a measured one. See
`policy/README.md`.

## Tests

```bash
python -m unittest discover -t . -s peptide_suite/tests
```

`-t .` matters. It lets the test package load the demonstration pack before any
test imports — without it the suite fails at the first coefficient read, which
is the boundary working as intended rather than a broken suite.

```bash
python tools/check_policy_boundary.py --scope all
```
