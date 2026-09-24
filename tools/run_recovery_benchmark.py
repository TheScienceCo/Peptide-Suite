#!/usr/bin/env python3
"""
Run the blind recovery benchmark and print the report.

    python tools/run_recovery_benchmark.py                      # motif ablation, all drugs
    python tools/run_recovery_benchmark.py --protocol temporal
    python tools/run_recovery_benchmark.py --drug semaglutide --goal binding_affinity
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peptide_suite.core.holdout import Protocol            # noqa: E402
from peptide_suite.core.recovery import run_benchmark, summarise   # noqa: E402
from peptide_suite.runtime import load_active_policy       # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["motif_ablation", "temporal"],
                        default="motif_ablation")
    parser.add_argument("--drug", action="append", default=[])
    parser.add_argument("--goal", default="binding_affinity")
    parser.add_argument("--policy", default="policy/demo.v1.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_active_policy(args.policy)
    results = run_benchmark(protocol=Protocol(args.protocol), drugs=args.drug,
                            goal=args.goal, seed=args.seed)
    for result in results:
        print(result.report())
        print()
    print("-" * 72)
    print(summarise(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
