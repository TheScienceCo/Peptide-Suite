"""
CLI entry point for Peptide Optimization Pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from peptide_suite.workflows.optimize import (
    OptimizeWorkflow,
    format_workflow_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Peptide Optimization Pipeline: substitution scanning with confidence scoring"
    )

    subparsers = parser.add_subparsers(dest="command", help="Workflow to run")

    # Workflow 1: Optimize
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize a peptide sequence (Workflow 1)",
    )
    optimize_parser.add_argument(
        "input",
        help="Peptide sequence or gene name (e.g., 'IGF1' or 'MGFPGLQPRR...')",
    )
    optimize_parser.add_argument(
        "--goal",
        default=None,
        help="Target goal (e.g., 'protease_resistance', 'binding_affinity'). "
        "If not provided, will be inferred.",
    )
    optimize_parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Auto-confirm inferred function (for testing)",
    )
    optimize_parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for charge calculations (default: 7.4)",
    )

    # Workflow 2: Find Peptides (placeholder)
    find_parser = subparsers.add_parser(
        "find",
        help="Find peptides for a functional goal (Workflow 2 - TODO)",
    )
    find_parser.add_argument(
        "goal",
        help="Functional goal (e.g., 'myelinating_peptides', 'wound_healing')",
    )

    # Test/calibration commands
    test_parser = subparsers.add_parser(
        "test",
        help="Run calibration tests on known peptides",
    )

    args = parser.parse_args()

    if args.command == "optimize":
        run_optimize(args)
    elif args.command == "find":
        run_find(args)
    elif args.command == "test":
        run_test_panel()
    else:
        parser.print_help()
        sys.exit(1)


def run_optimize(args):
    """Execute Workflow 1: Optimize This Peptide."""
    logger.info("Launching Workflow 1: Optimize This Peptide")

    workflow = OptimizeWorkflow()

    try:
        peptide_context, recommendations = workflow.run(
            input_sequence_or_name=args.input,
            confirmed_goal=args.goal,
            auto_confirm=args.auto_confirm,
            ph=args.ph,
        )

        # Display results
        output = format_workflow_summary(peptide_context, recommendations)
        print(output)

        # Save results to file
        output_file = Path(f"results_{peptide_context.name}.txt")
        with open(output_file, "w") as f:
            f.write(output)
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Workflow error: {e}", exc_info=True)
        sys.exit(1)


def run_find(args):
    """Execute Workflow 2: Find Peptides."""
    logger.error("Workflow 2: Find Peptides is not yet implemented (v1 roadmap)")
    sys.exit(1)


def run_test_panel():
    """Run calibration tests on known peptides (IGF-1, insulin, GLP-1, BPC-157)."""
    logger.info("Running test panel calibration...")

    test_cases = [
        ("IGF1", "MGFPGLQPRRVSCGQAKDEARGYLCFSQTPRAT", "binding_affinity"),
        ("INS", "GIVEQCCTSICSLYQLENYCN", "binding_affinity"),
        ("GCG", "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT", None),  # Will infer
        ("BPC157", "GEPGDWGDPEHPPGAGQGDSPGAGAGQGDQGAGAGAEAEQGAGAGRGRGDSPR", "tissue_repair"),
    ]

    workflow = OptimizeWorkflow()

    for gene_name, sequence, goal in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test case: {gene_name}")
        logger.info(f"{'='*60}")

        try:
            peptide_context, recommendations = workflow.run(
                input_sequence_or_name=sequence,
                confirmed_goal=goal,
                auto_confirm=True,
            )

            logger.info(f"✓ Completed: {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"✗ Failed: {e}")


if __name__ == "__main__":
    main()
