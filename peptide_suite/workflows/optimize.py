"""
Workflow 1: "Optimize This Peptide"

Single-position substitution scanning with quantified confidence.
Gain-of-function oriented (seeking improvements, not loss-of-function).

Tier 1 (v1): Canonical AAs only, sequence-based scoring.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from peptide_suite.core import (
    PeptideContext,
    SubstitutionRecommendation,
    ConfidenceLevel,
    min_homologs_for_conservation,
)
from peptide_suite.core.peptide_manager import PeptideManager, CANONICAL_AAS
from peptide_suite.core.evidence_retrieval import EvidenceRetriever
from peptide_suite.core.conservation import ConservationAnalyzer
from peptide_suite.core.substitution_predictor import SubstitutionPredictor
from peptide_suite.core.confidence_scoring import ConfidenceScorer

logger = logging.getLogger(__name__)


class OptimizeWorkflow:
    """
    End-to-end peptide optimization pipeline.
    """

    def __init__(self):
        self.peptide_manager = PeptideManager()
        self.evidence_retriever = EvidenceRetriever()
        self.conservation = ConservationAnalyzer()
        self.predictor = SubstitutionPredictor()
        self.confidence_scorer = ConfidenceScorer()

    def run(
        self,
        input_sequence_or_name: str,
        confirmed_goal: Optional[str] = None,
        auto_confirm: bool = False,
        ph: float = 7.4,
        homologs: Optional[List[str]] = None,
    ) -> Tuple[PeptideContext, List[SubstitutionRecommendation]]:
        """
        Run the full optimization workflow.

        Args:
            input_sequence_or_name: Raw AA sequence or gene name (e.g., "IGF1", or "MGFPGLQPRR...")
            confirmed_goal: If provided, skip function inference step
            auto_confirm: If True, auto-use inferred function (testing only)
            ph: pH for charge calculations
            homologs: Caller-supplied homologous sequences. This is the manual
                route to the conservation term until NCBI retrieval is wired up;
                supplying >= 3 distinct sequences activates entropy scoring.

        Returns:
            Tuple of (PeptideContext, list of SubstitutionRecommendation objects)
            These can then be formatted for display.
        """

        logger.info(f"=== Workflow 1: Optimize Peptide ===")
        logger.info(f"Input: {input_sequence_or_name[:50]}...")

        # Step 0: Parse input
        peptide_context = self._parse_input(input_sequence_or_name)

        # Validate that we have a sequence
        if not peptide_context.sequence:
            logger.error(f"Could not parse '{input_sequence_or_name}' as a valid sequence or recognized peptide name")
            return peptide_context, []

        logger.info(f"Parsed: {peptide_context.sequence[:50]}... (length {len(peptide_context.sequence)})")

        # Step 1: Infer function
        if confirmed_goal:
            peptide_context.inferred_function = confirmed_goal
            peptide_context.confirmed_goal = confirmed_goal
        else:
            inferred_fn, confidence = self.evidence_retriever.infer_function_from_name(
                peptide_context.name
            )
            peptide_context.inferred_function = inferred_fn
            logger.info(f"Inferred function: {inferred_fn} (conf={confidence:.2f})")

            if not auto_confirm:
                logger.info(
                    ">>> Awaiting user confirmation of inferred function before proceeding. "
                    "(In programmatic mode, pass confirmed_goal or auto_confirm=True)"
                )
                return peptide_context, []
            else:
                peptide_context.confirmed_goal = inferred_fn

        # Step 2: Retrieve homologs and compute conservation
        logger.info("Step 2: Retrieving homologs and computing conservation...")

        if homologs:
            logger.info(f"Using {len(homologs)} caller-supplied homolog(s)")
            homolog_source = "caller-supplied"
        else:
            homologs, homolog_status = self.evidence_retriever.retrieve_homologs(
                peptide_context.name
            )
            homolog_source = "NCBI"
            if not homologs:
                logger.warning(f"No homologs found ({homolog_status}). Using query peptide only.")
                homologs = [peptide_context.sequence]
            else:
                logger.info(f"Retrieved {len(homologs)} homologs from NCBI")

        # The query sequence is part of its own alignment, and only distinct
        # sequences carry information: duplicates would make every position look
        # perfectly conserved.
        distinct = list(dict.fromkeys([peptide_context.sequence] + list(homologs)))
        peptide_context.homolog_count = len(distinct)
        peptide_context.known_homologs = distinct
        min_homologs = min_homologs_for_conservation()
        peptide_context.conservation_available = len(distinct) >= min_homologs

        if peptide_context.conservation_available:
            msa = self.conservation.build_msa_from_sequences(distinct)
            conservation_profile = self.conservation.conservation_profile(msa)
            logger.info(
                f"Conservation entropy computed from {len(distinct)} distinct sequences "
                f"(range: {min(conservation_profile.values()):.2f} - {max(conservation_profile.values()):.2f})"
            )
        else:
            conservation_profile = {}
            note = (
                f"Conservation entropy NOT computed: only {len(distinct)} distinct sequence(s) "
                f"available, {min_homologs} required. Entropy over a single "
                f"sequence is 0 at every position by construction and carries no information, "
                f"so no conservation claim is made and no conservation penalty is applied."
            )
            peptide_context.data_notes.append(note)
            logger.warning(note)

        peptide_context.conservation_entropy = conservation_profile

        # Step 3: Run substitution scan
        logger.info("Step 3: Running substitution scan...")
        all_recommendations = self._run_substitution_scan(
            peptide_context, conservation_profile, ph
        )

        # Step 4: Rank and filter
        logger.info("Step 4: Ranking recommendations...")
        ranked = self._rank_recommendations(all_recommendations)

        # Return top 3-5
        top_recommendations = ranked[:5]
        logger.info(f"Top {len(top_recommendations)} recommendations selected")

        return peptide_context, top_recommendations

    def _parse_input(self, input_str: str) -> PeptideContext:
        """Parse input (sequence or name) into PeptideContext."""
        # Try loading as sequence first
        try:
            sequence, name = self.peptide_manager.load_sequence(input_str)
            return PeptideContext(sequence=sequence, name=name)
        except ValueError:
            # Might be a gene name; treat as such
            name = input_str.strip()
            # For now, we can't resolve to sequence without NCBI access
            # but store name for evidence retrieval
            logger.warning(f"Could not parse as sequence. Treating '{name}' as gene/peptide name.")
            return PeptideContext(sequence="", name=name)

    def _run_substitution_scan(
        self,
        peptide_context: PeptideContext,
        conservation_profile: Dict[int, float],
        ph: float,
    ) -> List[SubstitutionRecommendation]:
        """
        Scan every position for beneficial substitutions.

        For each position and each candidate AA:
        - Predict primary effect
        - Predict off-target effects
        - Score net benefit
        - Log prediction for calibration

        Returns:
            List of all SubstitutionRecommendation objects (unsorted)
        """
        all_recs = []
        sequence = peptide_context.sequence

        for position in range(len(sequence)):
            wt_aa = sequence[position]

            for mutant_aa in CANONICAL_AAS:
                if mutant_aa == wt_aa:
                    continue  # Skip identity substitution

                # Predict effects
                primary_effect, off_targets = self.predictor.predict_substitution_effect(
                    sequence=sequence,
                    position=position,
                    wild_type_aa=wt_aa,
                    mutant_aa=mutant_aa,
                    conservation_profile=conservation_profile,
                    inferred_goal=peptide_context.confirmed_goal or "generic_improvement",
                    ph=ph,
                    conservation_available=peptide_context.conservation_available,
                )

                # Combine scores
                all_effects = [primary_effect] + off_targets
                net_score, combined_confidence = self.confidence_scorer.combine_effect_scores(
                    all_effects
                )
                breakdown = self.confidence_scorer.explain_net_score(all_effects)
                limiting = self.confidence_scorer.limiting_effect(all_effects)
                breakdown["confidence_limited_by"] = limiting.description if limiting else ""

                # Determine net recommendation
                if net_score >= 0.5:
                    recommendation = "recommend"
                elif net_score >= 0.2:
                    recommendation = "recommend_with_caveats"
                else:
                    recommendation = "not_recommended"

                # Log for calibration
                self.confidence_scorer.log_prediction(
                    peptide_name=peptide_context.name,
                    position=position,
                    wt_aa=wt_aa,
                    mutant_aa=mutant_aa,
                    predicted_effect=primary_effect.description,
                    predicted_score=max(0.0, net_score),  # Clamp to [0, 1] for logging
                    predicted_confidence=combined_confidence.value,
                )

                # Create recommendation
                rec = SubstitutionRecommendation(
                    position=position,
                    wild_type_aa=wt_aa,
                    mutant_aa=mutant_aa,
                    is_noncanonical=False,
                    target_category=peptide_context.confirmed_goal or "unspecified",
                    primary_effect=primary_effect,
                    off_target_effects=off_targets,
                    net_recommendation=recommendation,
                    overall_confidence=combined_confidence.value,
                    net_score=net_score,
                    ranking_rationale=f"Score {net_score:.2f}, confidence {combined_confidence.value}",
                    score_breakdown=breakdown,
                )

                all_recs.append(rec)

        logger.info(f"Generated {len(all_recs)} candidate substitutions")
        return all_recs

    def _rank_recommendations(
        self, recommendations: List[SubstitutionRecommendation]
    ) -> List[SubstitutionRecommendation]:
        """
        Rank recommendations by net score (descending), filtering for quality.

        Rules:
        - Filter out 'not_recommended' unless net_score is borderline
        - Sort by net_score descending
        - Prefer "recommend" over "recommend_with_caveats"
        """

        # Filter for candidates worth considering
        good_recs = [
            r for r in recommendations if r.net_recommendation in ["recommend", "recommend_with_caveats"]
        ]

        # If too few, include top not_recommended
        if len(good_recs) < 3:
            not_recs = [r for r in recommendations if r.net_recommendation == "not_recommended"]
            not_recs.sort(key=lambda r: r.net_score, reverse=True)
            good_recs.extend(not_recs[: 3 - len(good_recs)])

        # Sort by score
        good_recs.sort(key=lambda r: r.net_score, reverse=True)

        return good_recs


def format_recommendation_for_display(rec: SubstitutionRecommendation) -> str:
    """Format a single recommendation for terminal display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Position {rec.position + 1}: {rec.wild_type_aa} → {rec.mutant_aa}")
    lines.append(f"{'='*60}")

    lines.append(f"Category: {rec.target_category}")
    lines.append(f"Net Recommendation: {rec.net_recommendation.upper()}")
    lines.append(f"Net Score: {rec.net_score:+.2f} | Confidence: {rec.overall_confidence}")

    if rec.primary_effect:
        lines.append(f"\n📌 PRIMARY EFFECT:")
        lines.append(f"   {rec.primary_effect.description}")
        lines.append(
            f"   Evidence: {rec.primary_effect.evidence_tier.name}, "
            f"Confidence: {rec.primary_effect.confidence.value}"
        )
        lines.append(f"   Score: {rec.primary_effect.score:.2f}")
        lines.append(f"   Reasoning: {rec.primary_effect.reasoning}")

    if rec.off_target_effects:
        lines.append(f"\n⚠️  OFF-TARGET EFFECTS:")
        for i, effect in enumerate(rec.off_target_effects, 1):
            lines.append(f"   ({i}) {effect.description}")
            lines.append(f"       Evidence: {effect.evidence_tier.name}, Score: {effect.score:.2f}")
            lines.append(f"       Reasoning: {effect.reasoning[:100]}...")

    lines.append(f"\nRanking Rationale: {rec.ranking_rationale}")

    return "\n".join(lines)


def format_workflow_summary(peptide_context: PeptideContext, recommendations: List[SubstitutionRecommendation]) -> str:
    """Format full workflow output for display."""
    lines = []

    lines.append("\n" + "="*70)
    lines.append("WORKFLOW 1: OPTIMIZE THIS PEPTIDE - RESULTS")
    lines.append("="*70)

    lines.append(f"\nPeptide: {peptide_context.name}")
    lines.append(f"Sequence: {peptide_context.sequence}")
    lines.append(f"Length: {len(peptide_context.sequence)} AA")
    lines.append(f"\nInferred Function: {peptide_context.inferred_function}")
    lines.append(f"Confirmed Goal: {peptide_context.confirmed_goal}")

    lines.append(f"\nConservation Profile Summary:")
    if peptide_context.conservation_entropy:
        entropies = list(peptide_context.conservation_entropy.values())
        lines.append(f"  - Entropy range: {min(entropies):.2f} - {max(entropies):.2f}")
        lines.append(f"  - Mean entropy: {sum(entropies)/len(entropies):.2f}")

    lines.append(f"\n{'='*70}")
    lines.append(f"TOP RECOMMENDATIONS ({len(recommendations)} suggestions)")
    lines.append(f"{'='*70}")

    for i, rec in enumerate(recommendations, 1):
        lines.append(format_recommendation_for_display(rec))

    lines.append(f"\n{'='*70}")
    lines.append("CALIBRATION STATUS:")
    calibration = peptide_context
    lines.append(f"  - Predictions logged for future validation")
    lines.append(f"  - See logs/prediction_calibration.jsonl for audit trail")

    return "\n".join(lines)
