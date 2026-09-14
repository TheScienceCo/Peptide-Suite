"""
Confidence scoring framework based on evidence tiers and Bayesian weighting.
Implements per-prediction confidence with audit trail for later calibration.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from . import EvidenceTier, ConfidenceLevel, Effect

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Implements evidence-tier-based confidence scoring.
    Tracks predictions for later Brier-score calibration against test panel.
    """

    # Evidence tier base weights (Bayesian-flavored)
    TIER_WEIGHTS = {
        EvidenceTier.DIRECT_EXPERIMENTAL: 1.0,
        EvidenceTier.HOMOLOG_EXPERIMENTAL: 0.8,
        EvidenceTier.BIOCHEMICAL_PRINCIPLE: 0.6,
        EvidenceTier.INFERENCE_ONLY: 0.3,
    }

    # Confidence thresholds (adjustable based on calibration)
    CONFIDENCE_THRESHOLDS = {
        ConfidenceLevel.HIGH: 0.7,
        ConfidenceLevel.MEDIUM: 0.4,
        ConfidenceLevel.LOW: 0.0,
    }

    def __init__(self, calibration_log_path: str = "logs/prediction_calibration.jsonl"):
        self.calibration_log = Path(calibration_log_path)
        self.calibration_log.parent.mkdir(parents=True, exist_ok=True)

    def score_effect(
        self,
        description: str,
        evidence_tier: EvidenceTier,
        reasoning: str,
        modifier: float = 1.0,
        sources: Optional[List[str]] = None,
        equation_refs: Optional[List[int]] = None,
    ) -> Effect:
        """
        Score a single effect based on evidence tier.

        Args:
            description: What the effect is
            evidence_tier: Where the evidence comes from
            reasoning: Explanation of why we believe this
            modifier: 0-1, used to apply domain-specific penalties
                     (e.g., if a claim contradicts existing data, multiply down)
            sources: Literature/database citations
            equation_refs: Which equations informed this

        Returns:
            Effect object with confidence score and level
        """
        # Base score from tier weight
        base_score = self.TIER_WEIGHTS[evidence_tier]

        # Apply modifier (e.g., if multiple papers agree, keep high;
        # if one marginal paper, multiply down)
        final_score = base_score * modifier

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        # Assign confidence level based on thresholds
        confidence_level = self._score_to_level(final_score)

        return Effect(
            category="effect",
            description=description,
            evidence_tier=evidence_tier,
            confidence=confidence_level,
            score=final_score,
            reasoning=reasoning,
            sources=sources or [],
            equation_refs=equation_refs or [],
        )

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= self.CONFIDENCE_THRESHOLDS[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif score >= self.CONFIDENCE_THRESHOLDS[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def combine_effect_scores(self, effects: List[Effect]) -> Tuple[float, ConfidenceLevel]:
        """
        Combine multiple effects (primary + off-targets) into a net score.

        Uses weighted average, with off-target negatives weighted by conviction.

        Args:
            effects: List of Effect objects (first should be primary)

        Returns:
            Tuple of (net_score, combined_confidence_level)
        """
        if not effects:
            return 0.0, ConfidenceLevel.LOW

        # Primary effect (first) counts as +score
        # Off-targets count as -score (penalty)
        net = 0.0
        weights_sum = 0.0

        for i, effect in enumerate(effects):
            weight = effect.score  # Weight by own confidence
            if i == 0:  # Primary effect, positive contribution
                net += weight
            else:  # Off-target, negative contribution
                net -= weight * 0.5  # Off-targets are weighted at half severity

            weights_sum += weight

        if weights_sum > 0:
            net = net / weights_sum
        else:
            net = 0.0

        net = max(-1.0, min(1.0, net))  # Clamp to [-1, 1]

        # Combine confidences (take worst of all effects)
        # Map to numeric values for comparison
        confidence_rank = {ConfidenceLevel.HIGH: 3, ConfidenceLevel.MEDIUM: 2, ConfidenceLevel.LOW: 1}
        min_confidence = min((e.confidence for e in effects), key=lambda conf: confidence_rank[conf])

        return net, min_confidence

    def log_prediction(
        self,
        peptide_name: str,
        position: int,
        wt_aa: str,
        mutant_aa: str,
        predicted_effect: str,
        predicted_score: float,
        predicted_confidence: str,
        actual_outcome: Optional[str] = None,
        actual_score: Optional[float] = None,
    ) -> None:
        """
        Log a prediction for later calibration against test panel.

        Args:
            peptide_name: E.g., "IGF-1"
            position: 0-indexed
            wt_aa: Wild-type amino acid
            mutant_aa: Mutant amino acid
            predicted_effect: Description
            predicted_score: 0-1 confidence
            predicted_confidence: "High"/"Medium"/"Low"
            actual_outcome: (later filled in) "confirmed" / "refuted" / "null"
            actual_score: (later filled in) empirical measurement

        Returns:
            None (logs to file)
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "peptide": peptide_name,
            "position": position,
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "predicted_effect": predicted_effect,
            "predicted_score": predicted_score,
            "predicted_confidence": predicted_confidence,
            "actual_outcome": actual_outcome,
            "actual_score": actual_score,
        }

        with open(self.calibration_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"Logged prediction: {peptide_name} {wt_aa}{position+1}{mutant_aa} "
            f"(conf={predicted_confidence}, score={predicted_score:.2f})"
        )

    def compute_brier_score(self) -> Optional[float]:
        """
        Compute Brier score (MSE between predicted and actual) for calibration.

        Only includes predictions where actual_outcome is filled in.

        Returns:
            Brier score (lower is better) or None if no completed predictions
        """
        if not self.calibration_log.exists():
            return None

        predictions = []
        actuals = []

        with open(self.calibration_log, "r") as f:
            for line in f:
                record = json.loads(line)
                if record.get("actual_outcome") is not None:
                    predictions.append(record["predicted_score"])
                    # Map outcome to 0/1: "confirmed" -> 1, else -> 0
                    actual = 1.0 if record["actual_outcome"] == "confirmed" else 0.0
                    actuals.append(actual)

        if not predictions:
            return None

        # Brier score = mean((predicted - actual)^2)
        brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
        return brier

    def get_calibration_summary(self) -> Dict:
        """Return summary of calibration data (for audit trail display)."""
        if not self.calibration_log.exists():
            return {"total_predictions": 0, "completed": 0, "brier_score": None}

        total = 0
        completed = 0

        with open(self.calibration_log, "r") as f:
            for line in f:
                record = json.loads(line)
                total += 1
                if record.get("actual_outcome") is not None:
                    completed += 1

        brier = self.compute_brier_score()

        return {
            "total_predictions": total,
            "completed": completed,
            "brier_score": brier,
            "calibration_log_path": str(self.calibration_log),
        }
