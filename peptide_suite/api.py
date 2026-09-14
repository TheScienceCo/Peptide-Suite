"""
HTTP API for the peptide suite.

Thin transport layer over the real Python pipeline: every number returned here
is produced by the same code the CLI runs. Nothing is computed in the browser,
so the UI cannot drift from the analysis engine.
"""

import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from peptide_suite.core import PeptideContext, SubstitutionRecommendation
from peptide_suite.core.confidence_scoring import ConfidenceScorer
from peptide_suite.core.evidence_retrieval import EvidenceRetriever
from peptide_suite.core.peptide_manager import PeptideManager
from peptide_suite.workflows.find_peptides import FindPeptidesWorkflow
from peptide_suite.workflows.optimize import OptimizeWorkflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Peptide Suite",
    description="Substitution scanning and peptide discovery with evidence-tiered confidence",
    version="1.1.0",
)

_optimize = OptimizeWorkflow()
_find = FindPeptidesWorkflow()
_peptides = PeptideManager()
_evidence = EvidenceRetriever()
_scorer = ConfidenceScorer()


# ---- serialization ---------------------------------------------------------

def encode(obj: Any) -> Any:
    """Recursively convert dataclasses and enums to JSON-safe structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: encode(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def encode_effect(effect) -> Dict:
    """Effects need their enum names preserved for display, not just values."""
    if effect is None:
        return None
    return {
        "category": effect.category,
        "description": effect.description,
        "evidence_tier": effect.evidence_tier.name,
        "evidence_weight": effect.evidence_tier.value,
        "confidence": effect.confidence.value,
        "score": round(effect.score, 3),
        "magnitude": round(effect.magnitude, 3),
        "contribution": round(effect.score * effect.magnitude, 3),
        "reasoning": effect.reasoning,
        "equation_refs": effect.equation_refs or [],
        "sources": effect.sources or [],
    }


def encode_recommendation(rec: SubstitutionRecommendation) -> Dict:
    return {
        "position": rec.position,
        "display_position": rec.position + 1,
        "wild_type_aa": rec.wild_type_aa,
        "mutant_aa": rec.mutant_aa,
        "label": f"{rec.wild_type_aa}{rec.position + 1}{rec.mutant_aa}",
        "is_noncanonical": rec.is_noncanonical,
        "target_category": rec.target_category,
        "primary_effect": encode_effect(rec.primary_effect),
        "off_target_effects": [encode_effect(e) for e in rec.off_target_effects],
        "net_recommendation": rec.net_recommendation,
        "overall_confidence": rec.overall_confidence,
        "net_score": round(rec.net_score, 3),
        "ranking_rationale": rec.ranking_rationale,
        "score_breakdown": rec.score_breakdown or {},
    }


def encode_context(ctx: PeptideContext) -> Dict:
    return {
        "sequence": ctx.sequence,
        "name": ctx.name,
        "length": len(ctx.sequence),
        "inferred_function": ctx.inferred_function,
        "confirmed_goal": ctx.confirmed_goal,
        "homolog_count": ctx.homolog_count,
        "conservation_available": ctx.conservation_available,
        "conservation_entropy": {str(k): round(v, 3) for k, v in (ctx.conservation_entropy or {}).items()},
        "data_notes": ctx.data_notes,
    }


# ---- request models --------------------------------------------------------

class InferRequest(BaseModel):
    input: str = Field(..., description="Peptide sequence or recognised name")


class OptimizeRequest(BaseModel):
    input: str = Field(..., description="Peptide sequence or recognised name")
    goal: Optional[str] = Field(None, description="Confirmed functional goal")
    ph: float = Field(7.4, ge=0.0, le=14.0)
    homologs: Optional[List[str]] = Field(
        None,
        description=(
            "Optional homologous sequences. Supplying 3 or more distinct sequences "
            "activates the per-position conservation term, which is otherwise "
            "reported as not computed."
        ),
    )


class FindRequest(BaseModel):
    query: str
    force_full: bool = False


# ---- endpoints -------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/goals")
def goals() -> Dict:
    """Goals the substitution predictor has a real scoring path for."""
    return {
        "goals": [
            {
                "id": "protease_resistance",
                "label": "Protease resistance",
                "quick_win": True,
                "description": (
                    "Highest-confidence lane. Scored by matching residues against documented "
                    "protease P1 specificities, so the evidence is a real motif match rather "
                    "than an estimate."
                ),
            },
            {
                "id": "binding_affinity",
                "label": "Binding affinity",
                "quick_win": False,
                "description": (
                    "Charge state (Henderson-Hasselbalch) and hydrophobicity change are computed, "
                    "but the DIRECTION of the effect on binding is not predicted without a "
                    "receptor structure. Magnitude reflects perturbation size only."
                ),
            },
            {
                "id": "generic_improvement",
                "label": "No goal specified",
                "quick_win": False,
                "description": (
                    "No primary benefit is claimed; ranking reflects off-target cost only."
                ),
            },
        ]
    }


@app.post("/api/infer-function")
def infer_function(req: InferRequest) -> Dict:
    """
    Step 1 of Workflow 1: infer the peptide's likely function and hand it back
    for confirmation. The scan deliberately does not run until the user confirms
    or overrides, per the spec.
    """
    raw = req.input.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Input is empty")

    sequence, name, parse_error = "", raw, None
    try:
        sequence, name = _peptides.load_sequence(raw)
    except ValueError as e:
        parse_error = str(e)

    inferred, confidence = _evidence.infer_function_from_name(name)

    return {
        "sequence": sequence,
        "name": name,
        "length": len(sequence),
        "parse_error": parse_error,
        "inferred_function": inferred,
        "inference_confidence": confidence,
        "properties": _peptides.basic_properties(sequence) if sequence else None,
        "prompt": (
            f"Based on the available records, you are seeking a form of: {inferred}. "
            f"Confirm or specify a different target function before proceeding."
        ),
    }


@app.post("/api/optimize")
def optimize(req: OptimizeRequest) -> Dict:
    """Workflow 1: run the substitution scan against a confirmed goal."""
    try:
        ctx, recs = _optimize.run(
            input_sequence_or_name=req.input,
            confirmed_goal=req.goal,
            auto_confirm=True,
            ph=req.ph,
            homologs=req.homologs,
        )
    except Exception as e:
        logger.exception("Optimize failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not ctx.sequence:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not parse '{req.input}' as a valid peptide sequence, and gene-name "
                f"resolution is not wired up in this build. Paste a raw amino acid sequence."
            ),
        )

    return {
        "context": encode_context(ctx),
        "ph": req.ph,
        "recommendations": [encode_recommendation(r) for r in recs],
        "scan_size": len(ctx.sequence) * 19,
    }


@app.post("/api/find-peptides")
def find_peptides(req: FindRequest) -> Dict:
    """Workflow 2: locate peptides associated with a functional goal."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        result = _find.run(req.query, force_full_workflow=req.force_full)
    except Exception as e:
        logger.exception("Find peptides failed")
        raise HTTPException(status_code=500, detail=str(e))

    payload = encode(result)
    # asdict() flattens enums to their values; restore tier names for display.
    for bucket in ("known_answers", "candidates"):
        for i, entry in enumerate(payload.get(bucket, [])):
            source = (getattr(result, bucket) or [])[i]
            entry["evidence_tier"] = source.evidence_tier.name
            entry["confidence"] = source.confidence.value
    return payload


@app.get("/api/calibration")
def calibration() -> Dict:
    """Prediction-vs-outcome log summary, for the Brier-score check."""
    return _scorer.get_calibration_summary()


# ---- static frontend -------------------------------------------------------

FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect width="16" height="16" rx="3" fill="#2a78d6"/>'
    '<path d="M4 11.5V4.5h3a2.2 2.2 0 0 1 0 4.4H5.6" stroke="#fff" '
    'stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>'
)


@app.get("/favicon.ico")
def favicon():
    return Response(content=FAVICON, media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
