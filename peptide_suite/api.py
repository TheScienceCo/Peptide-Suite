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

from peptide_suite.runtime import load_active_policy
from peptide_suite.core import PeptideContext, SubstitutionRecommendation, evidence_weight
from peptide_suite.core.confidence_scoring import ConfidenceScorer
from peptide_suite.core.evidence_retrieval import EvidenceRetriever
from peptide_suite.core.function_inference import FunctionInferencer
from peptide_suite.core.peptide_manager import PeptideManager
from peptide_suite.workflows.find_peptides import FindPeptidesWorkflow
from peptide_suite.workflows.optimize import OptimizeWorkflow
from peptide_suite.workflows.transform import TransformWorkflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Load the policy before the app object exists, so a misconfigured deployment
# refuses to start rather than serving requests that fail one at a time. The
# server has no usable behaviour without coefficients, so failing here is the
# whole point; there is nothing to degrade to.
_policy = load_active_policy()
logger.info("Loaded %s", _policy.banner())
if _policy.is_demonstration:
    logger.warning(
        "Running on a DEMONSTRATION policy pack. Numbers this server returns show the "
        "shape of a result and carry no claim about magnitude. %d of %d thresholds are "
        "placeholders.",
        len(_policy.placeholder_thresholds), _policy.n_thresholds,
    )

app = FastAPI(
    title="Peptide Suite",
    description="Substitution scanning and peptide discovery with evidence-tiered confidence",
    version="1.1.0",
)

_optimize = OptimizeWorkflow()
_find = FindPeptidesWorkflow()
_peptides = PeptideManager()
_evidence = EvidenceRetriever()
_inferencer = FunctionInferencer()
_transform = TransformWorkflow()
_scorer = ConfidenceScorer()


# ---- serialization ---------------------------------------------------------

def stamp_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach the policy provenance to a scored response.

    A score is not interpretable without knowing what weighted it. The banner in
    the server log is not visible to a caller, and a caller that received a
    number has already been given something it may act on, so the provenance
    travels with the number rather than being available on request.
    """
    payload["policy"] = {
        "policy_version": _policy.policy_version,
        "pack_kind": _policy.pack_kind,
        "digest": _policy.digest[:12],
        "is_demonstration": _policy.is_demonstration,
        "n_placeholder_thresholds": len(_policy.placeholder_thresholds),
        "n_thresholds": _policy.n_thresholds,
    }
    if _policy.is_demonstration:
        payload["policy"]["warning"] = (
            "Scores below were weighted by a demonstration policy pack. No weight in it "
            "was fitted to data. The ordering and the shape of the output are real; the "
            "magnitudes are not a claim."
        )
    return payload

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


def encode_claim(claim) -> Optional[Dict]:
    if claim is None:
        return None
    return {
        "type": claim.claim_type.value,
        "text": claim.text,
        "method": claim.method,
        "tier": claim.tier,
        "citations": claim.citations,
        "inference_step": claim.inference_step,
        "experimentally_tested": claim.experimentally_tested,
        "rendered": claim.render(),
    }


def encode_effect(effect) -> Dict:
    """Effects need their enum names preserved for display, not just values."""
    if effect is None:
        return None
    return {
        "category": effect.category,
        "description": effect.description,
        "evidence_tier": effect.evidence_tier.name,
        "evidence_weight": evidence_weight(effect.evidence_tier),
        "confidence": effect.confidence.value,
        "score": round(effect.score, 3),
        "magnitude": round(effect.magnitude, 3),
        "contribution": round(effect.score * effect.magnitude, 3),
        "reasoning": effect.reasoning,
        "equation_refs": effect.equation_refs or [],
        "sources": effect.sources or [],
    }


def encode_electrostatics(profile) -> Dict:
    """
    Per-residue titration across the compartment series.

    Sent as a table rather than a single pI: pI says where net charge crosses
    zero and nothing about which residue carries it, which is the part a
    substitution decision turns on.
    """
    return {
        "compartments": profile.compartments,
        "net_charge": {k: round(v, 3) for k, v in profile.net_charge.items()},
        "terminal_charges": {
            label: {k: round(v, 3) for k, v in charges.items()}
            for label, charges in profile.terminal_charges.items()
        },
        "switch_threshold": round(profile.switch_threshold, 3),
        "summary": profile.summary(),
        "notes": profile.notes,
        "titrations": [
            {
                "position": t.position,
                "display_position": t.display_position,
                "residue": t.residue,
                "pka": t.pka,
                "group": t.group,
                "cited_range": t.cited_range,
                "context_dependent": t.context_dependent,
                "protonation_swing": round(t.protonation_swing, 3),
                "charge_swing": round(t.charge_swing, 3),
                "is_switchable": t in profile.switchable,
                "note": t.note,
                "uncertainty_note": t.uncertainty_note(profile.switch_threshold),
                "points": [
                    {
                        "compartment": pt.compartment,
                        "ph": pt.ph,
                        "protonated_fraction": round(pt.protonated_fraction, 4),
                        "effective_charge": round(pt.effective_charge, 4),
                    }
                    for pt in t.points
                ],
            }
            for t in profile.titrations
        ],
    }


def encode_structure_template(decision) -> Dict:
    """
    Which structure conformational claims rest on, or why none was admitted.

    The rejections travel with the decision. A refusal that cannot say what it
    looked at is indistinguishable from a system that never looked.
    """
    return {
        "tier": decision.tier.value,
        "tier_label": decision.tier.label,
        "is_refusal": decision.is_refusal,
        "permits_conformational_claims": decision.permits_conformational_claims,
        "template": (
            {
                "identifier": decision.template.identifier,
                "is_experimental": decision.template.is_experimental,
                "same_peptide": decision.template.same_peptide,
                "homolog_identity": decision.template.homolog_identity,
                "source": decision.template.source,
            }
            if decision.template else None
        ),
        "considered_count": len(decision.considered),
        "summary": decision.summary(),
        "detail": decision.detail_lines(),
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
        "homolog_source": ctx.homolog_source,
        "homolog_source_detail": ctx.homolog_source_detail,
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


class TransformRequest(BaseModel):
    sequence: str
    formulation_ph: float = Field(7.4, ge=0.0, le=14.0)
    is_internal_fragment: bool = True


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
    resolved_from_name = ""
    also_readable_as_sequence = ""

    # Names are resolved BEFORE sequence parsing, because many peptide and
    # protein names are themselves valid amino acid strings: "laminin" is
    # L-A-M-I-N-I-N and "oxytocin" is O-X-Y-T-O-C-I-N. Parsing first would
    # silently analyse a 7-residue peptide nobody asked about.
    by_name = _inferencer.resolve_name(raw)
    if by_name:
        matched_name, entry = by_name
        sequence, name = entry["sequence"], matched_name
        resolved_from_name = (
            f"Interpreted '{raw.strip()}' as a name and resolved it to {matched_name} "
            f"({len(sequence)} residues)."
        )
        try:
            literal, _ = _peptides.load_sequence(raw)
            if literal and literal != sequence:
                also_readable_as_sequence = literal
        except ValueError:
            pass

    if not sequence:
        try:
            sequence, name = _peptides.load_sequence(raw)
        except ValueError as e:
            parse_error = str(e)

    protein_by_name = _inferencer.lookup_protein_by_name(raw)
    if protein_by_name and not by_name:
        literal = sequence
        return {
            "sequence": "", "name": raw.strip(), "length": 0, "parse_error": None,
            "inferred_function": "", "inference_confidence": 0.0,
            "suggested_goal": None, "properties": None,
            "protein_match": protein_by_name,
            "also_readable_as_sequence": literal,
        }

    if not sequence:
        # Still not a sequence. It may name a protein the ontology knows about,
        # in which case the answer is what is known plus the analysable motifs
        # derived from it — not a dead end.
        protein = _inferencer.lookup_protein_by_name(raw)
        return {
            "sequence": "", "name": name, "length": 0, "parse_error": parse_error,
            "inferred_function": "", "inference_confidence": 0.0,
            "suggested_goal": None, "properties": None,
            "protein_match": protein,
        }

    _cleaned, sequence_notes = _peptides.clean_sequence(raw)
    length_class = _peptides.classify_length(sequence)
    inference = _inferencer.infer(sequence, name=name, raw_input=raw)

    rec = inference.uniprot
    return {
        "sequence": sequence,
        "name": name,
        "length": len(sequence),
        "parse_error": parse_error,
        "sequence_notes": ([resolved_from_name] if resolved_from_name else []) + sequence_notes,
        "resolved_from_name": resolved_from_name,
        "also_readable_as_sequence": also_readable_as_sequence,
        "is_protein": length_class["is_protein"],
        "length_note": length_class["note"],
        "uniprot": (
            {
                "accession": rec.accession,
                "entry_name": rec.entry_name,
                "protein_name": rec.protein_name,
                "gene": rec.gene,
                "organism": rec.organism,
                "full_length": rec.length,
                "subcellular_location": rec.subcellular_location,
                "keywords": rec.keywords[:12],
                "url": rec.url,
            }
            if rec else None
        ),
        "inferred_function": inference.inferred_function,
        "inference_confidence": round(inference.confidence, 2),
        "inference_level": inference.level,
        "inference_basis": inference.basis,
        "is_identification": inference.is_identification,
        "matched_name": inference.matched_name,
        "parent_protein": inference.parent_protein,
        "native_context_note": inference.native_context_note,
        "suggested_goal": inference.suggested_goal,
        "suggested_goal_reason": inference.goal_reason,
        "alternatives": inference.alternatives,
        "caveats": inference.caveats,
        "claim": encode_claim(inference.claim),
        "properties": _peptides.basic_properties(sequence),
        "prompt": (
            f"Based on the sequence, you are seeking a form of: {inference.inferred_function} "
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

    return stamp_policy({
        "context": encode_context(ctx),
        "ph": req.ph,
        "recommendations": [encode_recommendation(r) for r in recs],
        "scan_size": len(ctx.sequence) * 19,
    })


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

    payload = stamp_policy(encode(result))
    # asdict() flattens enums to their values; restore tier names for display.
    for bucket in ("known_answers", "candidates"):
        for i, entry in enumerate(payload.get(bucket, [])):
            source = (getattr(result, bucket) or [])[i]
            entry["evidence_tier"] = source.evidence_tier.name
            entry["confidence"] = source.confidence.value
    return payload


def encode_transformation(t) -> Dict:
    scal = t.scalarize()
    return {
        "move": t.move.value,
        "position": t.position,
        "display_position": t.display_position,
        "description": t.description,
        "rationale": t.rationale,
        "evidence_tier": t.evidence_tier,
        "physics_tier_reached": t.physics_tier_reached,
        "tradeoff_label": t.tradeoff_label,
        "notes": t.notes,
        "scalarized": scal,
        "objective_deltas": [
            {
                "objective": d.objective.value,
                "direction": d.direction.value,
                "assessed": d.assessed,
                "magnitude": d.magnitude,
                "signed": d.signed,
                "basis": d.basis,
                "claim": encode_claim(d.claim),
            }
            for d in t.objective_deltas
        ],
        "preorganization": (
            {
                "helicity_delta": t.preorganization.helicity_delta,
                "basin_restriction": t.preorganization.basin_restriction,
                "rmsf_change": t.preorganization.rmsf_change,
                "mechanism": t.preorganization.mechanism,
                "rendered": t.preorganization.render(),
                "claim": encode_claim(t.preorganization.claim),
            }
            if t.preorganization else None
        ),
        "assumptions": [
            {"kind": a.kind.value, "statement": a.statement, "impact_if_wrong": a.impact_if_wrong}
            for a in t.assumptions
        ],
        "leakage_flag": (
            {
                "scaffold": t.leakage_flag.scaffold,
                "marketed_analogs": t.leakage_flag.marketed_analogs,
                "matching_modification": t.leakage_flag.matching_modification,
                "note": t.leakage_flag.note,
            }
            if t.leakage_flag else None
        ),
    }


def encode_physics(physics: Dict) -> Dict:
    t0 = physics["tier0"].data
    tiers = []
    for key in ("tier0", "tier1", "tier2", "tier3", "tier4"):
        tr = physics.get(key)
        if tr is None:
            tiers.append({
                "tier": int(key[-1]), "available": False,
                "unavailable_reason": physics.get("tier3_blocked_reason", "")
                if key == "tier3" else "Not applicable to this peptide.",
                "missing_dependency": "",
            })
            continue
        tiers.append({
            "tier": tr.tier,
            "available": tr.available,
            "unavailable_reason": tr.unavailable_reason,
            "missing_dependency": tr.missing_dependency,
            "illustrative_only": tr.illustrative_only,
        })

    return {
        "tier_reached": physics["tier_reached"],
        "summary": physics["summary"],
        "tiers": tiers,
        "hydrophobic_moment": t0["hydrophobic_moment"],
        "windowed_hydrophobic_moment": t0["windowed_hydrophobic_moment"],
        "mean_hydrophobicity": t0["mean_hydrophobicity"],
        "isoelectric_point": t0["isoelectric_point"],
        "net_charge_at_formulation_ph": t0["net_charge_at_formulation_ph"],
        "formulation_ph": t0["formulation_ph"],
        "charge_vs_ph": t0["charge_vs_ph"],
        "helical_wheel": t0["helical_wheel"],
        "liabilities": [
            {
                "motif": l.motif,
                "position": l.position,
                "display_position": l.display_position,
                "class": l.liability_class.value,
                "mechanism": l.mechanism,
                "severity": l.severity,
                "mitigation": l.mitigation,
            }
            for l in t0["liabilities"]
        ],
        "pka_residues": physics["tier1"].data.get("residues", []),
        "pocket_perturbation_applied": physics["tier1"].data.get("pocket_perturbation_applied", False),
    }


def encode_native_context(nc) -> Dict:
    ex = nc.excision
    return {
        "retrieval_status": nc.retrieval_status,
        "structure_status": nc.structure_status,
        "data_notes": nc.data_notes,
        "excision": (
            {
                "is_internal_fragment": ex.is_internal_fragment,
                "n_terminal_artifact": ex.n_terminal_artifact,
                "c_terminal_artifact": ex.c_terminal_artifact,
                "recommended_n_cap": ex.recommended_n_cap,
                "recommended_c_cap": ex.recommended_c_cap,
                "priority_note": ex.priority_note,
            }
            if ex else None
        ),
        "contacts": [
            {
                "partner": c.partner_description,
                "class": c.contact_class.value,
                "residues": c.residues_involved,
                "basis": c.classification_basis,
                "response": c.recommended_response,
            }
            for c in nc.contacts
        ],
        "partners": [
            {
                "name": p.partner_name,
                "class": p.partner_class.value,
                "rationale": p.rationale,
                "stoichiometry": p.stoichiometry,
                "engagement_order": p.engagement_order,
                "covalent_tether_assessment": p.covalent_tether_assessment,
                "evidence_required": p.evidence_required,
            }
            for p in nc.partners
        ],
        "assumptions": [
            {"kind": a.kind.value, "statement": a.statement, "impact_if_wrong": a.impact_if_wrong}
            for a in nc.assumptions
        ],
    }


@app.post("/api/transform")
def transform(req: TransformRequest) -> Dict:
    """
    Generate ranked discrete transformations with the full objective vector.

    Emits transformations, never an optimised sequence: a sequence hides which
    moves were made and why.
    """
    try:
        sequence, _ = _peptides.load_sequence(req.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = _transform.run(
            sequence,
            formulation_ph=req.formulation_ph,
            is_internal_fragment=req.is_internal_fragment,
        )
    except Exception as e:
        logger.exception("Transform failed")
        raise HTTPException(status_code=500, detail=str(e))

    return stamp_policy({
        "sequence": result["sequence"],
        "physics": encode_physics(result["physics"]),
        "native_context": encode_native_context(result["native_context"]),
        "electrostatics": encode_electrostatics(result["electrostatics"]),
        "structure_template": encode_structure_template(result["structure_template"]),
        "transformations": [encode_transformation(t) for t in result["transformations"]],
        "rejected": result["rejected"],
        "weights": result["weights"],
        "scalarization_note": result["scalarization_note"],
        "comparability_warning": result["comparability_warning"],
    })


@app.get("/api/calibration")
def calibration() -> Dict:
    """Prediction-vs-outcome log summary, for the Brier-score check."""
    return _scorer.get_calibration_summary()


@app.get("/api/policy")
def policy_info():
    """
    What produced the numbers this server returns.

    Exposed because a caller cannot interpret a score without knowing whether it
    came from a fitted policy or a demonstration pack, and a banner printed once
    into a server log is not available to the caller.
    """
    return {
        "policy_version": _policy.policy_version,
        "pack_kind": _policy.pack_kind,
        "digest": _policy.digest,
        "is_demonstration": _policy.is_demonstration,
        "free_parameters": _policy.free_parameters,
        "placeholder_thresholds": _policy.placeholder_thresholds,
        "banner": _policy.banner(),
        "derivation": _policy.provenance.get("derivation", ""),
    }


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
