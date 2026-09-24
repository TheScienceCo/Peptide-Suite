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
from peptide_suite.core.biological_context import BiologicalContext
from peptide_suite.core.biological_context import retrieve as retrieve_context
from peptide_suite.core.function_inference import FunctionInferencer
from peptide_suite.core.peptide_manager import PeptideManager
from peptide_suite.core.substitution_landscape import (
    DEFAULT_METRIC,
    METRICS,
    LandscapeError,
    build_landscape,
    encode_landscape,
)
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


def encode_classified_contact(c) -> Dict:
    """
    One classified native contact, with the evidence behind it.

    Citation and confidence travel with the classification rather than being
    available on request: a frozen residue footprint that cannot say what froze
    it is an assertion.
    """
    return {
        "contact": c.contact,
        "class": c.contact_class.value,
        "class_description": c.contact_class.description,
        "subclass": c.subclass.value if c.subclass else None,
        "also_classified": c.also_classified.value if c.also_classified else None,
        "residues": c.residues,
        "claim": c.claim,
        "citation": c.citation,
        "citation_precision": c.citation_precision,
        "independently_verified": c.independently_verified,
        "confidence": c.confidence,
        "verification_path": c.verification_path,
        "magnitude_source": c.magnitude_source,
        "notes": c.notes,
        "summary": c.summary(),
    }


def encode_partner_proposal(p) -> Dict:
    """A co-agent proposal. The mode and its shipping requirement travel together."""
    return {
        "primary": p.primary,
        "partner": p.partner,
        "mode": p.mode.value,
        "mode_description": p.mode.description,
        "shipping_requirement": p.mode.shipping_requirement,
        "rationale": p.rationale,
        "stoichiometry": p.stoichiometry,
        "engagement_order": p.engagement_order,
        "synergy_index": p.synergy_index,
        "accessory_protein": p.accessory_protein,
        "citation": p.citation,
        "citation_precision": p.citation_precision,
        "independently_verified": p.independently_verified,
        "confidence": p.confidence,
        "unmet_requirements": p.unmet_requirements(),
        "is_actionable": p.is_actionable,
        "notes": p.notes,
        "summary": p.summary(),
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


class LandscapeRequest(BaseModel):
    input: str = Field(..., description="Peptide sequence or recognised name")
    goal: Optional[str] = Field(None, description="Confirmed functional goal")
    metric: str = Field(DEFAULT_METRIC, description="Which computed quantity to colour by")
    ph: float = Field(7.4, ge=0.0, le=14.0)
    homologs: Optional[List[str]] = Field(
        None,
        description=(
            "Optional homologous sequences. The conservation metric is computable only "
            "when 3 or more distinct sequences are supplied; without them every cell of "
            "that metric is returned as not computed rather than as zero."
        ),
    )


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
    # Optional. When absent the sequence is identified against the local
    # reference set, because the native-contact rules key on the peptide's
    # identity and a caller should not have to know its name to get them.
    peptide_name: str = ""
    # Optional. Class B1 placement rules need to know which receptor is meant;
    # the reporting rule applies with or without it.
    receptor: str = ""


def encode_biological_context(context: BiologicalContext) -> Dict[str, Any]:
    """
    Serialise what is known about the peptide, before any score.

    Every section carries its own tier and source string rather than inheriting
    one from the object, because a record can hold a verified accession beside
    a curated domain layout and those are not the same claim. The interface
    renders the badge from these fields; it does not decide them.
    """
    sequence = context.mature_sequence
    return {
        "is_established": context.is_established,
        "summary": context.summary(),
        "name": context.name,
        "aliases": list(context.aliases),
        "gene": context.gene,
        "organism": context.organism,
        "uniprot": context.uniprot,
        "family": context.family,
        "form": context.form.value,
        "form_description": context.form.describe,
        "precursor_of": context.precursor_of,
        "mature_sequence": sequence,
        "mature_length": len(sequence),
        "sequence_matches_mature": context.sequence_matches_mature,
        "needs_verification": context.needs_verification,
        "function": context.function.encode() if context.function else None,
        "regions": [r.encode(sequence) for r in context.regions],
        "disulfides": [d.encode(sequence) for d in context.disulfides],
        "disulfide_inconsistencies": context.disulfide_inconsistencies(),
        "modifications": [m.encode() for m in context.modifications],
        "primary_receptors": [r.encode() for r in context.primary_receptors],
        "secondary_receptors": [r.encode() for r in context.secondary_receptors],
        "interfaces": [i.encode() for i in context.interfaces],
        # Rendered as its own state. "No interface annotation was retrieved" and
        # "the interface has no notable features" are different sentences and
        # an empty list alone reads as the second.
        "interface_available": bool(context.interfaces),
        "retrieval_notes": context.retrieval_notes,
        "unavailable_sources": context.unavailable_sources,
    }


def _inferred_name(sequence: str) -> str:
    """The peptide's name from identification, or empty if it was not recognised."""
    inference = _inferencer.infer(sequence)
    return inference.matched_name if inference.is_identification else ""


# ---- endpoints -------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict:
    return {"status": "ok", "version": app.version}


# The four kinds of reasoning that could bear on a binding question, and which
# of them this deployment can actually do. Presenting all four as one "binding
# affinity prediction" is the overstatement the label audit exists to fix: a
# sequence-derived perturbation size and a structure-supported interaction
# prediction are not the same claim and must not share a number.
BINDING_REASONING_KINDS = [
    {
        "kind": "sequence_derived",
        "label": "Sequence-derived perturbation",
        "available": True,
        "what_it_is": (
            "How large a physicochemical change the substitution is: charge at the "
            "requested pH by Henderson-Hasselbalch, and hydrophobicity by Kyte-Doolittle."),
        "what_it_is_not": (
            "Not a statement about whether the change helps or hurts binding. A large "
            "perturbation at a position that does not touch the receptor changes nothing."),
    },
    {
        "kind": "experimental_mutation",
        "label": "Experimentally measured mutation effect",
        "available": False,
        "what_it_is": (
            "A published measurement of this substitution in this peptide against this "
            "receptor."),
        "what_it_is_not": (
            "Not available: no empirical variant evidence store is populated in this "
            "build, and PubMed is unreachable from this environment."),
    },
    {
        "kind": "structure_supported",
        "label": "Structure-supported interaction prediction",
        "available": False,
        "what_it_is": (
            "What the substitution does to a resolved contact, from an experimental "
            "structure of the complex."),
        "what_it_is_not": (
            "Not available: the structure-template gate has supplied no interface "
            "structure, so no contact can be evaluated."),
    },
    {
        "kind": "learned_model",
        "label": "Learned model prediction",
        "available": False,
        "what_it_is": "A model trained on measured binding data.",
        "what_it_is_not": (
            "Not available: no model in this repository is trained on real labels."),
    },
]


@app.get("/api/goals")
def goals() -> Dict:
    """
    Goals the substitution predictor has a real scoring path for.

    Each goal declares what it computes and what it does not, structurally
    rather than in prose the interface is free to truncate. The labels were
    audited: "Binding affinity" implied a computed dissociation constant, and
    what is computed is the size of a physicochemical perturbation.
    """
    return {
        "binding_reasoning_kinds": BINDING_REASONING_KINDS,
        "goals": [
            {
                "id": "protease_resistance",
                "label": "Protease resistance (documented P1 motifs)",
                "short_label": "Protease resistance",
                "quick_win": True,
                "description": (
                    "Highest-confidence lane. Scored by matching residues against documented "
                    "protease P1 specificities, so the evidence is a real motif match rather "
                    "than an estimate."
                ),
                "computes": [
                    "Whether the wild-type residue matches a documented P1 specificity",
                    "Whether the mutant introduces a new P1 site",
                ],
                "does_not_compute": [
                    "kcat/KM, or any rate constant",
                    "Half-life in plasma or any other medium",
                    "Cleavage at sites outside the reference specificity set",
                ],
                "strongest_evidence": "DIRECT_EXPERIMENTAL",
            },
            {
                "id": "binding_affinity",
                "label": "Binding-interface perturbation (not a predicted Kd)",
                "short_label": "Binding perturbation",
                "quick_win": False,
                "description": (
                    "Charge state (Henderson-Hasselbalch) and hydrophobicity change are computed, "
                    "but the DIRECTION of the effect on binding is not predicted without a "
                    "receptor structure. Magnitude reflects perturbation size only."
                ),
                "computes": [
                    "Change in effective residue charge at the requested pH",
                    "Change in Kyte-Doolittle hydrophobicity",
                    "How large a perturbation those amount to",
                ],
                "does_not_compute": [
                    "Kd, Ki, IC50, EC50 or any affinity constant",
                    "Whether affinity goes up or down — the direction is not determined",
                    "Selectivity between a primary receptor and a cross-reactive one",
                ],
                "strongest_evidence": "BIOCHEMICAL_PRINCIPLE",
                "reasoning_kinds": BINDING_REASONING_KINDS,
            },
            {
                "id": "generic_improvement",
                "label": "No goal specified",
                "short_label": "No goal",
                "quick_win": False,
                "description": (
                    "No primary benefit is claimed; ranking reflects off-target cost only."
                ),
                "computes": ["Off-target cost only"],
                "does_not_compute": ["Any benefit — none is claimed without a goal"],
                "strongest_evidence": "INFERENCE_ONLY",
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

    # Known biology comes before prediction. The interface renders this section
    # above the engineering output, so a reader learns what the system thinks
    # the molecule is before seeing a number about it.
    context = retrieve_context(
        sequence,
        name=inference.matched_name or name,
        uniprot_client=_inferencer.uniprot,
    )

    return {
        "biological_context": encode_biological_context(context),
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


@app.get("/api/landscape-metrics")
def landscape_metrics() -> Dict:
    """The selectable quantities, with the encoding each one is entitled to."""
    return {
        "default": DEFAULT_METRIC,
        "metrics": [
            {
                "key": m.key,
                "label": m.label,
                "units": m.units,
                "encoding": m.encoding,
                "midpoint_meaning": m.midpoint_meaning,
                "description": m.description,
                "source": m.source,
            }
            for m in METRICS
        ],
    }


@app.post("/api/substitution-landscape")
def substitution_landscape(req: LandscapeRequest) -> Dict:
    """
    The full position x residue grid, rather than the top few of it.

    Runs the same scan as /api/optimize and skips the ranking cut. Nothing is
    recomputed per metric: one scan, re-projected.
    """
    try:
        ctx, recs = _optimize.scan(
            input_sequence_or_name=req.input,
            confirmed_goal=req.goal,
            auto_confirm=True,
            ph=req.ph,
            homologs=req.homologs,
        )
    except Exception as e:
        logger.exception("Landscape scan failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not ctx.sequence:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not parse '{req.input}' as a valid peptide sequence, and gene-name "
                f"resolution is not wired up in this build. Paste a raw amino acid sequence."
            ),
        )

    try:
        landscape = build_landscape(ctx, recs, metric_key=req.metric, ph=req.ph)
    except LandscapeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return stamp_policy({
        "landscape": encode_landscape(landscape),
        "context": encode_context(ctx),
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
        "nbo": ({
            "required": t.nbo_requirement.is_required,
            "outstanding": t.nbo_requirement.is_outstanding,
            "triggers": [x.value for x in t.nbo_requirement.triggers],
            "statement": t.nbo_requirement.statement(),
            "required_tooling": t.nbo_requirement.required_tooling,
        } if t.nbo_requirement else None),
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
            peptide_name=req.peptide_name or _inferred_name(sequence),
            receptor=req.receptor,
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
        "research_requests": result["research_requests"],
        "class_b1": result["class_b1"],
        "length_notes": result["length_notes"],
        "conformer_ensemble": {
            "status": result["conformer_ensemble"].status.value,
            "constraint": result["conformer_ensemble"].constraint.value,
            "sequence_length": result["conformer_ensemble"].sequence_length,
            "ceiling": result["conformer_ensemble"].ceiling,
            "reason": result["conformer_ensemble"].reason,
            "required_tooling": result["conformer_ensemble"].required_tooling,
            "notes": result["conformer_ensemble"].notes,
        },
        "feasibility": [
            {
                "code": f.code, "severity": f.severity.value, "positions": f.positions,
                "description": f.description, "remedy": f.remedy, "basis": f.basis,
                "summary": f.summary(),
            }
            for f in result["feasibility"]
        ],
        "partner_proposals": [encode_partner_proposal(p)
                              for p in result["partner_proposals"]],
        "classified_contacts": [encode_classified_contact(c)
                                for c in result["classified_contacts"]],
        "scaffold_opportunities": [encode_classified_contact(c)
                                   for c in result["scaffold_opportunities"]],
        "registry_is_empty": result["registry_is_empty"],
        "transformations": [encode_transformation(t) for t in result["transformations"]],
        "rejected": result["rejected"],
        "weights": result["weights"],
        "scalarization_note": result["scalarization_note"],
        "comparability_warning": result["comparability_warning"],
    })


@app.get("/api/parameterization")
def parameterization() -> Dict:
    """
    The parameterized-residue registry and the pipeline that would fill it.

    Exposed because the registry being empty is the reason most non-canonical
    proposals are research requests rather than recommendations, and that is
    better answered by a readable endpoint than inferred from an absence.
    """
    from peptide_suite.core.ncaa_registry import NCAARegistry, pipeline_specification
    return {
        "registry_is_empty": NCAARegistry.is_empty(),
        "parameterized_residues": sorted(NCAARegistry.usable()),
        # Separate from the usable list on purpose: a partial record is real
        # work and licenses nothing, and merging the two lists is exactly how a
        # half-parameterized residue ends up in a ranking.
        "in_progress_residues": {
            code: {
                "stages_completed": record.get("stages_completed", []),
                "stages_outstanding": [s.value for s in NCAARegistry.cost(code).stages_outstanding],
                "geometry_method": record.get("geometry_method", ""),
                "esp_rrms": record.get("esp_rrms"),
                "known_deficiencies": record.get("known_deficiencies", []),
                "validated_against": record.get("validated_against", ""),
            }
            for code in sorted(NCAARegistry.registry())
            if not NCAARegistry.status(code).permits_scoring
            for record in [NCAARegistry.registry()[code]]
        },
        "engines": __import__("peptide_suite.core.qm_engine",
                              fromlist=["engine_status"]).engine_status(),
        "pipeline": pipeline_specification(),
        "catalogue": [
            {
                "code": entry.code,
                "full_name": entry.full_name,
                "class": entry.residue_class,
                "status": NCAARegistry.status(entry.code).value,
                "absent_from": entry.absent_from,
                "torsion_scan_driver": entry.torsion_scan_driver,
                "note": entry.note,
            }
            for entry in sorted(NCAARegistry.catalogue().values(), key=lambda e: e.code)
        ],
        "note": (
            "An empty registry is the correct current state. Nothing here has been "
            "parameterized by this project, so every proposal using non-canonical "
            "chemistry is emitted as a research request rather than a recommendation."
        ),
    }


@app.get("/api/holdout")
def holdout() -> Dict:
    """
    The holdout protocols and why named-entity exclusion is not one of them.

    Exposed as an endpoint because the leakage argument is checkable from the
    data: for any drug in the golden set it lists which other drugs still carry
    each of its motifs after a name-based exclusion.
    """
    from peptide_suite.core.holdout import GoldenSet, Protocol
    return {
        "protocols": [
            {"name": p.value, "is_clean_holdout": p.is_clean_holdout,
             "description": p.description}
            for p in Protocol
        ],
        "golden_set": sorted(GoldenSet.drugs()),
        "named_entity_leakage": {
            drug: GoldenSet.named_entity_leakage(drug)
            for drug in sorted(GoldenSet.drugs())
        },
        "decoys": GoldenSet.decoys(),
        "reporting_contract": (
            "Never report a hit as binary. Report the rank of the true modification "
            "within the full proposal list and the count of proposals ranked above it, "
            "with the decoy false-positive rate alongside."
        ),
    }


@app.get("/api/ml/status")
def ml_status() -> Dict:
    """
    What the ML layer can and cannot do here.

    Separate from the analysis endpoints and clearly labelled, because ML
    predictions are a different provenance category from everything else this
    server returns and must not be read as the same kind of number.
    """
    try:
        from ml.datasets.registry import TASKS, trainable_tasks
        from ml.embeddings.encoder import available_encoders
    except ImportError as e:
        return {"available": False,
                "reason": f"The ML layer requires torch, which is not installed ({e}). "
                          f"The peptide analysis pipeline does not depend on it."}
    return {
        "available": True,
        "encoders": available_encoders(),
        "trainable_tasks": trainable_tasks(),
        "tasks": {
            name: {"status": spec.status.value, "kind": spec.kind.value,
                   "source": spec.source, "licence": spec.licence,
                   "guidance": spec.claim_guidance()}
            for name, spec in sorted(TASKS.items())
        },
        "note": ("No model is trained here: the labelled datasets are unreachable from "
                 "this environment. Nothing is backed by invented labels."),
    }


@app.get("/api/ml/split-comparison")
def ml_split_comparison(n_families: int = 40, per_family: int = 8,
                        seed: int = 0) -> Dict:
    """
    Random versus sequence-clustered evaluation, run live.

    The experiment is synthetic by design: the claim is about the evaluation
    protocol, not about biology, so it is settled on constructed sequences where
    the label is a known function of the input. Running it live rather than
    serving a stored number means the table cannot drift from the code.
    """
    try:
        from ml.experiments.split_gap import run
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"ML layer unavailable: {e}")

    n_families = max(4, min(n_families, 80))
    per_family = max(2, min(per_family, 16))
    result = run(n_families=n_families, per_family=per_family, seed=seed)
    return {
        "n_sequences": result.n_sequences,
        "n_clusters": result.n_clusters,
        "identity_threshold": result.threshold,
        "leakage": result.leakage,
        "arms": [
            {"arm": arm.name, "split": arm.split,
             "roc_auc": arm.metrics["roc_auc"].value,
             "ci_low": arm.metrics["roc_auc"].ci_low,
             "ci_high": arm.metrics["roc_auc"].ci_high,
             "n": arm.metrics["roc_auc"].n}
            for arm in result.arms
        ],
        "gaps": {name: result.gap_for(name)
                 for name in sorted({a.name for a in result.arms})},
        "skipped": result.skipped,
        "is_synthetic": True,
        "claim_guidance": ("This demonstrates a fact about evaluation protocol and "
                           "supports no biological claim, whatever the numbers say."),
        "report": result.report(),
    }


class RepresentationRequest(BaseModel):
    sequence: str = Field(..., description="Reference peptide sequence")
    encoder: str = Field(
        "deterministic-positional-onehot",
        description=(
            "Which encoder produces the vectors. Mixing encoders is refused. The default "
            "is the positional encoder because the composition one collapses distinct "
            "sequences onto each other -- roughly half of a single-substitution scan "
            "lands on top of something else -- and a plot that silently merges half its "
            "points is worse than one whose components carry little. The composition "
            "encoder remains selectable and concentrates far more variance into two "
            "dimensions; both limitations ride with the response."
        ),
    )
    candidates: Optional[List[str]] = Field(
        None,
        description=(
            "Extra sequences to place in the same space — multi-substitution designs, "
            "related peptides. Anything not the same length as the reference is "
            "reported as unrelated rather than given a substitution count it does "
            "not have."
        ),
    )
    max_points: int = Field(1200, ge=8, le=3000)


@app.post("/api/ml/representation")
def ml_representation(req: RepresentationRequest) -> Dict:
    """
    The reference and its single substitutions projected into two dimensions.

    Kept under /api/ml because a representation is a different provenance
    category from a computed physical quantity, and the response says which
    encoder made it and how much of the variation the picture actually carries.
    """
    try:
        from ml.embeddings.encoder import (
            DeterministicEncoder, EncoderUnavailable, ESM2Encoder, PositionalOneHotEncoder,
        )
        from ml.embeddings.explorer import (
            ProjectionError, project, single_substitution_variants,
        )
        from ml.explain.neighbours import NeighbourError, out_of_distribution
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"The ML layer is not installed ({e}).")

    sequence = "".join(req.sequence.split()).upper()
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sequence):
        raise HTTPException(
            status_code=400,
            detail="Give a reference sequence in one-letter codes; this projection has "
                   "no encoder for non-canonical residues.",
        )

    encoders = {
        DeterministicEncoder.model: lambda: DeterministicEncoder(max_length=max(len(sequence), 1)),
        PositionalOneHotEncoder.model: lambda: PositionalOneHotEncoder(max_length=max(len(sequence), 1)),
        ESM2Encoder().model: ESM2Encoder,
    }
    if req.encoder not in encoders:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown encoder '{req.encoder}'. Available: {', '.join(sorted(encoders))}",
        )
    encoder = encoders[req.encoder]()

    pairs = [("wild type", sequence)] + single_substitution_variants(sequence)
    for candidate in (req.candidates or []):
        clean = "".join(candidate.split()).upper()
        if clean:
            pairs.append((f"candidate: {clean[:12]}…" if len(clean) > 12 else f"candidate: {clean}",
                          clean))
    truncated = len(pairs) > req.max_points
    if truncated:
        # Keep the reference and any supplied candidates; the single-substitution
        # cloud is what gets cut, and the response says so.
        keep = [pairs[0]] + [p for p in pairs if p[0].startswith("candidate:")]
        room = req.max_points - len(keep)
        pairs = keep + [p for p in pairs[1:] if not p[0].startswith("candidate:")][:max(room, 0)]

    try:
        embeddings = [encoder.encode(seq) for _, seq in pairs]
    except EncoderUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        projection = project(embeddings, [label for label, _ in pairs], sequence)
    except ProjectionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Where a supplied candidate sits relative to the substitution cloud, in the
    # cloud's own terms. Only for candidates: asking whether a single
    # substitution of the reference is out of distribution relative to the other
    # single substitutions answers itself.
    cloud = [(label, emb) for (label, _), emb in zip(pairs, embeddings)
             if not label.startswith("candidate:")]
    ood: List[Dict] = []
    for (label, _), embedding in zip(pairs, embeddings):
        if not label.startswith("candidate:"):
            continue
        try:
            report = out_of_distribution(
                embedding,
                [e for _, e in cloud],
                [l for l, _ in cloud],
                query_label=label,
            )
        except NeighbourError as e:
            ood.append({"label": label, "available": False, "reason": str(e)})
            continue
        ood.append({
            "label": label,
            "available": True,
            "distance_to_nearest": report.distance_to_nearest,
            "reference_median_nn_distance": report.reference_median_nn_distance,
            "percentile": report.percentile,
            "is_outside": report.is_outside,
            "n_coincident": report.n_coincident,
            "verdict": report.verdict,
            "caveat": report.caveat,
            "nearest": [
                {"label": n.label, "sequence": n.sequence,
                 "distance": n.distance, "rank": n.rank}
                for n in report.nearest
            ],
        })

    return {
        "reference": sequence,
        "ood": ood,
        "encoder": {
            "model": projection.encoder_model,
            "version": projection.encoder_version,
            "kind": projection.encoder_kind,
            "has_learned_content": projection.encoder_has_learned_content,
            "input_dim": projection.input_dim,
        },
        "distinct_positions": projection.distinct_positions,
        "distinct_projected": projection.distinct_projected,
        "explained_variance_ratio": projection.explained_variance_ratio,
        "cumulative_explained": projection.cumulative_explained,
        "interpretation": projection.interpretation,
        "warnings": projection.warnings,
        "truncated": truncated,
        "points": [
            {"label": p.label, "sequence": p.sequence, "x": p.x, "y": p.y,
             "class": p.variant_class.value, "n_substitutions": p.n_substitutions}
            for p in projection.points
        ],
    }


@app.get("/api/ml/fusion-benefit")
def ml_fusion_benefit(n_families: int = 60, per_family: int = 8,
                      n_permutations: int = 6, seed: int = 0) -> Dict:
    """
    Single modality vs naive concatenation vs learned fusion, run live.

    Reports the verdict rule's answer rather than the winning number: arms whose
    bootstrap intervals overlap have not been separated by this test, and a
    comparison that cannot clear a shuffled-label null has not compared
    anything.
    """
    try:
        from ml.experiments.fusion_benefit import run
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"The ML layer is not installed ({e}).")

    try:
        result = run(n_families=n_families, per_family=per_family,
                     n_permutations=n_permutations, seed=seed)
    except Exception as e:
        logger.exception("Fusion comparison failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "modalities": result.modalities,
        "split": result.split,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "arms": [
            {
                "name": arm.name,
                "kind": arm.kind,
                "roc_auc": arm.metrics["roc_auc"].value,
                "ci_low": arm.metrics["roc_auc"].ci_low,
                "ci_high": arm.metrics["roc_auc"].ci_high,
                "n": arm.metrics["roc_auc"].n,
                "gate_share": arm.gate_share,
            }
            for arm in result.arms
        ],
        "permutation_null": result.permutation_null,
        "resolution": result.resolution,
        "verdict": result.verdict,
        "skipped": result.skipped,
        "is_synthetic": True,
        "claim_guidance": ("This demonstrates that the comparison can be run and read, and "
                           "supports no biological claim whatever the numbers say."),
        "report": result.report(),
    }


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
