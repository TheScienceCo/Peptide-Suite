"""
Blind recovery: does the scan propose what was actually made?  [Phase 4]

The benchmark this repository was missing. Take a peptide that was really
engineered, hide the engineered form, feed the parent in, and ask where the
modification that was actually made and validated lands in the proposal list.

Built on `holdout.py`'s protocols and its reporting contract, which already
settled the two arguments that decide whether such a number means anything.

WHY NOT BLIND ON THE NAME. The obvious protocol -- "ignore every mention of
semaglutide" -- is the one `holdout.Protocol.NAMED_ENTITY` exists to mark as
insufficient. Removing semaglutide leaves liraglutide, which carries Arg34 and
gamma-Glu-linked acylation at Lys26, and taspoglutide, which is [Aib8,
Aib35]-GLP-1. The union of those two is semaglutide's answer. A named-entity
run would score beautifully with the answer sitting in the corpus under two
other names, and `leakage_after_blinding` prints exactly which ones.

So: MOTIF_ABLATION removes every molecule carrying the chemistry, whatever it
is called. TEMPORAL freezes the corpus at a year and requires the modification
to have been first published after it.

THREE THINGS THAT MAKE A RECOVERY NUMBER HONEST, all of which this reports.

  Whether the answer was even proposable. The scan enumerates canonical single
  substitutions. Aib8 is a non-canonical residue and a C18 diacid is not a
  substitution at all, so for those the true answer is NOT IN THE CANDIDATE
  SPACE and no amount of cleverness could have found it. That is reported as
  unrecoverable-by-this-method, never as a miss -- scoring it as a miss
  understates the method, and quietly dropping it overstates it.

  The rank, with the total. `RankedOutcome` refuses a bare hit, because 2nd of
  3 and 2nd of 47 are different results.

  What rank chance alone produces. On a 31-residue peptide there are 589
  candidates, so "in the top 10" sounds impressive and is 1.7% of the list.
  The permutation null below is the only thing that makes the rank a claim
  rather than a number.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .holdout import GoldenSet, Protocol, RankedOutcome

#: Modifications the substitution scan could in principle propose. It
#: enumerates single canonical substitutions, so anything else -- a
#: non-canonical residue, an attached chain, a cyclisation, a truncation -- is
#: outside the candidate space by construction rather than by failure.
_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

#: `Arg34`, `A8G`, `Aib8`, `[Aib8, Aib35]`. Captures a residue token and a
#: position; whether the residue is canonical is decided separately.
_POINT_CHANGE = re.compile(r"\b([A-Z][a-z]{0,2})\s*(\d+)\b")


class CandidateSpace:
    """Whether a stated modification is one the scan could ever propose."""

    IN = "IN_CANDIDATE_SPACE"
    NON_CANONICAL = "NON_CANONICAL_RESIDUE"
    NOT_A_SUBSTITUTION = "NOT_A_SUBSTITUTION"
    UNPARSEABLE = "UNPARSEABLE"
    #: Read fine, but could not be placed in the stored sequence -- usually a
    #: missing numbering offset. Kept apart from UNPARSEABLE because they need
    #: different fixes: one is a modification this method cannot express, the
    #: other is a modification it could score if the data said where it goes.
    UNMAPPABLE = "UNMAPPABLE"

    @staticmethod
    def classify(modification: str) -> Tuple[str, Optional[int], str]:
        """
        Returns (verdict, 1-indexed position, residue) for a modification
        string as the golden set writes it.

        Deliberately conservative: anything it cannot read confidently is
        UNPARSEABLE rather than guessed at, because a misparse here silently
        scores the wrong substitution and the run still produces a number.
        """
        text = (modification or "").strip()
        if not text:
            return CandidateSpace.UNPARSEABLE, None, ""

        lowered = text.lower()
        if any(word in lowered for word in
               ("diacid", "palmitoyl", "acylation", "peg", "lipid", "linker",
                "oeg", "gamma-glu", "cyclis", "cycliz", "staple", "truncat",
                "amide", "acetyl", "myristoyl")):
            return CandidateSpace.NOT_A_SUBSTITUTION, None, text

        match = _POINT_CHANGE.search(text)
        if not match:
            return CandidateSpace.UNPARSEABLE, None, text
        residue, position = match.group(1), int(match.group(2))
        if len(residue) == 1 and residue in _CANONICAL:
            return CandidateSpace.IN, position, residue
        # Three-letter canonical names -- Arg34, Lys26 -- are still canonical.
        three = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
                 "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
                 "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
                 "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
        if residue in three:
            return CandidateSpace.IN, position, three[residue]
        return CandidateSpace.NON_CANONICAL, position, residue


@dataclass(frozen=True)
class Blinding:
    """
    What was removed before the scan ran, and whether it was enough.

    `verified` is the field that matters. A blinding that was requested and not
    actually applied produces the best-looking result in the whole benchmark,
    so the run refuses to report a rank unless the removal was checked against
    the data the scan actually reads.
    """
    protocol: Protocol
    ablated_motifs: Tuple[str, ...] = ()
    cutoff_year: Optional[int] = None
    removed_drugs: Tuple[str, ...] = ()
    residual_leakage: Dict[str, List[str]] = field(default_factory=dict)
    verified: bool = False
    note: str = ""

    @property
    def is_clean(self) -> bool:
        return self.verified and not self.residual_leakage

    def describe(self) -> str:
        if self.protocol is Protocol.MOTIF_ABLATION:
            head = (f"Ablated motif(s) {', '.join(self.ablated_motifs)}, removing "
                    f"{len(self.removed_drugs)} molecule(s): "
                    f"{', '.join(self.removed_drugs) or 'none'}.")
        elif self.protocol is Protocol.TEMPORAL:
            head = f"Corpus frozen at {self.cutoff_year}."
        else:
            head = f"Protocol {self.protocol.value}."
        if self.residual_leakage:
            head += (" RESIDUAL LEAKAGE: "
                     + "; ".join(f"{m} still present in {', '.join(d)}"
                                 for m, d in sorted(self.residual_leakage.items())))
        if not self.verified:
            head += " NOT VERIFIED — no rank may be reported from this run."
        return head + (f" {self.note}" if self.note else "")


def blind_by_motif(target: str, motifs: Sequence[str] = ()) -> Blinding:
    """
    Remove every molecule carrying the target's chemistry, not just its name.

    The removal is then checked: any motif still present in a molecule that
    survived is reported as residual leakage, and the blinding is not clean.
    """
    motifs = tuple(motifs) if motifs else tuple(GoldenSet.motifs_of(target))
    if not motifs:
        return Blinding(protocol=Protocol.MOTIF_ABLATION, verified=False,
                        note=f"'{target}' carries no recorded motifs, so there is "
                             f"nothing to ablate and nothing to hold out.")

    removed = sorted({d for motif in motifs for d in GoldenSet.drugs_carrying(motif)})
    survivors = [d for d in GoldenSet.drugs() if d not in removed]
    residual = {}
    for motif in motifs:
        still = [d for d in survivors if motif in GoldenSet.motifs_of(d)]
        if still:
            residual[motif] = still

    return Blinding(
        protocol=Protocol.MOTIF_ABLATION, ablated_motifs=motifs,
        removed_drugs=tuple(removed), residual_leakage=residual, verified=True,
        note=("Every molecule carrying the chemistry was removed, not only the one "
              "carrying the name."))


def blind_by_date(target: str, cutoff_year: Optional[int] = None) -> Blinding:
    """
    Freeze the corpus at a year. Refuses when the year is not recorded.

    A temporal holdout's entire claim is that the modification was first
    published after the cutoff. Without a first-publication year for the target
    there is no claim, and inventing a plausible year would produce a benchmark
    that looks rigorous and tests nothing.
    """
    entry = GoldenSet.drugs().get(target, {})
    first = entry.get("first_published")
    if first is None:
        return Blinding(
            protocol=Protocol.TEMPORAL, cutoff_year=cutoff_year, verified=False,
            note=(f"No first-publication year is recorded for '{target}', so a temporal "
                  f"holdout cannot be run. Add `first_published` to its golden-set "
                  f"entry. A year guessed here would make the benchmark look rigorous "
                  f"and test nothing."))
    if cutoff_year is None:
        cutoff_year = int(first) - 1
    if int(first) <= cutoff_year:
        return Blinding(
            protocol=Protocol.TEMPORAL, cutoff_year=cutoff_year, verified=False,
            note=(f"'{target}' was first published in {first}, at or before the cutoff "
                  f"{cutoff_year}, so it is inside the corpus and this is not a holdout."))
    return Blinding(
        protocol=Protocol.TEMPORAL, cutoff_year=cutoff_year, verified=True,
        note=f"'{target}' was first published in {first}, after the cutoff.")


def leakage_after_blinding(target: str) -> Dict[str, List[str]]:
    """What a NAMED-ENTITY exclusion of `target` would leave behind."""
    return GoldenSet.named_entity_leakage(target)


@dataclass
class NullBaseline:
    """Where the true modification lands when the ranking is shuffled."""
    trials: int
    median_rank: Optional[float]
    better_or_equal: int          # shuffles ranking it at least as well
    total_candidates: int

    @property
    def p_value(self) -> Optional[float]:
        """Fraction of shuffles that did at least as well. Not a test statistic."""
        if not self.trials:
            return None
        return (self.better_or_equal + 1) / (self.trials + 1)

    def describe(self) -> str:
        if not self.trials:
            return "No permutation null was run, so the rank has nothing to be read against."
        return (f"Permutation null over {self.trials} shuffles: median rank "
                f"{self.median_rank:.0f} of {self.total_candidates}, and "
                f"{self.better_or_equal} shuffle(s) placed it at least as high "
                f"(p ≈ {self.p_value:.3f}).")


@dataclass
class RecoveryResult:
    """One attempt to recover one modification."""
    drug: str
    parent: str
    modification: str
    blinding: Blinding
    candidate_space: str
    goal: str = ""
    outcome: Optional[RankedOutcome] = None
    null: Optional[NullBaseline] = None
    position_proposed_at_all: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_evaluable(self) -> bool:
        """
        Whether this case can produce a rank at all.

        Two ways it cannot, and they mean opposite things: the blinding did not
        hold (the run is invalid), or the modification was never in the
        candidate space (the run is valid and the method cannot express the
        answer).
        """
        return (self.blinding.is_clean
                and self.candidate_space == CandidateSpace.IN)

    def report(self) -> str:
        head = f"{self.drug} / {self.modification} (parent {self.parent}): "
        if self.candidate_space == CandidateSpace.UNMAPPABLE:
            return head + ("NO RANK REPORTED — the modification is a canonical "
                           "substitution this method could score, but it could not be "
                           "placed in the stored sequence. "
                           + " ".join(self.notes))
        if self.candidate_space != CandidateSpace.IN:
            return head + (
                f"NOT RECOVERABLE BY THIS METHOD — {self.candidate_space}. The scan "
                f"enumerates single canonical substitutions, so this modification was "
                f"never in the candidate space. Reported as out of scope, not as a "
                f"miss: scoring it as a miss understates the method and dropping it "
                f"overstates it. {self.blinding.describe()}")
        if not self.blinding.is_clean:
            return head + f"NO RANK REPORTED — {self.blinding.describe()}"
        lines = [head + self.outcome.report()]
        lines.append(f"  Scored under the '{self.goal}' goal.")
        purpose = GoldenSet.drugs().get(self.drug, {}).get("modification_purpose", {})
        stated = purpose.get(self.modification)
        if stated and stated != self.goal:
            lines.append(
                f"  MISMATCH: this modification was made for '{stated}', not "
                f"'{self.goal}'. A rank under the wrong objective measures whether the "
                f"system stumbles onto the right residue for the wrong reason, which is "
                f"not what the benchmark is for.")
        elif not stated:
            lines.append(
                "  No purpose is recorded for this modification, so it is not known "
                "whether the goal it was scored under is the one it was made for. Add "
                "`modification_purpose` to the golden-set entry.")
        if self.null:
            lines.append("  " + self.null.describe())
        if not self.position_proposed_at_all:
            lines.append("  The position itself was never proposed with any residue.")
        lines.extend("  " + n for n in self.notes)
        return "\n".join(lines)


# ---- running a case ---------------------------------------------------------

@dataclass(frozen=True)
class MappedChange:
    """A modification located in the STORED sequence's own numbering."""
    position: int                 # 1-indexed into the stored sequence
    wild_type: str
    mutant: str
    stated_position: int          # as the literature wrote it
    offset: int

    def describe(self) -> str:
        return (f"{self.wild_type}{self.position}{self.mutant} "
                f"(literature numbering: position {self.stated_position}, "
                f"offset {self.offset})")


class NumberingError(ValueError):
    """Raised when a stated position cannot be located in the stored sequence."""


def map_to_sequence(sequence: str, stated_position: int, mutant: str,
                    offset: Optional[int]) -> MappedChange:
    """
    Put a literature position into the stored sequence's numbering, and check it.

    This is the step that quietly ruins benchmarks. Semaglutide's `Arg34` is
    written in GLP-1 (7-37) numbering, where position 34 is index 28 of the
    31-residue stored chain. Applied without the offset it lands on a residue
    six places away, the scan is asked about the wrong substitution, and the
    run still produces a perfectly presentable number.

    So the offset must be supplied, never inferred: searching for the offset
    that makes the answer work is fitting the benchmark to the result. The
    mapped position is then checked to hold something other than the mutant --
    a position that already holds the target residue means the offset is wrong
    or the parent is not this molecule.
    """
    if offset is None:
        raise NumberingError(
            f"No numbering offset is recorded, so literature position "
            f"{stated_position} cannot be placed in the stored sequence. GLP-1 is "
            f"written in (7-37) numbering and IGF-1 in at least two schemes; applying "
            f"a position from the wrong one lands six residues away and still produces "
            f"a number. Add `numbering_offset` to the golden-set entry.")

    position = stated_position - offset
    if position < 1 or position > len(sequence):
        raise NumberingError(
            f"Literature position {stated_position} with offset {offset} maps to "
            f"{position}, which is outside the {len(sequence)}-residue stored sequence.")

    wild_type = sequence[position - 1]
    if wild_type == mutant:
        raise NumberingError(
            f"Literature position {stated_position} maps to {position}, which already "
            f"holds {mutant}. Either the offset is wrong or this parent is not the "
            f"molecule the modification was made to.")
    return MappedChange(position=position, wild_type=wild_type, mutant=mutant,
                        stated_position=stated_position, offset=offset)


def _permutation_null(n_candidates: int, true_rank: int, trials: int,
                      seed: int) -> NullBaseline:
    """
    Where the true modification lands when the ordering carries no information.

    A uniform draw rather than a reshuffle of the actual scores: under the null
    the ranking is uninformative, so every position is equally likely, and
    sampling that directly is both exact and cheaper than shuffling a list of
    589 objects a thousand times.
    """
    rng = random.Random(seed)
    draws = [rng.randint(1, n_candidates) for _ in range(trials)]
    better = sum(1 for d in draws if d <= true_rank)
    draws.sort()
    median = draws[len(draws) // 2] if draws else None
    return NullBaseline(trials=trials, median_rank=median, better_or_equal=better,
                        total_candidates=n_candidates)


def run_case(drug: str, modification: str, blinding: Blinding,
             goal: str = "binding_affinity", null_trials: int = 1000,
             seed: int = 0, workflow=None) -> RecoveryResult:
    """
    Blind, scan, and locate the true modification in the proposal list.

    Returns a result in every case, including the ones that produce no rank:
    a modification outside the candidate space and a blinding that did not hold
    are both real outcomes, and a benchmark that silently skipped them would
    report only the cases it happened to be able to score.
    """
    entry = GoldenSet.drugs().get(drug, {})
    parent = entry.get("parent", "")
    verdict, stated_position, residue = CandidateSpace.classify(modification)

    result = RecoveryResult(drug=drug, parent=parent, modification=modification,
                            blinding=blinding, candidate_space=verdict, goal=goal)
    if verdict != CandidateSpace.IN or not blinding.is_clean:
        return result

    # Resolve the parent to a SEQUENCE before scanning. The workflow takes a
    # name, but its name resolution is thinner than the API's -- "GLP-1 (7-37)"
    # resolves through `resolve_name` and not through the workflow -- and a
    # benchmark that scanned an empty sequence would report every case as
    # unmappable and look like a data problem. Resolving here also means the
    # result records exactly which residues were scanned.
    from .function_inference import FunctionInferencer

    resolved = FunctionInferencer().resolve_name(parent)
    if not resolved:
        result.notes.append(
            f"The parent '{parent}' did not resolve to a sequence, so nothing was "
            f"scanned. Set `parent` in the golden-set entry to a name the reference "
            f"set carries.")
        result.candidate_space = CandidateSpace.UNMAPPABLE
        return result
    sequence = resolved[1]["sequence"]

    from ..workflows.optimize import OptimizeWorkflow
    workflow = workflow or OptimizeWorkflow()
    context, proposals = workflow.scan(sequence, confirmed_goal=goal, auto_confirm=True)

    try:
        mapped = map_to_sequence(context.sequence, stated_position, residue,
                                 entry.get("numbering_offset"))
    except NumberingError as e:
        result.notes.append(str(e))
        result.candidate_space = CandidateSpace.UNMAPPABLE
        return result

    ranked = sorted(proposals, key=lambda r: r.net_score, reverse=True)
    rank = None
    for i, proposal in enumerate(ranked, start=1):
        if (proposal.position + 1 == mapped.position
                and proposal.mutant_aa.upper() == mapped.mutant):
            rank = i
            break

    result.position_proposed_at_all = any(
        p.position + 1 == mapped.position for p in ranked)
    result.notes.append(f"Mapped to {mapped.describe()}.")
    result.outcome = RankedOutcome(
        modification=modification, protocol=blinding.protocol, rank=rank,
        total_proposals=len(ranked),
        ablated_motifs=list(blinding.ablated_motifs),
        cutoff_year=blinding.cutoff_year)
    if rank is not None:
        result.null = _permutation_null(len(ranked), rank, null_trials, seed)
    return result


def run_benchmark(protocol: Protocol = Protocol.MOTIF_ABLATION,
                  drugs: Sequence[str] = (), goal: str = "binding_affinity",
                  seed: int = 0) -> List[RecoveryResult]:
    """
    Every recorded modification of every named drug, under one protocol.

    Runs the unscoreable cases too and reports why. A benchmark that quietly
    dropped them would report the subset it could score and call that the
    result.
    """
    names = list(drugs) if drugs else sorted(GoldenSet.drugs())
    results: List[RecoveryResult] = []
    workflow = None
    for drug in names:
        blinding = (blind_by_motif(drug) if protocol is Protocol.MOTIF_ABLATION
                    else blind_by_date(drug))
        for modification in GoldenSet.drugs()[drug].get("modifications", []):
            if workflow is None and blinding.is_clean:
                from ..workflows.optimize import OptimizeWorkflow
                workflow = OptimizeWorkflow()
            results.append(run_case(drug, modification, blinding, goal=goal,
                                    seed=seed, workflow=workflow))
    return results


def summarise(results: Sequence[RecoveryResult]) -> str:
    """The benchmark's headline, phrased so it cannot be read as a hit rate."""
    if not results:
        return "No recovery cases were run."

    evaluable = [r for r in results if r.is_evaluable and r.outcome]
    out_of_scope = [r for r in results
                    if r.candidate_space in (CandidateSpace.NON_CANONICAL,
                                             CandidateSpace.NOT_A_SUBSTITUTION,
                                             CandidateSpace.UNPARSEABLE)]
    unmappable = [r for r in results if r.candidate_space == CandidateSpace.UNMAPPABLE]
    unblinded = [r for r in results
                 if r.candidate_space == CandidateSpace.IN and not r.blinding.is_clean]
    proposed = [r for r in evaluable if r.outcome.was_proposed]

    lines = [
        f"{len(results)} case(s): {len(evaluable)} scoreable, "
        f"{len(out_of_scope)} outside the candidate space, "
        f"{len(unblinded)} with no valid blinding, "
        f"{len(unmappable)} unmappable into the stored numbering.",
    ]
    if out_of_scope:
        lines.append(
            f"The {len(out_of_scope)} outside the candidate space are not misses. The "
            f"scan enumerates single canonical substitutions, so non-canonical residues "
            f"and attached chains could not have been proposed by any ranking.")
    if not evaluable:
        lines.append(
            "No case produced a rank, so there is no recovery result to report -- which "
            "is a statement about what can currently be evaluated, not about the "
            "method's accuracy.")
        return "\n".join(lines)

    ranks = sorted(r.outcome.rank for r in proposed)
    lines.append(
        f"{len(proposed)} of {len(evaluable)} scoreable modification(s) were proposed "
        f"at all" + (f"; ranks {', '.join(map(str, ranks))}." if ranks else "."))
    lines.append(
        "Report these as ranks with their totals and their permutation nulls. A count "
        "of how many landed in some top-N is the form this contract exists to refuse.")
    return "\n".join(lines)
