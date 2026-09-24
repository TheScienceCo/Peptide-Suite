# What to feed the training layer

Short version: **single-change variants with a measured outcome, a stated
comparator, a stated assay, and a citation.** Everything below is why each of
those words is load-bearing, and what happens to a row that is missing one.

## First, what is and is not trained today

Nothing in this repository is trained on real labels. The ML layer exists, the
splits and leakage guards work, and the experiments that run are **synthetic by
construction** — they demonstrate that an evaluation protocol can be run and
read, and support no biological claim. `ml/datasets/registry.py` marks them
`SYNTHETIC_METHOD_ONLY` for exactly that reason.

So "training data" here means the thing that would change that: a store of
measured outcomes for modified peptides. It has two consumers, and they have
different appetites.

| Consumer | Needs | Useful at |
|---|---|---|
| **Experimental precedent** (shipping now) | one record | 1 record |
| **A trained model** | many records, many parents | hundreds, realistically |

The first is worth feeding immediately. A single well-formed record makes the
substitution scan say "somebody measured this exact change" instead of "no
precedent found", and that outranks anything the program computes. The second
needs volume, and volume of the right shape.

## One record, field by field

Fill `docs/variant_evidence_template.csv`. Delete the example row.

| Column | Required | Why |
|---|---|---|
| `parent` | yes | The molecule the change is relative to. Also the **split key** — see below. |
| `parent_sequence` | strongly | Lets the loader check the position actually holds the wild-type residue you claim |
| `variant_name` | yes | How the record is referred to |
| `modification_kind` | yes | `SUBSTITUTION`, `NCAA_SUBSTITUTION`, `LIPIDATION`, `CYCLIZATION`, `TRUNCATION`, … |
| `position` / `wild_type` / `mutant` | for substitutions | 1-indexed **into `parent_sequence`**, not a paper's numbering |
| `measure` | yes | `AFFINITY_KD`, `POTENCY_EC50`, `PLASMA_HALF_LIFE`, `PROTEASE_STABILITY`, … |
| `direction` | yes | `INCREASED` / `DECREASED` / `UNCHANGED` / `NOT_DETERMINED` |
| `fold_change` **or** `value`+`unit` | for a model | A direction alone is a classification label, not a regression target |
| `comparator` | **yes** | "Three-fold more potent" is not a fact until it says than *what* |
| `assay` | **yes** | Two numbers for one measure from different assays are not comparable |
| `target` | for affinity/potency | An affinity with no receptor names no interaction |
| `pmid` or `doi` | for the top tier | No citation → caps at `BIOCHEMICAL_PRINCIPLE` |
| `extraction` | yes | `CURATOR_READ_FULL_TEXT` / `CURATOR_READ_ABSTRACT` / `AUTOMATED_PARSE` |

### Numbering is the field that will bite you

GLP-1 is written in at least three numbering schemes and IGF-1 in two. A
position in the wrong scheme lands on the wrong residue and **still looks
plausible** — nothing about it reads as an error later. So `position` is
1-indexed into the `parent_sequence` you supply in the same row, and the loader
checks that `parent_sequence[position-1] == wild_type` and rejects the row when
it does not. Put the paper's own numbering in `note`.

## Four rows that will be rejected, and why

The loader refuses these loudly rather than accepting them and quietly
dropping them at training time.

1. **No measured outcome.** A modification with no result is a *design*. It is
   stored, it is displayed, and it can never support a claim about effect —
   training on it means inventing a target value.
2. **More than one change at once.** A construct with a backbone substitution,
   a sequence substitution and a lipid chain has one measured half-life and
   three changes. Attaching that number to any one of them teaches a model an
   attribution nobody made. Such rows are stored and marked unattributable.
3. **A direction with no magnitude.** Real result, not a regression target.
   Kept as a classification label, not coerced into a number.
4. **No citation.** Caps below the tier floor. A model trained on unverifiable
   numbers produces outputs that inherit the unverifiability while looking
   like predictions.

## What is actually worth your time

**Single-change variants beat marketed analogues.** Semaglutide, tirzepatide
and friends each differ from their parent in several places simultaneously.
The store will take them, mark them confounded, and refuse to let them support
any claim about the substitution — which is correct, and means the effort of
entering them buys the precedent display and nothing for training.

Alanine scans, single-point mutagenesis series and truncation ladders are the
high-value shape.

**Count parents, not rows.** Thirty variants of one peptide is **one molecule**
for splitting purposes, not thirty — `split_by_parent` puts whole parents on
one side of the split, because a random split would put a peptide in train and
its own single mutant in test and score the model on recall. So:

- 5 parents × 8 single variants each ≫ 1 parent × 40 variants
- below roughly 4–5 distinct parents there is nothing to hold out at all

**One measure at a time.** Ten Kd values against one receptor are worth more
than ten different measures across ten peptides, because a model is fit per
measure. Pick the one you care about most and go deep.

## Loading it

```bash
python tools/load_variant_evidence.py your_records.csv --dry-run   # validate only
python tools/load_variant_evidence.py your_records.csv             # write the store
```

`--dry-run` reports every rejected row with the reason and changes nothing.
The store is `peptide_suite/data/variant_evidence.json`.
