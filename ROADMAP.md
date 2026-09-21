# Roadmap

Work is sequenced by addendum. Each is finished before the next begins, because
each later layer assumes the guarantees of the earlier one — a learned model
built on top of a system that cannot say where its numbers came from inherits
that inability and makes it harder to see.

## Addendum 1 — IP boundary and cross-lingual corpus — COMPLETE

Engine/policy separation. No coefficient is hardcoded, defaulted, or inlined in
the engine; an absent policy artifact fails at startup rather than falling back.

- Policy schema, strict validation, fail-loud loader
- Opaque aggregate-term identifiers (HMAC under a private salt)
- Demonstration pack, every value marked `derived` or `placeholder_midpoint`
- The boundary made operative: the engine genuinely cannot score without a pack

Outstanding: boundary debt, tracked and ratcheted in `policy/BOUNDARY_DEBT.txt`.
It only moves down.

## Addendum 2 — Physical biochemistry layer — COMPLETE (pending tooling)

Build order from the addendum's section 13:

- [x] **6a** Provenance tier schema + licensing enforcement + parameter budget
- [x] **6b** Electrostatics — per-residue pKa, multi-pH protonation
- [x] **6c** Structure template hierarchy + confidence gates + refusal path
- [x] **6d** Parameterized-residue registry (empty, QM pipeline stubbed)
- [x] **6e** Parent-molecule context + contact classifier + golden cases
      (insulin, ghrelin, CCK, drosocin)
- [x] **6f** Partner-peptide module + RAMP schema requirement
- [x] **6g** Class B1 restricted-zone rule + affinity/efficacy/bias three-field output
- [x] **6h** Synthetic feasibility gate
- [x] **6i** Holdout protocol rewrite (motif ablation, temporal, decoys) +
      rank-based reporting

All nine build-order steps are done. What remains from Addendum 2 is the
explanatory depth that the addendum itself places after 6i:

- [x] Conformer ensembles — eligibility and the length refusal are built; the
      search itself is stubbed and raises. No engine in this deployment.
- [x] n->pi* backbone analysis — the requirement is enforced and proposals carry
      their outstanding obligation; the analysis is stubbed and raises.
- [x] SAPT contact decomposition — the component model, what each responds to,
      and the enforced prohibition on summing into a binding free energy. The
      decomposition itself is stubbed and raises.

All three are gates and contracts rather than computations, because no QM
engine, conformational search or NBO implementation is available here. What is
built is the part that has to be right when the engines arrive: which peptides
are eligible, which proposals owe a mechanistic account, and which operation is
forbidden. Wiring an engine in changes one value per module rather than the
shape of every consumer.

Remaining, and blocked on tooling rather than on design:

- [ ] Install a conformational search (CREST + GFN2-xTB) and make
      `conformer_ensemble.generate` return ensembles
- [ ] Install NBO and make `backbone_nbo.analyse` return stabilisation energies
- [ ] Install a QM package with SAPT0/DFT-SAPT and make
      `contact_decomposition.decompose` return real components
- [ ] Run the 6d parameterization pipeline for at least one residue, so the
      registry stops being empty

These come last rather than first because they are explanatory depth on a system
that had to be honest before it was deep. All three depend on the QM pipeline
stubbed in 6d, so the parameterized-residue registry is the prerequisite.

## Addendum 3 — ML and representation learning — IN PROGRESS

Built so far:

- [x] **Provenance** — the three ML categories, kept distinct from the existing
      tiers and from each other. None may be a regression target.
- [x] **Reproducibility** — seeds, device selection including MPS, run records
      that report a dirty tree as not reproducible rather than as its nearest
      commit.
- [x] **Splitting** — random versus sequence-clustered, union-find
      single-linkage with the no-cross-cluster guarantee tested exhaustively.
- [x] **Metrics** — ROC-AUC (tie-aware), PR-AUC, F1, precision, recall, MCC,
      MAE/RMSE/R², calibration curves, bootstrap intervals on everything.
- [x] **Dataset registry** — eight tasks with real sources and licences; seven
      UNAVAILABLE with reasons, one SYNTHETIC_METHOD_ONLY. No invented labels.
- [x] **Training** — checkpoints carrying their run record, early stopping that
      restores the best weights.
- [x] **Model cards** — generated, refusing to omit limitations or the
      similarity control.
- [x] **Experiment tracking** — JSONL, with a reproduce command that refuses to
      print for an unreproducible run.
- [x] **Encoders** — deterministic encoder present, ESM-2 fails loudly rather
      than substituting.
- [x] **The split-gap experiment**, visible in the UI and run live.
- [x] **Substitution landscape** — the whole position x residue grid, in its own
      tab, over five selectable quantities. A cell with no computed value is
      hatched and carries no value on the wire; the wild-type diagonal is its
      own state; a one-signed quantity gets a sequential encoding rather than a
      diverging one. Colour scale derived from the grid shown and declared as
      such. Diverging poles validated for colour-vision deficiency in both
      modes, with dark steps selected against the dark surface rather than
      flipped. The column marginal rides above it on a zero baseline: which
      positions tolerate change at all, reduced from the same cells.

Remaining, and mostly blocked on data and weights rather than design:

- [ ] Real labelled datasets. Every prediction task is UNAVAILABLE here; the
      registry names the source and licence for each.
- [ ] ESM-2 weights, then the four-arm comparison per task, and the same
      projection over a learned space — where the explorer's 19% would be the
      number to beat
      (physicochemical baseline / classical ML / pretrained embeddings /
      fine-tuned) that answers whether language models add predictive value.
- [x] **Representation explorer** — reference plus every single substitution
      projected onto two principal components, leading with the explained
      variance rather than the scatter. Refuses to mix encoder spaces, refuses
      a set with no variation, and fixes the SVD's arbitrary sign so the plot
      does not mirror itself between runs. Two deterministic encoders with
      opposite limitations, both stated; the pretrained arm still raises.
- [ ] Multimodal fusion: modality-specific encoders plus a fusion layer,
      compared against single modality and naive concatenation.
- [x] **Explainability, the parts that do not need a trained model** — nearest
      examples in representation space, and an out-of-distribution warning that
      places a candidate inside the reference set's own nearest-neighbour
      distance distribution rather than reporting a bare distance. Refuses a
      reference set too small to have a spread; refuses a distance across two
      encoder spaces. Residue sensitivity is the landscape's column marginal,
      computed over the real pipeline, and is not duplicated here.
- [ ] Attribution, once a model trained on real labels exists. Attributing the
      synthetic split-gap model's recall of constructed sequences would be a
      picture of nothing.
- [ ] The receptor-prediction validation idea — take the learned properties
      together with the physics pipeline's and predict the binding site,
      ignoring the literature, then check against known biology.

The goal is a credible computational peptide engineering platform, not
cosmetically added ML libraries. Existing analyses stay usable and
interpretable; new capability is modular.

**Provenance categories must stay distinct.** The existing system already
separates experimentally observed data, external annotations, and deterministic
descriptors. Addendum 3 adds three more that must never be silently merged with
those or with each other: pretrained-model embeddings, ML predictions, and
uncertainty estimates. Heuristic and proprietary scores remain their own
category.

Scope:

1. **PyTorch pipeline** — `ml/{datasets,embeddings,models,training,evaluation,inference,multimodal,configs}`.
   Deterministic splits, seeds, checkpointing, early stopping, batch inference.
   GPU/MPS/CPU; MPS matters because development happens on an M-series Mac.
   No synthetic labels: a task without real labelled data is marked
   experimental and uses a public benchmark instead.
2. **Protein language model embeddings** — ESM-2 or equivalent. Residue-level
   and pooled, with model version and preprocessing recorded. Cached. PCA/UMAP
   visualization of WT vs single vs multi-substitution candidates in
   representation space.
3. **Transfer-learning property prediction** — 2–4 tasks with genuinely good
   public data, the rest as declared placeholders. Every task compares four
   arms: physicochemical baseline, classical ML, pretrained embeddings,
   fine-tuned model where dataset size makes it defensible. The point is to
   show whether protein language models actually add predictive value.
4. **Evaluation** — a major feature, not an afterthought. ROC-AUC, PR-AUC, F1,
   MCC, MAE/RMSE/R², calibration curves, bootstrap CIs.
   **Sequence-similarity-aware splitting is the headline.** Random peptide
   splits overestimate generalization when near-duplicate sequences straddle
   train and test. Random-split versus clustered-split results are displayed
   side by side and the gap is shown prominently.
5. **Multimodal** — modality-specific encoders plus a fusion layer, not
   concatenated scalars. Compare single modality vs concatenation vs learned
   fusion, and report whether fusion actually helps on held-out data.
6. **Explainability** — nearest training examples in embedding space,
   residue-level sensitivity, attribution, uncertainty, OOD warning by embedding
   distance. Attribution is not causation and is not to be described as such.
7. **Model cards**, auto-generated, exposed in the UI.
8. **Experiment tracking** — config, architecture, hyperparameters, dataset
   version, git commit, seed, metrics, checkpoint. Reproducible from the stored
   config.
9. **Full-stack integration** — model predictions, embedding explorer,
   substitution landscape heatmap (position x amino acid), multimodal analysis,
   evaluation dashboard. Integrated into this app, not disconnected notebooks.
10. **API** — typed schemas for embeddings, prediction, batch scoring,
    substitution scans, model metadata, evaluation metrics.
11. **Testing** — unit, inference, API, data validation, deterministic
    regression on fixed examples. CI must not download multi-gigabyte weights;
    use tiny test models.
12. **Documentation** — README rewritten so an ML engineer or computational
    biologist understands the project in about two minutes. Architecture
    diagram, UI screenshots, one performance comparison table, and a section
    titled "Why sequence-aware evaluation matters". *Done: `docs/architecture.svg`
    (selected for dark mode rather than flipped), eight screenshots taken from
    a live run rather than mocked, the split-gap table, and the section. A test
    fails on a figure that is referenced but missing, or committed but shown
    nowhere.*

**Validation idea worth building toward:** take the learned properties together
with those from the existing pipeline and predict the receptor or binding site a
peptide would engage — ignoring what the literature says — then check the
prediction against known biology. That is a real test of whether the model
learned something, as opposed to a retrospective fit.

Standards: reproducibility, rigorous baselines, clear provenance, useful failure
analysis, modular architecture. Explicitly avoid toy notebooks, unjustified deep
learning, fake datasets, hard-coded demo results, scientific overclaims, data
leakage, and hiding negative results.

Deliverables at completion: implementation summary, repository architecture,
datasets and licenses, models implemented, evaluation results, screenshots,
known limitations, remaining roadmap, exact reproduction commands, and a short
statement of what the repository demonstrates to an ML hiring manager versus a
computational-biology hiring manager.
