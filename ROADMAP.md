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

## Addendum 2 — Physical biochemistry layer — IN PROGRESS

Build order from the addendum's section 13:

- [x] **6a** Provenance tier schema + licensing enforcement + parameter budget
- [x] **6b** Electrostatics — per-residue pKa, multi-pH protonation
- [x] **6c** Structure template hierarchy + confidence gates + refusal path
- [x] **6d** Parameterized-residue registry (empty, QM pipeline stubbed)
- [x] **6e** Parent-molecule context + contact classifier + golden cases
      (insulin, ghrelin, CCK, drosocin)
- [x] **6f** Partner-peptide module + RAMP schema requirement
- [ ] **6g** Class B1 restricted-zone rule + affinity/efficacy/bias three-field output
- [ ] **6h** Synthetic feasibility gate
- [ ] **6i** Holdout protocol rewrite (motif ablation, temporal, decoys) +
      rank-based reporting

Then: conformer ensembles, SAPT, NBO (addendum sections 4 and 5). Those are
explanatory depth on a system that must first be honest, which is why they come
last rather than first.

## Addendum 3 — ML and representation learning — QUEUED

Not started. Begins only after Addendum 2 is complete.

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
    titled "Why sequence-aware evaluation matters".

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
