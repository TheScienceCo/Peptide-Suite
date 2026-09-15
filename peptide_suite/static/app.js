/* Peptide Suite — frontend.
   All analysis happens server-side in the Python pipeline. This file renders
   what comes back and nothing more: no score is computed here, so the display
   cannot drift from the engine. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const EXAMPLES = {
  sequences: [
    { label: "GLP-1 (7-37)", seq: "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG" },
    { label: "Insulin A-chain", seq: "GIVEQCCTSICSLYQLENYCN" },
    { label: "Glucagon", seq: "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT" },
    { label: "Oxytocin", seq: "CYIQNCPLG" },
    { label: "LL-37", seq: "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES" },
    { label: "name: laminin", seq: "laminin" },
  ],
  queries: ["myelinating peptides", "wound healing", "angiogenesis", "antimicrobial", "appetite regulation"],
};

let GOALS = [];

/* ---------- API ---------- */

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

/* ---------- shared rendering ---------- */

function notice(text, kind = "warn", glyph = "!") {
  const n = el("div", `notice ${kind === "warn" ? "" : kind}`.trim());
  n.append(el("span", "glyph", glyph), el("span", null, text));
  return n;
}

// A score is meaningless without knowing what weighted it. Every scored
// response carries its policy provenance, and it is rendered above the results
// rather than below them: a caveat placed after the numbers is read after the
// reader has already formed a view.
function policyNotice(policy) {
  if (!policy) return null;
  if (!policy.is_demonstration) {
    return notice(
      `Scored under policy ${policy.policy_version} (${policy.digest}).`,
      "info", "i"
    );
  }
  const n = el("div", "notice");
  n.append(
    el("span", "glyph", "!"),
    (() => {
      const body = el("span", null);
      body.append(el("strong", null, "Demonstration policy pack. "));
      body.append(document.createTextNode(
        `No weight in this pack was fitted to data, and ${policy.n_placeholder_thresholds} of ` +
        `${policy.n_thresholds} thresholds are placeholders. The evidence tiers, the ` +
        `computed physical quantities and the reasoning behind each candidate are real. ` +
        `The overall ranking magnitudes are not a claim. (policy ${policy.policy_version}, ` +
        `${policy.digest})`
      ));
      return body;
    })()
  );
  return n;
}

function meter(label, value, variant) {
  const m = el("div", "meter");
  const l = el("div", "ml");
  l.append(el("span", null, label), el("b", null, value.toFixed(2)));
  const track = el("div", "track");
  const fill = el("div", `fill${variant ? " " + variant : ""}`);
  fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  track.append(fill);
  m.append(l, track);
  return m;
}

function busy(btn, on, label) {
  btn.disabled = on;
  btn.textContent = "";
  if (on) btn.append(el("span", "spinner"), document.createTextNode("Working…"));
  else btn.textContent = label;
}

/* ---------- Workflow 1 ---------- */

async function onAnalyze() {
  const btn = $("#btn-infer");
  const input = $("#seq").value.trim();
  $("#gate-slot").innerHTML = "";
  $("#opt-results").innerHTML = "";

  if (!input) {
    $("#gate-slot").append(notice("Enter a peptide sequence or a peptide name.", "error", "×"));
    return;
  }

  busy(btn, true);
  try {
    const info = await api("/api/infer-function", { input });
    renderGate(info);
  } catch (e) {
    $("#gate-slot").append(notice(e.message, "error", "×"));
  } finally {
    busy(btn, false, "Analyze");
  }
}

function renderGate(info) {
  const slot = $("#gate-slot");
  slot.innerHTML = "";

  // A name that matched a protein rather than a peptide: report what is known
  // and which derived motifs can actually be analysed.
  if (!info.sequence && info.protein_match) {
    const m = info.protein_match;
    const box = el("div", "gate");
    box.append(el("h2", null, `${m.name}${m.gene ? ` (${m.gene})` : ""}`));
    box.append(el("p", "reasoning", m.rationale));
    box.append(el("div", "kv",
      `Biological process: ${m.biological_process}${m.go_term ? ` (${m.go_term})` : ""}` +
      `${m.uniprot ? ` · UniProt ${m.uniprot}` : ""}`));

    if (m.derived_motifs.length) {
      box.append(notice(
        "This is a full-length protein, so there is no single sequence to optimise. " +
        "The derived motifs below are short, analysable, and have characterised activity — " +
        "paste one to run the workflows on it.", "info", "i"));
      m.derived_motifs.forEach((d) => {
        const row = el("div", "celltype");
        const h = el("div");
        h.append(el("span", "n", d.motif));
        row.append(h, el("div", "r", d.function));
        const use = el("button", "chip", `Analyse ${d.motif}`);
        use.addEventListener("click", () => { $("#seq").value = d.motif; onAnalyze(); });
        row.append(use);
        box.append(row);
      });
    } else {
      box.append(notice(
        "No short analysable motif is on file for this protein. Paste the specific region " +
        "you are interested in, or use the Find peptides tab to search by function.", "warn", "!"));
    }
    slot.append(box);
    return;
  }

  if (info.parse_error) {
    slot.append(notice(`Could not read that as a peptide sequence — ${info.parse_error}`, "error", "×"));
    return;
  }

  const gate = el("div", "gate");
  gate.append(el("h2", null, "Confirm the target function"));

  const LEVEL_LABEL = {
    0: "identified in UniProt",
    1: "matched a known peptide",
    2: "matched a characterised motif",
    3: "matched a family profile",
    4: "not recognised — default applied",
  };

  const p = el("p");
  p.append(
    document.createTextNode("Based on the sequence, you are seeking a form of: "),
    el("span", "quote", info.inferred_function)
  );
  gate.append(p);

  // What the parser had to clean out of the paste, and whether this is a
  // protein rather than a peptide — both change what the analysis means.
  (info.sequence_notes || []).forEach((n) => gate.append(notice(n, "info", "i")));
  if (info.is_protein) gate.append(notice(info.length_note, "warn", "!"));

  if (info.uniprot) {
    const u = info.uniprot;
    const box = el("div", "uniprot");
    box.append(el("b", null, "UniProt"));
    const line = el("div", "kv");
    const link = el("a", "acc", u.accession);
    link.href = u.url;
    link.target = "_blank";
    link.rel = "noopener";
    line.append(link, document.createTextNode(
      ` — ${u.protein_name}${u.gene ? ` (${u.gene})` : ""}` +
      `${u.organism ? `, ${u.organism}` : ""}${u.full_length ? `, ${u.full_length} aa` : ""}`));
    box.append(line);
    if (u.subcellular_location) box.append(el("div", "kv", `Location: ${u.subcellular_location}`));
    if ((u.keywords || []).length) box.append(el("div", "eqrefs", u.keywords.join(" · ")));
    gate.append(box);
  }

  // How the inference was reached, so a level-4 default never reads like a
  // level-0 identification.
  const prov = el("div", `provenance lvl${info.inference_level}`);
  prov.append(el("span", "lvl", `Level ${info.inference_level}`));
  prov.append(el("span", "lvl-label", LEVEL_LABEL[info.inference_level] || ""));
  prov.append(el("span", "lvl-conf", `confidence ${info.inference_confidence.toFixed(2)}`));
  gate.append(prov);
  gate.append(el("div", "kv", info.inference_basis));

  if (info.parent_protein) {
    gate.append(el("div", "kv", `Parent protein: ${info.parent_protein}`));
  }
  if (info.native_context_note) {
    gate.append(notice(info.native_context_note, "info", "i"));
  }
  (info.caveats || []).forEach((c) => gate.append(notice(c)));

  if ((info.alternatives || []).length) {
    gate.append(el("div", "kv",
      "Other motifs present: " +
      info.alternatives.map((a) => `${a.motif} @ ${a.position} (${a.parent_protein})`).join(", ")));
  }

  const row = el("div", "row");

  const goalWrap = el("div");
  goalWrap.append(el("label", null, "Target function"));
  const sel = el("select");
  sel.id = "goal-select";
  GOALS.forEach((g) => {
    const o = el("option", null, g.label + (g.quick_win ? "  — quick win" : ""));
    o.value = g.id;
    sel.append(o);
  });
  // Pre-select what was inferred rather than leaving the first option by default.
  if (info.suggested_goal && GOALS.some((g) => g.id === info.suggested_goal)) {
    sel.value = info.suggested_goal;
  }
  goalWrap.append(sel);

  const why = el("div", "kv goal-why");
  if (info.suggested_goal_reason) {
    why.append(el("b", null, "Why this goal "),
               document.createTextNode(info.suggested_goal_reason));
  }

  const desc = el("div", "kv");
  desc.id = "goal-desc";
  const syncDesc = () => {
    const g = GOALS.find((x) => x.id === sel.value);
    desc.textContent = g ? g.description : "";
    // The rationale explains the SUGGESTED goal; hide it once the user overrides.
    why.hidden = sel.value !== info.suggested_goal;
  };
  sel.addEventListener("change", syncDesc);
  syncDesc();

  const btnWrap = el("div", "shrink");
  const run = el("button", "primary", "Run substitution scan");
  run.id = "btn-scan";
  run.addEventListener("click", () => runScan(info.sequence, sel.value));
  btnWrap.append(run);

  row.append(goalWrap, btnWrap);
  gate.append(row, why, desc);

  const props = info.properties;
  if (props) {
    gate.append(el("div", "kv",
      `${props.length} residues · ${props.charged_residues} charged (${props.percent_charged}%) · ` +
      `${props.aromatic_count} aromatic · ${props.proline_count} Pro · ${props.cysteine_count} Cys`));
  }

  slot.append(gate);
}

async function runScan(sequence, goal) {
  const btn = $("#btn-scan");
  const out = $("#opt-results");
  out.innerHTML = "";
  busy(btn, true);

  const ph = parseFloat($("#ph").value);
  try {
    const homologs = $("#homologs").value
      .split(/[\n,]+/)
      .map((x) => x.trim().toUpperCase())
      .filter(Boolean);

    const data = await api("/api/optimize", {
      input: sequence,
      goal,
      ph: Number.isFinite(ph) ? ph : 7.4,
      homologs: homologs.length ? homologs : null,
    });
    renderOptimizeResults(data);
  } catch (e) {
    out.append(notice(e.message, "error", "×"));
  } finally {
    busy(btn, false, "Run substitution scan");
  }
}

function renderOptimizeResults(data) {
  const out = $("#opt-results");
  const pn = policyNotice(data.policy);
  if (pn) out.append(pn);
  const ctx = data.context;
  const hitPositions = new Set(data.recommendations.map((r) => r.position));

  // --- summary strip
  const stats = el("div", "stats");
  const addStat = (k, v, sm) => {
    const s = el("div", "stat");
    s.append(el("div", "k", k), el("div", `v${sm ? " sm" : ""}`, v));
    stats.append(s);
  };
  addStat("Length", `${ctx.length}`);
  addStat("Candidates scanned", `${data.scan_size}`);
  addStat("pH", `${data.ph}`);
  addStat("Homologs", `${ctx.homolog_count}`);
  addStat("Conservation", ctx.conservation_available ? "computed" : "unavailable", true);
  out.append(stats);

  // --- limitations, stated before any results
  ctx.data_notes.forEach((n) => out.append(notice(n)));

  // --- sequence with recommended positions marked
  const seqCard = el("div", "card");
  seqCard.append(el("h2", null, "Sequence"));
  const view = el("div", "seqview");
  [...ctx.sequence].forEach((aa, i) => {
    const r = el("div", `res${hitPositions.has(i) ? " hit" : ""}`, aa);
    if (i % 10 === 0) r.append(el("span", "pos", String(i + 1)));
    r.title = `${aa}${i + 1}`;
    view.append(r);
  });
  seqCard.append(view);
  seqCard.append(el("div", "kv", "Highlighted residues carry a ranked substitution below."));
  out.append(seqCard);

  // --- conservation
  out.append(renderConservation(ctx));

  // --- recommendations
  const recCard = el("div", "card");
  recCard.append(el("h2", null, `Ranked substitutions (${data.recommendations.length})`));

  if (!data.recommendations.length) {
    recCard.append(el("div", "empty-chart", "No substitutions were returned for this goal."));
  }

  data.recommendations.forEach((rec, i) => recCard.append(renderRec(rec, i + 1)));
  out.append(recCard);
}

function renderConservation(ctx) {
  const card = el("div", "card");
  card.append(el("h2", null, "Per-position conservation (Shannon entropy)"));

  if (!ctx.conservation_available) {
    const empty = el("div", "empty-chart");
    empty.append(el("strong", null, "Not computed"));
    empty.append(document.createTextNode(
      `Only ${ctx.homolog_count} distinct sequence(s) were available; at least 3 are required. ` +
      `Entropy over a single sequence is zero at every position by construction, so plotting it ` +
      `would show a conservation pattern that does not exist. No conservation penalty was applied ` +
      `to the scores below.`
    ));
    card.append(empty);
    return card;
  }

  const entries = Object.entries(ctx.conservation_entropy)
    .map(([k, v]) => [Number(k), v])
    .sort((a, b) => a[0] - b[0]);

  // The y-axis ceiling is log2(number of sequences) — the highest entropy the
  // alignment can actually produce — not log2(20). Scaling to the theoretical
  // 20-residue maximum would squash every bar of a small alignment into the
  // baseline and make a real conservation pattern look like no pattern at all.
  const achievableMax = Math.log2(Math.max(2, ctx.homolog_count));
  const max = Math.max(achievableMax, ...entries.map((e) => e[1]));
  const tickStep = max <= 1.2 ? 0.25 : max <= 2.5 ? 0.5 : 1;
  const ticks = [];
  for (let t = 0; t <= max + 1e-9; t += tickStep) ticks.push(Number(t.toFixed(2)));

  // Height encodes entropy; the band colour is redundant reinforcement, so the
  // chart stays readable for viewers who cannot separate the status hues.
  const legend = el("div", "chart-legend");
  [
    ["Conserved — substitution risky (< 0.5)", "var(--status-critical)"],
    ["Moderately conserved (0.5–2.0)", "var(--status-warning)"],
    ["Variable — tolerant (> 2.0)", "var(--status-good)"],
  ].forEach(([label, color]) => {
    const s = el("span");
    const sw = el("span", "swatch");
    sw.style.background = color;
    s.append(sw, document.createTextNode(label));
    legend.append(s);
  });
  card.append(legend);

  const W = 1000, H = 150, PAD_L = 34, PAD_B = 22, PAD_T = 8;
  const plotW = W - PAD_L - 8, plotH = H - PAD_B - PAD_T;
  const bw = plotW / entries.length;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Shannon entropy for ${entries.length} positions`);
  svg.style.display = "block";

  const ns = (tag, attrs) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };

  // gridlines + y labels
  ticks.forEach((t) => {
    const y = PAD_T + plotH - (t / max) * plotH;
    svg.append(ns("line", {
      x1: PAD_L, x2: W - 8, y1: y, y2: y,
      stroke: "var(--grid)", "stroke-width": 1,
    }));
    const lbl = ns("text", {
      x: PAD_L - 7, y: y + 3.5, "text-anchor": "end",
      fill: "var(--text-muted)", "font-size": 10,
      "font-family": "ui-monospace, monospace",
    });
    lbl.textContent = t.toFixed(2);
    svg.append(lbl);
  });

  entries.forEach(([pos, h], i) => {
    // A fully conserved position has zero entropy, which is the single most
    // important thing on this chart. It gets a deliberate floor mark so it reads
    // as "conserved", not as missing data.
    const barH = Math.max(3, (h / max) * plotH);
    const color = h < 0.5 ? "var(--status-critical)"
                : h < 2.0 ? "var(--status-warning)"
                : "var(--status-good)";
    const band = h < 0.5 ? "conserved" : h < 2.0 ? "moderately conserved" : "variable";
    // 2px surface gap between adjacent fills
    const g = ns("rect", {
      x: PAD_L + i * bw + 1,
      y: PAD_T + plotH - barH,
      width: Math.max(1, bw - 2),
      height: barH,
      fill: color,
      rx: Math.min(3, bw / 3),
    });
    const t = ns("title", {});
    t.textContent = `Position ${pos + 1} — entropy ${h.toFixed(2)} (${band})`;
    g.append(t);
    svg.append(g);
  });

  // baseline
  svg.append(ns("line", {
    x1: PAD_L, x2: W - 8, y1: PAD_T + plotH, y2: PAD_T + plotH,
    stroke: "var(--axis)", "stroke-width": 1,
  }));

  const xl = ns("text", {
    x: PAD_L, y: H - 5, fill: "var(--text-muted)", "font-size": 10,
    "font-family": "ui-monospace, monospace",
  });
  xl.textContent = `position 1 → ${entries.length}`;
  svg.append(xl);

  const wrap = el("div", "chart");
  wrap.append(svg);
  card.append(wrap);
  card.append(el("div", "kv",
    `Computed from ${ctx.homolog_count} distinct sequences. Axis maximum is ` +
    `log₂(${ctx.homolog_count}) = ${achievableMax.toFixed(2)}, the highest entropy this many ` +
    `sequences can produce. Hover a bar for its value.`));
  return card;
}

const VERDICT = {
  recommend: { glyph: "✓", word: "Recommend" },
  recommend_with_caveats: { glyph: "~", word: "With caveats" },
  not_recommended: { glyph: "✕", word: "Not recommended" },
};

function renderRec(rec, rank) {
  const d = el("details", "rec");
  if (rank === 1) d.open = true;

  const summary = document.createElement("summary");
  summary.className = "rec-head";

  const v = VERDICT[rec.net_recommendation] || { glyph: "?", word: rec.net_recommendation };
  const badge = el("span", `verdict ${rec.net_recommendation}`);
  badge.append(el("span", null, v.glyph), el("span", null, v.word));

  summary.append(
    el("span", "rank", String(rank)),
    el("span", "mut", rec.label),
    badge,
    el("span", "net", (rec.net_score >= 0 ? "+" : "") + rec.net_score.toFixed(2)),
    el("span", "caret", "›")
  );
  d.append(summary);

  const body = el("div", "rec-body");
  const limitedBy = rec.score_breakdown && rec.score_breakdown.confidence_limited_by;
  body.append(el("div", "kv",
    `Target category: ${rec.target_category} · overall confidence: ${rec.overall_confidence}` +
    (limitedBy ? ` — limited by: ${limitedBy}` : "")));

  const bd = rec.score_breakdown;
  if (bd && bd.formula) {
    const box = el("div", "derivation");
    box.append(el("b", null, "How this score was derived"));
    const eq = el("div", "eq");
    eq.append(
      el("span", null, `benefit ${bd.expected_benefit.toFixed(3)}`),
      el("span", "op", "−"),
      el("span", null, `off-target cost ${bd.combined_cost.toFixed(3)}`),
      el("span", "op", "="),
      el("span", "res", (bd.net >= 0 ? "+" : "") + bd.net.toFixed(3))
    );
    box.append(eq);
    box.append(el("div", "formula", bd.formula));
    body.append(box);
  }

  if (rec.primary_effect) body.append(renderEffect(rec.primary_effect, "Primary effect"));
  rec.off_target_effects.forEach((e, i) =>
    body.append(renderEffect(e, `Off-target ${i + 1}`)));

  d.append(body);
  return d;
}

function renderEffect(eff, kindLabel) {
  const box = el("div", "effect");

  const head = el("div", "effect-head");
  head.append(
    el("span", "effect-kind", kindLabel),
    el("span", "effect-desc", eff.description),
    el("span", "tier", eff.evidence_tier)
  );
  box.append(head);

  const m = el("div", "meters");
  m.append(
    meter("Confidence it is real", eff.score),
    meter("Magnitude if real", eff.magnitude, "mag"),
    meter("Contribution to score", eff.contribution, "contrib")
  );
  box.append(m);

  box.append(el("p", "reasoning", eff.reasoning));

  if (eff.equation_refs && eff.equation_refs.length) {
    box.append(el("div", "eqrefs", `equations: #${eff.equation_refs.join(", #")}`));
  }
  return box;
}

/* ---------- Workflow 2 ---------- */

async function onFind() {
  const btn = $("#btn-find");
  const out = $("#find-results");
  out.innerHTML = "";
  const query = $("#query").value.trim();

  if (!query) {
    out.append(notice("Enter a functional goal to search for.", "error", "×"));
    return;
  }

  busy(btn, true);
  try {
    const data = await api("/api/find-peptides", { query, force_full: $("#force").checked });
    renderFindResults(data);
  } catch (e) {
    out.append(notice(e.message, "error", "×"));
  } finally {
    busy(btn, false, "Search");
  }
}

function renderFindResults(r) {
  const out = $("#find-results");
  const pn = policyNotice(r.policy);
  if (pn) out.append(pn);

  if (r.cell_types.length) {
    const c = el("div", "card");
    c.append(el("h2", null, `Step 1 — cell types implicated (${r.cell_types.length})`));
    r.cell_types.forEach((ct) => {
      const b = el("div", "celltype");
      const h = el("div");
      h.append(el("span", "n", ct.name));
      if (ct.ontology_id) h.append(el("span", "oid", ct.ontology_id));
      b.append(h, el("div", "r", ct.relationship), el("div", "s", `source: ${ct.source}`));
      c.append(b);
    });
    out.append(c);
  }

  if (r.known_answer_found) {
    const c = el("div", "card");
    c.append(el("h2", null, "Step 2 — the literature already answers this"));
    c.append(notice(
      "A clear answer already exists for this function, so the full expression-comparison " +
      "ranking was not forced. These are established associations, not novel predictions.",
      "info", "i"
    ));
    r.known_answers.forEach((cand) => c.append(renderCandidate(cand)));
    out.append(c);
  }

  if (r.candidates.length) {
    const c = el("div", "card");
    c.append(el("h2", null, `Step 3 — ranked candidates (${r.candidates.length})`));
    r.candidates.forEach((cand) => c.append(renderCandidate(cand)));
    out.append(c);
  }

  r.data_notes.forEach((n) => out.append(notice(n)));
  out.append(el("div", "methodology", r.methodology_note));
}

function renderCandidate(c) {
  const box = el("div", "cand");

  const head = el("div", "cand-head");
  head.append(el("span", "n", c.name));
  if (c.gene) head.append(el("span", "g", c.gene));
  head.append(el("span", "sc", c.combined_score.toFixed(2)));
  box.append(head);

  box.append(el("div", "kv", `${c.confidence} confidence · ${c.evidence_tier}`));
  box.append(el("p", "rationale", c.rationale));

  const lit = el("div", "kv");
  lit.append(el("b", null, "Literature "), document.createTextNode(c.literature_note));
  box.append(lit);

  const exp = el("div", "kv");
  exp.append(el("b", null, "Expression "), document.createTextNode(c.expression_note));
  box.append(exp);

  if (c.citations.length) {
    box.append(el("div", "cite", c.citations.join(" · ")));
  }
  if (c.requires_verification) {
    box.append(el("div", "cite", "⚠ Requires independent verification against primary literature."));
  }
  return box;
}

/* ---------- wiring ---------- */

const TABS = ["optimize", "transform", "find"];

function switchTab(which) {
  TABS.forEach((name) => {
    const selected = name === which;
    $(`#tab-${name}`).setAttribute("aria-selected", String(selected));
    $(`#panel-${name}`).hidden = !selected;
  });
}

async function init() {
  TABS.forEach((name) => $(`#tab-${name}`).addEventListener("click", () => switchTab(name)));
  $("#btn-transform").addEventListener("click", onTransform);
  $("#btn-infer").addEventListener("click", onAnalyze);
  $("#btn-find").addEventListener("click", onFind);
  $("#query").addEventListener("keydown", (e) => { if (e.key === "Enter") onFind(); });

  EXAMPLES.sequences.forEach((ex) => {
    const b = el("button", "chip", ex.label);
    b.addEventListener("click", () => { $("#seq").value = ex.seq; });
    $("#seq-examples").append(b);
  });

  EXAMPLES.sequences.forEach((ex) => {
    const b = el("button", "chip", ex.label);
    b.addEventListener("click", () => { $("#tseq").value = ex.seq; });
    $("#tseq-examples").append(b);
  });

  EXAMPLES.queries.forEach((q) => {
    const b = el("button", "chip", q);
    b.addEventListener("click", () => { $("#query").value = q; onFind(); });
    $("#query-examples").append(b);
  });

  try {
    const [health, goals] = await Promise.all([api("/api/health"), api("/api/goals")]);
    $("#version").textContent = `v${health.version}`;
    GOALS = goals.goals;
  } catch {
    $("#version").textContent = "offline";
  }
}

init();

/* ---------- Transformations ---------- */

const OBJECTIVE_LABELS = {
  potency: "Potency",
  functional_selectivity_bias: "Functional selectivity / bias",
  proteolytic_half_life: "Proteolytic half-life",
  albumin_fcrn_engagement: "Albumin / FcRn engagement",
  aggregation_propensity: "Aggregation propensity",
  solubility_at_formulation_ph: "Solubility at formulation pH",
  immunogenicity_risk: "Immunogenicity risk",
  synthesizability: "Synthesizability",
};

const MOVE_LABELS = {
  backbone_constraint: "Backbone constraint",
  side_chain_substitution: "Side-chain substitution",
  lipidation: "Lipidation",
  cyclization_stapling: "Cyclization / stapling",
  terminal_capping: "Terminal capping",
  glycosylation: "Glycosylation",
  disulfide_surrogate: "Disulfide surrogate",
  charge_engineering: "Charge engineering",
  liability_removal: "Liability removal",
};

async function onTransform() {
  const btn = $("#btn-transform");
  const out = $("#transform-results");
  out.innerHTML = "";
  const sequence = $("#tseq").value.trim();

  if (!sequence) {
    out.append(notice("Enter a peptide sequence.", "error", "×"));
    return;
  }

  busy(btn, true);
  try {
    const ph = parseFloat($("#tph").value);
    const data = await api("/api/transform", {
      sequence,
      formulation_ph: Number.isFinite(ph) ? ph : 7.4,
      is_internal_fragment: $("#tfrag").checked,
    });
    renderTransformResults(data);
  } catch (e) {
    out.append(notice(e.message, "error", "×"));
  } finally {
    busy(btn, false, "Generate transformations");
  }
}

function renderTransformResults(d) {
  const out = $("#transform-results");
  const pn = policyNotice(d.policy);
  if (pn) out.append(pn);
  const p = d.physics;

  // --- how far the physics actually got
  const tierCard = el("div", "card");
  tierCard.append(el("h2", null, "Physics tiers"));
  tierCard.append(el("div", "kv", p.summary));

  const ladder = el("div", "tier-ladder");
  p.tiers.forEach((t) => {
    const row = el("div", `tier-row ${t.available ? "ok" : "off"}`);
    row.append(el("span", "tier-n", `Tier ${t.tier}`));
    row.append(el("span", "tier-state", t.available ? "ran" : "did not run"));
    const detail = el("span", "tier-detail");
    if (t.available) {
      detail.textContent = t.illustrative_only
        ? "ran, but without pocket perturbation — see the pKa note below"
        : "computed";
    } else {
      detail.textContent = t.unavailable_reason || "";
      if (t.missing_dependency) {
        detail.append(el("span", "tier-dep", ` Needs: ${t.missing_dependency}`));
      }
    }
    row.append(detail);
    ladder.append(row);
  });
  tierCard.append(ladder);
  out.append(tierCard);

  // --- Tier 0 numbers
  const stats = el("div", "stats");
  const addStat = (k, v, sm) => {
    const s = el("div", "stat");
    s.append(el("div", "k", k), el("div", `v${sm ? " sm" : ""}`, v));
    stats.append(s);
  };
  addStat("Isoelectric point", `${p.isoelectric_point}`);
  addStat(`Net charge @ pH ${p.formulation_ph}`, `${p.net_charge_at_formulation_ph >= 0 ? "+" : ""}${p.net_charge_at_formulation_ph}`);
  addStat("Max μH (11-mer)", `${p.windowed_hydrophobic_moment.max_moment}`);
  addStat("Mean hydrophobicity", `${p.mean_hydrophobicity >= 0 ? "+" : ""}${p.mean_hydrophobicity}`);
  addStat("Liabilities", `${p.liabilities.length}`);
  out.append(stats);

  // --- charge vs pH
  out.append(renderChargeCurve(p));
  const titration = renderTitration(d.electrostatics);
  if (titration) out.append(titration);

  // --- liabilities
  if (p.liabilities.length) {
    const c = el("div", "card");
    c.append(el("h2", null, `Liability motifs (${p.liabilities.length})`));
    p.liabilities.forEach((l) => {
      const row = el("div", `liability sev-${l.severity}`);
      const head = el("div", "liability-head");
      head.append(
        el("span", "sev", l.severity.toUpperCase()),
        el("span", "lmotif", `${l.motif} @ ${l.display_position}`),
        el("span", "lclass", l.class.replace(/_/g, " "))
      );
      row.append(head);
      row.append(el("div", "kv", l.mechanism));
      const mit = el("div", "kv");
      mit.append(el("b", null, "Mitigation "), document.createTextNode(l.mitigation));
      row.append(mit);
      c.append(row);
    });
    out.append(c);
  }

  // --- excision site
  const ex = d.native_context.excision;
  if (ex && ex.is_internal_fragment) {
    const c = el("div", "card");
    c.append(el("h2", null, "Excision-site check"));
    c.append(notice(ex.priority_note, "info", "i"));
    c.append(el("div", "kv", ex.n_terminal_artifact));
    c.append(el("div", "kv", ex.c_terminal_artifact));
    out.append(c);
  }

  d.native_context.data_notes.forEach((n) => out.append(notice(n)));

  // --- transformations
  const tc = el("div", "card");
  tc.append(el("h2", null, `Ranked transformations (${d.transformations.length})`));
  tc.append(notice(d.comparability_warning, "warn", "!"));
  tc.append(el("div", "kv", d.scalarization_note));

  const w = el("div", "weights");
  w.append(el("b", null, "Scalarization weights "));
  w.append(document.createTextNode(
    Object.entries(d.weights).map(([k, v]) => `${OBJECTIVE_LABELS[k] || k} ${v}`).join(" · ")
  ));
  tc.append(w);

  d.transformations.forEach((t, i) => tc.append(renderTransformation(t, i + 1)));

  if (d.rejected.length) {
    const r = el("div", "kv");
    r.append(el("b", null, `${d.rejected.length} transformation(s) withheld `));
    r.append(document.createTextNode(
      "for violating the output contract: " +
      d.rejected.map((x) => `${x.description} (${x.violations.join("; ")})`).join(" | ")
    ));
    tc.append(r);
  }

  out.append(tc);
}

// Per-residue titration across the compartment series. A table rather than a
// curve: the question is which residue carries the charge and whether that
// changes between compartments, and a net-charge curve answers neither.
//
// "Changes state" is marked with a glyph and a word, never by colour alone --
// the switchable and static rows have to stay distinguishable to a reader who
// cannot separate the two hues.
function renderTitration(e) {
  if (!e) return null;
  const card = el("div", "card");
  card.append(el("h2", null, "Protonation by compartment"));
  card.append(el("div", "kv", e.summary));

  const labels = Object.keys(e.compartments);

  if (e.titrations.length) {
    const table = el("table", "titration");
    const head = el("tr");
    head.append(el("th", null, "Residue"), el("th", null, "Group"),
              el("th", null, "Charge"), el("th", null, "pKa"));
    labels.forEach((l) => head.append(el("th", null, `${l} (pH ${e.compartments[l]})`)));
    head.append(el("th", null, "Across range"));
    table.append(head);

    e.titrations.forEach((t) => {
      const row = el("tr", t.is_switchable ? "switchable" : null);
      // Protonated fraction is the same physics for an acid and a base, but it
      // means opposite things: a fully protonated lysine is charged, a fully
      // protonated glutamate is neutral. The sign column says which, so the
      // percentages can stay comparable without being misread.
      const physio = t.points[0];
      const sign = physio.effective_charge > 0.05 ? "+"
                 : physio.effective_charge < -0.05 ? "\u2212" : "0";
      row.append(el("td", "res", `${t.residue}${t.display_position}`));
      row.append(el("td", "grp", t.group));
      const chg = el("td", "sign", sign);
      chg.title = `effective charge at pH ${physio.ph}: ${physio.effective_charge.toFixed(2)}`;
      row.append(chg);
      row.append(el("td", "num", `${t.pka}`));

      labels.forEach((label) => {
        const point = t.points.find((p) => p.compartment === label);
        const cell = el("td", "frac");
        const bar = el("div", "fracbar");
        const fill = el("div", "fracfill");
        fill.style.width = `${Math.round(point.protonated_fraction * 100)}%`;
        bar.append(fill);
        cell.append(el("span", "fracnum", `${Math.round(point.protonated_fraction * 100)}%`), bar);
        cell.title = `protonated fraction ${point.protonated_fraction.toFixed(3)}`;
        row.append(cell);
      });

      const verdict = el("td", "verdict");
      verdict.append(
        el("span", "glyph", t.is_switchable ? "\u25C6" : "\u25CB"),
        el("span", null, t.is_switchable
          ? `changes (${Math.round(t.protonation_swing * 100)} pts)`
          : "no change")
      );
      row.append(verdict);
      table.append(row);
    });

    const wrap = el("div", "tablewrap");
    wrap.append(table);
    card.append(wrap);
    card.append(el("div", "footnote",
      `Protonated fraction from Henderson-Hasselbalch. Protonated means charged for a base ` +
      `(K, R, H) and neutral for an acid (D, E, C, Y) \u2014 the charge column gives the sign ` +
      `at physiological pH. A residue counts as changing when its protonated fraction moves ` +
      `by at least ${Math.round(e.switch_threshold * 100)} points across the range.`));

    e.titrations.filter((t) => t.uncertainty_note).forEach((t) => {
      card.append(notice(`${t.residue}${t.display_position}: ${t.uncertainty_note}`));
    });
  }

  const net = el("div", "netcharge");
  labels.forEach((label) => {
    const b = el("div", "stat");
    b.append(el("div", "k", `net charge, ${label}`),
             el("div", "v", e.net_charge[label].toFixed(2)));
    net.append(b);
  });
  card.append(net);

  (e.notes || []).forEach((n) => card.append(notice(n)));
  return card;
}

function renderChargeCurve(p) {
  const card = el("div", "card");
  card.append(el("h2", null, "Net charge vs pH"));

  const pts = p.charge_vs_ph;
  const W = 1000, H = 190, PAD_L = 40, PAD_B = 26, PAD_T = 10, PAD_R = 12;
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_B - PAD_T;

  const charges = pts.map((q) => q[1]);
  const yMax = Math.ceil(Math.max(...charges, 1));
  const yMin = Math.floor(Math.min(...charges, -1));
  const x = (ph) => PAD_L + ((ph - pts[0][0]) / (pts[pts.length - 1][0] - pts[0][0])) * plotW;
  const y = (q) => PAD_T + plotH - ((q - yMin) / (yMax - yMin)) * plotH;

  const ns = (tag, attrs) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Net charge as a function of pH");
  svg.style.display = "block";

  for (let q = yMin; q <= yMax; q++) {
    const yy = y(q);
    svg.append(ns("line", {
      x1: PAD_L, x2: W - PAD_R, y1: yy, y2: yy,
      stroke: q === 0 ? "var(--axis)" : "var(--grid)",
      "stroke-width": q === 0 ? 1.5 : 1,
    }));
    const lbl = ns("text", {
      x: PAD_L - 7, y: yy + 3.5, "text-anchor": "end",
      fill: "var(--text-muted)", "font-size": 10, "font-family": "ui-monospace, monospace",
    });
    lbl.textContent = q > 0 ? `+${q}` : `${q}`;
    svg.append(lbl);
  }

  [2, 4, 6, 8, 10, 12, 14].forEach((ph) => {
    const lbl = ns("text", {
      x: x(ph), y: H - 8, "text-anchor": "middle",
      fill: "var(--text-muted)", "font-size": 10, "font-family": "ui-monospace, monospace",
    });
    lbl.textContent = `pH ${ph}`;
    svg.append(lbl);
  });

  svg.append(ns("path", {
    d: pts.map((q, i) => `${i ? "L" : "M"}${x(q[0]).toFixed(1)},${y(q[1]).toFixed(1)}`).join(" "),
    fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }));

  // pI marker: where the curve crosses zero
  const pI = p.isoelectric_point;
  svg.append(ns("line", {
    x1: x(pI), x2: x(pI), y1: PAD_T, y2: PAD_T + plotH,
    stroke: "var(--text-muted)", "stroke-width": 1, "stroke-dasharray": "3 3",
  }));
  const pilbl = ns("text", {
    x: x(pI) + 5, y: PAD_T + 12, fill: "var(--text-secondary)",
    "font-size": 11, "font-family": "ui-monospace, monospace",
  });
  pilbl.textContent = `pI ${pI}`;
  svg.append(pilbl);

  // formulation pH marker
  svg.append(ns("circle", {
    cx: x(p.formulation_ph), cy: y(p.net_charge_at_formulation_ph), r: 4,
    fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2,
  }));

  const wrap = el("div", "chart");
  wrap.append(svg);
  card.append(wrap);
  card.append(el("div", "kv",
    `Computed per residue via Henderson-Hasselbalch. Marker shows the formulation pH ` +
    `(${p.formulation_ph}, net ${p.net_charge_at_formulation_ph >= 0 ? "+" : ""}` +
    `${p.net_charge_at_formulation_ph}). Solubility is typically worst near pI.`));
  return card;
}

function renderTransformation(t, rank) {
  const d = el("details", "rec xform");
  if (rank === 1) d.open = true;

  const summary = document.createElement("summary");
  summary.className = "rec-head";
  const s = t.scalarized;

  summary.append(
    el("span", "rank", String(rank)),
    el("span", "movetag", MOVE_LABELS[t.move] || t.move),
    el("span", "xdesc", t.description)
  );
  if (t.tradeoff_label) summary.append(el("span", "trade-pill", "TRADE"));
  if (t.leakage_flag) summary.append(el("span", "leak-pill", "KNOWN ANALOG"));
  summary.append(
    el("span", "net", `${s.score >= 0 ? "+" : ""}${s.score.toFixed(2)}`),
    el("span", "cov", `${Math.round(s.coverage * 100)}%`),
    el("span", "caret", "›")
  );
  d.append(summary);

  const body = el("div", "rec-body");

  if (t.tradeoff_label) body.append(notice(t.tradeoff_label, "warn", "⇄"));
  if (t.leakage_flag) body.append(notice(t.leakage_flag.note, "info", "i"));

  body.append(el("p", "reasoning", t.rationale));

  // objective vector
  const ov = el("div", "objvec");
  ov.append(el("b", null, "Objective vector"));
  t.objective_deltas.forEach((od) => ov.append(renderObjectiveRow(od)));
  body.append(ov);

  body.append(el("div", "kv",
    `Scalarized ${s.score >= 0 ? "+" : ""}${s.score.toFixed(3)}. ${s.caveat}`));

  if (t.preorganization) {
    const pre = el("div", "derivation");
    pre.append(el("b", null, "Pre-organization proxy"));
    pre.append(el("div", "kv", t.preorganization.rendered));
    if (t.preorganization.claim) {
      pre.append(el("div", "eqrefs", t.preorganization.claim.rendered));
    }
    body.append(pre);
  }

  if (t.assumptions.length) {
    const a = el("div", "assumptions");
    a.append(el("b", null, "Assumptions"));
    t.assumptions.forEach((as) => {
      const row = el("div", "assumption");
      row.append(el("span", "akind", as.kind.replace(/_/g, " ")));
      row.append(el("div", "kv", as.statement));
      if (as.impact_if_wrong) {
        const imp = el("div", "kv");
        imp.append(el("b", null, "If wrong "), document.createTextNode(as.impact_if_wrong));
        row.append(imp);
      }
      a.append(row);
    });
    body.append(a);
  }

  t.notes.forEach((n) => body.append(el("div", "eqrefs", n)));
  body.append(el("div", "eqrefs",
    `evidence tier: ${t.evidence_tier || "n/a"} · physics tier reached: ${t.physics_tier_reached}`));

  d.append(body);
  return d;
}

function renderObjectiveRow(od) {
  const row = el("div", "objrow");
  row.append(el("span", "objname", OBJECTIVE_LABELS[od.objective] || od.objective));

  if (!od.assessed) {
    // Rendered as visibly absent, never as a zero: "not looked at" must not
    // read the same as "no effect".
    row.append(el("span", "objtrack unassessed"));
    const r = el("span", "objval unassessed-label", "not assessed");
    r.title = od.claim ? od.claim.inference_step : "";
    row.append(r);
    return row;
  }

  const track = el("span", "objtrack");
  const mid = el("span", "objmid");
  const bar = el("span", `objbar ${od.signed >= 0 ? "pos" : "neg"}`);
  const pct = Math.min(1, Math.abs(od.signed)) * 50;
  if (od.signed >= 0) {
    bar.style.left = "50%";
    bar.style.width = `${pct}%`;
  } else {
    bar.style.right = "50%";
    bar.style.width = `${pct}%`;
  }
  track.append(mid, bar);
  row.append(track);

  const val = el("span", "objval", `${od.signed >= 0 ? "+" : ""}${od.signed.toFixed(2)}`);
  row.append(val);
  row.append(el("span", "objclaim", od.claim ? od.claim.type : ""));
  if (od.claim) row.title = od.claim.rendered;
  return row;
}
