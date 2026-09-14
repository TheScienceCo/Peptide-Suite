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
    { label: "IGF-1 fragment", seq: "MGFPGLQPRRVSCGQAKDEARGYLCFSQTPRAT" },
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
    $("#gate-slot").append(notice("Enter a peptide sequence to analyze.", "error", "×"));
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

  if (info.parse_error) {
    slot.append(notice(`Could not read that as a peptide sequence — ${info.parse_error}`, "error", "×"));
    return;
  }

  const gate = el("div", "gate");
  gate.append(el("h2", null, "Confirm the target function"));

  const p = el("p");
  p.append(
    document.createTextNode("Based on the available records, you are seeking a form of: "),
    el("span", "quote", info.inferred_function)
  );
  gate.append(p);

  if (info.inference_confidence === 0) {
    gate.append(notice(
      "This peptide was not recognised, so no function could be inferred from records. " +
      "Select the target function yourself — the scan will not run until you do.",
      "info", "i"
    ));
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
  goalWrap.append(sel);

  const desc = el("div", "kv");
  desc.id = "goal-desc";
  const syncDesc = () => {
    const g = GOALS.find((x) => x.id === sel.value);
    desc.textContent = g ? g.description : "";
  };
  sel.addEventListener("change", syncDesc);
  syncDesc();

  const btnWrap = el("div", "shrink");
  const run = el("button", "primary", "Run substitution scan");
  run.id = "btn-scan";
  run.addEventListener("click", () => runScan(info.sequence, sel.value));
  btnWrap.append(run);

  row.append(goalWrap, btnWrap);
  gate.append(row, desc);

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

function switchTab(which) {
  const isOpt = which === "optimize";
  $("#tab-optimize").setAttribute("aria-selected", String(isOpt));
  $("#tab-find").setAttribute("aria-selected", String(!isOpt));
  $("#panel-optimize").hidden = !isOpt;
  $("#panel-find").hidden = isOpt;
}

async function init() {
  $("#tab-optimize").addEventListener("click", () => switchTab("optimize"));
  $("#tab-find").addEventListener("click", () => switchTab("find"));
  $("#btn-infer").addEventListener("click", onAnalyze);
  $("#btn-find").addEventListener("click", onFind);
  $("#query").addEventListener("keydown", (e) => { if (e.key === "Enter") onFind(); });

  EXAMPLES.sequences.forEach((ex) => {
    const b = el("button", "chip", ex.label);
    b.addEventListener("click", () => { $("#seq").value = ex.seq; });
    $("#seq-examples").append(b);
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
