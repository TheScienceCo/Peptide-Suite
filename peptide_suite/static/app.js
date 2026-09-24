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

// The identification the user is currently looking at, and the input it was
// made from. Keyed on the input so that editing the sequence invalidates it:
// showing GLP-1's receptors above a scan of something else is the exact
// confusion the two-step flow exists to prevent.
let IDENTIFIED = null;

function currentInput() {
  return $("#seq").value.trim();
}

/**
 * Step one. Work out what the molecule IS -- identity, receptors, binding
 * sites, domain structure, basic properties -- and stop there.
 *
 * Split out from Analyze because the two questions are different and the
 * second is only worth asking once the first has an answer. Previously one
 * button did both, so the goal selector appeared beside the identification and
 * a reader chose an engineering objective while still reading what the thing
 * was.
 */
async function runIdentify({ thenGate = false } = {}) {
  const btn = thenGate ? $("#btn-analyze") : $("#btn-identify");
  const input = currentInput();
  $("#gate-slot").innerHTML = "";
  $("#opt-results").innerHTML = "";

  if (!input) {
    $("#gate-slot").append(notice("Enter a peptide sequence or a peptide name.", "error", "×"));
    return;
  }

  busy(btn, true);
  try {
    const info = await api("/api/infer-function", { input });
    IDENTIFIED = { input, info };
    renderIdentity(info);
    // The buttons advance either way. Reaching the identification through
    // Analyze is not a different state from reaching it through Identify, and
    // leaving Identify on screen afterwards would offer a step already taken.
    advanceToAnalyze(info);
    if (thenGate) renderGoalGate(info);
  } catch (e) {
    IDENTIFIED = null;
    $("#gate-slot").append(notice(e.message, "error", "×"));
    resetSteps();          // buttons only: the error above is what to read
  } finally {
    busy(btn, false, thenGate ? "Analyze" : "Identify");
  }
}

/**
 * Step two. Choose what to optimise for and scan.
 *
 * Reuses the identification when it is still the one on screen rather than
 * re-fetching, so Analyze does not silently replace the context the user is
 * reading. If the sequence has been edited since, this identifies again first
 * -- Analyze on its own has to work, because a user who ignores step one is
 * not making a mistake.
 */
async function onAnalyze() {
  const input = currentInput();
  if (!IDENTIFIED || IDENTIFIED.input !== input) {
    return runIdentify({ thenGate: true });
  }
  renderGoalGate(IDENTIFIED.info);
  const gate = $("#gate-slot .gate");
  if (gate) gate.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * After a successful identification, Analyze becomes the only button.
 *
 * Two buttons side by side ask the user to decide which one they want. One
 * button, after the step that produced something to act on, says what the next
 * step is.
 */
function advanceToAnalyze(info) {
  const identify = $("#btn-identify");
  const analyze = $("#btn-analyze");
  if (!identify || !analyze) return;

  // Nothing to analyse: a full-length protein has no single sequence to
  // optimise, and an unparseable input has none at all. Offering the next step
  // where there is no next step is worse than offering both.
  const analysable = Boolean(info.sequence) && !info.parse_error;
  identify.hidden = analysable;
  analyze.hidden = false;
  analyze.classList.toggle("primary", analysable);
  analyze.classList.toggle("ghost", !analysable);
  analyze.disabled = !analysable;

  const hint = $("#step-hint");
  if (hint) {
    hint.hidden = false;
    hint.textContent = analysable
      ? "Identified. Next: choose what to optimise for and scan the substitutions."
      : "Nothing here can be scanned — see above for why.";
  }
}

/**
 * Back to the two-button start.
 *
 * `clear` also empties the results, and the input listener passes it: a
 * changed sequence makes everything on screen describe a different molecule,
 * and resetting the buttons while leaving GLP-1's receptors above the fold is
 * the confusion this flow exists to prevent. The error path does not pass it,
 * because the thing to read there is the error.
 */
function resetSteps({ clear = false } = {}) {
  if (clear) {
    $("#gate-slot").innerHTML = "";
    $("#opt-results").innerHTML = "";
    IDENTIFIED = null;
  }
  const identify = $("#btn-identify");
  const analyze = $("#btn-analyze");
  if (!identify || !analyze) return;
  identify.hidden = false;
  analyze.hidden = false;
  analyze.disabled = false;
  analyze.classList.remove("primary");
  analyze.classList.add("ghost");
  const hint = $("#step-hint");
  if (hint) hint.hidden = true;
}

/**
 * What the molecule is: known biology, then how the identification was
 * reached, then its measurable properties. No engineering controls.
 */
function renderIdentity(info) {
  const slot = $("#gate-slot");
  slot.innerHTML = "";

  // Identity, then what is known about it. The two were the other way round
  // and it read badly: a context card reporting an empty lookup sat above the
  // card that had just identified the molecule, so the page appeared to fail
  // and succeed at the same question in that order.
  //
  // Both still come before any engineering control. Clicking Analyze used to
  // land straight on those, so a reader saw a score before they saw what the
  // system thought the molecule was.
  identityCard(info);
  const bio = renderBiologicalContext(info.biological_context, {
    name: info.matched_name || "",
  });
  if (bio) slot.append(bio);

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

}

/**
 * The identity card: inference level, UniProt, caveats and basic properties.
 * Split from the goal gate so Identify can show it without asking the user to
 * pick an engineering objective in the same breath.
 */
function identityCard(info) {
  const slot = $("#gate-slot");
  const gate = el("div", "gate");
  gate.append(el("h2", null, "What this is"));

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

  const props = info.properties;
  if (props) {
    gate.append(el("div", "kv",
      `${props.length} residues · ${props.charged_residues} charged (${props.percent_charged}%) · ` +
      `${props.aromatic_count} aromatic · ${props.proline_count} Pro · ${props.cysteine_count} Cys`));
  }

  slot.append(gate);
}

/**
 * The goal gate: what to optimise for, and the button that scans.
 *
 * Appended below the identity rather than replacing it, so the context stays
 * on screen while the objective is chosen.
 */
function renderGoalGate(info) {
  const slot = $("#gate-slot");
  if ($("#goal-select")) return;          // already open; do not stack two
  if (!info.sequence || info.parse_error) return;

  const gate = el("div", "gate");
  gate.append(el("h2", null, "Confirm the target function"));

  const row = el("div", "row");

  const goalWrap = el("div");
  goalWrap.append(el("label", null, "Target function"));
  const sel = el("select");
  sel.id = "goal-select";
  GOALS.forEach((g) => {
    const o = el("option", null, g.label + (g.quick_win ? "  — quick win" : ""));
    o.title = g.description || "";
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
  run.addEventListener("click", () => {
    ACTIVE_GOAL = GOALS.find((g) => g.id === sel.value) || null;
    runScan(info.sequence, sel.value);
  });
  btnWrap.append(run);

  row.append(goalWrap, btnWrap);
  gate.append(row, why, desc);

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

const HOMOLOG_SOURCE_LABEL = {
  LIVE: "database",
  CACHED: "cached lookup",
  LOCAL_FIXTURE: "bundled fixture",
  UNAVAILABLE: "none obtained",
  CALLER_SUPPLIED: "supplied by you",
};

function renderOptimizeResults(data) {
  const out = $("#opt-results");
  const pn = policyNotice(data.policy);
  if (pn) out.append(pn);

  // What the selected lane computes, before any score from it.
  const cap = renderCapability(ACTIVE_GOAL);
  if (cap) out.append(cap);
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
  // The provenance sits next to the count, not in a footnote. The same number
  // of homologs means something different depending on where they came from,
  // and conservation is only as good as this.
  addStat("Homolog source", HOMOLOG_SOURCE_LABEL[ctx.homolog_source] || "\u2014", true);
  addStat("Conservation", ctx.conservation_available ? "computed" : "unavailable", true);
  out.append(stats);

  if (ctx.homolog_source_detail && ctx.homolog_source !== "CALLER_SUPPLIED") {
    out.append(notice(ctx.homolog_source_detail,
                      ctx.homolog_source === "LOCAL_FIXTURE" ? "warn" : "info",
                      ctx.homolog_source === "LOCAL_FIXTURE" ? "!" : "i"));
  }

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

// The goal the current scan ran against, so a recommendation can state the
// limits of the lane that produced it.
let ACTIVE_GOAL = null;

function limitsOf(rec) {
  const box = el("div", "limits");
  const goal = ACTIVE_GOAL
    || GOALS.find((g) => g.id === rec.target_category)
    || null;

  box.append(el("b", null, "What this does not establish"));
  const list = el("ul", "limitlist");
  const items = (goal && goal.does_not_compute) ? goal.does_not_compute.slice() : [];

  // The direction limit is already in the goal's does_not_compute list, so it
  // is not appended again here. It appeared twice in the first version.
  if (!items.length) items.push("No limits were declared for this goal.");
  items.forEach((t) => list.append(el("li", null, t)));
  box.append(list);

  if (goal && goal.reasoning_kinds) {
    box.append(el("div", "kv",
      "Four kinds of reasoning could bear on a binding question. This result uses "
      + "only the ones marked available:"));
    const table = el("div", "kinds");
    goal.reasoning_kinds.forEach((k) => {
      const row = el("div", `kind ${k.available ? "on" : "off"}`);
      const head = el("div", "khead");
      head.append(el("span", "kstate", k.available ? "used" : "unavailable"),
                  el("b", null, k.label));
      row.append(head);
      row.append(el("div", "kv", k.available ? k.what_it_is : k.what_it_is_not));
      table.append(row);
    });
    box.append(table);
  }
  return box;
}

function renderCapability(goal) {
  // The capability panel: what the selected lane computes and what it does
  // not. A label reading "Binding affinity" implied a computed dissociation
  // constant; what is computed is the size of a perturbation.
  if (!goal) return null;
  const box = el("div", "capability");
  const head = el("div", "caphead");
  head.append(el("span", "bctitle", "What this analysis computes"));
  if (goal.strongest_evidence) head.append(tierBadge(goal.strongest_evidence, false));
  box.append(head);

  const cols = el("div", "capcols");
  const yes = el("div", "capcol");
  yes.append(el("div", "capsub", "Computed"));
  const yesList = el("ul", "limitlist");
  (goal.computes || []).forEach((t) => yesList.append(el("li", null, t)));
  yes.append(yesList);

  const no = el("div", "capcol");
  no.append(el("div", "capsub", "Not computed"));
  const noList = el("ul", "limitlist");
  (goal.does_not_compute || []).forEach((t) => noList.append(el("li", null, t)));
  no.append(noList);

  cols.append(yes, no);
  box.append(cols);
  return box;
}

const VERDICT = {
  recommend: { glyph: "✓", word: "Recommend" },
  recommend_with_caveats: { glyph: "~", word: "With caveats" },
  not_recommended: { glyph: "✕", word: "Not recommended" },
};

// ---- experimental precedent -------------------------------------------------
//
// Known biology before prediction: what somebody measured outranks what this
// program computed, so it goes above the derivation rather than below it.
//
// An empty result is rendered, not omitted. "Nobody has measured this" and "we
// did not look" are the same blank space on a page and opposite facts, and a
// missing section reads as the first while meaning the second.

const MATCH_LABEL = {
  SAME_PEPTIDE_SAME_SUBSTITUTION: "Same change, same molecule",
  SAME_PEPTIDE_SAME_POSITION: "Same position, different residue",
  SAME_SUBSTITUTION_OTHER_PEPTIDE: "Same change, different peptide",
  SAME_PEPTIDE_OTHER_CHANGE: "Same molecule, different change",
};

function precedentOf(rec) {
  const p = rec.experimental_precedent;
  const box = el("div", "precedent");
  const head = el("div", "phead");
  head.append(el("b", null, "Experimental precedent"));

  if (!p || !p.searched) {
    head.append(el("span", "pstate none", "not searched"));
    box.append(head);
    box.append(el("div", "kv",
      "The variant-evidence store was not queried for this substitution, so nothing "
      + "here says whether a measurement exists."));
    return box;
  }

  const found = p.has_precedent;
  head.append(el("span", `pstate ${found ? "yes" : "none"}`,
                 found ? "measured" : "none found"));
  box.append(head);
  box.append(el("div", "kv", p.statement));

  if (p.store_is_empty) return box;

  (p.matches || []).forEach((m) => {
    const row = el("div", `pmatch ${m.usable_as_evidence ? "usable" : "context"}`);
    const top = el("div", "pmhead");
    top.append(el("b", null, m.record.name),
               el("span", "pkind", MATCH_LABEL[m.kind] || m.kind));
    if (m.record.strongest_tier) top.append(tierBadge(m.record.strongest_tier));
    row.append(top);
    row.append(el("div", "kv", m.caveat));

    // The confounding note comes before the outcomes, not after. A reader who
    // meets "12-fold" first has already attributed it to the substitution.
    const c = m.record.confounding;
    if (c && c.is_confounded) row.append(el("div", "confound", c.statement));

    if (!m.record.has_evidence) {
      row.append(el("div", "kv",
        "A modification with no measured outcome: a design, not evidence."));
    } else {
      const list = el("ul", "outcomes");
      m.record.outcomes.forEach((o) => list.append(el("li", null, o.description)));
      row.append(list);
    }
    box.append(row);
  });
  return box;
}

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

  // What this result does not establish. Placed before the arithmetic rather
  // than after it: a reader who meets the limits only after the number has
  // already formed a view about the number.
  body.append(limitsOf(rec));

  // Above the arithmetic, because a measurement outranks a computation.
  body.append(precedentOf(rec));

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

/* ---------- Model evaluation ---------- */

async function onRunML() {
  const btn = $("#btn-ml");
  const out = $("#ml-results");
  out.innerHTML = "";
  busy(btn, true);
  try {
    const families = parseInt($("#ml-families").value, 10) || 40;
    const per = parseInt($("#ml-per").value, 10) || 8;
    const d = await api(`/api/ml/split-comparison?n_families=${families}&per_family=${per}`);
    renderMLResults(d);
  } catch (e) {
    out.append(notice(e.message, "error", "×"));
  } finally {
    busy(btn, false, "Run the comparison");
  }
}

function renderMLResults(d) {
  const out = $("#ml-results");

  // The synthetic disclaimer goes first. A table of AUCs reads as a result
  // about peptides unless something says otherwise before it is read.
  out.append(notice(d.claim_guidance, "warn", "!"));

  const leak = el("div", "card");
  leak.append(el("h2", null, "Leakage"));
  const stats = el("div", "stats");
  const addStat = (k, v, sm) => {
    const s = el("div", "stat");
    s.append(el("div", "k", k), el("div", `v${sm ? " sm" : ""}`, v));
    stats.append(s);
  };
  addStat("Sequences", `${d.n_sequences}`);
  addStat("Clusters", `${d.n_clusters}`);
  addStat("Identity threshold", `${Math.round(d.identity_threshold * 100)}%`);
  addStat("Leaked, random", `${d.leakage.random}`);
  addStat("Leaked, clustered", `${d.leakage.clustered}`);
  leak.append(stats);
  leak.append(el("div", "footnote",
    "Test sequences with a neighbour above the identity threshold in training."));
  out.append(leak);

  const card = el("div", "card");
  card.append(el("h2", null, "ROC-AUC by split"));
  const wrap = el("div", "tablewrap");
  const table = el("table", "titration");
  const head = el("tr");
  head.append(el("th", null, "Arm"), el("th", null, "Random split"),
              el("th", null, "Clustered split"), el("th", null, "Gap"));
  table.append(head);

  const arms = [...new Set(d.arms.map((a) => a.arm))];
  arms.forEach((arm) => {
    const rnd = d.arms.find((a) => a.arm === arm && a.split === "random");
    const clu = d.arms.find((a) => a.arm === arm && a.split === "sequence_clustered");
    const gap = d.gaps[arm];
    const row = el("tr", gap > 0.3 ? "switchable" : null);
    const cell = (m) => {
      const td = el("td", "num");
      if (!m || m.roc_auc === null) { td.textContent = "—"; return td; }
      td.append(el("span", null, m.roc_auc.toFixed(3)));
      if (m.ci_low !== null) {
        td.append(el("div", "footnote",
          `${m.ci_low.toFixed(2)}\u2013${m.ci_high.toFixed(2)}`));
      }
      return td;
    };
    row.append(el("td", "armname", arm));
    row.append(cell(rnd), cell(clu));
    const g = el("td", "verdict");
    g.append(el("span", "glyph", gap > 0.3 ? "\u25C6" : "\u25CB"),
             el("span", null, gap === null ? "—" : `${gap >= 0 ? "+" : ""}${gap.toFixed(3)}`));
    row.append(g);
    table.append(row);
  });
  wrap.append(table);
  card.append(wrap);
  card.append(el("div", "footnote",
    "Intervals are 95% bootstrap. On a test set this size the interval is usually " +
    "wider than the differences people report."));
  (d.skipped || []).forEach((sk) => card.append(notice(sk, "info", "i")));
  out.append(card);

  const note = el("div", "card");
  note.append(el("h2", null, "Reading the table"));
  d.report.split("\n").forEach((line) => {
    if (line.trim()) note.append(el("p", "reasoning", line));
  });
  out.append(note);
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

const TABS = ["optimize", "transform", "landscape", "ml", "find"];

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
  $("#btn-identify").addEventListener("click", () => runIdentify());
  $("#btn-analyze").addEventListener("click", onAnalyze);
  // Editing the sequence invalidates the identification on screen, so the flow
  // returns to its start rather than letting Analyze scan one peptide under
  // another one's receptors.
  $("#seq").addEventListener("input", () => {
    if (IDENTIFIED && IDENTIFIED.input !== currentInput()) resetSteps({ clear: true });
  });
  $("#btn-find").addEventListener("click", onFind);
  $("#btn-ml").addEventListener("click", onRunML);
  $("#btn-fusion").addEventListener("click", onFusion);
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

  // After GOALS is populated: the landscape's goal select is built from it.
  await initLandscape();
  await initRepresentation();
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
  // What the input's length means for the analysis, stated before the results
  // rather than after: a protein run through a peptide frame produces output
  // that looks ordinary, and the caveat is only useful ahead of it.
  (d.length_notes || []).forEach((n) => out.append(notice(n, "warn", "!")));
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
  const template = renderStructureTemplate(d.structure_template, d.rejected);
  if (template) out.append(template);
  const contacts = renderClassifiedContacts(d.classified_contacts);
  if (contacts) out.append(contacts);
  const ens = renderEnsemble(d.conformer_ensemble);
  if (ens) out.append(ens);
  const feas = renderFeasibility(d.feasibility);
  if (feas) out.append(feas);
  const b1 = renderClassB1(d.class_b1);
  if (b1) out.append(b1);
  const partners = renderPartnerProposals(d.partner_proposals);
  if (partners) out.append(partners);
  const requests = renderResearchRequests(d.research_requests, d.registry_is_empty);
  if (requests) out.append(requests);

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
// Which structure conformational claims rest on, or why none was admitted.
// Rendered as a card rather than a footnote because a refusal is a result: it
// changes which proposals exist below, and a reader who does not see it will
// read a shorter list as "nothing else applies".
// Proposals that cannot be recommendations, because their chemistry has no
// force-field parameters. Rendered as their own section rather than as
// low-ranked entries in the main list: a research request with a rank beside it
// reads as a recommendation with a caveat, which is the failure this gate
// exists to prevent.
// Contacts the peptide makes in its native context, and what each one permits.
// Rendered above the proposals because the class determines which proposals are
// allowed to exist: an ESSENTIAL footprint is frozen, and a reader who meets
// that rule only as a rejection further down has already wondered why an
// obvious move is missing.
const CONTACT_CLASS_GLYPH = {
  SCAFFOLD: "\u25C7",     // removable by design
  PROTECTIVE: "\u25D1",   // replace, do not remove
  ESSENTIAL: "\u25C6",    // frozen
  UNRESOLVED: "\u25CB",   // unverified
};

// Co-agent proposals. Their own section, never mixed into the ranked list: a
// partner proposal beside single-peptide ones reads as a comparable
// alternative, and it is not — it changes what the product is.
// The class B1 placement rule, when a receptor was named. Shown as a band
// rather than prose: the restricted span is a fact about positions, and a
// reader deciding where to put a lipid wants to see the range.
// Synthesis and stability liabilities. Its own card because it answers a
// different question from the rest of the output: not "is this a good molecule"
// but "can this molecule be made as specified". A biophysics-only reading
// misses the whole class.
// Whether a conformer ensemble is admissible, and why not. Two failure modes
// with different remedies — too long for a search to be complete, versus no
// engine installed — so they are reported separately rather than as one
// "unavailable".
function renderEnsemble(e) {
  if (!e) return null;
  const card = el("div", "card");
  card.append(el("h2", null, "Conformer ensemble"));
  const head = el("div", "contact-head");
  head.append(
    el("span", "glyph", e.status === "eligible" ? "\u25C6" : "\u25CB"),
    el("span", "n", e.status.replace(/_/g, " ")),
    el("span", "tag", e.constraint.replace(/_/g, " ")),
    el("span", "tag", `${e.sequence_length} residues, ceiling ${e.ceiling}`)
  );
  card.append(head);
  card.append(el("div", "r", e.reason));
  if ((e.required_tooling || []).length) {
    card.append(el("div", "footnote", `Needs: ${e.required_tooling.join("; ")}`));
  }
  (e.notes || []).forEach((n) => card.append(notice(n, "info", "i")));
  return card;
}

// A proposal's outstanding n→π* obligation. Rendered on the proposal rather
// than in a footnote: for Aib, N-methylation and proline-rich segments it IS
// the mechanism, and without it the steric account below stands in for one.
function nboNotice(nbo) {
  if (!nbo || !nbo.required) return null;
  const n = notice(nbo.statement, nbo.outstanding ? "warn" : "info",
                   nbo.outstanding ? "!" : "i");
  return n;
}

function renderFeasibility(flags) {
  if (!flags || !flags.length) return null;
  const card = el("div", "card");
  card.append(el("h2", null, `Synthetic feasibility (${flags.length})`));
  flags.forEach((f) => {
    const box = el("div", `contact contact-${f.severity === "blocking" ? "essential" :
                            f.severity === "high" ? "protective" : "scaffold"}`);
    const head = el("div", "contact-head");
    head.append(el("span", "n", f.code.replace(/_/g, " ")),
                el("span", "tag", f.severity));
    if (f.positions && f.positions.length) {
      head.append(el("span", "tag", `pos ${f.positions.join(", ")}`));
    }
    box.append(head);
    box.append(el("div", "r", f.description));
    if (f.remedy) box.append(el("div", "footnote", `Remedy: ${f.remedy}`));
    box.append(el("div", "eqrefs", f.basis));
    card.append(box);
  });
  return card;
}

function renderClassB1(b1) {
  if (!b1 || !b1.applies) return null;
  const card = el("div", "card");
  card.append(el("h2", null, "Class B1 placement"));
  card.append(el("div", "kv", b1.rule));
  const span = el("div", "kv");
  span.textContent =
    `Restricted: positions ${b1.restricted_positions[0]}\u2013` +
    `${b1.restricted_positions[b1.restricted_positions.length - 1]}. ` +
    `Conjugation permitted from position ${b1.permitted_positions[0]} onward.`;
  card.append(span);
  card.append(el("div", "footnote", b1.worked_example));
  return card;
}

function renderPartnerProposals(proposals) {
  if (!proposals || !proposals.length) return null;
  const card = el("div", "card");
  card.append(el("h2", null, `Partner peptides (${proposals.length})`));
  card.append(el("div", "kv",
    "A second peptide participating in the activity. The mode decides what ships: " +
    "an obligate pair sold as a loose combination is two inactive peptides."));

  proposals.forEach((p) => {
    const box = el("div", "contact");
    const head = el("div", "contact-head");
    head.append(el("span", "n", `${p.primary} + ${p.partner}`),
                el("span", "tag", p.mode));
    box.append(head);
    box.append(el("div", "r", p.rationale));
    box.append(el("div", "footnote", p.mode_description));
    box.append(el("div", "footnote", `Ships as: ${p.shipping_requirement}`));

    const facts = [
      p.stoichiometry && `stoichiometry ${p.stoichiometry}`,
      p.engagement_order && `order: ${p.engagement_order}`,
      p.synergy_index && `synergy ${p.synergy_index}`,
      p.accessory_protein && `accessory ${p.accessory_protein}`,
    ].filter(Boolean);
    if (facts.length) box.append(el("div", "kv", facts.join(" \u00b7 ")));

    // What is still missing is shown, not hidden: an unmet requirement is the
    // difference between a proposal and something someone could act on.
    (p.unmet_requirements || []).forEach((u) => box.append(notice(u, "warn", "!")));

    const evidence = el("div", "eqrefs");
    evidence.textContent =
      `${p.citation} (${p.citation_precision}) \u00b7 confidence ${p.confidence} \u00b7 ` +
      (p.independently_verified ? "verified here" : "not independently verified here");
    box.append(evidence);
    (p.notes || []).forEach((n) => box.append(el("div", "footnote", n)));
    card.append(box);
  });
  return card;
}

function renderClassifiedContacts(contacts) {
  if (!contacts || !contacts.length) return null;
  const card = el("div", "card");
  card.append(el("h2", null, `Native contacts (${contacts.length})`));
  card.append(el("div", "kv",
    "Contacts this peptide makes in its native setting. The class decides what a " +
    "proposal may do: SCAFFOLD is a disruption target, PROTECTIVE must be replaced " +
    "rather than removed, ESSENTIAL is frozen."));

  contacts.forEach((c) => {
    const box = el("div", `contact contact-${c.class.toLowerCase()}`);
    const head = el("div", "contact-head");
    head.append(
      el("span", "glyph", CONTACT_CLASS_GLYPH[c.class] || "\u25CB"),
      el("span", "n", c.contact),
      el("span", "tag", c.subclass ? `${c.class} / ${c.subclass}` : c.class)
    );
    if (c.also_classified) head.append(el("span", "tag", `also ${c.also_classified}`));
    box.append(head);

    if (c.claim) box.append(el("div", "r", c.claim));
    box.append(el("div", "footnote", c.class_description));
    if (c.verification_path) {
      box.append(el("div", "footnote", `Verification: ${c.verification_path}`));
    }

    // Evidence sits with the classification, not behind a click. A frozen
    // footprint that cannot say what froze it is an assertion.
    const evidence = el("div", "eqrefs");
    evidence.textContent =
      `${c.citation} (${c.citation_precision}) \u00b7 confidence ${c.confidence} \u00b7 ` +
      (c.independently_verified ? "verified here" : "not independently verified here");
    box.append(evidence);
    if (c.magnitude_source) {
      box.append(el("div", "footnote", `Magnitude from: ${c.magnitude_source}`));
    }
    (c.notes || []).forEach((n) => box.append(el("div", "footnote", n)));
    card.append(box);
  });
  return card;
}

function renderResearchRequests(requests, registryEmpty) {
  if (!requests || !requests.length) return null;
  const card = el("div", "card");
  card.append(el("h2", null, `Research requests (${requests.length})`));
  card.append(el("div", "kv",
    "These modifications may well be the right ones. The system has no basis to say " +
    "so: none of the chemistry below exists in a standard force field, so it cannot " +
    "be scored, simulated or ranked."));
  if (registryEmpty) {
    card.append(notice(
      "The parameterized-residue registry is empty. That is its correct current state, " +
      "not a gap \u2014 nothing has been parameterized by this project, so every " +
      "non-canonical proposal lands here.", "info", "i"));
  }

  requests.forEach((r) => {
    const box = el("div", "request");
    const head = el("div", "request-head");
    head.append(el("span", "n", r.proposal));
    if (r.position) head.append(el("span", "pos", `position ${r.position}`));
    box.append(head);

    r.residues.forEach((res) => {
      const row = el("div", "request-res");
      const title = el("div");
      title.append(el("b", null, res.residue), el("span", "tag", res.status));
      row.append(title);
      row.append(el("div", "r", res.rationale));
      row.append(el("div", "footnote", res.cost_summary));
      if (res.stages_outstanding && res.stages_outstanding.length) {
        row.append(el("div", "eqrefs", res.stages_outstanding.join(" \u2192 ")));
      }
      (res.notes || []).forEach((n) => row.append(el("div", "footnote", n)));
      box.append(row);
    });
    card.append(box);
  });
  return card;
}

function renderStructureTemplate(st, rejected) {
  if (!st) return null;
  const card = el("div", "card");
  card.append(el("h2", null, "Structure template"));

  const ladder = el("div", "tier-ladder");
  const TIERS = [
    [1, "Experimental structure of this peptide bound to this receptor"],
    [2, "Experimental structure of a close homolog bound to the same receptor"],
    [3, "Predicted complex, confidence gate passed"],
    [4, "Refuse \u2014 block all conformation-dependent proposals"],
  ];
  TIERS.forEach(([n, label]) => {
    const row = el("div", `tier-row${n === st.tier ? " reached" : ""}`);
    row.append(
      el("span", "glyph", n === st.tier ? "\u25C6" : "\u25CB"),
      el("span", "tier-n", `Tier ${n}`),
      el("span", null, label)
    );
    ladder.append(row);
  });
  card.append(ladder);
  card.append(el("div", "kv", st.summary));

  (st.detail || []).forEach((d) => card.append(notice(d, "info", "i")));

  const blocked = (rejected || []).filter(
    (r) => r.blocked_by === "structure_template_refusal");
  if (blocked.length) {
    const box = el("div", "blocked-list");
    box.append(el("div", "footnote",
      `${blocked.length} conformation-dependent proposal(s) were generated and then ` +
      `blocked. They are listed so the absence is visible rather than silent.`));
    blocked.forEach((b) => {
      const row = el("div", "celltype");
      const h = el("div");
      h.append(el("span", "n", b.description));
      row.append(h, el("div", "r", "blocked \u2014 no admissible structure template"));
      box.append(row);
    });
    card.append(box);
  }
  return card;
}

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
  // Before the rationale, not after: the rationale is the steric account, and
  // a reader who meets the caveat afterwards has already taken it as the
  // mechanism.
  const nbo = nboNotice(t.nbo);
  if (nbo) body.append(nbo);

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

/* ---------- Substitution landscape ---------- */

let LANDSCAPE_METRICS = [];

// The steps of each ramp, in order outward from the neutral middle. Kept as
// token names rather than hex so light and dark swap in the stylesheet, where
// the two modes are each selected against their own surface rather than one
// being an inversion of the other.
const DIV_NEG = ["--div-neg-1", "--div-neg-2", "--div-neg-3", "--div-neg-4"];
const DIV_POS = ["--div-pos-1", "--div-pos-2", "--div-pos-3", "--div-pos-4"];
const SEQ_STEPS = ["--seq-100", "--seq-200", "--seq-350", "--seq-450", "--seq-600"];

// Height of the column-marginal strip, in px. Kept in step with the .hc.marg
// rule in the stylesheet: bar heights are computed against it.
const MARGIN_STRIP_PX = 34;

function cellColor(value, encoding, bound) {
  // No range to spread a scale across: every computed cell sits at the bottom
  // of the ramp rather than being scattered by rounding noise.
  if (!bound || bound <= 0) {
    return encoding === "diverging" ? "var(--div-mid)" : `var(${SEQ_STEPS[0]})`;
  }
  if (encoding === "diverging") {
    const n = Math.max(-1, Math.min(1, value / bound));
    const step = Math.ceil(Math.abs(n) * DIV_POS.length);
    if (step === 0) return "var(--div-mid)";
    const ramp = n < 0 ? DIV_NEG : DIV_POS;
    return `var(${ramp[Math.min(ramp.length, step) - 1]})`;
  }
  const n = Math.max(0, Math.min(1, value / bound));
  const idx = Math.min(SEQ_STEPS.length, Math.max(1, Math.ceil(n * SEQ_STEPS.length)));
  return `var(${SEQ_STEPS[idx - 1]})`;
}

function fmtValue(v, metric) {
  if (v === undefined || v === null) return "—";
  return metric.key === "hydrophobicity_delta" ? v.toFixed(1)
       : metric.encoding === "diverging" ? (v >= 0 ? "+" : "") + v.toFixed(2)
       : v.toFixed(2);
}

function landscapeLegend(ls) {
  const m = ls.metric;
  const wrap = el("div", "heat-legend");

  // No computed cell means no scale to legend. A ramp drawn over an empty grid
  // invites the reader to place the hatched cells somewhere on it.
  const ramp = el("div", "ramp");
  const bound = ls.scale_bound;
  if (ls.n_computed === 0) {
    ramp.append(el("span", null, "no value scale — nothing in this grid was computed"));
  } else if (m.encoding === "diverging") {
    ramp.append(el("span", null, bound ? `−${fmtValue(bound, m).replace("+", "")}` : "low"));
    const steps = el("div", "steps");
    [...DIV_NEG].reverse().forEach((t) => {
      const i = el("i"); i.style.background = `var(${t})`; steps.append(i);
    });
    const mid = el("i"); mid.style.background = "var(--div-mid)"; steps.append(mid);
    DIV_POS.forEach((t) => {
      const i = el("i"); i.style.background = `var(${t})`; steps.append(i);
    });
    ramp.append(steps, el("span", null, bound ? `+${fmtValue(bound, m).replace("+", "")}` : "high"));
  } else {
    ramp.append(el("span", null, "0"));
    const steps = el("div", "steps");
    SEQ_STEPS.forEach((t) => {
      const i = el("i"); i.style.background = `var(${t})`; steps.append(i);
    });
    ramp.append(steps, el("span", null, bound ? fmtValue(bound, m) : "high"));
  }
  wrap.append(ramp);

  if (ls.n_computed > 0 && m.encoding === "diverging" && m.midpoint_meaning) {
    wrap.append(el("span", null, `middle: ${m.midpoint_meaning}`));
  }

  const wt = el("div", "key");
  const wtSwatch = el("i", "wt-cell");
  wt.append(wtSwatch, el("span", null, "wild-type residue — not a substitution"));
  wrap.append(wt);

  if (ls.n_not_computed > 0) {
    const nc = el("div", "key");
    nc.append(el("i", "nc"), el("span", null, `not computed (${ls.n_not_computed})`));
    wrap.append(nc);
  }
  return wrap;
}

function landscapeGrid(ls) {
  const m = ls.metric;
  const wrap = el("div", "heat-wrap");
  const scroll = el("div", "heat-scroll");
  const grid = el("div", "heat");
  grid.style.setProperty("--cols", String(ls.length));

  // Row 1: position ticks. Every fifth, plus the two ends, so the axis reads
  // without the numbers colliding at 18px cells.
  grid.append(el("div", "corner"));
  for (let i = 0; i < ls.length; i++) {
    const n = i + 1;
    const show = n === 1 || n === ls.length || n % 5 === 0;
    grid.append(el("div", "tick", show ? String(n) : ""));
  }

  // Row 2: the wild-type sequence itself, so a reader can see which residue
  // each column is a substitution away from.
  grid.append(el("div", "corner"));
  for (let i = 0; i < ls.length; i++) grid.append(el("div", "wt", ls.sequence[i]));

  const byKey = new Map();
  ls.cells.forEach((c) => byKey.set(`${c.position}:${c.aa}`, c));

  // Column marginal, sharing the grid's columns so it reads as a reduction of
  // the chart above rather than as a second chart that happens to sit nearby.
  // Height encodes the most favourable computed value in the column; a column
  // with nothing computed gets the hatch, not a zero-height bar, because a bar
  // of no height is a claim that nothing helps.
  const profile = ls.profile || [];
  if (profile.length === ls.length) {
    const computed = profile.filter((p) => p.n_computed > 0);
    const span = computed.length
      ? Math.max(...computed.map((p) => Math.abs(m.encoding === "diverging" ? p.best : p.best)), 1e-9)
      : 0;
    grid.append(el("div", "rowlab marglab", "best"));
    for (let pos = 0; pos < ls.length; pos++) {
      const col = profile[pos];
      const cell = el("div", "hc marg");
      if (!col || col.n_computed === 0) {
        cell.classList.add("nc");
      } else {
        const bar = el("div", "margbar");
        const frac = span > 0 ? Math.min(1, Math.abs(col.best) / span) : 0;
        bar.style.background = cellColor(col.best, m.encoding, ls.scale_bound);
        if (m.encoding === "diverging") {
          // Anchored on a zero baseline, growing up for a gain and down for a
          // loss. Bare height would encode magnitude alone, so a position where
          // the best available substitution is strongly harmful would draw a
          // tall bar and read as the best position on the strip.
          cell.classList.add("marg-div");
          // In px, not a percentage: a floor of a percent or two of the strip
          // rounds a small negative to under a pixel, and a bar that does not
          // render reads as a column with no result rather than as a column
          // whose best option is still a loss.
          bar.style.height = `${Math.max(3, frac * (MARGIN_STRIP_PX / 2))}px`;
          if (col.best >= 0) bar.style.bottom = "50%"; else bar.style.top = "50%";
        } else {
          bar.style.bottom = "0";
          bar.style.height = `${Math.max(3, frac * MARGIN_STRIP_PX)}px`;
        }
        cell.append(bar);
      }
      cell.dataset.margin = String(pos);
      grid.append(cell);
    }
  }

  ls.rows.forEach((aa) => {
    grid.append(el("div", "rowlab", aa));
    for (let pos = 0; pos < ls.length; pos++) {
      const c = byKey.get(`${pos}:${aa}`);
      const cell = el("div", "hc");
      if (!c) {
        cell.classList.add("nc");
      } else if (c.state === "WILD_TYPE") {
        cell.classList.add("wt-cell");
      } else if (c.state === "NOT_COMPUTED") {
        cell.classList.add("nc");
      } else {
        cell.style.background = cellColor(c.value, m.encoding, ls.scale_bound);
      }
      if (c) cell.dataset.key = `${pos}:${aa}`;
      grid.append(cell);
    }
  });

  scroll.append(grid);

  // Per-mark hover: a cell chart with no tooltip forces the reader to guess the
  // value from the colour, which is the one thing colour cannot do precisely.
  const tip = el("div", "heat-tip");
  tip.hidden = true;
  wrap.append(scroll, tip);

  const show = (ev) => {
    const cell = ev.target.closest(".hc");
    if (!cell) { tip.hidden = true; return; }

    if (cell.dataset.margin !== undefined) {
      const col = (ls.profile || [])[Number(cell.dataset.margin)];
      if (!col) { tip.hidden = true; return; }
      tip.innerHTML = "";
      tip.append(el("b", null, `Position ${col.position + 1} (${col.wt}) — column summary`));
      if (col.n_computed > 0) {
        tip.append(el("div", "val",
          `best ${fmtValue(col.best, m)} · mean ${fmtValue(col.mean, m)} · ` +
          `worst ${fmtValue(col.worst, m)}`));
      }
      tip.append(el("div", null, col.detail));
      if (col.n_not_computed > 0) {
        tip.append(el("div", null, `${col.n_not_computed} of ${col.n_not_computed + col.n_computed} not computed.`));
      }
      tip.hidden = false;
      placeTip(cell);
      return;
    }

    if (!cell.dataset.key) { tip.hidden = true; return; }
    const c = byKey.get(cell.dataset.key);
    if (!c) { tip.hidden = true; return; }
    tip.innerHTML = "";
    const wt = ls.sequence[c.position];
    tip.append(el("b", null,
      c.state === "WILD_TYPE"
        ? `Position ${c.position + 1}: ${wt} (wild type)`
        : `Position ${c.position + 1}: ${wt} → ${c.aa}`));
    if (c.state === "COMPUTED") {
      const line = el("div");
      line.append(document.createTextNode(`${m.label}: `));
      line.append(el("span", "val", `${fmtValue(c.value, m)}`));
      line.append(document.createTextNode(` ${m.units.split(",")[0]}`));
      tip.append(line);
    } else if (c.state === "NOT_COMPUTED") {
      tip.append(el("div", "val", "not computed"));
    }
    tip.append(el("div", null, c.detail));
    if (c.confidence && c.state === "COMPUTED") {
      tip.append(el("div", null, `Overall confidence for this substitution: ${c.confidence}`));
    }
    tip.hidden = false;
    placeTip(cell);
  };

  function placeTip(cell) {
    const box = wrap.getBoundingClientRect();
    const cb = cell.getBoundingClientRect();
    const left = Math.min(
      Math.max(4, cb.left - box.left + cb.width / 2 - tip.offsetWidth / 2),
      Math.max(4, box.width - tip.offsetWidth - 4)
    );
    const above = cb.top - box.top - tip.offsetHeight - 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${above > 0 ? above : cb.bottom - box.top + 8}px`;
  }

  scroll.addEventListener("mousemove", show);
  scroll.addEventListener("mouseleave", () => { tip.hidden = true; });
  return wrap;
}

function landscapeTable(ls) {
  const m = ls.metric;
  const det = el("details");
  det.append(el("summary", null, `Table view — all ${ls.cells.length} cells`));
  const wrap = el("div", "heat-table-wrap");
  const table = el("table", "titration heat-table");
  const thead = el("thead");
  const hr = el("tr");
  ["Pos", "WT", "Sub", "State", m.label, "Detail"].forEach((h) => hr.append(el("th", null, h)));
  thead.append(hr);
  const tbody = el("tbody");
  ls.cells.forEach((c) => {
    const tr = el("tr");
    tr.append(el("td", "num", String(c.position + 1)));
    tr.append(el("td", "res", ls.sequence[c.position]));
    tr.append(el("td", "res", c.aa));
    tr.append(el("td", "st", c.state === "COMPUTED" ? "computed"
                          : c.state === "WILD_TYPE" ? "wild type" : "not computed"));
    tr.append(el("td", "num", c.state === "COMPUTED" ? fmtValue(c.value, m) : "—"));
    tr.append(el("td", "det", c.detail));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  wrap.append(table);
  det.append(wrap);
  return det;
}

function renderLandscape(d) {
  const out = $("#landscape-results");
  out.innerHTML = "";
  const ls = d.landscape;
  const m = ls.metric;

  const pn = policyNotice(d.policy);
  if (pn) out.append(pn);

  const card = el("div", "card");
  card.append(el("h2", null,
    `${ls.name && ls.name !== "unnamed_peptide" ? ls.name : "Peptide"} — ${m.label.toLowerCase()}`));
  card.append(el("div", "kv",
    `${ls.length} residues × ${ls.rows.length} residue substitutions · goal: ${ls.goal || "unspecified"} · pH ${ls.ph}`));
  card.append(el("p", "footnote", m.description));

  if (ls.n_computed === 0) {
    card.append(notice(
      `No cell of this grid has a computed value. ${ls.not_computed_reason || ""}`.trim(),
      "warn", "!"
    ));
  } else if (ls.n_not_computed > 0) {
    card.append(notice(
      `${ls.n_not_computed} of ${ls.n_not_computed + ls.n_computed + ls.length} cells are hatched ` +
      `because the pipeline did not compute them. ${ls.not_computed_reason || ""}`.trim(),
      "warn", "!"
    ));
  }

  card.append(el("p", "footnote",
    m.encoding === "diverging"
      ? "Top strip: the most favourable computed substitution at each position, on a zero " +
        "baseline — above the line the best available change is a predicted gain, below it " +
        "the best available change is still a loss. The column marginal of the grid below, " +
        "not a separate calculation."
      : "Top strip: the least costly computed substitution at each position — the " +
        "column marginal of the grid below, not a separate calculation."));
  card.append(landscapeGrid(ls));
  card.append(landscapeLegend(ls));
  if (ls.scale_basis) card.append(el("p", "footnote", ls.scale_basis));
  card.append(el("p", "footnote", `Computed by: ${m.source}.`));
  card.append(landscapeTable(ls));

  (ls.notes || []).forEach((n) => card.append(notice(n, "info", "i")));
  out.append(card);
}

async function onLandscape() {
  const btn = $("#btn-landscape");
  const out = $("#landscape-results");
  const seq = $("#lseq").value.trim();
  if (!seq) { out.innerHTML = ""; out.append(notice("Enter a sequence first.", "error", "!")); return; }

  out.innerHTML = "";
  busy(btn, true);
  const ph = parseFloat($("#lph").value);
  try {
    const homologs = $("#lhomologs").value
      .split(/[\n,]+/).map((x) => x.trim().toUpperCase()).filter(Boolean);
    const d = await api("/api/substitution-landscape", {
      input: seq,
      goal: $("#lgoal").value || null,
      metric: $("#lmetric").value,
      ph: Number.isFinite(ph) ? ph : 7.4,
      homologs: homologs.length ? homologs : null,
    });
    renderLandscape(d);
  } catch (e) {
    out.append(notice(e.message, "error", "!"));
  } finally {
    busy(btn, false, "Build the grid");
  }
}

async function initLandscape() {
  $("#btn-landscape").addEventListener("click", onLandscape);

  EXAMPLES.sequences.forEach((ex) => {
    const b = el("button", "chip", ex.label);
    b.addEventListener("click", () => { $("#lseq").value = ex.seq; });
    $("#lseq-examples").append(b);
  });

  const goalSel = $("#lgoal");
  GOALS.forEach((g) => {
    const o = el("option", null, g.label);
    o.value = g.id;
    goalSel.append(o);
  });

  const metricSel = $("#lmetric");
  const desc = $("#lmetric-desc");
  try {
    const d = await api("/api/landscape-metrics");
    LANDSCAPE_METRICS = d.metrics;
    LANDSCAPE_METRICS.forEach((m) => {
      const o = el("option", null, m.label);
      o.value = m.key;
      metricSel.append(o);
    });
    metricSel.value = d.default;
  } catch {
    desc.textContent = "Could not load the metric list.";
    return;
  }
  const syncDesc = () => {
    const m = LANDSCAPE_METRICS.find((x) => x.key === metricSel.value);
    if (!m) return;
    desc.textContent =
      `${m.description} Encoded as a ${m.encoding} scale` +
      (m.encoding === "diverging" ? `, where the middle means: ${m.midpoint_meaning}.` : ".");
  };
  metricSel.addEventListener("change", syncDesc);
  syncDesc();
}

/* ---------- Representation explorer ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
};

// Categorical slots in the palette's fixed order, assigned in sequence. The
// fourth class deliberately takes no hue: the first three slots are what clear
// the all-pairs separation gate that a scatter has to meet, so "unrelated" is
// carried by shape instead.
const VARIANT_SERIES = [
  { key: "WILD_TYPE", label: "wild type", token: "--series-1" },
  { key: "SINGLE", label: "single substitution", token: "--series-2" },
  { key: "MULTI", label: "multi-substitution", token: "--series-3" },
];
const VARIANT_BY_KEY = new Map(VARIANT_SERIES.map((v) => [v.key, v]));

function scatterPlot(d) {
  const W = 660, H = 420;
  const M = { top: 14, right: 14, bottom: 44, left: 56 };
  const wrap = el("div", "scatter-wrap");
  const svg = svgEl("svg", {
    class: "scatter", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": "Principal component projection of substitution variants",
  });

  const pts = d.points;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const pad = (lo, hi) => { const s = (hi - lo) || 1; return [lo - s * 0.06, hi + s * 0.06]; };
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs));
  const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const sx = (v) => M.left + ((v - x0) / (x1 - x0)) * (W - M.left - M.right);
  const sy = (v) => H - M.bottom - ((v - y0) / (y1 - y0)) * (H - M.top - M.bottom);

  // Recessive grid first, so nothing draws over a data mark.
  for (let i = 0; i <= 4; i++) {
    const gx = M.left + (i / 4) * (W - M.left - M.right);
    const gy = M.top + (i / 4) * (H - M.top - M.bottom);
    svg.append(svgEl("line", { class: "grid", x1: gx, x2: gx, y1: M.top, y2: H - M.bottom }));
    svg.append(svgEl("line", { class: "grid", x1: M.left, x2: W - M.right, y1: gy, y2: gy }));
  }
  svg.append(svgEl("line", { class: "axis", x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom }));
  svg.append(svgEl("line", { class: "axis", x1: M.left, x2: M.left, y1: M.top, y2: H - M.bottom }));

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const xLabel = svgEl("text", { class: "axlabel", x: (M.left + W - M.right) / 2, y: H - 10, "text-anchor": "middle" });
  xLabel.textContent = `PC1 — ${pct(d.explained_variance_ratio[0])} of the variation`;
  const yLabel = svgEl("text", {
    class: "axlabel", x: 14, y: (M.top + H - M.bottom) / 2, "text-anchor": "middle",
    transform: `rotate(-90 14 ${(M.top + H - M.bottom) / 2})`,
  });
  yLabel.textContent = `PC2 — ${pct(d.explained_variance_ratio[1])}`;
  svg.append(xLabel, yLabel);

  const marks = svgEl("g");
  const nodes = [];
  // Reference last so it sits on top of the cloud it is the centre of.
  const ordered = [...pts].sort((a, b) => (a.class === "WILD_TYPE") - (b.class === "WILD_TYPE"));
  ordered.forEach((p) => {
    const series = VARIANT_BY_KEY.get(p.class);
    let node;
    if (!series) {
      node = svgEl("rect", { class: "other", x: sx(p.x) - 4, y: sy(p.y) - 4, width: 8, height: 8 });
    } else {
      const isRef = p.class === "WILD_TYPE";
      node = svgEl("circle", {
        class: "pt", cx: sx(p.x), cy: sy(p.y), r: isRef ? 5.5 : 3,
        fill: `var(${series.token})`,
      });
    }
    node.__point = p;
    nodes.push(node);
    marks.append(node);
  });
  svg.append(marks);

  // One direct label, on the reference. Labelling 590 points would be noise,
  // and labelling none leaves the plot's anchor unnamed.
  const ref = pts.find((p) => p.class === "WILD_TYPE");
  if (ref) {
    const t = svgEl("text", { class: "direct", x: sx(ref.x) + 9, y: sy(ref.y) - 7 });
    t.textContent = "wild type";
    svg.append(t);
  }

  wrap.append(svg);

  const tip = el("div", "heat-tip");
  tip.hidden = true;
  wrap.append(tip);

  let focused = null;
  svg.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const py = ((ev.clientY - box.top) / box.height) * H;
    let best = null, bestDist = Infinity;
    nodes.forEach((n) => {
      const p = n.__point;
      const dist = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
      // Ties go to the reference. Variants can land exactly on top of it --
      // under the composition encoder that is common -- and reporting one of
      // them while the reference's own mark is the thing under the cursor
      // reads as the reference having moved.
      if (dist < bestDist - 1e-9
          || (dist <= bestDist + 1e-9 && p.class === "WILD_TYPE")) {
        bestDist = dist;
        best = n;
      }
    });
    // A generous hit radius: the marks are small because the cloud is dense,
    // and a hit target the size of the mark would be unusable.
    if (!best || bestDist > 18 ** 2) {
      tip.hidden = true;
      if (focused) { focused.classList.remove("focus"); focused = null; }
      return;
    }
    if (focused !== best) {
      if (focused) focused.classList.remove("focus");
      best.classList.add("focus");
      focused = best;
    }
    const p = best.__point;
    tip.innerHTML = "";
    tip.append(el("b", null, p.label));
    tip.append(el("div", null,
      p.n_substitutions === null
        ? "Different length from the reference — no substitution count."
        : `${p.n_substitutions} substitution${p.n_substitutions === 1 ? "" : "s"} from the reference`));
    tip.append(el("div", "val", `PC1 ${p.x.toFixed(3)} · PC2 ${p.y.toFixed(3)}`));
    tip.hidden = false;
    const wbox = wrap.getBoundingClientRect();
    const left = Math.min(Math.max(4, ev.clientX - wbox.left + 12),
                          Math.max(4, wbox.width - tip.offsetWidth - 4));
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(4, ev.clientY - wbox.top - tip.offsetHeight - 10)}px`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.hidden = true;
    if (focused) { focused.classList.remove("focus"); focused = null; }
  });

  return wrap;
}

function scatterLegend(d) {
  const present = new Set(d.points.map((p) => p.class));
  const wrap = el("div", "scatter-legend");
  VARIANT_SERIES.forEach((s) => {
    if (!present.has(s.key)) return;
    const key = el("div", "key");
    const swatch = el("i");
    swatch.style.background = `var(${s.token})`;
    const n = d.points.filter((p) => p.class === s.key).length;
    key.append(swatch, el("span", null, `${s.label} (${n})`));
    wrap.append(key);
  });
  const others = d.points.filter((p) => !VARIANT_BY_KEY.has(p.class));
  if (others.length) {
    const key = el("div", "key");
    key.append(el("i", "other"),
               el("span", null, `different length — unrelated (${others.length})`));
    wrap.append(key);
  }
  return wrap;
}

function representationTable(d) {
  const ref = d.points.find((p) => p.class === "WILD_TYPE");
  const dist = (p) => (ref ? Math.hypot(p.x - ref.x, p.y - ref.y) : Math.hypot(p.x, p.y));
  const rows = [...d.points].sort((a, b) => dist(b) - dist(a));
  const det = el("details");
  det.append(el("summary", null, `Table view — all ${rows.length} points, furthest from the reference first`));
  const wrap = el("div", "heat-table-wrap");
  const table = el("table", "titration heat-table");
  const thead = el("thead");
  const hr = el("tr");
  ["Variant", "Class", "Subs", "PC1", "PC2", "Distance from reference"].forEach((h) => hr.append(el("th", null, h)));
  thead.append(hr);
  const tbody = el("tbody");
  rows.forEach((p) => {
    const tr = el("tr");
    tr.append(el("td", "res", p.label));
    tr.append(el("td", "st", (VARIANT_BY_KEY.get(p.class) || { label: "unrelated" }).label));
    tr.append(el("td", "num", p.n_substitutions === null ? "—" : String(p.n_substitutions)));
    tr.append(el("td", "num", p.x.toFixed(3)));
    tr.append(el("td", "num", p.y.toFixed(3)));
    tr.append(el("td", "num", dist(p).toFixed(3)));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  wrap.append(table);
  det.append(wrap);
  return det;
}

function renderOOD(entries) {
  // One block per supplied candidate. Deliberately not a chart: a handful of
  // items with a verdict each is a list, and a chart here would be decoration
  // over five numbers.
  const wrap = el("div");
  wrap.append(el("div", "label", "WHERE EACH CANDIDATE SITS"));
  wrap.append(el("p", "footnote",
    "Each candidate's distance to its nearest neighbour in the substitution cloud, placed " +
    "inside the cloud's own nearest-neighbour distances. A raw distance answers nothing — " +
    "the same number is close in one space and remote in another — so the reference set " +
    "supplies the comparison."));

  entries.forEach((o) => {
    const block = el("div", "ood");
    block.append(el("div", "t", o.label));
    if (!o.available) {
      block.append(notice(o.reason, "warn", "!"));
      wrap.append(block);
      return;
    }
    // Status wears an icon and a word, never colour alone.
    block.append(notice(o.verdict, o.is_outside ? "warn" : "info", o.is_outside ? "!" : "i"));
    block.append(el("div", "kv",
      `distance to nearest ${o.distance_to_nearest.toFixed(3)} · ` +
      `the cloud's own median nearest-neighbour distance ${o.reference_median_nn_distance.toFixed(3)} · ` +
      `${(o.percentile * 100).toFixed(0)}th percentile`));
    const chips = el("div", "chips");
    o.nearest.forEach((n) => {
      chips.append(el("span", "chip static", `${n.label}  ${n.distance.toFixed(3)}`));
    });
    block.append(chips);
    block.append(el("p", "footnote", o.caveat));
    wrap.append(block);
  });
  return wrap;
}

function renderRepresentation(d) {
  const out = $("#representation-results");
  out.innerHTML = "";
  const card = el("div", "card");
  card.append(el("h2", null, "Principal components of the substitution set"));
  card.append(el("div", "kv",
    `${d.points.length} sequences · ${d.encoder.model} v${d.encoder.version} · ` +
    `${d.encoder.input_dim}-dimensional input`));

  // The headline is the number, not the picture. A scatter that carries a small
  // share of the variation invites a reading it cannot support, so the share
  // goes first and at size.
  const hero = el("div", "hero-stat");
  hero.append(el("div", "figure", `${(d.cumulative_explained * 100).toFixed(1)}%`));
  hero.append(el("div", "caption",
    "of the variation between these sequences is in the two dimensions plotted below. " +
    "The rest is in directions not drawn."));
  card.append(hero);

  // The projection already emits a warning when the encoder has no learned
  // content; repeating it here as a second notice said the same thing twice.
  (d.warnings || []).forEach((w) => card.append(notice(w, "warn", "!")));
  if (d.truncated) {
    card.append(notice(
      "The single-substitution cloud was capped for this projection; supplied candidates " +
      "and the reference were kept. The components shown are those of the points plotted.",
      "warn", "!"));
  }

  card.append(scatterPlot(d));
  card.append(scatterLegend(d));
  card.append(el("p", "footnote", d.interpretation));
  if ((d.ood || []).length) card.append(renderOOD(d.ood));
  card.append(representationTable(d));
  out.append(card);
}

async function onRepresentation() {
  const btn = $("#btn-representation");
  const out = $("#representation-results");
  const seq = $("#rseq").value.trim();
  if (!seq) { out.innerHTML = ""; out.append(notice("Enter a reference sequence first.", "error", "!")); return; }
  out.innerHTML = "";
  busy(btn, true);
  try {
    const candidates = $("#rcandidates").value
      .split(/[\n,]+/).map((x) => x.trim().toUpperCase()).filter(Boolean);
    const d = await api("/api/ml/representation", {
      sequence: seq,
      encoder: $("#rencoder").value,
      candidates: candidates.length ? candidates : null,
    });
    renderRepresentation(d);
  } catch (e) {
    out.append(notice(e.message, "error", "!"));
  } finally {
    busy(btn, false, "Project");
  }
}

async function initRepresentation() {
  $("#btn-representation").addEventListener("click", onRepresentation);

  EXAMPLES.sequences.filter((ex) => !ex.seq.includes(":") && /^[A-Z]+$/.test(ex.seq))
    .forEach((ex) => {
      const b = el("button", "chip", ex.label);
      b.addEventListener("click", () => { $("#rseq").value = ex.seq; });
      $("#rseq-examples").append(b);
    });

  const sel = $("#rencoder");
  const desc = $("#rencoder-desc");
  let encoders = {};
  try {
    const status = await api("/api/ml/status");
    encoders = status.encoders || {};
  } catch {
    desc.textContent = "Could not load the encoder list.";
    return;
  }
  Object.entries(encoders).forEach(([name, info]) => {
    const o = el("option", null, info.available ? name : `${name} — unavailable`);
    o.value = name;
    // An unavailable encoder stays visible and stays unselectable: hiding it
    // would make the deterministic fallback look like the intended one.
    o.disabled = !info.available;
    sel.append(o);
  });
  // Prefer the positional encoder where it is available: the composition one
  // maps distinct sequences to one vector, and a default that silently merges
  // half a substitution scan is the wrong thing to show first.
  const preferred = "deterministic-positional-onehot";
  const firstAvailable = Object.entries(encoders).find(([, i]) => i.available);
  if (encoders[preferred] && encoders[preferred].available) sel.value = preferred;
  else if (firstAvailable) sel.value = firstAvailable[0];
  const syncDesc = () => {
    const info = encoders[sel.value];
    desc.textContent = info ? info.note : "";
  };
  sel.addEventListener("change", syncDesc);
  syncDesc();
}

/* ---------- Does fusion help? ---------- */

function renderFusion(d) {
  const out = $("#fusion-results");
  out.innerHTML = "";
  const card = el("div", "card");
  card.append(el("h2", null, "Single modality, concatenation, learned fusion"));
  card.append(el("div", "kv",
    `modalities: ${d.modalities.join(" + ")} · split: ${d.split} · ` +
    `${d.n_train} train, ${d.n_test} test`));

  card.append(notice(d.claim_guidance, "warn", "!"));

  (d.skipped || []).forEach((note) => card.append(notice(note, "warn", "!")));

  if (d.arms.length) {
    const table = el("table", "titration");
    const thead = el("thead");
    const hr = el("tr");
    ["Arm", "ROC-AUC", "95% interval", "Gate"].forEach((h) => hr.append(el("th", null, h)));
    thead.append(hr);
    const tbody = el("tbody");
    d.arms.forEach((a) => {
      const tr = el("tr");
      tr.append(el("td", "armname", a.name));
      tr.append(el("td", "num", a.roc_auc === null ? "—" : a.roc_auc.toFixed(3)));
      tr.append(el("td", "num",
        a.ci_low === null ? "—" : `${a.ci_low.toFixed(3)}–${a.ci_high.toFixed(3)}`));
      // A gate that puts everything on one modality is the single most useful
      // thing this table can say, so it gets a column rather than a footnote.
      tr.append(el("td", "grp", a.gate_share
        ? a.gate_share.map((w, i) => `${d.modalities[i]} ${(w * 100).toFixed(0)}%`).join(" · ")
        : "—"));
      tbody.append(tr);
    });
    table.append(thead, tbody);
    // Scroll the table, not the page: a horizontal page scroll at phone width
    // makes every other card on the page unreadable too.
    const scroll = el("div", "table-scroll");
    scroll.append(table);
    card.append(scroll);
  }

  if ((d.permutation_null || []).length) {
    const lo = Math.min(...d.permutation_null), hi = Math.max(...d.permutation_null);
    card.append(el("p", "footnote",
      `Shuffled-label null: ${d.permutation_null.length} runs, ROC-AUC ` +
      `${lo.toFixed(3)}–${hi.toFixed(3)}. A single run of it spans most of that range, ` +
      `which is why the check is a permutation rather than one control.`));
  }

  if (d.resolution) card.append(notice(d.resolution, "info", "i"));
  if (d.verdict) card.append(notice(d.verdict, "info", "i"));
  out.append(card);
}

async function onFusion() {
  const btn = $("#btn-fusion");
  const out = $("#fusion-results");
  out.innerHTML = "";
  busy(btn, true);
  const families = parseInt($("#fusion-families").value, 10);
  const perms = parseInt($("#fusion-perms").value, 10);
  try {
    const params = new URLSearchParams({
      n_families: Number.isFinite(families) ? families : 60,
      n_permutations: Number.isFinite(perms) ? perms : 6,
    });
    renderFusion(await api(`/api/ml/fusion-benefit?${params}`));
  } catch (e) {
    out.append(notice(e.message, "error", "!"));
  } finally {
    busy(btn, false, "Run the comparison");
  }
}

/* ---------- Biological context: known biology, before any score ---------- */

// Evidence tiers, rendered as a badge. The badge exists so a reader can tell a
// database record from a curated note from a calculation without opening the
// JSON, which was previously the only way.
const TIER_LABEL = {
  DIRECT_EXPERIMENTAL: "experimental",
  HOMOLOG_EXPERIMENTAL: "database",
  BIOCHEMICAL_PRINCIPLE: "curated",
  INFERENCE_ONLY: "inferred",
};

function tierBadge(tier, needsVerification) {
  const badge = el("span", `tier tier-${(tier || "INFERENCE_ONLY").toLowerCase()}`);
  badge.textContent = TIER_LABEL[tier] || "inferred";
  if (needsVerification) {
    badge.textContent += " · unverified";
    badge.title = "Curated into this repository and not checked against its primary "
                + "source. It cannot be reported as an experimental observation.";
  }
  return badge;
}

// Element.append() returns undefined, so it cannot be chained. A helper, so the
// mistake has one place to not be made in.
function sectionHead(title) {
  const head = el("div", "bchead");
  head.append(el("span", "bctitle", title));
  return head;
}

function contextRow(label, value, extra) {
  if (!value) return null;
  const row = el("div", "bcrow");
  row.append(el("span", "bclabel", label));
  const body = el("span", "bcvalue");
  body.append(document.createTextNode(value));
  if (extra) body.append(extra);
  row.append(body);
  return row;
}

function renderReceptor(r) {
  const box = el("div", "receptor");
  const head = el("div", "rhead");
  head.append(el("b", null, r.target_gene || r.target_name));
  if (r.is_primary) head.append(el("span", "pill primary", "primary"));
  else head.append(el("span", "pill", r.interaction_type.replace(/_/g, " ").toLowerCase()));
  head.append(tierBadge(r.tier, r.needs_verification));
  box.append(head);

  if (r.target_name && r.target_name !== r.target_gene) {
    box.append(el("div", "kv", r.target_name));
  }
  if (r.receptor_class) box.append(el("div", "kv", r.receptor_class));
  if (r.role) box.append(el("div", "r", r.role));

  // An unmeasured affinity is stated as unmeasured. The alternative -- an empty
  // field, or worse a zero -- reads as a number.
  if (r.affinity_available) {
    const a = r.affinity;
    box.append(el("div", "kv", `${a.kind} ${a.value} ${a.unit}${r.assay_context ? " · " + r.assay_context : ""}`));
  } else {
    box.append(el("div", "kv muted-note", "No measured affinity was retrieved with a source, so none is shown."));
  }
  if (r.notes) box.append(el("p", "footnote", r.notes));
  box.append(el("p", "footnote", `Source: ${r.source}`));
  return box;
}

/**
 * What is KNOWN about the molecule, as opposed to what it is.
 *
 * `identity` is the identification from the same response, and it is passed in
 * to keep two true sentences from contradicting each other on screen. This
 * card used to open "No established peptide identity was found for this
 * sequence" directly above a card reading "matched a known peptide — exact
 * sequence match to GLP-1 (7-36) amide". Both were correct: the inferencer
 * recognised the sequence and the curated context layer, which is keyed on
 * GLP-1 (7-37), had no record for it. Printed together they read as the system
 * disagreeing with itself, and a reader has no way to tell which half to
 * believe.
 *
 * So the empty state now says which of the two things is missing.
 */
function renderBiologicalContext(bc, identity) {
  if (!bc) return null;
  const card = el("div", "card bio-context");
  card.append(el("div", "label", "BIOLOGICAL CONTEXT"));

  if (!bc.is_established) {
    const named = identity && identity.name;
    card.append(notice(
      (named
        ? `This sequence was identified as ${identity.name}, but no curated biological `
          + "record for it was retrieved. Identification and biological context are "
          + "separate lookups, and this one came back empty. "
        : "No established peptide identity was found for this sequence. ")
      + "So nothing below asserts a receptor, a domain structure or a function, and "
      + "none is inferred from resemblance to a peptide family.",
      "warn", "!"));
    (bc.unavailable_sources || []).forEach((u) =>
      card.append(el("p", "footnote", `Not consulted or unreachable — ${u}`)));
    return card;
  }

  // --- identification ---
  const ident = el("div", "bcsection");
  ident.append(el("h3", null, bc.name));
  const rows = el("div", "bcrows");
  [["Gene", bc.gene], ["Organism", bc.organism], ["UniProt", bc.uniprot],
   ["Family", bc.family], ["Form", bc.form_description]]
    .forEach(([k, v]) => { const r = contextRow(k, v); if (r) rows.append(r); });
  if (bc.aliases.length) {
    const r = contextRow("Also called", bc.aliases.join(", "));
    if (r) rows.append(r);
  }
  ident.append(rows);
  if (bc.needs_verification) {
    ident.append(notice(
      "The records below were curated into this repository and have not been checked "
      + "against UniProt or primary literature from this environment. They are shown as "
      + "curated, and none of them is reported as an experimental observation.",
      "warn", "!"));
  }
  (bc.retrieval_notes || []).forEach((n) => ident.append(el("p", "footnote", n)));
  card.append(ident);

  // --- function ---
  if (bc.function) {
    const fn = el("div", "bcsection");
    const head = el("div", "bchead");
    head.append(el("span", "bctitle", "Function"), tierBadge(bc.function.tier, bc.function.needs_verification));
    fn.append(head, el("p", "r", bc.function.value));
    card.append(fn);
  }

  // --- structure ---
  if (bc.regions.length || bc.disulfides.length || bc.precursor_of) {
    const st = el("div", "bcsection");
    st.append(sectionHead("Structure"));
    if (bc.mature_length) {
      st.append(el("div", "kv", `Mature peptide: ${bc.mature_length} residues`));
    }
    if (bc.precursor_of) st.append(el("p", "footnote", bc.precursor_of));
    bc.regions.forEach((r) => {
      const row = el("div", "region");
      const head = el("div", "rhead");
      head.append(el("b", null, r.name), el("span", "pill", `${r.start}–${r.end}`),
                  tierBadge(r.tier, r.needs_verification));
      row.append(head);
      if (r.residues) row.append(el("div", "seqline", r.residues));
      if (r.role) row.append(el("div", "r", r.role));
      st.append(row);
    });
    if (bc.disulfides.length) {
      const ds = el("div", "kv");
      ds.append(document.createTextNode(
        "Disulfides: " + bc.disulfides.map((b) => `C${b.first}–C${b.second}`).join(", ")));
      st.append(ds);
      // A pairing that does not land on cysteines means the table and the
      // sequence disagree, and one of them is wrong.
      if ((bc.disulfide_inconsistencies || []).length) {
        st.append(notice(
          `Curated disulfide pair(s) ${bc.disulfide_inconsistencies.join(", ")} do not land `
          + `on cysteines in the stored sequence. Neither is used until that is resolved.`,
          "error", "×"));
      }
    }
    card.append(st);
  }

  // --- receptors ---
  const rec = el("div", "bcsection");
  rec.append(sectionHead("Receptors and molecular targets"));
  if (!bc.primary_receptors.length && !bc.secondary_receptors.length) {
    rec.append(notice("No established receptor or molecular target was retrieved for this peptide.",
                      "warn", "!"));
  } else {
    bc.primary_receptors.forEach((r) => rec.append(renderReceptor(r)));
    if (bc.secondary_receptors.length) {
      rec.append(el("p", "footnote",
        "Secondary interactions — cross-reactivity, binding proteins and proteases. "
        + "Design-relevant, and not the principal receptor."));
      bc.secondary_receptors.forEach((r) => rec.append(renderReceptor(r)));
    }
  }
  card.append(rec);

  // --- binding interface ---
  const iface = el("div", "bcsection");
  iface.append(sectionHead("Binding interface"));
  if (!bc.interface_available) {
    iface.append(notice(
      "No sufficiently supported binding-interface annotation was retrieved. Residue-level "
      + "contacts come from structures and mutagenesis; none is substituted from "
      + "hydrophobicity or charge reasoning.",
      "warn", "!"));
  } else {
    bc.interfaces.forEach((i) => {
      const box = el("div", "receptor");
      const ihead = el("div", "rhead");
      ihead.append(el("b", null, i.target_gene), tierBadge(i.tier, i.needs_verification));
      box.append(ihead);
      if (i.ligand_regions.length) box.append(el("div", "kv", `Ligand side: ${i.ligand_regions.join(", ")}`));
      if (i.receptor_regions.length) box.append(el("div", "kv", `Receptor side: ${i.receptor_regions.join(", ")}`));
      if (i.critical_residues.length) box.append(el("div", "kv", `Critical residues: ${i.critical_residues.join(", ")}`));
      if (i.activation_mechanism) box.append(el("div", "r", i.activation_mechanism));
      if (i.structures.length) box.append(el("div", "kv", `Structures: ${i.structures.join(", ")}`));
      iface.append(box);
    });
  }
  card.append(iface);

  // --- data availability ---
  if ((bc.unavailable_sources || []).length) {
    const avail = el("div", "bcsection");
    avail.append(sectionHead("Data availability"));
    bc.unavailable_sources.forEach((u) => avail.append(el("div", "kv", u)));
    avail.append(el("p", "footnote",
      "A source that could not be reached is reported as unreachable, not as an absence of "
      + "information. The two are different answers."));
    card.append(avail);
  }
  return card;
}
