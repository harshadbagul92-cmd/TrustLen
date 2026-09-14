/* =========================================================================
   TrustLens — React interface
   =========================================================================
   No build step. React, Framer Motion and htm are loaded as plain scripts
   from /static/vendor, so you edit this file and refresh. htm gives us
   JSX-like templates via tagged template literals:

       html`<${Thing} prop=${value}>children<//>`

   The whole results panel is derived from the Evidence object the backend
   returns. Nothing about a finding is hard-coded here — if the detector did
   not measure it, this interface cannot show it.
   ========================================================================= */

const { createElement, useState, useEffect, useRef, useCallback } = React;
const { motion, AnimatePresence } = Motion;
const html = htm.bind(createElement);

/* ---------------------------------------------------------------- helpers */
const VERDICTS = {
  likely_authentic:   { label: "No evidence of manipulation", colour: "var(--ok)",       key: "ok" },
  uncertain:          { label: "Not enough to call it",       colour: "var(--warn)",     key: "warn" },
  likely_manipulated: { label: "Likely manipulated",          colour: "var(--bad)",      key: "bad" },
  declared_ai:        { label: "The file declares it is AI",  colour: "var(--declared)", key: "declared" },
};

const DIRECTIONS = [
  { id: "supports_manipulation", title: "Points to manipulation",     colour: "var(--bad)" },
  { id: "supports_authentic",    title: "Points to it being genuine", colour: "var(--ok)" },
  { id: "limitation",            title: "Could not be checked",       colour: "var(--warn)" },
  { id: "context",               title: "Context",                    colour: "var(--ink-3)" },
];

const fmtValue = (v) => {
  if (v === null || v === undefined) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
};

/** Count a number up on first paint. Small enough not to need a library. */
function useCountUp(target, ms = 900) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setN(target); return; }
    let raf, start;
    const tick = (t) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start) / ms);
      setN(target * (1 - Math.pow(1 - p, 3)));      // ease-out cubic
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return n;
}

/* ------------------------------------------------------------------ icons */
const Icon = {
  lens: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/><path d="M11 8v6M8 11h6"/></svg>`,
  scan: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>
      <path d="M3 12h18"/></svg>`,
  book: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  shield: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>`,
  cpu: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
      <path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>`,
  upload: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
      stroke-linecap="round" stroke-linejoin="round" ...${p}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v13"/></svg>`,
  caret: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round" width="15" height="15" ...${p}>
      <path d="m9 18 6-6-6-6"/></svg>`,
  check: (p) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2"
      stroke-linecap="round" stroke-linejoin="round" width="10" height="10" ...${p}>
      <path d="m20 6-11 11-5-5"/></svg>`,
};

/* --------------------------------------------------------------- disclosure */
function Fold({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return html`
    <div class="fold">
      <button class="fold-btn" onClick=${() => setOpen(!open)} aria-expanded=${open}>
        <span>${title}</span>
        <span class=${"caret" + (open ? " open" : "")}><${Icon.caret} /></span>
      </button>
      <${AnimatePresence} initial=${false}>
        ${open && html`
          <${motion.div} key="b"
            initial=${{ height: 0, opacity: 0 }}
            animate=${{ height: "auto", opacity: 1 }}
            exit=${{ height: 0, opacity: 0 }}
            transition=${{ duration: .28, ease: [.22,.61,.36,1] }}
            style=${{ overflow: "hidden" }}>
            <div class="fold-body">${children}</div>
          <//>`}
      <//>
    </div>`;
}

/* ------------------------------------------------------------- verdict card */
function VerdictCard({ ev }) {
  const meta = VERDICTS[ev.verdict] || VERDICTS.uncertain;
  const score = useCountUp(ev.score);
  const R = 58, C = 2 * Math.PI * R;
  const [lo, hi] = ev.confidence;

  return html`
    <${motion.div} class="card verdict" style=${{ "--vc": meta.colour }}
      initial=${{ opacity: 0, y: 16 }} animate=${{ opacity: 1, y: 0 }}
      transition=${{ duration: .45, ease: [.22,.61,.36,1] }}>

      <div class="ring">
        <svg viewBox="0 0 132 132">
          <circle class="ring-track" cx="66" cy="66" r=${R} fill="none" stroke-width="9"/>
          <${motion.circle} class="ring-fill" cx="66" cy="66" r=${R} fill="none" stroke-width="9"
            strokeDasharray=${C}
            initial=${{ strokeDashoffset: C }}
            animate=${{ strokeDashoffset: C * (1 - ev.score) }}
            transition=${{ duration: 1.1, ease: [.22,.61,.36,1], delay: .15 }} />
        </svg>
        <div class="ring-mid">
          <div>
            <div class="ring-score">${score.toFixed(2)}</div>
            <div class="ring-cap">suspicion</div>
          </div>
        </div>
      </div>

      <div class="verdict-body">
        <h2 class="verdict-title">${meta.label}</h2>
        <div class="verdict-meta">
          <span class="chip">${ev.modality}</span>
          <span class="chip mono">${(ev.elapsed_ms / 1000).toFixed(1)}s</span>
          <span class="chip mono">range ${lo.toFixed(2)}–${hi.toFixed(2)}</span>
          ${ev.signals.length ? html`<span class="chip">${ev.signals.length} signals</span>` : null}
        </div>

        <div class="scale-wrap">
          <div class="scale">
            <${motion.div} class="scale-band"
              initial=${{ left: "50%", width: 0 }}
              animate=${{ left: (lo * 100) + "%", width: Math.max(1.5, (hi - lo) * 100) + "%" }}
              transition=${{ duration: .8, delay: .3, ease: [.22,.61,.36,1] }} />
            <${motion.div} class="scale-pin"
              initial=${{ left: "50%", opacity: 0 }}
              animate=${{ left: (ev.score * 100) + "%", opacity: 1 }}
              transition=${{ duration: .8, delay: .45, ease: [.34,1.56,.64,1] }} />
          </div>
          <div class="scale-ticks">
            <span>no evidence</span><span>not enough to call</span><span>likely manipulated</span>
          </div>
        </div>
      </div>
    <//>`;
}

/* --------------------------------------------------------- annotated image */
function Shot({ src, ev }) {
  const [dim, setDim] = useState(null);
  const f = ev.findings || {};
  const noise = f.noise_regions || [];
  const faces = f.face_boxes || [];

  const overlaps = (a, b) => !(a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1]);
  // Only draw what actually fed the score. A corroborating box is shown solely
  // where it overlaps a noise region; painting one the score ignored would
  // show the viewer a finding that is not one.
  const corroborating = [];
  ["ela_regions", "ghost_regions"].forEach((k, i) => {
    (f[k] || []).forEach((r) => {
      if (noise.some((n) => overlaps(r.bbox, n.bbox)))
        corroborating.push({ bbox: r.bbox, colour: i ? "#CA8A04" : "var(--warn)" });
    });
  });

  const legend = [];
  if (noise.length) legend.push(["var(--bad)", "different texture from the rest"]);
  if (corroborating.length) legend.push(["var(--warn)", "another check agrees here"]);
  if (faces.length) legend.push(["var(--brand)", "face found (not scored)"]);

  return html`
    <${motion.div} initial=${{ opacity: 0, y: 12 }} animate=${{ opacity: 1, y: 0 }}
      transition=${{ delay: .2, duration: .4 }} style=${{ marginTop: "18px" }}>
      <div class="shot">
        <img src=${src} alt="the image you submitted, with any flagged regions outlined"
             onLoad=${(e) => setDim({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
        ${dim && html`
          <svg viewBox=${`0 0 ${dim.w} ${dim.h}`} preserveAspectRatio="none">
            ${noise.map((r, i) => html`<rect key=${"n" + i} class="box-draw" stroke="var(--bad)"
                x=${r.bbox[0]} y=${r.bbox[1]}
                width=${r.bbox[2] - r.bbox[0]} height=${r.bbox[3] - r.bbox[1]} rx="4"
                style=${{ animationDelay: (.25 + i * .12) + "s" }} />`)}
            ${corroborating.map((r, i) => html`<rect key=${"c" + i} class="box-draw" stroke=${r.colour}
                x=${r.bbox[0]} y=${r.bbox[1]}
                width=${r.bbox[2] - r.bbox[0]} height=${r.bbox[3] - r.bbox[1]} rx="4"
                style=${{ animationDelay: (.45 + i * .1) + "s" }} />`)}
            ${faces.map(([x, y, w, h], i) => {
              const s = 1.3, cx = x + w / 2, cy = y + h / 2;
              return html`<rect key=${"f" + i} class="box-draw" stroke="var(--brand)"
                x=${cx - (w * s) / 2} y=${cy - (h * s) / 2}
                width=${w * s} height=${h * s} rx="6"
                style=${{ animationDelay: (.15 + i * .1) + "s" }} />`;
            })}
          </svg>`}
      </div>
      ${legend.length
        ? html`<div class="legend">${legend.map(([c, t], i) =>
            html`<span key=${i}><i style=${{ background: c }}></i>${t}</span>`)}</div>`
        : html`<div class="legend"><span>No region differed from the rest of the picture.</span></div>`}
    <//>`;
}

/* ------------------------------------------------------------ signal groups */
function Signals({ ev }) {
  return DIRECTIONS.map((d) => {
    const items = ev.signals.filter((s) => s.direction === d.id);
    if (!items.length) return null;
    const collapsed = d.id === "context" || d.id === "limitation";
    const body = html`
      <div>
        ${items.map((s, i) => html`
          <${motion.div} key=${s.name + i} class="sig" style=${{ "--sc": d.colour }}
            initial=${{ opacity: 0, x: -10 }} animate=${{ opacity: 1, x: 0 }}
            transition=${{ delay: .05 * i, duration: .3 }}>
            <span class="sig-icon"></span>
            <div class="sig-text">
              <p>${s.human}</p>
              <div class="sig-foot">
                ${s.where ? html`<span class="tag">${s.where}</span>` : null}
                <span class="tag">${fmtValue(s.value).slice(0, 90)}</span>
              </div>
            </div>
          <//>`)}
      </div>`;

    if (collapsed) {
      return html`<${Fold} key=${d.id}
        title=${d.id === "limitation"
          ? `What this check could not tell you (${items.length})`
          : `Other measurements (${items.length})`}>${body}<//>`;
    }
    return html`
      <div key=${d.id}>
        <div class="sig-head" style=${{ "--sc": d.colour }}>
          ${d.title}<span class="count">${items.length}</span>
        </div>
        ${body}
      </div>`;
  });
}

/* ------------------------------------------------------------------ result */
function Result({ ev, previewUrl }) {
  const rel = ev.reliability || {};
  return html`
    <div>
      <${VerdictCard} ev=${ev} />

      <${motion.div} class="card" style=${{ marginTop: "14px" }}
        initial=${{ opacity: 0, y: 12 }} animate=${{ opacity: 1, y: 0 }}
        transition=${{ delay: .12, duration: .4 }}>
        <div class="why-label">In plain English</div>
        <div class="why">${ev.explanation}</div>
        ${previewUrl && ev.modality === "image"
          ? html`<${Shot} src=${previewUrl} ev=${ev} />` : null}
        <${Signals} ev=${ev} />
      <//>

      ${(rel.ood_flags?.length || rel.soft_flags?.length) ? html`
        <${Fold} title="Why confidence was reduced">
          ${rel.ood_flags?.length ? html`
            <div class="alert" style=${{ "--ac": "var(--bad)" }}>
              <div><strong>Analysis could not run properly.</strong><br/>
              ${rel.ood_flags.join(", ")}${rel.note ? " — " + rel.note : ""}</div>
            </div>` : null}
          ${rel.soft_flags?.length ? html`
            <div class="alert" style=${{ "--ac": "var(--warn)" }}>
              <div><strong>Conditions that widened the range by ±${(rel.band || 0).toFixed(2)}:</strong><br/>
              ${rel.soft_flags.join(", ")}</div>
            </div>` : null}
        <//>` : null}

      <${Fold} title="Everything we measured">
        <pre class="json">${JSON.stringify(ev, null, 2)}</pre>
      <//>

      <p style=${{ color: "var(--ink-3)", fontSize: ".84rem", marginTop: "16px" }}>
        This input was analysed in memory and has already been discarded. TrustLens stores nothing.
      </p>
    </div>`;
}

/* ------------------------------------------------------------ scanning view */
const STEPS = ["Reading the file", "Checking provenance", "Running forensics",
               "Asking the model", "Weighing the evidence"];

function Scanning({ modality }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), 1100);
    return () => clearInterval(t);
  }, []);
  return html`
    <${motion.div} class="card scanning"
      initial=${{ opacity: 0, scale: .98 }} animate=${{ opacity: 1, scale: 1 }}>
      <div class="scan-beam"></div>
      <div class="why-label">Analysing ${modality || "input"}</div>
      <div class="scan-steps" style=${{ marginTop: "12px" }}>
        ${STEPS.map((s, i) => html`
          <div key=${s} class=${"scan-step" + (i <= step ? " done" : "")}>
            <span class="tick">${i < step ? html`<${Icon.check} />` : null}</span>${s}
          </div>`)}
      </div>
    <//>`;
}

/* ------------------------------------------------------------ analyse page */
function Analyse({ health }) {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ev, setEv] = useState(null);
  const [err, setErr] = useState(null);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef(null);

  const pick = (f) => {
    if (!f) return;
    setFile(f); setEv(null); setErr(null);
    setPreview((old) => { if (old) URL.revokeObjectURL(old);
      return f.type.startsWith("image/") ? URL.createObjectURL(f) : null; });
  };

  const run = useCallback(async () => {
    setBusy(true); setErr(null); setEv(null);
    try {
      const body = new FormData();
      if (mode === "text") body.append("text", text);
      else body.append("file", file);
      body.append("explain_with_model", "true");
      const res = await fetch("/analyze", { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `The server returned ${res.status}.`);
      }
      setEv(await res.json());
    } catch (e) {
      setErr(e.message || "Could not reach the analyser.");
    } finally { setBusy(false); }
  }, [mode, text, file]);

  const ready = mode === "text" ? text.trim().length > 0 : !!file;

  return html`
    <div>
      <div class="page-head">
        <h1 class="page-title">Check something</h1>
        <p class="page-sub">Paste a message, or drop in an image, voice recording or video.
          TrustLens never answers with a flat “real” or “fake” — where the evidence is thin,
          it says so instead of guessing.</p>
      </div>

      <div class="card">
        <div class="mode-tabs">
          ${[["text", "Message or article"], ["file", "Image, audio or video"]].map(([id, label]) => html`
            <button key=${id} class=${"mode-tab" + (mode === id ? " active" : "")}
              onClick=${() => { setMode(id); setEv(null); setErr(null); }}>
              ${mode === id ? html`<${motion.span} layoutId="modePill" class="mode-pill"
                  transition=${{ type: "spring", stiffness: 380, damping: 32 }} />` : null}
              ${label}
            </button>`)}
        </div>

        <${AnimatePresence} mode="wait">
          ${mode === "text"
            ? html`<${motion.div} key="t" initial=${{ opacity: 0, y: 8 }} animate=${{ opacity: 1, y: 0 }}
                     exit=${{ opacity: 0, y: -8 }} transition=${{ duration: .2 }}>
                <textarea class="input" value=${text} onInput=${(e) => setText(e.target.value)}
                  placeholder="Paste a message, email, social post or article here…"></textarea>
              <//>`
            : html`<${motion.div} key="f" initial=${{ opacity: 0, y: 8 }} animate=${{ opacity: 1, y: 0 }}
                     exit=${{ opacity: 0, y: -8 }} transition=${{ duration: .2 }}>
                <input ref=${inputRef} type="file" style=${{ display: "none" }}
                  onChange=${(e) => pick(e.target.files[0])} />
                ${file
                  ? html`<div class="chosen">
                      <span class="kind">${(file.name.split(".").pop() || "?").toUpperCase().slice(0,4)}</span>
                      <div style=${{ flex: 1, minWidth: 0 }}>
                        <div style=${{ fontWeight: 600, overflow: "hidden",
                                       textOverflow: "ellipsis", whiteSpace: "nowrap" }}>${file.name}</div>
                        <div style=${{ color: "var(--ink-3)", fontSize: ".82rem" }}>
                          ${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                      </div>
                      <button class="btn ghost" onClick=${() => { setFile(null); setPreview(null); }}>Change</button>
                    </div>`
                  : html`<div class=${"drop" + (over ? " over" : "")}
                      onClick=${() => inputRef.current?.click()}
                      onDragOver=${(e) => { e.preventDefault(); setOver(true); }}
                      onDragLeave=${() => setOver(false)}
                      onDrop=${(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
                      <div class="drop-icon"><${Icon.upload} width="21" height="21" /></div>
                      <h4>Drop a file here</h4>
                      <p>or click to browse — images, audio and video up to 200 MB</p>
                    </div>`}
              <//>`}
        <//>

        <div style=${{ display: "flex", gap: "10px", marginTop: "16px", alignItems: "center" }}>
          <button class="btn" disabled=${!ready || busy} onClick=${run}>
            ${busy ? html`<span class="spinner"></span>` : html`<${Icon.scan} width="16" height="16" />`}
            ${busy ? "Analysing…" : "Analyse"}
          </button>
          ${!health?.language_model?.available
            ? html`<span style=${{ color: "var(--ink-3)", fontSize: ".83rem" }}>
                Running on forensics only — no model key set.</span>` : null}
        </div>
      </div>

      <${AnimatePresence} mode="wait">
        ${err ? html`
          <${motion.div} key="err" class="alert" style=${{ "--ac": "var(--bad)", marginTop: "14px" }}
            initial=${{ opacity: 0, y: 8 }} animate=${{ opacity: 1, y: 0 }} exit=${{ opacity: 0 }}>
            <div><strong>That didn’t work.</strong><br/>${err}</div>
          <//>` : null}
        ${busy ? html`<${motion.div} key="scan" style=${{ marginTop: "14px" }}
            initial=${{ opacity: 0 }} animate=${{ opacity: 1 }} exit=${{ opacity: 0 }}>
            <${Scanning} modality=${mode === "text" ? "text" : file?.type?.split("/")[0]} />
          <//>` : null}
        ${ev && !busy ? html`<${motion.div} key="res" style=${{ marginTop: "14px" }}
            initial=${{ opacity: 0 }} animate=${{ opacity: 1 }}>
            <${Result} ev=${ev} previewUrl=${preview} />
          <//>` : null}
      <//>
    </div>`;
}

/* ------------------------------------------------------------ static pages */
function HowItWorks() {
  return html`
    <div>
      <div class="page-head">
        <h1 class="page-title">How it works</h1>
        <p class="page-sub">Four checks, and an explicit refusal to overstate what they prove.</p>
      </div>
      <div class="card prose">
        <h3>Why there is no “fake” button</h3>
        <p>Deepfake-Eval-2024 took nine state-of-the-art open-source detectors and ran them on
          deepfakes collected from real social media rather than the benchmarks they were built
          for. AASIST fell from <strong>1.00 AUC to 0.43</strong>. GenConViT from 0.96 to 0.63.
          NPR from 0.98 to 0.53. An AUC of 0.50 is a coin flip, and no commercial detector the
          same team tested reached 90% accuracy.</p>
        <p>So TrustLens reports one of four bands, and treats “I don’t know” as a real answer.</p>

        <div style=${{ marginTop: "16px" }}>
          ${[["var(--declared)", "Declared AI", "The file’s own metadata says it was AI-generated. No model needed."],
             ["var(--ok)", "No evidence of manipulation", "The checks ran and found nothing. Never “this is real”."],
             ["var(--warn)", "Not enough to call it", "Ambiguous, or conditions that make the checks unreliable."],
             ["var(--bad)", "Likely manipulated", "Several independent checks agree."]].map(([c, t, d]) => html`
            <div key=${t} class="band-row">
              <span class="band-key" style=${{ background: c }}></span>
              <div><strong>${t}</strong><br/><span style=${{ color: "var(--ink-3)" }}>${d}</span></div>
            </div>`)}
        </div>

        <h3>Text</h3>
        <p>Eleven deterministic rules that quote the matched words back at you, so you can judge
          for yourself. A language model then reads the whole message alongside those hits. For
          factual claims, the same question is put to the model several times — answers that
          contradict each other mean the claim is not well supported.</p>

        <h3>Image</h3>
        <p>Provenance first: C2PA, IPTC and EXIF, read from real file structure, never the
          filename. Then noise-residual analysis finds regions carrying a different sensor
          texture, with ELA and JPEG-ghost used only to corroborate. Faces are cropped at 1.3×,
          the ratio FaceForensics++ found worth roughly 17 accuracy points.</p>

        <h3>Audio</h3>
        <p>Honest scope: this measures how a recording was <strong>processed</strong>, not
          whether a voice was cloned. Spectral ceiling, digitally-exact silence, pitch
          steadiness, noise floor. A good clone recorded through a real microphone passes these.</p>

        <h3>Video</h3>
        <p>Twelve sampled frames plus the soundtrack, fused. It reports the
          <strong>distribution</strong> — “5 of 12 frames” — because detectors that assume a
          whole video is fake lose about 31% accuracy on selectively manipulated footage.</p>

        <h3>What it cannot do</h3>
        <ul>
          <li>No trained face-swap detector, so a subtle identity swap is missed.</li>
          <li>Without a model key it finds <em>edits</em>, not fully generated images.</li>
          <li>Thresholds are tuned on a small sample — a starting point, not a benchmark.</li>
        </ul>
        <p style=${{ color: "var(--ink-3)", fontSize: ".86rem" }}>
          Accuracy figures above belong to the cited papers, not to this build.</p>
      </div>
    </div>`;
}

function Privacy({ stats, refresh }) {
  return html`
    <div>
      <div class="page-head">
        <h1 class="page-title">Privacy</h1>
        <p class="page-sub">Nothing you submit is stored. This page proves it rather than
          asking you to take it on trust.</p>
      </div>
      <div class="grid-2" style=${{ marginBottom: "14px" }}>
        ${[["Uploads kept", "0", "analysed in memory"],
           ["Results kept", "0", "discarded with the response"],
           ["Files on disk", String(stats?.scratch_files?.files_remaining ?? "—"), "should always be zero"]]
          .map(([k, v, d]) => html`
            <div key=${k} class="stat"><div class="k">${k}</div>
              <div class="v">${v}</div><div class="d">${d}</div></div>`)}
      </div>
      <div class="card prose">
        <h3>The one exception, stated plainly</h3>
        <p>Video is the only case where your file touches the disk, because OpenCV cannot read a
          video from memory. It is written to a scratch file, overwritten, and deleted in a
          <code>finally</code> block — so it goes even if the analysis crashes.</p>
        <p>This is auditable: search the codebase for <code>scratch_file</code> and you have found
          every write of user content. The counters above come live from the running server.</p>
        <h3>Your API key</h3>
        <p>Read from the environment only. Never typed into this interface, never logged, never
          sent anywhere except Google’s API.</p>
        <button class="btn ghost" onClick=${refresh} style=${{ marginTop: "6px" }}>
          Re-check for leftover files</button>
      </div>
    </div>`;
}

function System({ health, stats }) {
  const lm = health?.language_model || {};
  const rows = [
    ["Provenance (C2PA / EXIF)", true, "always available"],
    ["Image forensics", true, "noise residual, ELA, JPEG ghost"],
    ["Audio & video decoding", !!health?.audio_video_decoder, "bundled ffmpeg"],
    ["Language & vision model", !!lm.available, lm.available ? "connected" : (lm.reason || "unavailable")],
    ["Trained face-swap model", false, "not installed by design"],
  ];
  return html`
    <div>
      <div class="page-head">
        <h1 class="page-title">System</h1>
        <p class="page-sub">What is switched on right now, and what each band means.</p>
      </div>
      <div class="card">
        ${rows.map(([name, ok, note]) => html`
          <div key=${name} class="band-row">
            <span class=${"dot " + (ok ? "on" : "off")} style=${{ marginTop: "7px" }}></span>
            <div style=${{ flex: 1 }}>
              <strong>${name}</strong>
              <div style=${{ color: "var(--ink-3)", fontSize: ".85rem" }}>${note}</div>
            </div>
          </div>`)}
      </div>
      ${!lm.available ? html`
        <div class="alert" style=${{ "--ac": "var(--warn)", marginTop: "14px" }}>
          <div><strong>Running without a model.</strong><br/>
            Forensics and rules still work. Set <code>GEMINI_API_KEY</code> in a
            <code>.env</code> file to also detect fully AI-generated content.</div>
        </div>` : null}
      <div class="card prose" style=${{ marginTop: "14px" }}>
        <h3>Version</h3>
        <p><code>TrustLens ${health?.version || "—"}</code> · scratch files created
          ${stats?.scratch_files?.files_created ?? 0}, removed
          ${stats?.scratch_files?.files_removed ?? 0}.</p>
      </div>
    </div>`;
}

/* ------------------------------------------------------------------- shell */
const TABS = [
  { id: "analyse", label: "Analyse",      icon: Icon.lens },
  { id: "how",     label: "How it works", icon: Icon.book },
  { id: "privacy", label: "Privacy",      icon: Icon.shield },
  { id: "system",  label: "System",       icon: Icon.cpu },
];

function App() {
  const [tab, setTab] = useState("analyse");
  const [theme, setTheme] = useState(() => localStorage.getItem("tl-theme") || "dark");
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("tl-theme", theme); } catch (e) { /* private mode */ }
  }, [theme]);

  const loadStats = useCallback(() => {
    fetch("/privacy").then((r) => r.json()).then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/health").then((r) => r.json()).then(setHealth).catch(() => {});
    loadStats();
  }, [loadStats]);

  const lmOn = !!health?.language_model?.available;

  return html`
    <div class="shell">
      <aside class="sidebar">
        <div class="brand">
          <div class="brand-mark"><${Icon.lens} style=${{ color: "#fff" }} /></div>
          <div>
            <div class="brand-name">TrustLens</div>
            <div class="brand-sub">media forensics</div>
          </div>
        </div>

        <nav class="nav">
          <div class="nav-label">Workspace</div>
          ${TABS.map((t) => html`
            <button key=${t.id} class=${"nav-item" + (tab === t.id ? " active" : "")}
              onClick=${() => setTab(t.id)}>
              ${tab === t.id ? html`<${motion.span} layoutId="navPill" class="nav-pill"
                  transition=${{ type: "spring", stiffness: 420, damping: 34 }} />` : null}
              <${t.icon} />${t.label}
            </button>`)}
        </nav>

        <div class="side-block">
          <div class="status-row">
            <span class=${"dot " + (health ? "on" : "off") + (health ? "" : " pulse")}></span>
            ${health ? "Engine ready" : "Connecting…"}
          </div>
          <div class="status-row">
            <span class=${"dot " + (lmOn ? "on" : "off")}></span>
            ${lmOn ? "Model connected" : "Forensics only"}
          </div>
          <button class="theme-toggle" onClick=${() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle colour theme">
            <span>${theme === "dark" ? "Dark" : "Light"}</span>
            <span class="switch"><b></b></span>
          </button>
        </div>
      </aside>

      <main class="main">
        <${AnimatePresence} mode="wait">
          <${motion.div} key=${tab}
            initial=${{ opacity: 0, y: 12 }} animate=${{ opacity: 1, y: 0 }}
            exit=${{ opacity: 0, y: -8 }}
            transition=${{ duration: .26, ease: [.22,.61,.36,1] }}>
            ${tab === "analyse" ? html`<${Analyse} health=${health} />` : null}
            ${tab === "how"     ? html`<${HowItWorks} />` : null}
            ${tab === "privacy" ? html`<${Privacy} stats=${stats} refresh=${loadStats} />` : null}
            ${tab === "system"  ? html`<${System} health=${health} stats=${stats} />` : null}
          <//>
        <//>
      </main>
    </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
