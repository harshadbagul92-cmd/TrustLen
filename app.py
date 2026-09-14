"""
TrustLens - Streamlit interface.

    streamlit run app.py

Layout principle: one answer, then the evidence, then the raw data. Someone
worried about a message should get the verdict and a plain-English reason
without scrolling; everything else is one click away behind an expander.

Nothing rendered here is hard-coded per verdict. Every sentence describing
what was found comes from the Evidence object, so the interface cannot
describe a finding the detector did not actually make.
"""
import io

import streamlit as st
from PIL import Image, ImageDraw

from trustlens import __version__, config, detectors, llm, privacy
from trustlens.explainer import explain
from trustlens.schemas import Direction, Evidence, Modality, Verdict

st.set_page_config(page_title="TrustLens", page_icon="🔎",
                   layout="centered", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------
# theme
# --------------------------------------------------------------------------
BRAND = "#2563EB"
COLOURS = {
    Verdict.LIKELY_AUTHENTIC:   ("#16A34A", "#ECFDF3", "No evidence of manipulation"),
    Verdict.UNCERTAIN:          ("#D97706", "#FFFBEB", "Not enough to call it"),
    Verdict.LIKELY_MANIPULATED: ("#DC2626", "#FEF2F2", "Likely manipulated"),
    Verdict.DECLARED_AI:        ("#7C3AED", "#F5F3FF", "The file declares it is AI-generated"),
}
DIRECTION_STYLE = {
    Direction.MANIPULATION: ("#DC2626", "Points to manipulation"),
    Direction.AUTHENTIC:    ("#16A34A", "Points to it being genuine"),
    Direction.LIMITATION:   ("#D97706", "Could not be checked"),
    Direction.CONTEXT:      ("#64748B", "Context"),
}

st.markdown(f"""
<style>
  #MainMenu, footer {{visibility: hidden;}}
  .block-container {{padding-top: 2.4rem; max-width: 820px;}}

  .tl-brand {{font-size: 2.7rem; font-weight: 800; letter-spacing: -1.2px;
              color: {BRAND}; line-height: 1;}}
  .tl-tag {{color: #64748B; font-size: 1.02rem; margin: .35rem 0 1.6rem;}}

  .tl-verdict {{border-radius: 14px; padding: 1.3rem 1.5rem; margin: .4rem 0 1.1rem;
                border: 1px solid var(--edge); background: var(--wash);
                border-left: 7px solid var(--edge);}}
  .tl-verdict h2 {{margin: 0; font-size: 1.55rem; font-weight: 750; color: var(--edge);
                   letter-spacing: -.4px;}}
  .tl-verdict .tl-sub {{color: #475569; font-size: .92rem; margin-top: .3rem;}}

  .tl-scale {{height: 12px; border-radius: 99px; position: relative; margin: .9rem 0 .3rem;
              background: linear-gradient(90deg,#16A34A 0%,#16A34A 35%,#F59E0B 35%,
                                          #F59E0B 65%,#DC2626 65%,#DC2626 100%); opacity:.30;}}
  .tl-band {{position: absolute; top: -3px; height: 18px; border-radius: 99px;
             background: var(--edge); opacity: .38;}}
  .tl-pin {{position: absolute; top: -6px; width: 3px; height: 24px; border-radius: 2px;
            background: var(--edge);}}
  .tl-ticks {{display:flex; justify-content:space-between; font-size:.73rem; color:#94A3B8;}}

  .tl-why {{background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;
            padding:1.15rem 1.35rem; font-size:1.06rem; line-height:1.65; color:#0F172A;}}

  .tl-sig {{background:#FFFFFF; border:1px solid #E9EDF3; border-left:4px solid var(--c);
            border-radius:9px; padding:.75rem .95rem; margin-bottom:.55rem;}}
  .tl-sig .h {{font-size:1rem; color:#0F172A; line-height:1.5;}}
  .tl-sig .m {{font-size:.78rem; color:#64748B; margin-top:.35rem;}}
  .tl-head {{font-weight:700; margin:.9rem 0 .45rem; font-size:.83rem;
             letter-spacing:.05em; text-transform:uppercase;}}
  .tl-pill {{display:inline-block; background:#EFF4FB; color:#334155; border-radius:5px;
             padding:1px 7px; font-size:.73rem; margin-right:.35rem;}}
  .tl-note {{font-size:.86rem; color:#64748B;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# header + sidebar
# --------------------------------------------------------------------------
st.markdown("<div class='tl-brand'>TrustLens</div>", unsafe_allow_html=True)
st.markdown("<div class='tl-tag'>Checks a message, image, voice recording or video &mdash; "
            "and explains, in plain words, why it was flagged.</div>", unsafe_allow_html=True)

model_ok, model_reason = llm.available()
from trustlens.detectors import media as _media

with st.sidebar:
    st.markdown(f"<div style='font-weight:800;color:{BRAND};font-size:1.3rem'>TrustLens</div>",
                unsafe_allow_html=True)
    st.caption(f"v{__version__}")
    st.divider()
    st.markdown("**Checks available**")
    st.markdown(
        "🟢 Provenance (C2PA / EXIF)  \n"
        "🟢 Image forensics  \n"
        f"{'🟢' if _media.ffmpeg_available() else '🔴'} Audio and video decoding  \n"
        f"{'🟢' if model_ok else '🔴'} Language and vision model"
        f"{'' if model_ok else f' — `{model_reason}`'}  \n"
        "🔴 Trained face-swap model — not installed by design")
    if not model_ok:
        st.info("Set `GEMINI_API_KEY` to enable detection of fully AI-generated "
                "content and model-written explanations.", icon="🔑")
    st.divider()
    use_model = st.toggle("Model-written explanations", value=model_ok,
                          disabled=not model_ok,
                          help="Off uses the built-in template, which reads the same findings.")
    st.divider()
    st.markdown("**Privacy**")
    st.caption("Nothing you submit is stored. Files are analysed in memory and "
               "discarded once the result is shown.")
    if st.button("Check for leftover files", use_container_width=True):
        st.write(privacy.stats())


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
tab_text, tab_file = st.tabs(["Message or article", "Image, audio or video"])
submitted = None

with tab_text:
    text_input = st.text_area("Paste the text you want checked", height=140,
                              label_visibility="collapsed",
                              placeholder="Paste a message, email, post or article here…")
    if st.button("Check this text", type="primary", use_container_width=True):
        if text_input.strip():
            submitted = ("text", text_input.strip(), "", "")
        else:
            st.warning("Paste some text first.")

with tab_file:
    upload = st.file_uploader(
        "Upload a file", label_visibility="collapsed",
        type=[e.lstrip(".") for e in
              detectors.IMAGE_EXT + detectors.AUDIO_EXT + detectors.VIDEO_EXT])
    if st.button("Check this file", type="primary", use_container_width=True):
        if upload is not None:
            submitted = ("file", upload.getvalue(), upload.name, upload.type or "")
        else:
            st.warning("Choose a file first.")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def annotate(raw: bytes, evidence: Evidence):
    """
    Draw only what fed the score: the noise regions, any corroborating region
    that overlaps one, and detected faces. Painting a region the score ignored
    would show the viewer a finding that is not one.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None, []
    draw = ImageDraw.Draw(img)
    width = max(3, int(min(img.size) * 0.006))
    legend, f = [], evidence.findings or {}

    noise = [tuple(r["bbox"]) for r in (f.get("noise_regions") or [])]
    for box in noise:
        draw.rectangle(box, outline="#DC2626", width=width)
    if noise:
        legend.append(("#DC2626", "different texture from the rest"))

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for key, colour, label in (("ela_regions", "#D97706", "compression error agrees"),
                               ("ghost_regions", "#CA8A04", "JPEG history agrees")):
        for region in (f.get(key) or []):
            box = tuple(region["bbox"])
            if any(overlaps(box, nb) for nb in noise):
                draw.rectangle(box, outline=colour, width=max(2, width - 1))
                if (colour, label) not in legend:
                    legend.append((colour, label))

    for (x, y, w, h) in (f.get("face_boxes") or []):
        cx, cy = x + w / 2, y + h / 2
        nw, nh = w * config.IMG_FACE_CROP_SCALE, h * config.IMG_FACE_CROP_SCALE
        draw.rectangle([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2],
                       outline=BRAND, width=max(2, width - 1))
    if f.get("face_boxes"):
        legend.append((BRAND, "face found (not scored)"))
    return img, legend


def render_signals(evidence: Evidence, directions):
    for direction in directions:
        group = evidence.by_direction(direction)
        if not group:
            continue
        colour, heading = DIRECTION_STYLE[direction]
        st.markdown(f"<div class='tl-head' style='color:{colour}'>{heading}</div>",
                    unsafe_allow_html=True)
        for s in group:
            where = f"<span class='tl-pill'>{s.where}</span>" if s.where else ""
            st.markdown(f"<div class='tl-sig' style='--c:{colour}'>"
                        f"<div class='h'>{s.human}</div>"
                        f"<div class='m'>{where}<code>{s.value}</code></div></div>",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------
# run and render
# --------------------------------------------------------------------------
if submitted is None:
    st.markdown("<div class='tl-note'>TrustLens never answers with a flat "
                "&ldquo;real&rdquo; or &ldquo;fake&rdquo;. Published detectors fall to near "
                "chance on real-world media, so where the evidence is thin this tool says "
                "so rather than guessing.</div>", unsafe_allow_html=True)
    st.stop()

kind, payload, filename, content_type = submitted

with st.spinner("Checking…"):
    if kind == "text":
        evidence = detectors.get(Modality.TEXT).analyse(payload, {})
    else:
        evidence = detectors.analyse(payload, filename, content_type)
    evidence.explanation = explain(evidence, use_model=use_model)

edge, wash, headline = COLOURS[evidence.verdict]
st.markdown(
    f"<div class='tl-verdict' style='--edge:{edge};--wash:{wash}'>"
    f"<h2>{headline}</h2>"
    f"<div class='tl-sub'>{evidence.modality.value.title()} checked in "
    f"{evidence.elapsed_ms / 1000:.1f}s &middot; suspicion score {evidence.score:.2f} "
    f"&middot; plausible range {evidence.confidence[0]:.2f}&ndash;{evidence.confidence[1]:.2f}"
    f"</div></div>", unsafe_allow_html=True)

low, high = evidence.confidence
st.markdown(
    f"<div class='tl-scale' style='--edge:{edge}'>"
    f"<div class='tl-band' style='left:{low * 100:.1f}%;width:{max(1.5, (high - low) * 100):.1f}%'></div>"
    f"<div class='tl-pin' style='left:{evidence.score * 100:.1f}%'></div></div>"
    f"<div class='tl-ticks'><span>no evidence</span><span>not enough to call</span>"
    f"<span>likely manipulated</span></div>", unsafe_allow_html=True)

st.markdown("")
st.markdown(f"<div class='tl-why'>{evidence.explanation}</div>", unsafe_allow_html=True)

if evidence.modality == Modality.IMAGE and kind == "file":
    picture, legend = annotate(payload, evidence)
    if picture is not None:
        st.markdown("")
        st.image(picture, use_container_width=True)
        if legend:
            st.markdown(" &nbsp; ".join(
                f"<span style='color:{c};font-weight:800'>&#9646;</span> "
                f"<span class='tl-note'>{t}</span>" for c, t in legend),
                unsafe_allow_html=True)
        else:
            st.markdown("<div class='tl-note'>No region differed from the rest of "
                        "the picture.</div>", unsafe_allow_html=True)

st.markdown("")
render_signals(evidence, [Direction.MANIPULATION, Direction.AUTHENTIC])

limits = evidence.limitations
if limits:
    with st.expander(f"What this check could not tell you ({len(limits)})"):
        render_signals(evidence, [Direction.LIMITATION])

context = evidence.by_direction(Direction.CONTEXT)
if context:
    with st.expander(f"Other measurements ({len(context)})"):
        render_signals(evidence, [Direction.CONTEXT])

if evidence.reliability.ood_flags or evidence.reliability.soft_flags:
    with st.expander("Why confidence was reduced"):
        if evidence.reliability.ood_flags:
            st.error("Analysis could not run properly: "
                     + ", ".join(f"`{x}`" for x in evidence.reliability.ood_flags))
        if evidence.reliability.soft_flags:
            st.warning(f"Conditions that widened the range by ±{evidence.reliability.band:.2f}: "
                       + ", ".join(f"`{x}`" for x in evidence.reliability.soft_flags))
        if evidence.reliability.note:
            st.caption(evidence.reliability.note)

with st.expander("Everything we measured"):
    st.json(evidence.model_dump(), expanded=False)

st.markdown("")
st.markdown("<div class='tl-note'>This input was analysed in memory and has already been "
            "discarded. TrustLens stores nothing.</div>", unsafe_allow_html=True)

del payload
