import json
import requests
import streamlit as st
from schemas import Evidence, VerdictEnum, ModalityEnum
from detectors.text import TextDetector
from detectors.audio import AudioDetector
from detectors.image import ImageDetector
from detectors.video import VideoDetector
from explainer import explain
from backend.services.image_analysis import ImageAnalysisService
from backend.schemas.detector_result import DetectorResult

# Page configuration
st.set_page_config(
    page_title="Truth Lens — GenAI Misinformation & Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.55rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .badge-authentic {
        background: linear-gradient(135deg, #10B981, #059669);
    }
    .badge-synthetic {
        background: linear-gradient(135deg, #EC4899, #BE185D);
    }
    .badge-manipulated {
        background: linear-gradient(135deg, #EF4444, #DC2626);
    }
    .badge-uncertain {
        background: linear-gradient(135deg, #F59E0B, #D97706);
    }
    .badge-declared {
        background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    }
    .card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
        color: #F8FAFC;
    }
    .explanation-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border-left: 5px solid #6366F1;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .signal-item {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .signal-title {
        font-weight: 600;
        color: #38BDF8;
        font-size: 0.95rem;
    }
    .signal-where {
        font-size: 0.8rem;
        background: #1E293B;
        color: #94A3B8;
        padding: 2px 8px;
        border-radius: 4px;
        float: right;
    }
    .confidence-container {
        margin: 1.2rem 0;
        padding: 1rem;
        background: #0F172A;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .confidence-label {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #CBD5E1;
    }
    .interval-bar-bg {
        height: 24px;
        background: #1E293B;
        border-radius: 12px;
        position: relative;
        overflow: hidden;
        margin-top: 0.5rem;
        border: 1px solid #475569;
    }
    .interval-band {
        height: 100%;
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        position: absolute;
        opacity: 0.7;
        border-radius: 6px;
    }
    .interval-point {
        width: 6px;
        height: 100%;
        background: #F43F5E;
        position: absolute;
        top: 0;
        z-index: 10;
        box-shadow: 0 0 8px #F43F5E;
    }
</style>
""", unsafe_allow_html=True)

# Initialize standalone fallback detectors
text_det = TextDetector()
audio_det = AudioDetector()
image_det = ImageDetector()
video_det = VideoDetector()
phase2_image_service = ImageAnalysisService()

# Sidebar setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric-headers/100/search.png", width=64)
    st.title("Truth Lens Settings")
    st.markdown("**Hackathon Configuration**")
    
    backend_mode = st.radio("API Connection Mode", ["FastAPI Server (http://127.0.0.1:8000)", "Direct Embedded (Offline Demo)"])
    fastapi_url = st.text_input("FastAPI Endpoint", "http://127.0.0.1:8000/analyze")
    fastapi_image_url = st.text_input("FastAPI Image Endpoint", "http://127.0.0.1:8000/analyze/image")
    
    st.divider()
    st.markdown("### LLM Explainer Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI (gpt-4o-mini)", "Gemini (gemini-1.5-flash)"])
    llm_key = st.text_input("API Key (Optional)", type="password", help="Leave blank to use grounded mock explainer")
    
    st.divider()
    st.info("💡 **Non-Negotiable Contract**: Truth Lens never displays binary 'Real vs Fake' verdicts. Scores are calibrated and bounded by reliability gates.")

# Header
st.markdown("<div class='main-header'>🔍 Truth Lens</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>GenAI Multimodal Classifier & Explainer — Text, Image, Audio & Video</div>", unsafe_allow_html=True)

# Preset Quick Demo Buttons
st.markdown("##### ⚡ Quick Demo Presets (Select to load sample data)")
col1, col2, col3, col4 = st.columns(4)

preset_text = None
preset_filename = None
preset_modality = None

if col1.button("🚨 Text Bank Scam", use_container_width=True):
    preset_text = "URGENT: Your PCCOE bank account is suspended due to suspicious activity! Click http://bit.ly/secure-verify-now immediately to restore access or your funds will be frozen."
    preset_modality = "Text"

if col2.button("📷 AI Image (Midjourney)", use_container_width=True):
    preset_filename = "c2pa_midjourney_photo.jpg"
    preset_modality = "Image"

if col3.button("🎙️ Voice Clone Deepfake", use_container_width=True):
    preset_filename = "elevenlabs_clone_speech.wav"
    preset_modality = "Audio"

if col4.button("🎥 Video Face-Swap", use_container_width=True):
    preset_filename = "video_face_swap_fake.mp4"
    preset_modality = "Video"

# Input Selection
tab_text, tab_file = st.tabs(["📝 Text Analysis", "📁 Media File Upload (Image / Audio / Video)"])

input_text = ""
uploaded_file = None

with tab_text:
    default_val = preset_text if preset_text else ""
    input_text = st.text_area("Paste message, social media post, or article text to inspect:", value=default_val, height=120)
    submit_text = st.button("Inspect Text", type="primary", key="btn_text")

with tab_file:
    uploaded_file = st.file_uploader("Upload Image, Audio, or Video file:", type=["jpg", "jpeg", "png", "webp", "mp3", "wav", "m4a", "mp4", "mov", "webm"])
    if uploaded_file and (uploaded_file.type or "").startswith("image/"):
        st.image(uploaded_file, caption=f"Preview: {uploaded_file.name}", use_column_width=False, width=320)
    submit_file = st.button("Inspect File", type="primary", key="btn_file")


import os
import io

def get_preset_bytes(filename: str) -> bytes:
    possible_paths = [
        os.path.join("tests", "data", "images", filename),
        os.path.join("scratch", filename),
        filename
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    
    # Fallback to generating a valid dummy JPEG in memory
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Preset: {filename}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_analysis(txt: str = None, file_obj = None, preset_fn: str = None):
    evidence_data = None
    phase2_image_result = None
    
    # Try FastAPI server first if selected
    if "FastAPI Server" in backend_mode:
        try:
            files = None
            data = {}
            if llm_key:
                data["api_key"] = llm_key
                data["provider"] = "gemini" if "Gemini" in provider else "openai"
                
            if txt:
                data["text"] = txt
                resp = requests.post(fastapi_url, data=data, timeout=5)
                if resp.status_code == 200:
                    evidence_data = Evidence(**resp.json())
            elif file_obj:
                files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type or "application/octet-stream")}
                fn = file_obj.name.lower()
                if any(fn.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    # Call Phase 2 /analyze/image
                    resp = requests.post(fastapi_image_url, files=files, timeout=5)
                    if resp.status_code == 200:
                        phase2_image_result = DetectorResult(**resp.json()["result"])
                if not phase2_image_result:
                    resp = requests.post(fastapi_url, data=data, files=files, timeout=5)
                    if resp.status_code == 200:
                        evidence_data = Evidence(**resp.json())
            elif preset_fn:
                preset_bytes = get_preset_bytes(preset_fn)
                files = {"file": (preset_fn, preset_bytes, "image/jpeg")}
                if any(ext in preset_fn.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    resp = requests.post(fastapi_image_url, files=files, timeout=5)
                    if resp.status_code == 200:
                        phase2_image_result = DetectorResult(**resp.json()["result"])
                if not phase2_image_result:
                    resp = requests.post(fastapi_url, data=data, files=files, timeout=5)
                    if resp.status_code == 200:
                        evidence_data = Evidence(**resp.json())

        except Exception as e:
            st.warning(f"FastAPI server unreachable ({e}). Falling back to Embedded Engine.")

    # Fallback to Embedded Engine
    if not evidence_data and not phase2_image_result:
        meta = {}
        if llm_key:
            meta["api_key"] = llm_key
            meta["provider"] = "gemini" if "Gemini" in provider else "openai"

        if txt:
            evidence_data = text_det.analyse(txt, metadata=meta)
        elif file_obj:
            fn = file_obj.name.lower()
            b = file_obj.getvalue()
            meta.update({"filename": file_obj.name, "content_type": file_obj.type})
            if any(fn.endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".webp"]):
                phase2_image_result = phase2_image_service.analyze_image(b, filename=file_obj.name)
            elif any(fn.endswith(ext) for ext in [".mp3", ".wav", ".m4a"]):
                evidence_data = audio_det.analyse(b, meta)
            elif any(fn.endswith(ext) for ext in [".mp4", ".mov", ".webm"]):
                evidence_data = video_det.analyse(b, meta)
            else:
                evidence_data = text_det.analyse(b.decode("utf-8", errors="ignore"), meta)
        elif preset_fn:
            meta.update({"filename": preset_fn})
            preset_bytes = get_preset_bytes(preset_fn)
            if any(ext in preset_fn.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                phase2_image_result = phase2_image_service.analyze_image(preset_bytes, filename=preset_fn)
            elif "clone" in preset_fn:
                evidence_data = audio_det.analyse(preset_bytes, meta)
            elif "swap" in preset_fn:
                evidence_data = video_det.analyse(preset_bytes, meta)

        if evidence_data:
            evidence_data.explanation = explain(evidence_data, api_key=llm_key, provider="gemini" if "Gemini" in provider else "openai")

    return evidence_data, phase2_image_result


# Process trigger & session state management
if "active_evidence" not in st.session_state:
    st.session_state["active_evidence"] = None
if "active_phase2_image" not in st.session_state:
    st.session_state["active_phase2_image"] = None

if (submit_text and input_text) or preset_text:
    st.session_state["active_evidence"], st.session_state["active_phase2_image"] = run_analysis(txt=input_text or preset_text)
elif (submit_file and uploaded_file) or preset_filename:
    st.session_state["active_evidence"], st.session_state["active_phase2_image"] = run_analysis(file_obj=uploaded_file, preset_fn=preset_filename)

active_evidence = st.session_state["active_evidence"]
active_phase2_image = st.session_state["active_phase2_image"]


# Render Results
if active_phase2_image:
    st.divider()
    st.subheader("🖼️ Phase 2: AI Image Authenticity Analysis")
    
    # Disclaimer Banner
    st.warning("⚠️ **Notice**: Truth Lens assessments are AI-assisted probabilistic evaluations, not absolute proof.")

    col_v, col_ds, col_c, col_cl = st.columns([2, 1, 1, 1])
    
    with col_v:
        pred = active_phase2_image.prediction
        if pred == "declared_ai":
            badge_html = "<div class='verdict-badge badge-declared'>📜 AI Generation Declared by Provenance</div>"
        elif pred == "likely_synthetic":
            badge_html = "<div class='verdict-badge badge-synthetic'>🤖 Likely AI-Generated</div>"
        elif pred == "likely_authentic":
            badge_html = "<div class='verdict-badge badge-authentic'>✅ No Strong Evidence of Manipulation</div>"
        elif pred == "likely_manipulated":
            badge_html = "<div class='verdict-badge badge-manipulated'>🚨 Likely Manipulated</div>"
        else:
            badge_html = "<div class='verdict-badge badge-uncertain'>⚠️ Uncertain — More Evidence Needed</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

    with col_ds:
        st.metric("Detector Score", f"{active_phase2_image.detector_score:.4f}")

    with col_c:
        st.metric("Decision Confidence", f"{active_phase2_image.decision_confidence * 100:.0f}%")

    with col_cl:
        st.metric("Confidence Level", active_phase2_image.confidence_level.upper())

    # Grounded Narrative Explanation
    if active_phase2_image.explanation:
        st.markdown("##### 💬 Plain-English Explanation")
        st.markdown(f"<div class='explanation-box'>{active_phase2_image.explanation}</div>", unsafe_allow_html=True)

    # Confidence Interval / Uncertainty Bar
    conf_pct = active_phase2_image.confidence * 100
    unc_pct = active_phase2_image.uncertainty * 100

    st.markdown(f"""
    <div class='confidence-container'>
        <div class='confidence-label'>
            Confidence: <strong>{conf_pct:.0f}%</strong> ({active_phase2_image.confidence_level.upper()}) 
            <span style='float:right; color:#94A3B8;'>Uncertainty Factor: {unc_pct:.0f}%</span>
        </div>
        <div class='interval-bar-bg'>
            <div class='interval-band' style='left: 0%; width: {conf_pct}%;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if active_phase2_image.uncertainty_reason:
        st.info(f"ℹ️ **Uncertainty Note**: {active_phase2_image.uncertainty_reason}")

    # Grounded Evidence List
    st.markdown("##### 🔬 Why Was It Flagged? (Grounded Evidence)")
    if active_phase2_image.evidence:
        for ev in active_phase2_image.evidence:
            loc_tag = f"<span class='signal-where'>📍 {ev.location}</span>" if ev.location else ""
            st.markdown(f"""
            <div class='signal-item'>
                {loc_tag}
                <div class='signal-title'>• Category: <code>{ev.category}</code> (Score: {ev.score:.4f})</div>
                <div style='color: #E2E8F0; margin-top: 4px;'>{ev.description}</div>
            </div>
            """, unsafe_allow_html=True)

    # Provenance Information
    st.markdown("##### 📜 Provenance & Metadata")
    prov = active_phase2_image.provenance
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("EXIF Found", "Yes" if prov.exif_found else "No")
    p_col2.metric("C2PA Credentials", "Yes" if prov.c2pa_present or prov.c2pa_ai_declared else "No")
    p_col3.metric("Software Tag", prov.software or "None")

    # Warnings / Limitations
    if active_phase2_image.warnings:
        st.markdown("##### ⚠️ Important Limitations")
        for warn in active_phase2_image.warnings:
            st.caption(f"• {warn}")

    # Expandable Technical Evidence Section (Task 15)
    with st.expander("🛠️ Technical Evidence & Decision Diagnostics"):
        st.markdown("#### ⚡ Decision Engine Rules Triggered")
        for r in active_phase2_image.decision_rules_triggered:
            st.code(f"• {r}")
        
        st.markdown("#### 🔬 Visual Detector Outputs")
        for det in active_phase2_image.detector_outputs:
            st.markdown(f"**Model**: `{det.model}` | **Task**: `{det.task}` | **Signal**: `{det.signal}` | **Reliability**: `{det.reliability}`")
            st.markdown(f"**Raw Label**: `{det.raw_label}` | **Scores**: `{det.scores}`")
            if det.limitations:
                for lim in det.limitations:
                    st.caption(f"Limitation: {lim}")
        
        st.divider()
        st.json(active_phase2_image.model_dump())

elif active_evidence:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    col_v, col_m, col_s = st.columns([2, 1, 1])
    
    with col_v:
        verdict = active_evidence.verdict
        if verdict == VerdictEnum.DECLARED_AI:
            badge_html = f"<div class='verdict-badge badge-declared'>📜 AI Generation Declared by Provenance</div>"
        elif verdict == VerdictEnum.LIKELY_AUTHENTIC:
            badge_html = f"<div class='verdict-badge badge-authentic'>✅ No Strong Evidence of Manipulation</div>"
        elif verdict == VerdictEnum.LIKELY_SYNTHETIC:
            badge_html = f"<div class='verdict-badge badge-synthetic'>🤖 Likely AI-Generated</div>"
        elif verdict == VerdictEnum.LIKELY_MANIPULATED:
            badge_html = f"<div class='verdict-badge badge-manipulated'>🚨 Likely Manipulated</div>"
        else: # UNCERTAIN
            badge_html = f"<div class='verdict-badge badge-uncertain'>⚠️ Uncertain — More Evidence Needed</div>"
            
        st.markdown(badge_html, unsafe_allow_html=True)

    with col_m:
        st.metric("Modality", active_evidence.modality.value.upper())

    with col_s:
        st.metric("Calibrated Score", f"{active_evidence.score:.2f}")

    # Confidence Interval Visualizer
    low_pct = active_evidence.confidence[0] * 100
    high_pct = active_evidence.confidence[1] * 100
    width_pct = max(2.0, high_pct - low_pct)
    point_pct = active_evidence.score * 100

    st.markdown(f"""
    <div class='confidence-container'>
        <div class='confidence-label'>
            Confidence Interval Band: <strong>[{active_evidence.confidence[0]:.2f} — {active_evidence.confidence[1]:.2f}]</strong> 
            <span style='float:right; color:#94A3B8;'>Score Point: {active_evidence.score:.2f}</span>
        </div>
        <div class='interval-bar-bg'>
            <div class='interval-band' style='left: {low_pct}%; width: {width_pct}%;'></div>
            <div class='interval-point' style='left: {point_pct}%;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; margin-top: 4px;'>
            <span>0.0 (Authentic Band < 0.35)</span>
            <span>0.35 - 0.65 (Uncertain)</span>
            <span>1.0 (Manipulated > 0.65)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Explanation Box
    st.markdown("##### 💬 Plain-English Explanation")
    st.markdown(f"<div class='explanation-box'>{active_evidence.explanation}</div>", unsafe_allow_html=True)

    # Grounded Signals List
    st.markdown("##### 🔬 Grounded Detection Signals")
    if active_evidence.signals:
        for sig in active_evidence.signals:
            where_tag = f"<span class='signal-where'>📍 {sig.where}</span>" if sig.where else ""
            st.markdown(f"""
            <div class='signal-item'>
                {where_tag}
                <div class='signal-title'>• {sig.name} (Weight: {sig.weight:.2f})</div>
                <div style='color: #E2E8F0; margin-top: 4px;'>{sig.human}</div>
                <div style='font-size: 0.8rem; color: #94A3B8; margin-top: 2px;'>Extracted Metric: <code>{sig.value}</code></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No suspicious signals detected.")

    # OOD Reliability Note if applicable
    if active_evidence.reliability.ood_flags:
        st.warning(f"⚠️ **Reliability Gate Warning**: {', '.join(active_evidence.reliability.ood_flags)} — {active_evidence.reliability.note}")

    # Collapsed "How we decided" Drawer with raw JSON
    with st.expander("🛠️ How We Decided (Raw Evidence JSON)"):
        st.json(active_evidence.model_dump())

else:
    st.info("👆 Enter text or upload a media file above, or select one of the Quick Demo Presets to run detection.")
