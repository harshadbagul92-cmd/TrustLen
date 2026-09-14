"""
Video lane: frames plus the voice track.

Two design choices come straight from Deepfake-Eval-2024's error analysis:

  1. Report the DISTRIBUTION of flagged frames, not just an average. Published
     detectors assume the whole video is fake and lose ~31% accuracy on videos
     where only some faces or some moments are manipulated. "5 of 12 frames"
     survives that, and is more explainable than "0.51" anyway.

  2. Check the audio track separately and fuse late. A video whose frames look
     fine but whose voice was synthesised is exactly the case a frames-only
     pipeline misses.

Per-frame work is deliberately the cheap subset of the image lane - noise
residual and face detection. The full JPEG-ghost sweep would mean ~9 re-encodes
per frame, which is not worth the seconds on a live demo.
"""
import os
from typing import Any, Dict, List

import numpy as np

from .. import config, schemas
from ..schemas import Direction, Evidence, Modality, Reliability, Signal
from . import forensics_image as fx
from . import media, vision_llm
from .audio import AudioDetector
from .base import Detector


class VideoDetector(Detector):
    modality = Modality.VIDEO

    def __init__(self):
        self._audio = AudioDetector()

    def _run(self, data: Any, meta: Dict[str, Any]) -> Evidence:
        if not isinstance(data, (bytes, bytearray)):
            return self.abstain("not_bytes", "This input was not a video file.")
        data = bytes(data)
        suffix = os.path.splitext(str(meta.get("filename", "")))[1] or ".mp4"

        signals: List[Signal] = []
        reliability = Reliability()
        contributions: List[float] = []
        findings: Dict[str, Any] = {"bytes": len(data)}

        container = media.probe(data) if media.ffmpeg_available() else {}
        findings["container"] = container

        # -- frames ---------------------------------------------------------
        frames, frame_meta = media.sample_frames(data, suffix)
        findings["video"] = frame_meta
        if not frames:
            return self.abstain(
                frame_meta.get("error", "no_frames"),
                "No video frames could be read from this file, so nothing was checked.",
                flag="undecodable_video", extra={"container": container})

        flagged: List[Dict[str, Any]] = []
        brightness: List[float] = []
        face_counts: List[int] = []
        analysed = 0

        for i, frame in enumerate(frames):
            gray = np.asarray(frame[:, :, 1], dtype=np.float32)
            brightness.append(float(gray.mean()))

            flat = fx.flatness(gray)
            if (flat["mean_gradient"] < config.IMG_MIN_GRADIENT
                    or flat["pixel_std"] < config.IMG_MIN_PIXEL_STD):
                continue                      # black or blank frame, skip
            analysed += 1

            h, w = frame.shape[:2]
            zmap, _ = fx.noise_residual_zmap(frame, block=config.IMG_NOISE_BLOCK)
            regions = fx.cluster_outliers(
                zmap, config.IMG_NOISE_BLOCK, w, h, config.IMG_Z_THRESHOLD,
                config.IMG_NOISE_MIN_BLOCKS, config.IMG_MIN_COVERAGE_PCT)

            faces, _ = fx.detect_faces(frame)
            face_counts.append(len(faces))

            if regions:
                top = regions[0]
                flagged.append({
                    "frame_index": frame_meta.get("frame_indices", list(range(len(frames))))[i]
                    if i < len(frame_meta.get("frame_indices", [])) else i,
                    "position": i,
                    "region": top["region"],
                    "bbox": top["bbox"],
                    "peak_z": top["peak_z"],
                    "coverage_pct": top["coverage_pct"],
                })

        findings["frames_sampled"] = len(frames)
        findings["frames_analysed"] = analysed
        findings["frames_flagged"] = len(flagged)
        findings["flagged_detail"] = flagged[:6]
        findings["faces_per_frame"] = face_counts

        if analysed == 0:
            reliability.ood_flags.append("all_frames_blank")
            reliability.note = "Every sampled frame was too dark or flat to analyse."
        else:
            ratio = len(flagged) / analysed
            findings["flagged_ratio"] = round(ratio, 3)
            if flagged:
                where = ", ".join(sorted({f["region"] for f in flagged}))
                if ratio >= config.VIDEO_FRAME_FLAG_RATIO:
                    contributions.append(min(0.55, 0.20 + 0.5 * ratio))
                    direction, weight = Direction.MANIPULATION, 0.70
                    human = (f"{len(flagged)} of the {analysed} frames checked contain a region "
                             f"whose texture differs from the rest of that frame, around {where}. "
                             f"A consistent patch like this across many frames is what edited or "
                             f"pasted-in video looks like.")
                else:
                    contributions.append(0.12)
                    direction, weight = Direction.MANIPULATION, 0.35
                    human = (f"Only {len(flagged)} of the {analysed} frames checked showed an odd "
                             f"region ({where}). Too few to mean much - compression alone can do "
                             f"this on isolated frames.")
                signals.append(Signal(
                    name="frame_region_anomalies", direction=direction, human=human,
                    value={"flagged": len(flagged), "analysed": analysed,
                           "ratio": round(ratio, 3)},
                    weight=weight,
                    where="; ".join(f"frame {f['frame_index']} {f['region']}" for f in flagged[:3])))
            else:
                signals.append(Signal(
                    name="no_frame_anomalies", direction=Direction.AUTHENTIC,
                    human=f"None of the {analysed} frames checked contained a region that differs from the rest of that frame.",
                    value=0, weight=0.25, where=f"{analysed} sampled frames"))

        # -- brightness flicker ---------------------------------------------
        if len(brightness) >= 4:
            arr = np.array(brightness)
            mean = float(arr.mean())
            flicker = float(np.std(np.diff(arr))) / mean if mean > 1 else 0.0
            findings["brightness_flicker"] = round(flicker, 4)
            if flicker > 0.12:
                contributions.append(0.10)
                signals.append(Signal(
                    name="brightness_flicker", direction=Direction.MANIPULATION,
                    human=(f"Overall brightness jumps noticeably between sampled frames "
                           f"({flicker * 100:.0f}% variation). Scene cuts cause this too, so it "
                           f"is only a hint."),
                    value=round(flicker, 4), weight=0.30, where="across sampled frames"))

        if face_counts and len(set(face_counts)) > 1:
            findings["face_count_varies"] = True
            reliability.soft_flags.append("face_count_varies_across_frames")

        # -- faces present, but no specialist model -------------------------
        if any(face_counts):
            reliability.soft_flags.append("faces_present_no_specialist_model")
            reliability.band = max(reliability.band, 0.15)
            signals.append(Signal(
                name="faces_present_unscored", direction=Direction.LIMITATION,
                human=(f"Faces appear in the sampled frames (up to {max(face_counts)} at once), but "
                       f"this build has no trained face-swap detector. The faces themselves were "
                       f"not scored - only the picture around them was checked."),
                value={"max_faces": max(face_counts)}, weight=0.25,
                where="sampled frames"))

        # -- audio track -----------------------------------------------------
        audio_evidence = None
        if container.get("has_audio_stream") and media.ffmpeg_available():
            audio_evidence = self._audio.analyse(data, {"filename": meta.get("filename", "video")})
            findings["audio_lane"] = {
                "verdict": audio_evidence.verdict.value,
                "score": audio_evidence.score,
                "soft_flags": audio_evidence.reliability.soft_flags,
            }
            # Fold in only the audio findings that argue for manipulation, at a
            # reduced weight - the two modalities are independent evidence.
            for s in audio_evidence.by_direction(Direction.MANIPULATION):
                contributions.append(min(0.30, s.weight * 0.35))
                signals.append(Signal(
                    name=f"audio_{s.name}", direction=Direction.MANIPULATION,
                    human="In the soundtrack: " + s.human,
                    value=s.value, weight=min(1.0, s.weight * 0.8),
                    where=s.where or "soundtrack"))
            for s in audio_evidence.by_direction(Direction.AUTHENTIC)[:1]:
                signals.append(Signal(
                    name=f"audio_{s.name}", direction=Direction.AUTHENTIC,
                    human="In the soundtrack: " + s.human,
                    value=s.value, weight=s.weight * 0.6, where=s.where or "soundtrack"))
        else:
            reliability.soft_flags.append("no_audio_track_checked")
            signals.append(Signal(
                name="no_audio_track", direction=Direction.LIMITATION,
                human=("No soundtrack was analysed - this file has no audio stream, or audio "
                       "decoding is unavailable. A video can look genuine and still carry a "
                       "synthesised voice."),
                value=container.get("has_audio_stream", False), weight=0.20, where="soundtrack"))

        # -- vision model on a couple of keyframes ---------------------------
        vlm_ok, vlm_reason = None, None
        try:
            from PIL import Image
            import cv2
            ok, reason = __import__("trustlens.llm", fromlist=["llm"]).available()
            vlm_ok, vlm_reason = ok, reason
            if ok and frames:
                picks = [frames[len(frames) // 2]]
                if len(frames) > 3:
                    picks.append(frames[-2])
                seen = []
                for frame in picks:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = vision_llm.assess(pil)
                    if result["available"]:
                        seen.extend(result["consistent"])
                findings["vlm_keyframes"] = seen
                for e in seen[:3]:
                    contributions.append(e["ratio"] * (e["severity"] / 5.0) * 0.5)
                    signals.append(Signal(
                        name=f"vlm_{e['artifact']}", direction=Direction.MANIPULATION,
                        human=(f"In a sampled frame, a vision model saw "
                               f"{e['artifact'].replace('_', ' ')} ({e['detail']}) in "
                               f"{e['region']}, repeated across {e['agreement']} looks."),
                        value={"agreement": e["agreement"]}, weight=0.55, where="keyframe"))
        except Exception:
            pass

        if not vlm_ok:
            reliability.soft_flags.append("no_synthetic_video_detector")
            signals.append(Signal(
                name="synthetic_detector_unavailable", direction=Direction.LIMITATION,
                human=("No vision model is available, so fully AI-generated video would not be "
                       f"caught by these checks. (reason: {vlm_reason})"),
                value=vlm_reason, weight=0.30, where="whole video"))

        score = schemas.combine(contributions)
        findings["contributions"] = [round(c, 3) for c in contributions]
        return self.build(self.modality, score, signals, reliability, {}, findings)
