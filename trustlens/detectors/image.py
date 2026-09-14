"""
Image lane: provenance, forensics, and an optional vision model.

What it can do
  - Read a file's own declaration of AI origin (C2PA / IPTC / EXIF).
  - Localise regions whose noise texture, compression error or JPEG history
    differs from the rest of the frame - the signature of pasted-in content.
  - With a key set, ask a vision model for observable artifacts and keep only
    what it repeats across samples.

What it cannot do, and says so on every result
  - No trained face-swap CNN. A subtle identity swap will be missed.
  - Without a vision model it finds EDITS, not fully generated images.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .. import config, schemas
from ..schemas import Direction, Evidence, Modality, Reliability, Signal
from . import forensics_image as fx
from . import vision_llm
from .base import Detector


def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class ImageDetector(Detector):
    modality = Modality.IMAGE

    def _run(self, data: Any, meta: Dict[str, Any]) -> Evidence:
        if not isinstance(data, (bytes, bytearray)):
            return self.abstain("not_bytes", "This input was not an image file.")
        data = bytes(data)

        signals: List[Signal] = []
        reliability = Reliability()
        provenance: Dict[str, Any] = {}
        findings: Dict[str, Any] = {"bytes": len(data)}
        contributions: List[float] = []

        # -- decode --------------------------------------------------------
        try:
            pil = Image.open(__import__("io").BytesIO(data))
            pil.load()
            rgb = pil.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            return self.abstain(
                f"{type(exc).__name__}",
                "This file could not be opened as an image, so nothing was checked.",
                flag="undecodable_image")

        width, height = rgb.size
        findings["resolution"] = f"{width}x{height}"

        # -- 1. provenance beats every measurement -------------------------
        prov = fx.read_provenance(data, pil)
        provenance.update({k: v for k, v in prov.items() if v not in (None, False, 0)})
        if prov["declared_ai"]:
            signals.append(Signal(
                name="declared_ai_provenance", direction=Direction.MANIPULATION,
                human=f"The file itself declares it was AI-generated. {prov['detail']}.",
                value=prov["method"], weight=1.0, where="file metadata"))
            findings["short_circuit"] = "provenance"
            return self.build(self.modality, 0.98, signals, reliability,
                              provenance, findings)

        if prov.get("has_c2pa_container"):
            signals.append(Signal(
                name="c2pa_container_present", direction=Direction.CONTEXT,
                human="The file carries Content Credentials, but they do not claim AI generation.",
                value="c2pa_no_ai_assertion", weight=0.10, where="file metadata"))

        # -- 2. file-level facts -------------------------------------------
        quality = fx.jpeg_quality_from_qtables(pil)
        findings["jpeg_quality"] = quality
        if quality is not None:
            heavy = quality < 60
            if heavy:
                reliability.soft_flags.append("heavy_jpeg_compression")
                reliability.band = max(reliability.band, 0.12)
            signals.append(Signal(
                name="jpeg_compression", direction=Direction.CONTEXT,
                human=(f"Saved at JPEG quality about {quality} out of 100."
                       + (" Heavy compression erases the fine detail these checks "
                          "rely on, so findings are less reliable." if heavy else "")),
                value=quality, weight=0.25 if heavy else 0.10, where="whole file"))

        exif = fx.exif_completeness(pil)
        findings["exif_tags"] = exif["tag_count"]
        if exif["tag_count"] == 0:
            reliability.soft_flags.append("exif_stripped")
            contributions.append(0.03)
            signals.append(Signal(
                name="no_exif", direction=Direction.MANIPULATION,
                human=("No camera metadata. A very weak hint on its own - messaging "
                       "apps strip metadata from genuine photos too."),
                value=0, weight=0.03, where="file metadata"))
        elif exif["has_camera_tags"]:
            signals.append(Signal(
                name="camera_metadata_present", direction=Direction.AUTHENTIC,
                human=f"Carries camera metadata ({exif['make_model']}), which generated images usually lack.",
                value=exif["make_model"], weight=0.15, where="file metadata"))

        hint = fx.generator_dimension_hint((width, height))
        if hint:
            contributions.append(0.08 if exif["tag_count"] == 0 else 0.04)
            signals.append(Signal(
                name="generator_typical_dimensions", direction=Direction.MANIPULATION,
                human=f"{hint}. A weak hint on its own.",
                value=f"{width}x{height}", weight=0.08, where="whole image"))

        # -- 3. localized forensics ----------------------------------------
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        flat = fx.flatness(gray)
        findings["flatness"] = {k: round(v, 2) for k, v in flat.items()}
        too_flat = (flat["mean_gradient"] < config.IMG_MIN_GRADIENT
                    or flat["pixel_std"] < config.IMG_MIN_PIXEL_STD)

        noise_regions: List[dict] = []
        ela_regions: List[dict] = []
        ghost_regions: List[dict] = []
        forensics_ran = False

        if too_flat:
            reliability.soft_flags.append("too_little_detail_for_forensics")
            reliability.band = max(reliability.band, 0.15)
            signals.append(Signal(
                name="insufficient_detail", direction=Direction.LIMITATION,
                human=("This picture is almost entirely flat or dark, so there is not "
                       "enough detail for the compression and noise checks to mean "
                       "anything. No regions are reported."),
                value=findings["flatness"], weight=0.20, where="whole image"))
        else:
            noise_z, n_stats = fx.noise_residual_zmap(bgr, block=config.IMG_NOISE_BLOCK)
            ela_z, e_stats = fx.ela_zmap(rgb, block=config.IMG_ELA_BLOCK)
            ghost, g_stats = fx.jpeg_ghost_map(rgb, block=config.IMG_ELA_BLOCK)
            findings.update({"noise": n_stats, "ela": e_stats, "ghost": g_stats})
            forensics_ran = noise_z is not None

            noise_regions = fx.cluster_outliers(
                noise_z, config.IMG_NOISE_BLOCK, width, height,
                config.IMG_Z_THRESHOLD, config.IMG_NOISE_MIN_BLOCKS,
                config.IMG_MIN_COVERAGE_PCT)
            ela_regions = fx.cluster_outliers(
                ela_z, config.IMG_ELA_BLOCK, width, height,
                config.IMG_Z_THRESHOLD, config.IMG_ELA_MIN_BLOCKS,
                config.IMG_MIN_COVERAGE_PCT)
            ghost_regions = fx.cluster_outliers(
                ghost, config.IMG_ELA_BLOCK, width, height,
                config.IMG_GHOST_DEV_STEPS, config.IMG_ELA_MIN_BLOCKS,
                config.IMG_MIN_COVERAGE_PCT)

        findings["noise_regions"] = noise_regions[:4]
        findings["ela_regions"] = ela_regions[:4]
        findings["ghost_regions"] = ghost_regions[:4]

        if not forensics_ran and not too_flat:
            reliability.soft_flags.append("forensics_unavailable")

        if noise_regions:
            top = noise_regions[0]
            corroborators = []
            if any(_overlap(top["bbox"], r["bbox"]) for r in ela_regions):
                corroborators.append("compression error")
            if any(_overlap(top["bbox"], r["bbox"]) for r in ghost_regions):
                corroborators.append("JPEG history")

            # Size matters as much as peak strength: a large anomalous area is
            # far harder to explain away than one sharp block.
            size_term = min(0.18, 0.06 * top["coverage_pct"])
            strength = min(0.50, 0.18 + 0.06 * (top["peak_z"] - config.IMG_Z_THRESHOLD) + size_term)

            if corroborators:
                strength = min(0.74, strength + 0.22)
                human = (f"A region in {top['region']} has a different noise texture from "
                         f"the rest of the picture, and the {' and '.join(corroborators)} "
                         f"check(s) flag the same area. Independent checks agreeing on one "
                         f"region is what pasted-in content looks like.")
                name, weight = "localized_splice_corroborated", 0.85
            else:
                human = (f"A region in {top['region']} has a different noise texture from the "
                         f"rest of the picture, covering about {top['coverage_pct']}% of the "
                         f"frame. Editing causes this, but so can a crop, a filter or a "
                         f"sticker, so alone it is suggestive rather than conclusive.")
                name, weight = "localized_noise_anomaly", 0.55

            contributions.append(strength)
            signals.append(Signal(
                name=name, direction=Direction.MANIPULATION, human=human,
                value={"peak_z": top["peak_z"], "coverage_pct": top["coverage_pct"],
                       "corroborated_by": corroborators},
                weight=weight, where=f"{top['region']} (box {top['bbox']})"))
        elif forensics_ran:
            signals.append(Signal(
                name="no_localized_editing_traces", direction=Direction.AUTHENTIC,
                human=("Three separate checks - noise texture, compression error and JPEG "
                       "history - found no region that differs from the rest of the picture."),
                value=0, weight=0.20, where="whole image"))

        # -- 4. faces (detected, geometry prepared, deliberately unscored) --
        faces, face_error = fx.detect_faces(bgr)
        findings["faces_detected"] = len(faces)
        findings["face_boxes"] = faces[:8]
        if face_error:
            reliability.soft_flags.append("face_detection_failed")
        if faces:
            findings["face_crop_sizes"] = [fx.crop_face_13x(bgr, b).size for b in faces[:4]]
            reliability.soft_flags.append("faces_present_no_specialist_model")
            reliability.band = max(reliability.band, 0.15)
            where = ", ".join(sorted({fx.coarse_region_name(x + w // 2, y + h // 2, width, height)
                                      for (x, y, w, h) in faces}))
            signals.append(Signal(
                name="faces_present_unscored", direction=Direction.LIMITATION,
                human=(f"{len(faces)} face(s) found, in {where}. This build has no trained "
                       f"face-swap detector, so the faces themselves were not scored - only "
                       f"the picture around them was checked."),
                value=len(faces), weight=0.20,
                where="; ".join(f"face at {b}" for b in faces[:3])))

        # -- 5. vision model ------------------------------------------------
        vlm = vision_llm.assess(rgb)
        findings["vlm"] = {k: vlm[k] for k in ("available", "reason", "samples_ok", "score")}
        if vlm["available"]:
            findings["vlm_consistent"] = vlm["consistent"]
            findings["vlm_discarded"] = vlm["discarded"]
            for e in vlm["consistent"]:
                contributions.append(e["ratio"] * (e["severity"] / 5.0) * 0.8)
                signals.append(Signal(
                    name=f"vlm_{e['artifact']}", direction=Direction.MANIPULATION,
                    human=(f"A vision model saw {e['artifact'].replace('_', ' ')} "
                           f"({e['detail']}) in {e['region']}, and reported it in "
                           f"{e['agreement']} independent looks."),
                    value={"agreement": e["agreement"], "severity": e["severity"]},
                    weight=min(1.0, 0.5 + 0.1 * e["severity"]), where=e["region"]))
            if vlm["discarded"]:
                signals.append(Signal(
                    name="vlm_inconsistent_discarded", direction=Direction.CONTEXT,
                    human=(f"{len(vlm['discarded'])} other observation(s) appeared in only a "
                           f"minority of looks and were discarded as guesswork, not counted."),
                    value=[e["artifact"] for e in vlm["discarded"]], weight=0.05,
                    where="whole image"))
            if not vlm["consistent"]:
                signals.append(Signal(
                    name="vlm_found_nothing", direction=Direction.AUTHENTIC,
                    human=f"A vision model examined the image {vlm['samples_ok']} times and reported no specific artifacts.",
                    value=0, weight=0.25, where="whole image"))
        else:
            reliability.soft_flags.append("no_synthetic_image_detector")
            reliability.band = max(reliability.band, 0.15)
            signals.append(Signal(
                name="synthetic_detector_unavailable", direction=Direction.LIMITATION,
                human=("No vision model is available, so this check looked only for signs of "
                       "editing. A fully AI-generated picture would not be caught by these "
                       f"checks alone. (reason: {vlm['reason']})"),
                value=vlm["reason"], weight=0.30, where="whole image"))

        # -- 6. fuse --------------------------------------------------------
        if not forensics_ran and not vlm["available"]:
            reliability.ood_flags.append("no_detector_available")
            reliability.note = "Neither forensic analysis nor a vision model could run on this file."
            score = config.ABSTAIN_SCORE
        else:
            score = schemas.combine(contributions)

        findings["contributions"] = [round(c, 3) for c in contributions]
        return self.build(self.modality, score, signals, reliability, provenance, findings)
