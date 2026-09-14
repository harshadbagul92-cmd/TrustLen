"""
Audio lane.

Be clear about what this is. Research-grade audio anti-spoofing (AASIST and
similar) needs torch, and Deepfake-Eval-2024 measured those same systems at
0.43 AUC on real-world audio off the shelf - barely better than a coin flip.
This lane makes no claim to that capability, and says so on every result.

What it does instead is measure how the recording was PROCESSED: its spectral
ceiling, whether its silence is digitally exact, how steady the pitch is, its
noise floor. Those are real properties, and several of them together are
genuinely suggestive of a synthesised or re-encoded track. Individually they
are weak, so the lane abstains often. That is the correct behaviour, not a
shortcoming to be tuned away.
"""
from typing import Any, Dict, List

from .. import config, schemas
from ..schemas import Direction, Evidence, Modality, Reliability, Signal
from . import forensics_audio as fa
from . import media
from .base import Detector


def _stamp(start_s: float, end_s: float) -> str:
    def fmt(t: float) -> str:
        return f"{int(t // 60):d}:{int(t % 60):02d}"
    return f"{fmt(start_s)}-{fmt(end_s)}"


class AudioDetector(Detector):
    modality = Modality.AUDIO

    def _run(self, data: Any, meta: Dict[str, Any]) -> Evidence:
        if not isinstance(data, (bytes, bytearray)):
            return self.abstain("not_bytes", "This input was not an audio file.")
        data = bytes(data)

        if not media.ffmpeg_available():
            return self.abstain(
                "ffmpeg_missing",
                "Audio decoding is unavailable on this machine, so the file was not "
                "analysed. Install imageio-ffmpeg to enable the audio lane.",
                flag="decoder_unavailable")

        samples = media.decode_audio(data)
        if samples is None or samples.size == 0:
            return self.abstain(
                "decode_failed",
                "This file could not be decoded as audio, so nothing was checked.",
                flag="undecodable_audio")

        sr = config.AUDIO_SAMPLE_RATE
        duration = samples.size / sr
        signals: List[Signal] = []
        reliability = Reliability()
        contributions: List[float] = []
        findings: Dict[str, Any] = {"bytes": len(data)}

        if duration < config.AUDIO_MIN_SECONDS:
            return self.abstain(
                f"too_short_{duration:.2f}s",
                f"This clip is only {duration:.1f} seconds long - too short to measure "
                f"anything meaningful.",
                flag="audio_too_short", extra={"duration_s": round(duration, 2)})

        # -- measurements ---------------------------------------------------
        basic = fa.basic_stats(samples, sr)
        spectral = fa.spectral_profile(samples, sr)
        pitch = fa.pitch_track(samples, sr)
        silence = fa.silence_structure(samples, sr)
        floor = fa.noise_floor(samples, sr)
        clipping = fa.clipping_pct(samples)
        digital_silence = fa.digital_silence_pct(samples)

        findings.update({"basic": basic, "spectral": spectral, "pitch": pitch,
                         "silence": silence, "noise_floor": floor,
                         "clipping_pct": clipping,
                         "digital_silence_pct": digital_silence,
                         "container": media.probe(data)})

        whole = _stamp(0, duration)

        # -- 1. spectral ceiling --------------------------------------------
        bw = spectral.get("bandwidth_hz")
        if bw is not None:
            ratio = spectral.get("bandwidth_ratio", 1.0)
            if ratio < 0.55:
                reliability.soft_flags.append("narrowband_audio")
                reliability.band = max(reliability.band, 0.12)
                contributions.append(0.12)
                signals.append(Signal(
                    name="hard_spectral_ceiling", direction=Direction.MANIPULATION,
                    human=(f"All sound stops abruptly above {bw:.0f} Hz, well below the "
                           f"{spectral['nyquist_hz']:.0f} Hz this file could carry. Codecs and "
                           f"voice-synthesis tools cut off at a fixed frequency like this; a "
                           f"natural recording usually fades out gradually. It also means the "
                           f"fine detail other checks rely on has been discarded."),
                    value={"bandwidth_hz": round(bw), "ratio": ratio},
                    weight=0.45, where=whole))
            else:
                signals.append(Signal(
                    name="full_bandwidth", direction=Direction.AUTHENTIC,
                    human=(f"Sound extends to {bw:.0f} Hz with no hard cut-off, which is what "
                           f"an unprocessed wideband recording looks like."),
                    value={"bandwidth_hz": round(bw), "ratio": ratio},
                    weight=0.25, where=whole))

        # -- 2. digital-exact silence ---------------------------------------
        if digital_silence > 1.0:
            contributions.append(min(0.40, 0.15 + digital_silence / 100.0))
            signals.append(Signal(
                name="digital_exact_silence", direction=Direction.MANIPULATION,
                human=(f"{digital_silence:.1f}% of this file is perfectly silent - not quiet, "
                       f"but mathematically zero. A microphone always picks up some room "
                       f"noise, so true digital silence means the audio was generated, "
                       f"gated, or cut and joined rather than recorded in one take."),
                value=digital_silence, weight=0.60, where=whole))
        elif floor.get("noise_floor_rms", 0) > 1e-5:
            signals.append(Signal(
                name="natural_noise_floor", direction=Direction.AUTHENTIC,
                human=("The quiet parts still carry a faint, varying room-noise floor, which "
                       "is what a real microphone recording sounds like."),
                value=floor.get("noise_floor_rms"), weight=0.25, where=whole))

        # -- 3. pitch steadiness --------------------------------------------
        voiced = pitch.get("voiced_frames", 0)
        if voiced >= 25:
            cv = pitch.get("f0_cv")
            if cv is not None and cv < config.AUDIO_PITCH_CV_FLAT:
                contributions.append(0.22)
                signals.append(Signal(
                    name="unusually_steady_pitch", direction=Direction.MANIPULATION,
                    human=(f"The speaking pitch barely moves (variation {cv * 100:.1f}%, around "
                           f"{pitch.get('f0_mean_hz')} Hz). Human voices wobble slightly all the "
                           f"time. A flat monotone delivery can also cause this, so it is a hint "
                           f"rather than a finding."),
                    value={"f0_cv": cv, "f0_mean_hz": pitch.get("f0_mean_hz")},
                    weight=0.45, where=whole))
            else:
                signals.append(Signal(
                    name="natural_pitch_variation", direction=Direction.AUTHENTIC,
                    human=(f"Speaking pitch varies naturally around {pitch.get('f0_mean_hz')} Hz, "
                           f"with the small constant wobble a human voice has."),
                    value={"f0_cv": cv, "f0_mean_hz": pitch.get("f0_mean_hz")},
                    weight=0.25, where=whole))
        else:
            reliability.soft_flags.append("little_voiced_speech")
            signals.append(Signal(
                name="not_enough_speech", direction=Direction.LIMITATION,
                human=(f"Only {voiced} short stretches of voiced speech were found, which is "
                       f"too few to judge whether the voice sounds natural. Music, noise or "
                       f"silence make up most of this file."),
                value=voiced, weight=0.20, where=whole))

        # -- 4. conditions that degrade everything above --------------------
        if clipping > config.AUDIO_CLIPPING_PCT:
            reliability.soft_flags.append("clipped_audio")
            reliability.band = max(reliability.band, 0.10)
            signals.append(Signal(
                name="clipping", direction=Direction.CONTEXT,
                human=(f"{clipping:.1f}% of the audio is distorted by being recorded too loud, "
                       f"which damages the detail these checks look at."),
                value=clipping, weight=0.20, where=whole))

        if silence.get("silence_pct", 0) > 70:
            reliability.soft_flags.append("mostly_silence")
            signals.append(Signal(
                name="mostly_silence", direction=Direction.LIMITATION,
                human=f"{silence['silence_pct']:.0f}% of this file is silence, leaving little to analyse.",
                value=silence.get("silence_pct"), weight=0.20, where=whole))

        # -- 5. the standing limitation, on every audio result --------------
        reliability.soft_flags.append("no_voice_clone_model")
        reliability.band = max(reliability.band, 0.15)
        signals.append(Signal(
            name="no_voice_clone_model", direction=Direction.LIMITATION,
            human=("This build has no trained voice-cloning detector. These checks describe how "
                   "the recording was processed, not whether a voice was cloned. A high-quality "
                   "clone recorded through a real microphone would pass them."),
            value="aasist_not_installed", weight=0.30, where=whole))

        score = schemas.combine(contributions)
        findings["contributions"] = [round(c, 3) for c in contributions]
        return self.build(self.modality, score, signals, reliability, {}, findings)
