"""
Acoustic measurements on decoded PCM.

Every function returns a number taken from the samples. None of it is a
trained model, and none of it is a deepfake detector in the research sense -
AASIST-grade systems need torch and, per Deepfake-Eval-2024, still fall to
0.43 AUC on real-world audio anyway.

What these measurements honestly support is narrower: they describe how the
recording was PROCESSED. A hard spectral ceiling, digital-exact silence and
an unnaturally steady pitch are all real, checkable properties that synthesis
and re-encoding pipelines tend to leave behind. They are evidence about
processing, not proof of synthesis, and audio.py weights them accordingly.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Split into overlapping frames (n_frames, frame)."""
    if x.size < frame:
        return np.empty((0, frame), dtype=np.float32)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def basic_stats(x: np.ndarray, sr: int) -> Dict[str, float]:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return {
        "duration_s": round(x.size / sr, 2),
        "sample_rate": sr,
        "peak": round(peak, 4),
        "rms": round(float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0, 5),
        "dc_offset": round(float(np.mean(x)) if x.size else 0.0, 6),
    }


def clipping_pct(x: np.ndarray, threshold: float = 0.999) -> float:
    """Share of samples pinned at full scale. High values mean a damaged or
    over-driven recording, which degrades every other measurement."""
    if x.size == 0:
        return 0.0
    return round(100.0 * float(np.mean(np.abs(x) >= threshold)), 3)


def digital_silence_pct(x: np.ndarray) -> float:
    """
    Share of samples that are EXACTLY zero.

    A microphone always captures a noise floor, so a genuine recording has
    almost no exact zeros. Long runs of true digital silence indicate the
    audio was generated, gated, or spliced rather than recorded in one take.
    """
    if x.size == 0:
        return 0.0
    return round(100.0 * float(np.mean(x == 0.0)), 3)


def spectral_profile(x: np.ndarray, sr: int, frame: int = 1024,
                     hop: int = 512) -> Dict[str, float]:
    """
    Average spectrum, and the frequency above which there is effectively
    nothing left.

    `bandwidth_hz` is the highest frequency still within 55 dB of the spectral
    peak. Natural wideband speech decays gradually; codecs and many synthesis
    pipelines cut hard at a fixed frequency, so a ceiling well below Nyquist
    is a real, measurable fingerprint of processing.
    """
    out: Dict[str, float] = {}
    fr = _frames(x, frame, hop)
    if fr.shape[0] == 0:
        return out

    window = np.hanning(frame).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr * window, axis=1))
    mean_spec = spec.mean(axis=0)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)

    power = mean_spec ** 2 + 1e-12
    db = 10.0 * np.log10(power / power.max())
    above = np.where(db > -55.0)[0]
    out["bandwidth_hz"] = float(freqs[above[-1]]) if above.size else 0.0
    out["nyquist_hz"] = float(sr / 2)
    out["bandwidth_ratio"] = round(out["bandwidth_hz"] / out["nyquist_hz"], 3)

    # Spectral flatness: 1.0 is white noise, near 0 is strongly tonal.
    gmean = float(np.exp(np.mean(np.log(power))))
    amean = float(np.mean(power))
    out["spectral_flatness"] = round(gmean / amean, 5) if amean > 0 else 0.0

    centroid = float(np.sum(freqs * mean_spec) / (np.sum(mean_spec) + 1e-12))
    out["spectral_centroid_hz"] = round(centroid, 1)

    # How sharp is the cut at the ceiling? A cliff is more telling than a roll-off.
    if above.size and above[-1] + 4 < db.size:
        edge = above[-1]
        drop = float(db[edge] - db[min(db.size - 1, edge + 4)])
        out["ceiling_drop_db_per_4bins"] = round(drop, 1)
    return out


def pitch_track(x: np.ndarray, sr: int, fmin: float = 60.0,
                fmax: float = 400.0) -> Dict[str, float]:
    """
    Frame-wise F0 by autocorrelation.

    Returns voiced-frame count and the coefficient of variation of F0. Human
    speech carries constant micro-variation in pitch (jitter); some synthetic
    voices are noticeably steadier. This is a weak cue on its own - a person
    speaking in a flat monotone also scores low - so it only ever contributes
    alongside something else.
    """
    out: Dict[str, float] = {"voiced_frames": 0, "total_frames": 0}
    frame = int(0.040 * sr)
    hop = int(0.010 * sr)
    fr = _frames(x, frame, hop)
    out["total_frames"] = int(fr.shape[0])
    if fr.shape[0] == 0:
        return out

    min_lag, max_lag = int(sr / fmax), int(sr / fmin)
    if max_lag >= frame:
        max_lag = frame - 1
    if min_lag >= max_lag:
        return out

    f0s: List[float] = []
    energies = np.sqrt(np.mean(fr ** 2, axis=1))
    # Only look at frames with real energy; silence has no pitch to measure.
    threshold = max(1e-4, float(np.median(energies)) * 0.6)

    for i in range(fr.shape[0]):
        if energies[i] < threshold:
            continue
        seg = fr[i] - fr[i].mean()
        norm = float(np.dot(seg, seg))
        if norm <= 1e-9:
            continue
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        window = corr[min_lag:max_lag]
        if window.size == 0:
            continue
        lag = int(np.argmax(window)) + min_lag
        # Require a genuinely periodic frame, not just the largest bump.
        if corr[lag] / norm < 0.35:
            continue
        f0s.append(sr / float(lag))

    out["voiced_frames"] = len(f0s)
    if len(f0s) >= 12:
        arr = np.array(f0s, dtype=np.float64)
        mean = float(arr.mean())
        out["f0_mean_hz"] = round(mean, 1)
        out["f0_std_hz"] = round(float(arr.std()), 2)
        out["f0_cv"] = round(float(arr.std()) / mean, 4) if mean > 0 else 0.0
        # Median absolute frame-to-frame change: natural speech jitters.
        out["f0_jitter_hz"] = round(float(np.median(np.abs(np.diff(arr)))), 2)
    return out


def silence_structure(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Pause count and length. Continuous speech with no pauses at all is
    unusual for a genuine recording of a person."""
    out: Dict[str, float] = {}
    frame, hop = int(0.02 * sr), int(0.02 * sr)
    fr = _frames(x, frame, hop)
    if fr.shape[0] == 0:
        return out
    energy = np.sqrt(np.mean(fr ** 2, axis=1))
    # Threshold relative to the LOUD parts, not the quiet ones. Keying off a
    # low percentile made every frame of a steady-amplitude recording count as
    # silent, because that percentile sits just under the signal itself.
    loud = float(np.percentile(energy, 95))
    floor = max(1e-5, loud * 0.08)
    quiet = energy < floor

    out["silence_pct"] = round(100.0 * float(quiet.mean()), 2)
    runs, current = [], 0
    for q in quiet:
        if q:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    pauses = [r for r in runs if r >= 10]          # >= 200 ms
    out["pause_count"] = len(pauses)
    out["longest_pause_s"] = round(max(runs) * hop / sr, 2) if runs else 0.0
    return out


def noise_floor(x: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Level and steadiness of the quietest parts.

    A real room has a floor that breathes. A perfectly constant floor, or none
    at all, suggests the audio was synthesised or heavily processed.
    """
    out: Dict[str, float] = {}
    frame = int(0.02 * sr)
    fr = _frames(x, frame, frame)
    if fr.shape[0] < 5:
        return out
    energy = np.sqrt(np.mean(fr ** 2, axis=1))
    quietest = np.sort(energy)[: max(3, len(energy) // 10)]
    mean = float(quietest.mean())
    out["noise_floor_rms"] = round(mean, 6)
    out["noise_floor_cv"] = round(float(quietest.std()) / mean, 4) if mean > 1e-9 else 0.0
    return out
