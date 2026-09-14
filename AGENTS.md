# TrustLens — rules for anyone (or anything) editing this code

## The one rule

> **Every finding shown to a user must correspond to a measurement that was
> actually taken.** If a check could not run, say so. Never fill the gap with
> a plausible-sounding number.

This is not a style preference. The entire value of this project is that its
output can be trusted, and a single invented finding destroys that.

### What this has already caught

An earlier version of this codebase did all of the following. There are
regression tests for each, in `tests/test_trustlens.py::TestNoFabrication`:

- **Scored on the filename.** A file named `deepfake.jpg` was given 0.86
  automatically. Renaming a file must never change a verdict.
- **Faked localization.** A function called `_compute_coarse_gradcam_regions`
  emitted "around the mouth" and "around the eyes" for *any* image containing
  a face, with no model and no gradients behind it.
- **Never opened the file.** The audio and video lanes were `if/else` chains
  over the filename that invented figures like "F0 StdDev: 1.2 Hz" and
  "Mismatch Lag: +140ms" without decoding a single byte.
- **Narrated unmeasured properties.** Signals claimed "natural biological
  textures" and "authentic camera sensor noise distribution" when no such
  measurement existed.
- **Hid a crash.** A missing dependency was caught and turned into a valid
  `uncertain` result, making a broken detector indistinguishable from an
  unsure one.
- **Hard-coded UI prose.** The results panel printed "anomalies were
  identified in facial boundary blending, lighting symmetry, or generative
  network patterns" for every manipulated verdict, regardless of what fired.

A smoothly-varying invented score is **worse** than an obviously broken
constant, because it hides the failure. When a detector is unavailable,
abstain and say so.

---

## Architecture

Every lane implements `Detector._run(data, meta) -> Evidence`. The explainer
and the UI read only `Evidence`, so they never need to know which lane
produced a result, and a new lane costs them no changes.

```
trustlens/config.py       every threshold, with what it was measured against
trustlens/schemas.py      the Evidence contract  ← treat as frozen
trustlens/privacy.py      the ONLY place user content touches disk
trustlens/detectors/      one file per lane, plus the measurement modules
```

**`schemas.py` is a contract.** Changing `Evidence`, `Signal` or `Reliability`
affects all four lanes, the API and the UI. Propose it, do not just do it.

---

## Signal rules

- `direction` is required in spirit: `MANIPULATION`, `AUTHENTIC`, `CONTEXT` or
  `LIMITATION`. The UI groups by it, so a mislabelled signal misleads.
- `human` is one sentence a non-technical person understands. No jargon, no
  model names, no acronyms.
- `value` carries the actual measurement. A signal with no value is narration.
- `where` is a real location — image region, seconds range, frame number, line.

## Reliability: hard vs soft

- `ood_flags` — **hard**. Any one forces `UNCERTAIN`. Use for "this could not
  be analysed": undecodable file, no detector available.
- `soft_flags` — widen the confidence band only. Use for degrading conditions
  such as heavy JPEG compression. Nearly every real-world image is compressed;
  forcing `UNCERTAIN` on all of them would make the lane useless.

---

## Dependencies

`torch` and `transformers` are **not installed and must not be added.**
Published deepfake CNNs fall to 0.43–0.63 AUC on real-world media
(Deepfake-Eval-2024), so gigabytes of download buy very little. If you think
you need one, say so and explain what it gets us.

Consequences, which must stay visible in the UI and the pitch:

- No trained face-swap detector, so subtle identity swaps are missed.
- Without an API key, the image lane finds *edits*, not generated images.
- Audio checks describe **processing**, not synthesis.

---

## Privacy

Nothing the user submits is stored. The only write of user content is
`privacy.scratch_file`, used because OpenCV cannot read a video from memory —
and it deletes in a `finally`. If you add a second write path, you have broken
the guarantee the README makes. Don't.

---

## Before you commit

```bash
python -m unittest discover tests -v      # 32 tests, all must pass
```

Never commit: API keys, `.env`, or any media. `samples/`, `demo/` and all
image, audio and video extensions are gitignored — these are real people's
photographs and they must not leave the machine.
