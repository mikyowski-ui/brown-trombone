"""The speaker's voice for a prosody schedule (``prosody.py transfer``):
the baseline (tract length, tenseness, breathiness, pitch) as a profile
the schedule carries, and per phone the voice quality and the vowel
formants the engine is moved towards.

* **baseline** -- ``match_voice.match`` with every attempt spoken by the
  engine *following the schedule* (so the pitch and pace already are the
  speaker's and the rest is matched under them): median pitch, formant
  scale (tract length), H1*-H2* (tenseness), HNR and tilt (posterior gap,
  lung pressure). The profile goes into the schedule as ``"voice"`` with
  ``rate`` and ``expressiveness`` left out (the schedule decides them);
  ``say --match-prosody`` lays it over the voice (``--prosody-voice off``
  does not).
* **per phone** (``phone_voice``) -- on 10 ms frames: pitch
  (``match_voice.track_pitch``), formants (pitch-smoothed LPC,
  ``match_voice.formant_tracks``), H1*-H2* (``match_voice.harmonic_measures``)
  and the tilt (level 1-5 kHz over 0-1 kHz). A phone's voice quality is
  the median over the voiced frames of its middle 60 %; a vowel's F1 and F2
  the median over its middle 40 %, kept only when at least two frames
  agree within ``F_SPREAD`` and lie where such a formant can.
* **tenseness** (``tenseness_step``) -- the engine's H1*-H2* against its own
  mean, phone by phone, moved towards the speaker's against hers: a phone
  whose H1*-H2* stands higher than hers (relative to the means) is made
  tenser by the difference over ``H12_PER_SHARE``. The tilt is measured
  and reported but does not drive it: in this engine a looser voice gets
  brighter (alpha ratio -8.1 dB at tenseness 0.65, -6.6 at 0.5), where a
  breathier speaker's gets darker.
* **formants** (``formant_step``) -- each vowel's F1 and F2, each over its
  voice's formant scale, the speaker's against the engine's: the share
  the vowel's formants are moved by (``PhoneSpan::formants`` in the
  crate), within ``FORMANT_RANGE``; a step that would move a formant by
  more than ``F_JUMP`` is a tracking error and is not taken.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402

SR = 16_000
HOP = 0.01
#: The vowels whose formants are followed: the monophthongs, but not /ɚ/,
#: whose r-colouring (F3) the babble map does not shape.
FORMANT_VOWELS = set(A.VOWELS) - {"ER"}
#: Two formant frames agree when they are within this share of their median.
F_SPREAD = 0.15
#: Where an F1 and an F2 can be, Hz at a formant scale of 1 (a man's).
F1_RANGE, F2_RANGE = (180.0, 1000.0), (500.0, 2900.0)
#: How far a vowel's formants may be moved, a share either way (the
#: crate's ``Tuning::follow_formant_range``).
FORMANT_RANGE = 0.15
#: A step larger than this share is a tracking error, not a vowel.
F_JUMP = 0.3
#: Vowels measured in both renderings it takes to measure the speaker's
#: overall formant bias.
BIAS_TOKENS = 4
#: H1*-H2* of the engine per unit of tenseness share, dB (it falls as the
#: voice tightens): from a sweep of the tenseness over the four Harvard
#: sentences (deep voice: 0.55 -> 0.75 takes it from 8.7 to 6.7 dB; at a
#: woman's pitch and tract 8.6 to 7.2 dB), about 8 dB per unit of
#: tenseness, at 0.65 about 5 dB per unit of share.
H12_PER_SHARE = 5.0
#: Range of a phone's tenseness share (the crate's
#: ``Tuning::follow_tenseness_low`` and ``_high``).
TENSENESS_RANGE = (0.9, 1.15)
#: Most a phone's tenseness share moves in one step of the loop.
TENSENESS_STEP = 0.06
#: The profile keys a schedule's voice keeps: the rest (rate,
#: expressiveness) the schedule itself decides.
VOICE_KEYS = ("engine", "pitch", "tract_length", "tenseness", "tuning")
#: Settings the baseline match leaves as they are: the lung pressure, which
#: the matcher otherwise took down to its floor (4.5 cmH2O) for the
#: breathiness of the LibriSpeech reader, where with the breath of a
#: quiet phone (half of it) the folds no longer vibrate and every other
#: round of the match measured a whisper.
BASELINE_FIXED = {"lung_pressure": 8.0}
#: A baseline match that ends further than this from the recording (formant
#: scale, %; median pitch, %) is not trusted: the schedule gets no voice.
BASELINE_TRUST = {"formant_pct": 8.0, "pitch_pct": 15.0}


def frame_tilt(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Level 1-5 kHz over 0-1 kHz per 10 ms frame (30 ms window), dB."""
    hop, win = int(HOP * SR), int(0.03 * SR)
    xp = np.pad(np.asarray(x, np.float64), (win // 2, win))
    idx = np.arange(n_frames)[:, None] * hop + np.arange(win)[None, :]
    spec = np.abs(np.fft.rfft(xp[idx] * np.hanning(win), 1024, axis=1)) ** 2
    f = np.fft.rfftfreq(1024, 1 / SR)
    lo = spec[:, (f >= 50) & (f < 1000)].sum(axis=1)
    hi = spec[:, (f >= 1000) & (f < 5000)].sum(axis=1)
    return 10 * np.log10(np.maximum(hi, 1e-20) / np.maximum(lo, 1e-20))


def frame_measures(x: np.ndarray, order: int) -> dict:
    """Per 10 ms frame: f0 (0 unvoiced), F1..F5, H1*-H2* and tilt."""
    import match_voice as M
    import prosody as Pr

    x = np.asarray(x, np.float64)
    p = M.track_pitch(x, SR, 55.0, 450.0)
    f0 = Pr.clean_pitch(p.f0)
    n = len(f0)
    v = f0 > 0
    forms, _ = M.formant_tracks(x, f0, v, order, SR)
    h12 = np.full(n, np.nan)
    if v.any():
        _, h12c, _, _ = M.harmonic_measures(x, f0, forms, SR)
        h12[v] = h12c
    return {"f0": f0, "forms": forms, "h12": h12, "tilt": frame_tilt(x, n), "db": p.db}


def phone_voice(frames: dict, start: np.ndarray, end: np.ndarray, phones: list[str], scale: float = 1.0) -> dict:
    """Per phone, from ``frame_measures``: H1*-H2* and tilt (median over the
    voiced frames of its middle 60 %, NaN where none) and how many frames;
    for the vowels of ``FORMANT_VOWELS``, F1 and F2 at the nucleus (median
    of the middle 40 %, NaN where the estimate is not reliable). ``scale``
    is the voice's formant scale (1 for a man's), for the plausible
    ranges."""
    t = np.arange(len(frames["f0"])) * HOP
    n = len(phones)
    out = {k: np.full(n, np.nan) for k in ("h12", "tilt", "f1", "f2")}
    out["frames"] = np.zeros(n, int)
    v = frames["f0"] > 0
    for k, (a, b, p) in enumerate(zip(start, end, phones)):
        d = b - a
        m = (t >= a + 0.2 * d) & (t <= b - 0.2 * d) & v
        if not m.any():
            mid = int(round(0.5 * (a + b) / HOP))
            m = np.zeros(len(t), bool)
            if 0 <= mid < len(t) and v[mid]:
                m[mid] = True
        if m.any():
            h = frames["h12"][m]
            h = h[np.isfinite(h)]
            if len(h):
                out["h12"][k] = float(np.median(h))
                out["tilt"][k] = float(np.median(frames["tilt"][m]))
                out["frames"][k] = len(h)
        if p not in FORMANT_VOWELS:
            continue
        m = (t >= a + 0.3 * d) & (t <= b - 0.3 * d) & v
        if m.sum() < 2:
            mid = 0.5 * (a + b)
            m = (np.abs(t - mid) <= 0.0151) & v
        F = frames["forms"][m][:, :2]
        F = F[np.all(np.isfinite(F), axis=1)]
        if len(F) < 2:
            continue
        med = np.median(F, axis=0)
        agree = np.all(np.abs(F / med - 1) <= F_SPREAD, axis=1)
        if agree.sum() < 2:
            continue
        f1, f2 = np.median(F[agree], axis=0)
        if not (F1_RANGE[0] * scale <= f1 <= F1_RANGE[1] * scale and F2_RANGE[0] * scale <= f2 <= F2_RANGE[1] * scale
                and f2 > 1.3 * f1):  # fmt: skip
            continue
        out["f1"][k], out["f2"][k] = float(f1), float(f2)
    return out


def deviations(h: np.ndarray, frames: np.ndarray, phones: list[str], smooth: bool = True) -> np.ndarray:
    """Each sonorant's H1*-H2* (or tilt) against the frame-weighted mean of
    the utterance's sonorants, NaN for the rest; with ``smooth``, averaged
    (by frames) with the sonorants next to it, which a phone's few frames
    need: the measure moves by several dB from frame to frame."""
    import prosody as Pr

    son = np.array([p in Pr.SONORANTS for p in phones]) & np.isfinite(h) & (frames > 0)
    out = np.full(len(h), np.nan)
    if son.sum() < 2:
        return out
    mean = float(np.sum(h[son] * frames[son]) / frames[son].sum())
    out[son] = h[son] - mean
    if not smooth:
        return out
    w = np.where(son, frames, 0).astype(float)
    d = np.where(son, out, 0.0)
    sm = out.copy()
    for k in np.flatnonzero(son):
        lo, hi = max(0, k - 1), min(len(h), k + 2)
        sm[k] = float(np.sum(d[lo:hi] * w[lo:hi]) / np.sum(w[lo:hi]))
    return sm


def contour_fit(engine: np.ndarray, speaker: np.ndarray) -> dict:
    """How well the engine's deviations follow the speaker's (phones with
    both): correlation and RMS difference, dB."""
    ok = np.isfinite(engine) & np.isfinite(speaker)
    if ok.sum() < 3:
        return {"r": None, "rms_db": None, "n": int(ok.sum())}
    r = float(np.corrcoef(engine[ok], speaker[ok])[0, 1])
    return {"r": r, "rms_db": float(np.sqrt(np.mean((engine[ok] - speaker[ok]) ** 2))), "n": int(ok.sum())}


def tenseness_step(sched: dict, eng: dict, spk: dict) -> tuple[dict, dict]:
    """One step of moving each sonorant's tenseness share so that its
    H1*-H2* against the engine's mean stands where the speaker's does
    against hers; phones with few frames move less. Returns the new
    schedule and the fit before the step (``contour_fit``)."""
    phones = [p["phone"] for p in sched["phones"]]
    de = deviations(eng["h12"], eng["frames"], phones)
    ds = deviations(spk["h12"], spk["frames"], phones)
    new = json.loads(json.dumps(sched))
    for k, p in enumerate(new["phones"]):
        share = p.get("tenseness", 1.0)
        if np.isfinite(de[k]) and np.isfinite(ds[k]):
            n = min(eng["frames"][k], spk["frames"][k])
            step = (de[k] - ds[k]) / H12_PER_SHARE * n / (n + 2)
            share += float(np.clip(step, -TENSENESS_STEP, TENSENESS_STEP))
        p["tenseness"] = round(float(np.clip(share, *TENSENESS_RANGE)), 4)
    return new, contour_fit(de, ds)


def formant_bias(eng: dict, spk: dict, s_eng: float, s_spk: float) -> tuple[float, float]:
    """How much higher the speaker's F1 and F2 stand than the engine's over
    all the vowels measured in both, each over its voice's formant scale:
    the median ratio (1 with fewer than ``BIAS_TOKENS`` vowels)."""
    out = []
    for f in ("f1", "f2"):
        ok = np.isfinite(eng[f]) & np.isfinite(spk[f])
        r = (spk[f][ok] / s_spk) / (eng[f][ok] / s_eng)
        r = r[np.abs(r - 1) <= F_JUMP]
        out.append(float(np.median(r)) if len(r) >= BIAS_TOKENS else 1.0)
    return out[0], out[1]


def formant_error(eng: dict, spk: dict, s_eng: float, s_spk: float) -> dict:
    """The engine's vowel formants against the speaker's, each over its
    voice's formant scale: mean and median |error| per cent, F1 and F2,
    over the vowels measured in both; and the same with the speaker's
    overall bias (``formant_bias``) taken out (``*_pattern_pct``)."""
    out = {}
    bias = formant_bias(eng, spk, s_eng, s_spk)
    for f, b in zip(("f1", "f2"), bias):
        ok = np.isfinite(eng[f]) & np.isfinite(spk[f])
        e = (eng[f][ok] / s_eng) / (spk[f][ok] / s_spk) - 1
        p = (eng[f][ok] / s_eng) / (spk[f][ok] / s_spk / b) - 1
        out[f + "_pct"] = float(100 * np.mean(np.abs(e))) if ok.any() else None
        out[f + "_median_pct"] = float(100 * np.median(np.abs(e))) if ok.any() else None
        out[f + "_bias_pct"] = float(100 * np.mean(e)) if ok.any() else None
        out[f + "_pattern_pct"] = float(100 * np.mean(np.abs(p))) if ok.any() else None
        out["n"] = int(ok.sum())
    return out


def formant_step(sched: dict, eng: dict, spk: dict, s_eng: float, s_spk: float) -> dict:
    """One step of moving each vowel's formant shares towards the speaker's
    formants (each over its voice's formant scale), vowel by vowel: the
    median step over the tokens of a vowel measured in both renderings is
    taken by all its tokens (one token's nucleus is too few frames to
    trust, and the engine's vowel is one shape however often it comes).
    The speaker's overall bias (``formant_bias``) is taken out first: what
    moves is how her vowels stand against each other, not how much higher
    her first formant stands on the whole, which is as much the
    recording, her voice source and the tracker as her vowels (the Polly
    sample's F1 stood 12 % under the engine's on every vowel, after both
    were divided by their formant scales, and every vowel went to the
    limit; the recognizer heard more of its vowels wrong)."""
    new = json.loads(json.dumps(sched))
    phones = [p["phone"] for p in new["phones"]]
    bias = formant_bias(eng, spk, s_eng, s_spk)
    for v in sorted(set(phones) & FORMANT_VOWELS):
        toks = [k for k, p in enumerate(phones) if p == v]
        steps = []
        for f in ("f1", "f2"):
            s = [(spk[f][k] / s_spk) / (eng[f][k] / s_eng) for k in toks
                 if np.isfinite(eng[f][k]) and np.isfinite(spk[f][k])]  # fmt: skip
            s = [x for x in s if abs(x - 1) <= F_JUMP]
            steps.append(float(np.median(s)) / bias[len(steps)] if s else 1.0)
        for k in toks:
            p = new["phones"][k]
            r = list(p.get("formants") or [1.0, 1.0])
            r = [float(np.clip(r[j] * steps[j], 1 - FORMANT_RANGE, 1 + FORMANT_RANGE)) for j in range(2)]
            if r != [1.0, 1.0]:
                p["formants"] = [round(r[0], 4), round(r[1], 4)]
    return new


# --------------------------------------------------------------------------
# The baseline


def voice_of(profile: dict) -> dict:
    """The part of a ``match_voice`` profile a schedule carries."""
    return {k: v for k, v in profile.items() if k in VOICE_KEYS}


def baseline(sched: dict, audio: np.ndarray, voice: str = "deep", iters: int = 6, log=None) -> dict:
    """Match the engine's voice to the recording while it follows the
    schedule (see the module). Returns ``match_voice.match``'s report with
    ``voice``: the profile for the schedule."""
    import match_voice as M
    import prosody as Pr

    plain = {k: v for k, v in sched.items() if k != "voice"}

    def renderer(text: str, profile: dict | None, out: Path) -> Path:
        s = dict(plain)
        if profile:
            s["voice"] = voice_of(profile)
        Pr.render(s, out, voice=voice)
        return out

    rep = M.match("recording", sched["text"], iters=iters, transcript=True, log=log, renderer=renderer,
                  audio=np.asarray(audio, np.float64), fixed=BASELINE_FIXED, steady=True)  # fmt: skip
    e = rep["errors"]
    rep["trusted"] = all(abs(e[k]) <= v for k, v in BASELINE_TRUST.items())
    rep["voice"] = voice_of(rep["profile"]) if rep["trusted"] else None
    return rep


def five(rec: dict, syn: dict) -> dict:
    """The five things the baseline matches, engine against the speaker:
    pitch (median, %, and range, st), formant scale (%), H1*-H2* (dB),
    HNR (dB) and pace (speech time, %)."""
    return {
        "pitch_pct": 100 * (syn["f0_median"] / rec["f0_median"] - 1),
        "range_st": syn["range_st"] - rec["range_st"],
        "formant_scale_pct": 100 * (syn["scale"] / rec["scale"] - 1),
        "h1h2_db": syn["h1h2"] - rec["h1h2"],
        "hnr_db": syn["hnr"] - rec["hnr"],
        "pace_pct": 100 * (rec["speech_s"] / syn["speech_s"] - 1),
    }


def analyse_like(x: np.ndarray, ref: dict, order: int | None) -> dict:
    """``match_voice.analyse`` against ``ref`` at LPC ``order`` (chosen
    from the recording when ``None``), with the steadier formant scale of
    ``match_voice.steady_scale``, as a dict."""
    import match_voice as M

    return M.steady_scale(M.analyse(np.asarray(x, np.float64), order=order, ref=ref)).as_dict()


def reference_for(sched: dict, voice: str = "deep") -> dict:
    """What a formant scale is measured against: the default voice
    following the schedule (``match_voice.reference_analysis``)."""
    import match_voice as M
    import prosody as Pr
    import render as R

    plain = {k: v for k, v in sched.items() if k != "voice"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ref.wav"
        Pr.render(plain, out, voice=voice)
        return M.reference_analysis(R.load_audio(out).astype(np.float64))
