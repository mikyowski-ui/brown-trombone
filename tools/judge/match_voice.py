"""Match a voice: listen to a recording and write a profile that makes the
physical engine sound like that speaker.

    uv run python match_voice.py recording.wav --out profile.json
    uv run python match_voice.py rec.flac --text "what the speaker said" --transcript
    uv run python match_voice.py rec.wav --analyse      # measurements only
    uv run python match_voice.py --calibrate            # rebuild data/voice_calibration.json

What is measured (``analyse``), on any recording, mono or stereo, any rate
(resampled to 16 kHz):

* pitch: median F0 and the 10th..90th percentile range in semitones, from an
  autocorrelation tracker (Boersma's normalised autocorrelation, four
  candidate peaks per frame, a Viterbi path that charges for octave jumps,
  and a guard that drops frames more than 0.8 octave from the median);
* formant scale: pitch-smoothed LPC over the steady voiced frames gives
  formant candidates. Their long-term distribution over log frequency is
  lined up with the default voice's (saying the same text) by a frequency
  scale factor: that is the formant scale (> 1: shorter tract, smaller
  speaker). The formant spacing ``dF`` of a uniform tube fitted to the median
  F1..F4 is the guide where the alignment is ambiguous (``density_scale``);
* H1*-H2* (the first two harmonics' levels, corrected for the boost the first
  two formants give them, after Iseli and Alwan), the uncorrected H1-H2, and
  spectral tilt (alpha ratio: energy 1..5 kHz over 0..1 kHz);
* harmonics-to-noise ratio (from the autocorrelation peak) and cepstral peak
  prominence for breathiness;
* syllable rate (energy-envelope peaks in voiced speech per second of speech)
  and speech time, pauses of 0.25 s or more excluded.

How the profile is found: ``data/voice_calibration.json`` holds the same
measurements taken on the engine itself at 129 settings (a Latin hypercube
over pitch, tract length, tenseness, expressiveness, rate, posterior gap and
lung pressure), and a quadratic model of each measurement is fitted to them
at load time. The model is inverted for a first guess; then the engine says
the comparison text with the guess, the result is analysed with the same
code, and the guess is corrected (``_next``): pitch, tract length and rate
by ratio, expressiveness by the pitch-range error over its slope, and the
three voice-quality settings by a Broyden step on H1*-H2*, HNR and tilt.
The best round is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, resample_poly

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "data" / "voice_calibration.json"
SR = 16_000
HOP = 0.01

#: Text the engine speaks when it is compared with a recording and no
#: ``--text`` is given (Harvard sentences: varied vowels, few pauses).
DEFAULT_TEXT = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
    "It is easy to tell the depth of a well. These days a chicken leg is a rare dish."
)


def say_bin() -> Path:
    return Path(os.environ.get("SAY_BIN") or HERE / "bin" / "say").resolve()


# --------------------------------------------------------------------------
# audio


def load_audio(path: Path | str, sr: int = SR) -> np.ndarray:
    """Mono float64 audio at ``sr`` Hz; soundfile first, ffmpeg otherwise."""
    import soundfile as sf

    try:
        data, rate = sf.read(str(path), dtype="float64", always_2d=True)
        x = data.mean(axis=1)
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise
        res = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        )
        x, rate = np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64), sr
    if rate != sr:
        g = math.gcd(int(rate), sr)
        x = resample_poly(x, sr // g, int(rate) // g)
    x = x - x.mean() if x.size else x
    if x.size < sr // 2:
        raise ValueError(f"{path}: less than half a second of audio")
    return x


def frames(x: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames of length ``win`` centred every ``hop`` samples; centres."""
    pad = win // 2
    xp = np.pad(x, (pad, pad + win))
    n = 1 + (len(x) - 1) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    return xp[idx], np.arange(n) * hop


# --------------------------------------------------------------------------
# pitch


@dataclass
class Pitch:
    f0: np.ndarray  #: Hz per frame, 0 where unvoiced
    strength: np.ndarray  #: normalised autocorrelation at the chosen lag
    db: np.ndarray  #: frame level, dB


def track_pitch(x: np.ndarray, sr: int = SR, fmin: float = 55.0, fmax: float = 600.0) -> Pitch:
    """Boersma (1993) style autocorrelation pitch with a Viterbi path."""
    hop = int(round(HOP * sr))
    win = int(round(3.0 / fmin * sr))
    fr, _ = frames(x, win, hop)
    fr = fr - fr.mean(axis=1, keepdims=True)
    w = np.hanning(win + 2)[1:-1]
    frw = fr * w
    nfft = 1 << int(math.ceil(math.log2(2 * win)))
    ra = np.fft.irfft(np.abs(np.fft.rfft(frw, nfft)) ** 2, nfft)[:, :win]
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, nfft)) ** 2, nfft)[:win]
    rw = rw / rw[0]
    energy = ra[:, 0].copy()
    r = ra / np.maximum(energy[:, None], 1e-20) / np.maximum(rw[None, :], 1e-3)
    lo, hi = int(math.floor(sr / fmax)), min(int(math.ceil(sr / fmin)), win // 2)
    seg = r[:, lo - 1 : hi + 2]
    mid = seg[:, 1:-1]
    peak = (mid > seg[:, :-2]) & (mid >= seg[:, 2:]) & (mid > 0.2)
    # Parabolic interpolation of every peak.
    a, b, c = seg[:, :-2], mid, seg[:, 2:]
    den = a - 2 * b + c
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    val = np.minimum(b - 0.25 * (a - c) * shift, 1.0)
    lag = np.arange(lo, hi + 1)[None, :] + shift
    freq = sr / lag
    octave_cost = 0.01
    score = np.where(peak, val - octave_cost * np.log2(fmin / freq), -np.inf)
    k = 4
    top = np.argsort(-score, axis=1)[:, :k]
    rows = np.arange(len(r))[:, None]
    cand_s = score[rows, top]
    cand_f = freq[rows, top]
    cand_v = val[rows, top]
    ok = np.isfinite(cand_s)
    # Level and unvoiced candidate.
    rms = np.sqrt(np.mean(fr**2, axis=1)) + 1e-12
    db = 20 * np.log10(rms)
    peak_abs = np.max(np.abs(fr), axis=1)
    global_peak = np.max(np.abs(x)) + 1e-12
    vt, st = 0.45, 0.03
    unv = vt + np.maximum(0.0, 2.0 - (peak_abs / global_peak) / (st / (1.0 + vt)))
    # Viterbi: states 0..k-1 voiced candidates, k unvoiced.
    n = len(r)
    S = np.full((n, k + 1), -np.inf)
    S[:, :k] = np.where(ok, cand_s, -np.inf)
    S[:, k] = unv
    F = np.zeros((n, k + 1))
    F[:, :k] = np.where(ok, cand_f, 0.0)
    jump, vuv = 0.35, 0.14
    acc = S[0].copy()
    back = np.zeros((n, k + 1), dtype=np.int64)
    for t in range(1, n):
        fp, fc = F[t - 1][:, None], F[t][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = np.where(
                (fp > 0) & (fc > 0),
                jump * np.abs(np.log2(np.where(fp > 0, fp, 1) / np.where(fc > 0, fc, 1))),
                np.where((fp > 0) ^ (fc > 0), vuv, 0.0),
            )
        tot = acc[:, None] - cost
        back[t] = np.argmax(tot, axis=0)
        acc = tot[back[t], np.arange(k + 1)] + S[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(acc))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    f0 = F[np.arange(n), path]
    strength = np.zeros(n)
    voiced = path < k
    strength[voiced] = cand_v[np.arange(n)[voiced], path[voiced]]
    # Quiet frames are not voice, whatever the correlation says.
    loud = db > np.percentile(db, 95) - 35
    f0 = np.where(loud & voiced, f0, 0.0)
    # Octave guard: anything far from the median is an error, not a voice.
    v = f0 > 0
    if v.sum() > 10:
        med = np.median(f0[v])
        bad = v & (np.abs(np.log2(np.where(v, f0, med) / med)) > 0.8)
        f0[bad] = 0.0
    # Lone voiced frames are noise.
    v = f0 > 0
    lone = v & ~np.roll(v, 1) & ~np.roll(v, -1)
    f0[lone] = 0.0
    strength[f0 == 0] = 0.0
    return Pitch(f0, strength, db)


# --------------------------------------------------------------------------
# LPC formants


def lpc(frames_w: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation-method LPC for every row (Levinson, vectorised)."""
    nfft = 1 << int(math.ceil(math.log2(2 * frames_w.shape[1])))
    R = np.fft.irfft(np.abs(np.fft.rfft(frames_w, nfft)) ** 2, nfft)[:, : order + 1]
    R[:, 0] = R[:, 0] * (1 + 1e-6) + 1e-12
    n = len(R)
    a = np.zeros((n, order + 1))
    a[:, 0] = 1.0
    err = R[:, 0].copy()
    for i in range(1, order + 1):
        acc = R[:, i] + np.sum(a[:, 1:i] * R[:, i - 1 : 0 : -1], axis=1)
        k = -acc / err
        prev = a[:, 1:i].copy()
        a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        err = err * (1 - k * k)
    return a



def _box_rows(P: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Moving average of every row of ``P`` over ``2 half + 1`` bins."""
    W = int(half.max())
    C = np.cumsum(np.pad(P, ((0, 0), (W + 1, W)), mode="edge"), axis=1)
    k = np.arange(P.shape[1])[None, :] + W + 1
    hi = np.take_along_axis(C, k + half[:, None], axis=1)
    lo = np.take_along_axis(C, k - half[:, None] - 1, axis=1)
    return (hi - lo) / (2 * half[:, None] + 1)


#: LPC orders the formant tracker chooses between (at 16 kHz): fewer poles
#: for a short tract (few formants below 8 kHz), more for a long one.
ORDERS = (20, 22, 26)


def order_for(dF: float) -> int:
    """LPC order suited to a formant spacing (Hz) found with order 22."""
    if dF >= 1200:
        return 20
    if dF < 1000:
        return 26
    return 22


def formant_tracks(x: np.ndarray, f0: np.ndarray, sel: np.ndarray, order: int, sr: int = SR):
    """F1..F5 (Hz, NaN where not found) for the frames ``sel``, and every
    formant candidate of those frames pooled (for ``candidate_density``).

    The power spectrum of each frame is first smoothed over one pitch
    harmonic spacing (a triangle as wide as 2 F0), so the all-pole fit sees
    the envelope and not the harmonics: without this, high voices put poles
    on harmonics. Candidates below 1.15 F0, or broader than 400 Hz, are not
    formants.
    """
    out = np.full((len(sel), 5), np.nan)
    if not sel.any():
        return out, np.zeros(0)
    y = np.append(x[0], x[1:] - 0.95 * x[:-1])
    hop = int(round(HOP * sr))
    win = int(round(0.03 * sr))
    pad = win // 2
    yp = np.pad(y, (pad, pad + win))
    centres = np.flatnonzero(sel) * hop
    fr = yp[centres[:, None] + np.arange(win)[None, :]] * np.hanning(win)
    nfft = 2048
    P = np.abs(np.fft.rfft(fr, nfft)) ** 2
    half = np.maximum(1, np.round(f0[sel] / (sr / nfft) / 2)).astype(int)
    P = _box_rows(_box_rows(P, half), half)
    R = np.fft.irfft(P, nfft)[:, : order + 1]
    n = len(R)
    a = np.zeros((n, order + 1))
    a[:, 0] = 1.0
    err = R[:, 0] * (1 + 1e-9) + 1e-20
    for i in range(1, order + 1):
        acc = R[:, i] + np.sum(a[:, 1:i] * R[:, i - 1 : 0 : -1], axis=1)
        k = -acc / err
        prev = a[:, 1:i].copy()
        a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        err = err * (1 - k * k)
    comp = np.zeros((n, order, order))
    comp[:, 0, :] = -a[:, 1:]
    comp[:, np.arange(1, order), np.arange(order - 1)] = 1.0
    roots = np.linalg.eigvals(comp)
    f = np.angle(roots) * sr / (2 * np.pi)
    bw = -np.log(np.maximum(np.abs(roots), 1e-12)) * sr / np.pi
    above = (roots.imag > 0) & (f > np.maximum(180.0, 1.15 * f0[sel])[:, None])
    pooled = f[above & (f < 7500) & (bw < 500)]
    good = above & (f < sr / 2 - 200) & (bw < 400)
    f = np.sort(np.where(good, f, np.inf), axis=1)[:, :5]
    f[~np.isfinite(f)] = np.nan
    out[sel] = f
    return out, pooled


#: Log-frequency grid of the formant-candidate density: 200 Hz to 7.5 kHz,
#: 1/192 octave.
DENSITY_NPO = 192
DENSITY_F = 200.0 * 2.0 ** (np.arange(int(DENSITY_NPO * math.log2(7500 / 200))) / DENSITY_NPO)


def candidate_density(pooled: np.ndarray, n_frames: int) -> np.ndarray:
    """How often a formant is found at each frequency (``DENSITY_F``),
    per frame, smoothed by 3 %: the long-term distribution of formant
    frequencies, which a longer tract shifts down as a whole."""
    from scipy.ndimage import gaussian_filter1d

    edges = np.append(DENSITY_F, DENSITY_F[-1] * 2 ** (1 / DENSITY_NPO)) * 2 ** (-0.5 / DENSITY_NPO)
    h, _ = np.histogram(pooled, bins=edges)
    return gaussian_filter1d(h.astype(float), 0.03 / math.log(2) * DENSITY_NPO, mode="constant") / max(n_frames, 1)


def density_scale(dens: np.ndarray, ref: np.ndarray, guide: float, lo: float = 600.0, hi: float = 4500.0,
                  smin: float = 0.55, smax: float = 1.8) -> tuple[float, float, str]:  # fmt: skip
    """Formant scale ``s`` that lines up ``dens(s f)`` with ``ref(f)``.

    The correlation over ``lo..hi`` Hz (reference frequencies) is computed
    for every ``s`` in ``smin..smax``. When one alignment stands out (its
    correlation beats every other local peak by 0.12 or more, as it does
    for the engine) that is the scale. Speech whose formant distribution has
    a different shape from the engine's lines up almost equally well at
    several scales about 25 % apart; then ``guide`` (the formant-spacing
    ratio) chooses among the good ones (within 0.12 of the best), and is
    itself the answer when none lies within 15 % of it.

    Returns the scale, the correlation there and how it was chosen
    (``"aligned"``, ``"guided"`` or ``"spacing"``).
    """
    band = (DENSITY_F >= lo) & (DENSITY_F <= hi)
    i0, m = int(np.flatnonzero(band)[0]), int(band.sum())
    rb = ref[band] - ref[band].mean()
    ks = np.arange(int(math.floor(DENSITY_NPO * math.log2(smin))), int(math.ceil(DENSITY_NPO * math.log2(smax))) + 1)
    cs = np.full(len(ks), -1.0)
    for n, k in enumerate(ks):
        j = i0 + k
        if j < 0 or j + m > len(dens):
            continue
        u = dens[j : j + m] - dens[j : j + m].mean()
        den = math.sqrt(float(np.dot(u, u) * np.dot(rb, rb)))
        if den > 0:
            cs[n] = float(np.dot(u, rb)) / den
    peaks = [i for i in range(1, len(cs) - 1) if cs[i] >= cs[i - 1] and cs[i] > cs[i + 1]]
    if not peaks:
        return guide, float("nan"), "spacing"
    peaks.sort(key=lambda i: -cs[i])

    def refined(i: int) -> float:
        c2 = cs[i - 1] - 2 * cs[i] + cs[i + 1]
        d = 0.5 * (cs[i - 1] - cs[i + 1]) / c2 if c2 < 0 else 0.0
        return 2.0 ** ((ks[i] + d) / DENSITY_NPO)

    best = peaks[0]
    if len(peaks) == 1 or cs[best] - cs[peaks[1]] >= 0.12:
        return refined(best), float(cs[best]), "aligned"
    good = [i for i in peaks if cs[i] >= cs[best] - 0.12]
    near = min(good, key=lambda i: abs(math.log(refined(i) / guide)))
    if abs(math.log(refined(near) / guide)) <= math.log(1.15):
        return refined(near), float(cs[near]), "guided"
    return guide, float("nan"), "spacing"


# --------------------------------------------------------------------------
# harmonics, cepstrum, tilt


def _resonance_db(f: np.ndarray, F: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Gain (dB) of an all-pole resonance at F with bandwidth B, 0 dB at DC."""
    hb = B / 2
    num = F**2 + hb**2
    den = np.sqrt(((f - F) ** 2 + hb**2) * ((f + F) ** 2 + hb**2))
    return 20 * np.log10(num / den)


def harmonic_measures(x: np.ndarray, f0: np.ndarray, forms: np.ndarray, sr: int = SR):
    """Per voiced frame: H1, H2 (dB), corrected H1*-H2*, CPP (dB)."""
    sel = f0 > 0
    hop = int(round(HOP * sr))
    win = int(round(0.048 * sr))
    pad = win // 2
    xp = np.pad(x, (pad, pad + win))
    centres = np.flatnonzero(sel) * hop
    fr = xp[centres[:, None] + np.arange(win)[None, :]]
    fr = fr * np.hanning(win)
    nfft = 8192
    spec = np.abs(np.fft.rfft(fr, nfft)) + 1e-12
    db = 20 * np.log10(spec)
    bin_hz = sr / nfft
    f = f0[sel]

    def peak(mult: float) -> tuple[np.ndarray, np.ndarray]:
        lo = np.floor(f * mult * 0.9 / bin_hz).astype(int)
        hi = np.ceil(f * mult * 1.1 / bin_hz).astype(int)
        span = int((hi - lo).max()) + 1
        idx = lo[:, None] + np.arange(span)[None, :]
        idx = np.minimum(idx, db.shape[1] - 1)
        vals = np.take_along_axis(db, idx, axis=1)
        vals = np.where(idx <= hi[:, None], vals, -np.inf)
        j = np.argmax(vals, axis=1)
        return vals[np.arange(len(j)), j], (lo + j) * bin_hz

    h1, fh1 = peak(1.0)
    h2, fh2 = peak(2.0)
    F1, F2 = forms[sel, 0], forms[sel, 1]
    F1 = np.where(np.isfinite(F1), F1, np.nanmedian(F1) if np.isfinite(F1).any() else 500.0)
    F2 = np.where(np.isfinite(F2), F2, np.nanmedian(F2) if np.isfinite(F2).any() else 1500.0)
    B1 = 80.0 + 0.05 * F1
    B2 = 80.0 + 0.05 * F2
    corr1 = _resonance_db(fh1, F1, B1) + _resonance_db(fh1, F2, B2)
    corr2 = _resonance_db(fh2, F1, B1) + _resonance_db(fh2, F2, B2)
    h12c = (h1 - corr1) - (h2 - corr2)
    # Cepstral peak prominence (Hillenbrand): dB cepstrum of the dB spectrum.
    ceps = np.fft.irfft(db, nfft)[:, : nfft // 2]
    q = np.arange(nfft // 2) / sr
    qlo, qhi = int(sr / 600), int(sr / 55)
    cseg = ceps[:, qlo:qhi]
    jpk = np.argmax(cseg, axis=1) + qlo
    pk = ceps[np.arange(len(jpk)), jpk]
    qq = q[qlo:qhi]
    A = np.vstack([qq, np.ones_like(qq)]).T
    coef, *_ = np.linalg.lstsq(A, cseg.T, rcond=None)
    line = coef[0] * q[jpk] + coef[1]
    cpp = pk - line
    # Alpha ratio from the average spectrum of the voiced frames.
    pw = np.mean(spec**2, axis=0)
    fq = np.arange(len(pw)) * bin_hz
    lo_band = pw[(fq >= 50) & (fq < 1000)].sum()
    hi_band = pw[(fq >= 1000) & (fq < 5000)].sum()
    alpha = 10 * np.log10(hi_band / max(lo_band, 1e-20))
    return h1 - h2, h12c, cpp, alpha


# --------------------------------------------------------------------------
# rate


def syllable_rate(x: np.ndarray, f0: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Syllable nuclei per second of speech (pauses excluded); speech seconds."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3000], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    hop = int(round(HOP * sr))
    n = len(f0)
    pad = np.pad(y**2, (0, n * hop + hop))
    e = pad[: n * hop].reshape(n, hop).mean(axis=1)
    # 40 ms smoothing.
    k = np.hanning(7)
    e = np.convolve(e, k / k.sum(), mode="same")
    db = 10 * np.log10(e + 1e-14)
    top = np.percentile(db, 97)
    speech = db > top - 30
    # Pauses: silences of 0.25 s or more; shorter gaps (stop closures) count.
    sil = ~speech
    run_start, pause = None, np.zeros(n, dtype=bool)
    for i in range(n + 1):
        s = sil[i] if i < n else False
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            if i - run_start >= 25:
                pause[run_start:i] = True
            run_start = None
    # Leading and trailing silence are never speech.
    idx = np.flatnonzero(speech)
    if idx.size:
        pause[: idx[0]] = True
        pause[idx[-1] + 1 :] = True
    speech_s = float((~pause).sum() * HOP)
    pk, _ = find_peaks(db, prominence=3.0, distance=8)
    voiced = f0 > 0
    vnear = np.convolve(voiced.astype(float), np.ones(5), mode="same") > 0
    pk = [p for p in pk if db[p] > top - 22 and vnear[p] and not pause[p]]
    return (len(pk) / speech_s if speech_s > 0.3 else float("nan")), speech_s


# --------------------------------------------------------------------------
# the analysis


@dataclass
class Analysis:
    f0_median: float  #: Hz
    f0_p10: float
    f0_p90: float
    range_st: float  #: 10th..90th percentile of F0, semitones
    formants: list[float]  #: median F1..F5, Hz
    dF: float  #: formant spacing fitted to F1..F4, Hz
    order: int  #: LPC order the formants were tracked with
    h1h2: float  #: corrected H1*-H2*, dB
    h1h2_raw: float  #: uncorrected H1-H2, dB
    alpha: float  #: alpha ratio (1..5 kHz over 0..1 kHz), dB
    cpp: float  #: cepstral peak prominence, dB
    hnr: float  #: harmonics-to-noise ratio from the autocorrelation, dB
    syll_rate: float  #: syllables per second of speech
    voiced_s: float
    speech_s: float
    dF_by_order: dict = field(default_factory=dict)
    #: formant scale relative to the reference voice (``ref`` of ``analyse``):
    #: ``scale`` from lining up the formant-candidate densities
    #: (``density_scale``), guided by ``scale_lpc`` (the ratio of the
    #: formant spacings); ``scale_corr`` is the correlation where they line
    #: up and ``scale_how`` how the scale was chosen.
    scale: float = float("nan")
    scale_lpc: float = float("nan")
    scale_corr: float = float("nan")
    scale_how: str = ""
    density: list | None = None  #: kept for references only

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        if d["density"] is None:
            del d["density"]
        return d


def formant_spacing(F: np.ndarray) -> float:
    """dF of a uniform tube fitted to F1..F4 (``F_i = (i - 1/2) dF``)."""
    F = np.asarray(F, dtype=float)[:4]
    w = np.arange(1, 5) - 0.5
    ok = np.isfinite(F)
    if not ok.any():
        return float("nan")
    return float(np.sum(F[ok] * w[ok]) / np.sum(w[ok] ** 2))


def analyse(x: np.ndarray, sr: int = SR, order: int | None = None, all_orders: bool = False,
            ref: dict | None = None, keep_density: bool = False) -> Analysis:  # fmt: skip
    """Measure a recording (16 kHz mono, see ``load_audio``).

    ``order`` fixes the LPC order of the formant tracker; by default it is
    chosen from a first pass at order 22 (``order_for``). A recording and the
    engine's attempts at matching it are measured with the same order.
    ``ref`` (the analysis of the default voice, with ``keep_density`` and
    ``all_orders``) is what the formant scale is measured against.
    """
    p = track_pitch(x, sr)
    v = p.f0 > 0
    if v.sum() < 20:
        raise ValueError("not enough voiced speech to analyse (under 0.2 s)")
    f0 = p.f0[v]
    st = 12 * np.log2(f0 / np.median(f0))
    # Formants and harmonics only where the voice is steady and loud.
    strong = v & (p.strength > 0.6) & (p.db > np.percentile(p.db[v], 25) - 6)
    if strong.sum() < 15:
        strong = v
    by_order: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}

    def at(o: int):
        if o not in by_order:
            F, pooled = formant_tracks(x, p.f0, strong, o, sr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a formant never found
                med = np.nanmedian(F[strong], axis=0)
            by_order[o] = (F, formant_spacing(med), pooled)
        return by_order[o]

    if order is None:
        order = order_for(at(22)[1])
    forms, dF, _ = at(order)
    if all_orders:
        for o in ORDERS:
            at(o)
    dens = candidate_density(at(22)[2], int(strong.sum()))
    scale = scale_lpc = corr = float("nan")
    how = ""
    if ref is not None:
        scale_lpc = dF / ref["dF_by_order"][str(order)]
        scale, corr, how = density_scale(dens, np.asarray(ref["density"]), scale_lpc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Fmed = np.nanmedian(forms[strong], axis=0)
    f0s = np.where(strong, p.f0, 0.0)
    h12, h12c, cpp, alpha = harmonic_measures(x, f0s, forms, sr)
    r = np.clip(p.strength[v], 1e-4, 1 - 1e-4)
    hnr = 10 * np.log10(r / (1 - r))
    rate, speech_s = syllable_rate(x, p.f0, sr)
    return Analysis(
        f0_median=float(np.median(f0)),
        f0_p10=float(np.percentile(f0, 10)),
        f0_p90=float(np.percentile(f0, 90)),
        range_st=float(np.percentile(st, 90) - np.percentile(st, 10)),
        formants=[float(f) for f in Fmed],
        dF=dF,
        order=order,
        h1h2=float(np.nanmedian(h12c)),
        h1h2_raw=float(np.nanmedian(h12)),
        alpha=float(alpha),
        cpp=float(np.median(cpp)),
        hnr=float(np.median(hnr)),
        syll_rate=float(rate),
        voiced_s=float(v.sum() * HOP),
        speech_s=speech_s,
        dF_by_order={str(o): d for o, (_, d, _) in sorted(by_order.items())},
        scale=scale,
        scale_lpc=scale_lpc,
        scale_corr=corr,
        scale_how=how,
        density=[round(float(d), 5) for d in dens] if keep_density else None,
    )


def reference_analysis(x: np.ndarray) -> dict:
    """What a formant scale is measured against: the default voice."""
    return analyse(x, all_orders=True, keep_density=True).as_dict()


# --------------------------------------------------------------------------
# rendering


def render(text: str, profile: dict | None, out: Path) -> Path:
    cmd = [str(say_bin()), "--engine", "physical"]
    if profile:
        prof = out.with_suffix(".json")
        prof.write_text(json.dumps(profile), encoding="utf-8")
        cmd += ["--profile", str(prof)]
    cmd += ["--out", str(out), text]
    res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if res.returncode != 0 or not out.exists():
        tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"say failed ({res.returncode}): {' | '.join(tail)}")
    return out


def render_and_analyse(text: str, profile: dict | None, workdir: Path, name: str, renderer=None,
                       **kw) -> Analysis:  # fmt: skip
    """``renderer(text, profile, out) -> out`` speaks it (``render`` by
    default: the rules)."""
    return analyse(load_audio((renderer or render)(text, profile, Path(workdir) / f"{name}.wav")), **kw)


# --------------------------------------------------------------------------
# the voice parameters the matcher sets, and the profile they make

#: name, default, lo, hi, log-scaled.  ``lo..hi`` is where the matcher may
#: go (inside what the engine accepts); the calibration samples it.
PARAMS = [
    ("pitch", 105.0, 55.0, 400.0, True),
    ("tract_length", 1.0, 0.6, 1.5, True),
    ("tenseness", 0.65, 0.2, 0.95, False),
    ("expressiveness", 0.5, 0.0, 2.0, False),
    ("rate", 1.0, 0.5, 2.0, True),
    ("posterior_gap", 0.25, 0.05, 0.5, False),
    ("lung_pressure", 8.0, 4.5, 14.0, True),
]
NAMES = [p[0] for p in PARAMS]
TUNING_KEYS = {"posterior_gap", "lung_pressure"}
#: Calibration samples a little inside the bounds, where voices are sane.
CAL_RANGE = {
    "pitch": (75.0, 340.0),
    "tract_length": (0.64, 1.4),
    "tenseness": (0.25, 0.92),
    "expressiveness": (0.0, 2.0),
    "rate": (0.65, 1.6),
    "posterior_gap": (0.05, 0.5),
    "lung_pressure": (5.0, 13.0),
}


def _to_u(name: str, value: float) -> float:
    """Parameter value -> -1..1 over its bounds (log for log-scaled)."""
    _, _, lo, hi, lg = PARAMS[NAMES.index(name)]
    if lg:
        value, lo, hi = math.log(value), math.log(lo), math.log(hi)
    return 2 * (value - lo) / (hi - lo) - 1


def _from_u(name: str, u: float) -> float:
    _, _, lo, hi, lg = PARAMS[NAMES.index(name)]
    u = min(1.0, max(-1.0, u))
    if lg:
        return math.exp(math.log(lo) + (u + 1) / 2 * (math.log(hi) - math.log(lo)))
    return lo + (u + 1) / 2 * (hi - lo)


def u_of(values: dict) -> np.ndarray:
    return np.array([_to_u(n, values.get(n, PARAMS[i][1])) for i, n in enumerate(NAMES)])


def values_of(u: np.ndarray) -> dict:
    out = {}
    for i, n in enumerate(NAMES):
        v = _from_u(n, float(u[i]))
        out[n] = round(v, 1 if n == "pitch" else 4)
    return out


def profile_of(values: dict) -> dict:
    """A profile file: only keys of the profile format, inside their ranges."""
    prof: dict = {"engine": "Physical"}
    tuning = {}
    for i, n in enumerate(NAMES):
        if n not in values:
            continue
        _, _, lo, hi, _ = PARAMS[i]
        v = min(hi, max(lo, float(values[n])))
        if n in TUNING_KEYS:
            tuning[n] = v
        else:
            prof[n] = v
    if tuning:
        prof["tuning"] = tuning
    return prof


def values_of_profile(profile: dict) -> dict:
    flat = {k: v for k, v in profile.items() if k in NAMES}
    flat.update({k: v for k, v in profile.get("tuning", {}).items() if k in NAMES})
    return flat


# --------------------------------------------------------------------------
# calibration: the engine measured by the same analysis


def measures(a: Analysis | dict) -> dict:
    """The quantities the matcher compares, from an analysis."""
    d = a.as_dict() if isinstance(a, Analysis) else a
    return {
        "lf0": math.log(d["f0_median"]),
        "lscale": math.log(d["scale"]),
        "h1h2": d["h1h2"],
        "range": d["range_st"],
        "lsyl": math.log(max(d["syll_rate"], 0.3)),
        "hnr": d["hnr"],
        "alpha": d["alpha"],
    }


#: How much each mismatch matters to the matcher (one unit of error).
SIGMA = {"lf0": 0.01, "lscale": 0.006, "h1h2": 0.6, "range": 0.8, "lsyl": 0.04, "hnr": 1.5, "alpha": 1.5}
#: Pull towards the default voice, in -1..1 units (where measurements say
#: little, as for lung pressure and posterior gap, this decides).
PRIOR = {"pitch": 10, "tract_length": 10, "tenseness": 2, "expressiveness": 3, "rate": 5,
         "posterior_gap": 0.7, "lung_pressure": 0.5}  # fmt: skip


def _features(u: np.ndarray) -> np.ndarray:
    u = np.atleast_2d(u)
    cols = [np.ones(len(u))]
    cols += [u[:, i] for i in range(u.shape[1])]
    for i in range(u.shape[1]):
        for j in range(i, u.shape[1]):
            cols.append(u[:, i] * u[:, j])
    return np.stack(cols, axis=1)


class Calibration:
    """The engine's measurements over a grid of settings, and a quadratic
    model of each measurement fitted to them (ridge least squares)."""

    def __init__(self, data: dict):
        self.data = data
        self.samples = data["samples"]
        U = np.array([u_of(s["values"]) for s in self.samples])
        X = _features(U)
        lam = 1e-3 * len(U)
        self.coef: dict[str, np.ndarray] = {}
        self.fit_rms: dict[str, float] = {}
        keys = list(SIGMA)
        targets = {k: [] for k in keys}
        for s in self.samples:
            m = measures(s["analysis"])
            for k in keys:
                targets[k].append(m[k])
        reg = lam * np.eye(X.shape[1])
        reg[0, 0] = 0
        for k, y in targets.items():
            y = np.array(y)
            keep = np.isfinite(y)
            # Twice: drop the samples the analysis got badly wrong (an
            # octave error on a near-whisper, say) and fit again.
            for _ in range(3):
                Xk, yk = X[keep], y[keep]
                c = np.linalg.solve(Xk.T @ Xk + reg, Xk.T @ yk)
                res = np.where(np.isfinite(y), X @ c - np.nan_to_num(y), np.inf)
                mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-9
                keep = np.abs(res) < 4 * mad
            self.coef[k] = c
            self.fit_rms[k] = float(np.sqrt(np.mean(res[keep] ** 2)))

    @classmethod
    def load(cls, path: Path = CALIBRATION) -> Calibration:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `uv run python match_voice.py --calibrate`")
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            import hashlib

            if hashlib.sha1(say_bin().read_bytes()).hexdigest() != data.get("say_sha1"):
                print(f"note: {say_bin()} is not the binary {path.name} was measured with; "
                      "matching still corrects by rendering, but `--calibrate` gives a better start",
                      file=sys.stderr)  # fmt: skip
        except OSError:
            pass
        return cls(data)

    @property
    def reference(self) -> dict:
        """The default voice saying the calibration text, analysed."""
        return self.data["reference"]

    def predict(self, u: np.ndarray) -> dict:
        X = _features(u)[0]
        return {k: float(X @ c) for k, c in self.coef.items()}

    def solve(self, target: dict, offset: dict | None = None, start: np.ndarray | None = None,
              trust: float | None = None, keys: list[str] | None = None,
              free: list[str] | None = None) -> np.ndarray:  # fmt: skip
        """Settings (-1..1 units) whose predicted measurements ``keys``, plus
        ``offset``, come closest to ``target`` (weighted by ``SIGMA``), near
        ``start`` when ``trust`` is given. Only the parameters ``free`` move
        (all by default; the others stay at ``start``)."""
        from scipy.optimize import least_squares

        offset = offset or {}
        keys = keys or list(SIGMA)
        prior_u = u_of({})
        prior_w = np.array([1 / PRIOR[n] for n in NAMES])
        x0 = prior_u.copy() if start is None else np.asarray(start, dtype=float)
        idx = [NAMES.index(n) for n in (free or NAMES)]

        def full(z):
            u = x0.copy()
            u[idx] = z
            return u

        def resid(z):
            u = full(z)
            pred = self.predict(u)
            r = [(pred[k] + offset.get(k, 0.0) - target[k]) / SIGMA[k] for k in keys]
            r += list(((u - prior_u) * prior_w)[idx])
            if trust is not None and start is not None:
                r += list(((u - x0) / trust)[idx])
            return np.array(r)

        best = None
        starts = [x0[idx]] if start is not None else [x0[idx], np.zeros(len(idx))]
        for s in starts:
            res = least_squares(resid, np.clip(s, -0.999, 0.999), bounds=(-1.0, 1.0), method="trf")
            if best is None or res.cost < best.cost:
                best = res
        return full(best.x)

    def slope(self, u: np.ndarray, key: str, name: str, h: float = 0.05) -> float:
        """d(prediction ``key``) / d(parameter ``name`` in its own units) at ``u``."""
        i = NAMES.index(name)
        a, b = u.copy(), u.copy()
        a[i] = max(-1.0, u[i] - h)
        b[i] = min(1.0, u[i] + h)
        va, vb = values_of(a)[name], values_of(b)[name]
        return (self.predict(b)[key] - self.predict(a)[key]) / (vb - va) if vb != va else 0.0


def calibration_samples(n: int, seed: int = 7) -> list[dict]:
    """Latin hypercube over ``CAL_RANGE`` (log for log-scaled), plus the default voice."""
    rng = np.random.default_rng(seed)
    cols = []
    for name in NAMES:
        lo, hi = CAL_RANGE[name]
        lg = PARAMS[NAMES.index(name)][4]
        q = (rng.permutation(n) + rng.random(n)) / n
        if lg:
            cols.append(np.exp(math.log(lo) + q * (math.log(hi) - math.log(lo))))
        else:
            cols.append(lo + q * (hi - lo))
    out = [{n: round(float(c[i]), 4) for n, c in zip(NAMES, cols)} for i in range(n)]
    return [{n: p[1] for n, p in zip(NAMES, PARAMS)}] + out


def calibrate(n: int = 128, jobs: int = 4, path: Path = CALIBRATION) -> dict:
    """Render ``DEFAULT_TEXT`` over a Latin hypercube of settings and save
    what the analysis measures on each."""
    import hashlib

    samples = calibration_samples(n)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="match-cal-") as wd:
        reference = reference_analysis(load_audio(render(DEFAULT_TEXT, None, Path(wd) / "ref.wav")))

        def one(i: int) -> dict:
            vals = samples[i]
            a = render_and_analyse(DEFAULT_TEXT, profile_of(vals), Path(wd), f"c{i}", ref=reference)
            return {"values": vals, "analysis": a.as_dict()}

        done = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for k, row in enumerate(pool.map(one, range(len(samples)))):
                done.append(row)
                if (k + 1) % 16 == 0:
                    print(f"  {k + 1}/{len(samples)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    data = {
        "about": "physical engine measured by match_voice.analyse; rebuild with `match_voice.py --calibrate`",
        "say_sha1": hashlib.sha1(say_bin().read_bytes()).hexdigest(),
        "text": DEFAULT_TEXT,
        "reference": reference,
        "samples": done,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")
    cal = Calibration(data)
    print(f"{len(done)} settings in {time.perf_counter() - t0:.0f} s -> {path}")
    print("model fit (rms): " + ", ".join(f"{k} {v:.3f}" for k, v in cal.fit_rms.items()))
    return data


# --------------------------------------------------------------------------
# matching


#: The gate: matched voice vs recording.
TOL = {"pitch_pct": 5.0, "formant_pct": 3.0, "h1h2_db": 3.0}


def errors(rec: Analysis, syn: Analysis, transcript: bool = False) -> dict:
    """Matched voice minus recording. ``rate_pct`` is how much faster the
    matched voice speaks: by speech time when both say the same words
    (``transcript``), else by syllable rate."""
    if transcript:
        rate = 100 * (rec.speech_s / syn.speech_s - 1)
    else:
        rate = 100 * (syn.syll_rate / rec.syll_rate - 1)
    return {
        "pitch_pct": 100 * (syn.f0_median / rec.f0_median - 1),
        "formant_pct": 100 * (syn.scale / rec.scale - 1),
        "h1h2_db": syn.h1h2 - rec.h1h2,
        "range_st": syn.range_st - rec.range_st,
        "rate_pct": rate,
        "hnr_db": syn.hnr - rec.hnr,
        "alpha_db": syn.alpha - rec.alpha,
    }


def _score(e: dict) -> float:
    gate = max(abs(e["pitch_pct"]) / TOL["pitch_pct"], abs(e["formant_pct"]) / TOL["formant_pct"],
               abs(e["h1h2_db"]) / TOL["h1h2_db"])  # fmt: skip
    rest = abs(e["range_st"]) / 1.0 + abs(e["rate_pct"]) / 10 + abs(e["hnr_db"]) / 3 + abs(e["alpha_db"]) / 3
    return gate + 0.05 * rest


def _good_enough(e: dict) -> bool:
    return (abs(e["pitch_pct"]) < 1.0 and abs(e["formant_pct"]) < 0.8 and abs(e["h1h2_db"]) < 0.6
            and abs(e["range_st"]) < 0.6 and abs(e["rate_pct"]) < 3 and abs(e["hnr_db"]) < 1.0
            and abs(e["alpha_db"]) < 1.0)  # fmt: skip


#: Voice-quality settings, and what they are matched on (together).
QUALITY = ["tenseness", "posterior_gap", "lung_pressure"]
QUALITY_KEYS = ["h1h2", "hnr", "alpha"]


def _quality_step(cal: Calibration, state: dict, u: np.ndarray, rec: Analysis, syn: Analysis) -> np.ndarray:
    """Next voice-quality settings (-1..1 units) by a Broyden step.

    The Jacobian of (H1*-H2*, HNR, tilt) with respect to (tenseness,
    posterior gap, lung pressure) starts as the calibration model's and is
    corrected after every round by what the round actually measured.
    """
    from scipy.optimize import lsq_linear

    qi = [NAMES.index(n) for n in QUALITY]
    sig = np.array([SIGMA[k] for k in QUALITY_KEYS])
    y = np.array([measures(syn)[k] for k in QUALITY_KEYS]) / sig
    yt = np.array([measures(rec)[k] for k in QUALITY_KEYS]) / sig
    uq = u[qi].copy()
    if "J" not in state:
        J = np.zeros((3, 3))
        h = 0.05
        for j, i in enumerate(qi):
            a, b = u.copy(), u.copy()
            a[i], b[i] = max(-1, u[i] - h), min(1, u[i] + h)
            pa, pb = cal.predict(a), cal.predict(b)
            J[:, j] = [(pb[k] - pa[k]) / (b[i] - a[i]) for k in QUALITY_KEYS]
        state["J"] = J / sig[:, None]
    else:
        du, dy = uq - state["uq"], y - state["y"]
        if float(du @ du) > 1e-6:
            state["J"] = state["J"] + np.outer(dy - state["J"] @ du, du) / float(du @ du)
    state["uq"], state["y"] = uq, y
    prior = u_of({})[qi]
    pw = np.array([1 / PRIOR[n] for n in QUALITY])
    trust = 0.35
    A = np.vstack([state["J"], np.diag(pw), np.eye(3) / trust])
    b = np.concatenate([yt - y, -(uq - prior) * pw, np.zeros(3)])
    d = lsq_linear(A, b, bounds=(-1 - uq, 1 - uq)).x
    out = u.copy()
    out[qi] = uq + d
    return out


def _next(cal: Calibration, values: dict, rec: Analysis, syn: Analysis, rounds: list, state: dict) -> np.ndarray:
    """The next settings to try, from what this round measured.

    Pitch, tract length and rate each have one measurement that answers to
    them almost alone, so they are corrected by ratio (formants scale as
    1 / tract length, speech time as 1 / rate). Expressiveness is corrected
    by the pitch range error over its slope (measured between rounds once
    there are two, else the model's). The three voice-quality settings take
    a Broyden step (``_quality_step``).
    """
    v = dict(values)
    v["pitch"] = values["pitch"] * rec.f0_median / syn.f0_median
    v["tract_length"] = values["tract_length"] * syn.scale / rec.scale
    v["rate"] = values["rate"] / (1 + rounds[-1]["errors"]["rate_pct"] / 100)
    slope = None
    if len(rounds) >= 2:
        a, b = rounds[-2], rounds[-1]
        de = b["values"]["expressiveness"] - a["values"]["expressiveness"]
        if abs(de) > 0.05:
            s = (b["errors"]["range_st"] - a["errors"]["range_st"]) / de
            if s > 0.5:
                slope = s
    if slope is None:
        slope = max(1.0, cal.slope(u_of(values), "range", "expressiveness"))
    v["expressiveness"] = values["expressiveness"] - (syn.range_st - rec.range_st) / slope
    q = _quality_step(cal, state, u_of(values), rec, syn)
    u = u_of({n: min(PARAMS[i][3], max(PARAMS[i][2], v[n])) for i, n in enumerate(NAMES)})
    for n in QUALITY:
        u[NAMES.index(n)] = q[NAMES.index(n)]
    return u


def match(recording: Path | str, text: str | None = None, iters: int = 8, cal: Calibration | None = None,
          transcript: bool = False, log=print, renderer=None, audio: np.ndarray | None = None,
          fixed: dict | None = None, steady: bool = False) -> dict:  # fmt: skip
    """Find the profile; returns a report (see ``main`` for the table).

    ``text`` is what the engine says while it is compared with the
    recording; ``transcript`` says it is exactly what the recording says,
    so speaking rate is matched by speech time rather than syllable rate.
    ``renderer(text, profile, out) -> out`` speaks each attempt (``render``,
    the rules, by default; ``prosody.py`` speaks the recording's own
    schedule), ``audio`` is the recording already loaded, and ``fixed``
    holds settings that stay as given (``{"lung_pressure": 8.0}``);
    ``steady`` measures the formant scale as ``steady_scale`` does, on the
    recording and on every attempt.
    """
    t0 = time.perf_counter()
    cal = cal or Calibration.load()
    text = text or DEFAULT_TEXT
    rounds: list = []
    state: dict = {}
    best = None
    with tempfile.TemporaryDirectory(prefix="match-") as wd:
        # The formant scale is measured against the default voice saying the
        # comparison text.
        if text == cal.data["text"] and renderer is None:
            ref = cal.reference
        else:
            ref = reference_analysis(load_audio((renderer or render)(text, None, Path(wd) / "ref.wav")))
        rec = analyse(load_audio(recording) if audio is None else audio, ref=ref)
        if steady:
            rec = steady_scale(rec)
        order = rec.order
        target = measures(rec)
        fixed = fixed or {}
        if fixed:
            u = cal.solve(target, start=u_of(fixed), free=[n for n in NAMES if n not in fixed])
        else:
            u = cal.solve(target)
        for k in range(iters):
            values = values_of(u)
            prof = profile_of(values)
            syn = render_and_analyse(text, prof, Path(wd), f"it{k}", renderer, order=order, ref=ref)
            if steady:
                syn = steady_scale(syn)
            e = errors(rec, syn, transcript)
            rounds.append({"values": values, "errors": e})
            if log:
                log(f"  round {k}: " + "  ".join(f"{n} {values[n]:g}" for n in NAMES) + "  ->  "
                    + "  ".join(f"{n} {v:+.2f}" for n, v in e.items()))  # fmt: skip
            if best is None or _score(e) < _score(best["errors"]):
                best = {"values": values, "profile": prof, "analysis": syn, "errors": e, "round": k}
            if _good_enough(e):
                break
            u = _next(cal, values, rec, syn, rounds, state)
            for n, v in fixed.items():
                u[NAMES.index(n)] = _to_u(n, v)
    return {
        "recording": str(recording),
        "text": text,
        "transcript": transcript,
        "order": order,
        "profile": best["profile"],
        "recording_analysis": rec.as_dict(),
        "matched_analysis": best["analysis"].as_dict(),
        "errors": best["errors"],
        "pass": all(abs(best["errors"][k]) <= TOL[k] for k in TOL),
        "rounds": rounds,
        "best_round": best["round"],
        "seconds": time.perf_counter() - t0,
    }


#: The two estimates of a formant scale (``Analysis.scale`` from the
#: formant-candidate densities, ``scale_lpc`` from the formant spacing)
#: disagree when they differ by more than this, ln units.
SCALE_AGREE = 0.06


def steady_scale(a: Analysis) -> Analysis:
    """The analysis with a formant scale that a few seconds of speech can
    be trusted with: the geometric mean of the density and spacing
    estimates where they agree within ``SCALE_AGREE``, the spacing's
    where they do not. On the 16 LibriSpeech anchors (one reader, 3 to
    10 s each) the density estimate ranged from 0.69 to 1.30, the spacing
    from 0.99 to 1.18; they parted by more than 6 % on six of them."""
    import copy

    out = copy.copy(a)
    s, lpc = a.scale, a.scale_lpc
    if math.isfinite(lpc) and lpc > 0:
        if not (math.isfinite(s) and s > 0) or abs(math.log(s / lpc)) > SCALE_AGREE:
            out.scale = lpc
        else:
            out.scale = math.sqrt(s * lpc)
    return out


def table(rep: dict) -> str:
    r, s = rep["recording_analysis"], rep["matched_analysis"]
    rows = [
        ("median pitch, Hz", r["f0_median"], s["f0_median"], f"{rep['errors']['pitch_pct']:+.1f} %", "5 %"),
        ("pitch range p10-p90, st", r["range_st"], s["range_st"], f"{rep['errors']['range_st']:+.2f} st", ""),
        ("formant scale (x default voice)", r["scale"], s["scale"], f"{rep['errors']['formant_pct']:+.1f} %", "3 %"),
        ("  from formant spacing alone", r["scale_lpc"], s["scale_lpc"], "", ""),
        ("  formant spacing dF, Hz", r["dF"], s["dF"], "", ""),
        ("H1*-H2* (F1/F2-corrected), dB", r["h1h2"], s["h1h2"], f"{rep['errors']['h1h2_db']:+.1f} dB", "3 dB"),
        ("H1-H2 uncorrected, dB", r["h1h2_raw"], s["h1h2_raw"], f"{s['h1h2_raw'] - r['h1h2_raw']:+.1f} dB", ""),
        ("spectral tilt (alpha ratio), dB", r["alpha"], s["alpha"], f"{rep['errors']['alpha_db']:+.1f} dB", ""),
        ("HNR, dB", r["hnr"], s["hnr"], f"{rep['errors']['hnr_db']:+.1f} dB", ""),
        ("CPP, dB", r["cpp"], s["cpp"], f"{s['cpp'] - r['cpp']:+.2f} dB", ""),
        ("syllable rate, /s", r["syll_rate"], s["syll_rate"], f"{100 * (s['syll_rate'] / r['syll_rate'] - 1):+.1f} %", ""),
        ("speech time (pauses out), s", r["speech_s"], s["speech_s"], f"{100 * (s['speech_s'] / r['speech_s'] - 1):+.1f} %", ""),
    ]
    w = max(len(x[0]) for x in rows)
    out = [f"{'':{w}}  {'recording':>10}  {'matched':>10}  {'difference':>11}  {'gate':>5}"]
    for name, a, b, d, g in rows:
        out.append(f"{name:{w}}  {a:10.3f}  {b:10.3f}  {d:>11}  {g:>5}")
    verdict = "PASS" if rep["pass"] else "FAIL"
    out.append(f"gate (pitch 5 %, formant scale 3 %, H1*-H2* 3 dB): {verdict}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?", type=Path, help="speech recording (wav, flac, ...; ffmpeg for others)")
    ap.add_argument("--out", type=Path, default=None, help="profile to write (default: <recording>.profile.json)")
    ap.add_argument("--text", default=None, help="what the engine says when compared (default: four Harvard "
                    "sentences); give the recording's transcript when you have it")  # fmt: skip
    ap.add_argument("--transcript", action="store_true", help="--text is exactly what the recording says "
                    "(speaking rate is then matched by speech time)")  # fmt: skip
    ap.add_argument("--iters", type=int, default=8, help="render / re-analyse rounds at most (default 8)")
    ap.add_argument("--report", type=Path, default=None, help="also write the full report as JSON")
    ap.add_argument("--analyse", action="store_true", help="only print the measurements of the recording")
    ap.add_argument("--quiet", action="store_true", help="no per-round log")
    ap.add_argument("--calibrate", action="store_true", help="rebuild data/voice_calibration.json")
    ap.add_argument("--samples", type=int, default=128, help="settings to render for --calibrate")
    ap.add_argument("--jobs", type=int, default=4, help="parallel renders for --calibrate")
    args = ap.parse_args(argv)
    if args.calibrate:
        calibrate(args.samples, args.jobs)
        return 0
    if args.recording is None:
        ap.error("a recording is needed (or --calibrate)")
    if args.analyse:
        ref = Calibration.load().reference if CALIBRATION.exists() else None
        print(json.dumps(analyse(load_audio(args.recording), ref=ref).as_dict(), indent=2))
        return 0
    if args.transcript and not args.text:
        ap.error("--transcript needs --text")
    rep = match(args.recording, args.text, args.iters, transcript=args.transcript, log=None if args.quiet else print)
    out = args.out or args.recording.with_suffix(".profile.json")
    out.write_text(json.dumps(rep["profile"], indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(table(rep))
    print(f"profile -> {out}  ({rep['seconds']:.1f} s, {len(rep['rounds'])} renders, best round {rep['best_round']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
