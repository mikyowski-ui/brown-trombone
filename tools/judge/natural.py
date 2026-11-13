"""Naturalness: how a system's speech moves, against people saying the same.

The recognizer judges whether the words come through; it does not see that
speech can be intelligible and still sound like a machine.  This compares a
system's renders of the reference texts with recordings of people saying
those texts, on what a listener hears as natural or robotic:

* **spectrum in time** -- log-mel distance to the human saying the same
  sentence, after dynamic time warping, overall and by the class of the
  human's segment (vowel, fricative, stop, nasal, approximant, silence);
* **loudness contour** -- how a phrase starts and ends (rise and fall
  times), how individual segments rise and fall, how often and how long the
  level drops far below the sentence peak inside a sentence (the dead gaps of
  a machine), how steep the level steps are, and the modulation spectrum of
  the envelope (speech peaks near 4 Hz);
* **pitch** -- range, declination, final fall, movement from syllable to
  syllable and inside vowels, micro-variation;
* **fricatives** -- per fricative: rise and fall time of the hiss, its
  duration, level against the neighbouring vowel, spectral centroid, and how
  broad the noise is (flatness, the width of the peak, the peak's
  prominence: broad noise against a narrow whistle);
* **timing** -- segment durations per class and the speaking rate.

Segments come from a forced alignment of the reference phonemes (the CTC
phoneme model of ``judge.py``, Viterbi path, boundaries halfway between the
spikes of neighbouring phonemes): crude to a frame or two, but the same
aligner measures every system and the humans.

References: the 16 LibriSpeech sentences of ``material/human.tsv`` (one male
reader, clean) give the human value of each measure and, sentence against
sentence, its spread ("what a person scores"); a second recording, the
listener's own two sentences, can be given with ``--listener`` (any
format ``soundfile`` reads; it is read, never copied).  There is no second
human saying the same sentences, so the time-aligned distance has no
human-to-human reference: read it against eSpeak and the classic engine.

    uv run python judge.py natural physical espeak classic --listener ~/my-sentences.wav
    uv run python judge.py natural physical --name physical-new      # SAY_BIN=... for another build
    uv run python judge.py natural-report human espeak classic physical physical-new

Each system writes ``out/natural-<name>.json`` (every sentence, every
fricative); ``natural-report`` puts any of them side by side in
``report/natural.md``.  Everything below the model is plain numpy and is
tested on synthetic signals (``tests/test_natural.py``).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SR = 16_000
HOP = 160  # 10 ms frames
WIN = 400  # 25 ms analysis window
FRAME_S = HOP / SR
FINE_HOP = 40  # 2.5 ms: the fine envelope for rise times and steps
FINE_WIN = 160  # 10 ms
FINE_S = FINE_HOP / SR
ALIGN_FRAME_S = 0.02  # the phoneme model's frame
#: Inside a sentence the level counts as a dip below this, dB under the
#: sentence peak, and as dead silence below ``DEAD_DB``.
DIP_DB = -30.0
DEAD_DB = -60.0
#: A phrase has started once the level is this far under the sentence peak
#: (``EDGE_LO``) and is fully up at ``EDGE_HI``.
EDGE_LO, EDGE_HI = -30.0, -6.0
#: A level step steeper than this, dB per 10 ms, counts as a wall.
WALL_DB = 20.0
#: A rise or fall of the envelope this deep, dB, counts as a segment edge.
SEGMENT_EDGE_DB = 15.0
#: Pauses: at least this long below ``PAUSE_DB`` under the peak.
PAUSE_S, PAUSE_DB = 0.15, -35.0
N_MELS, FMIN, FMAX = 40, 100.0, 7600.0
MEL_FLOOR_DB = -60.0

STOPS = {"P", "B", "T", "D", "K", "G", "DX"}
AFFRICATES = {"CH", "JH"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH"}
NASALS = {"M", "N", "NG"}
APPROXIMANTS = {"L", "R", "W", "Y", "HH"}
VOWELS = {"AA", "AE", "AH", "AO", "AX", "EH", "ER", "IH", "IY", "UH", "UW", "AY", "AW", "EY", "OW", "OY"}
CLASSES = ["vowel", "fricative", "stop", "nasal", "approximant", "silence"]
#: Fricative groups reported together, and the band their hiss is measured in.
FRIC_GROUPS = {
    "s z": ({"S", "Z"}, (3000.0, 8000.0)),
    "sh zh": ({"SH", "ZH"}, (1500.0, 8000.0)),
    "f v th dh": ({"F", "V", "TH", "DH"}, (1500.0, 8000.0)),
}


def phone_class(p: str) -> str:
    if p in VOWELS:
        return "vowel"
    if p in FRICATIVES:
        return "fricative"
    if p in STOPS or p in AFFRICATES:
        return "stop"
    if p in NASALS:
        return "nasal"
    return "approximant"


def fric_group(p: str) -> str | None:
    for name, (members, _) in FRIC_GROUPS.items():
        if p in members:
            return name
    return None


# --------------------------------------------------------------------------
# Signal basics
# --------------------------------------------------------------------------


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Frames of ``win`` samples centred every ``hop`` samples."""
    pad = win // 2
    xp = np.pad(np.asarray(x, np.float64), (pad, pad + win))
    n = 1 + (len(x) - 1) // hop if len(x) else 0
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    return xp[idx]


def level_db(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    """Frame level, dB (mean square, Hann window), floored at -150 dB."""
    w = np.hanning(win + 2)[1:-1]
    fr = frames(x, win, hop) * w
    ms = np.mean(fr**2, axis=1) / np.mean(w**2)
    return 10 * np.log10(np.maximum(ms, 1e-15))


def band_level_db(x: np.ndarray, lo: float, hi: float, win: int = FINE_WIN * 2, hop: int = FINE_HOP) -> np.ndarray:
    """Level of the band ``lo..hi`` Hz per frame, dB."""
    w = np.hanning(win + 2)[1:-1]
    spec = np.abs(np.fft.rfft(frames(x, win, hop) * w, axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / SR)
    sel = (f >= lo) & (f <= hi)
    ms = spec[:, sel].sum(axis=1) * 2 / (win * np.sum(w**2))
    return 10 * np.log10(np.maximum(ms, 1e-15))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges where ``mask`` is true."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def crossing(db: np.ndarray, level: float, start: int, step: int) -> float | None:
    """Fractional index where ``db`` first reaches ``level`` walking from
    ``start`` by ``step`` (+1 or -1), interpolated; ``None`` if never."""
    k = start
    while 0 <= k < len(db):
        if db[k] >= level:
            prev = k - step
            if 0 <= prev < len(db) and db[prev] < level and db[k] != db[prev]:
                return prev + step * (level - db[prev]) / (db[k] - db[prev])
            return float(k)
        k += step
    return None


def speech_bounds(db: np.ndarray, peak: float, rel: float = EDGE_LO) -> tuple[int, int]:
    """First and last frame within ``rel`` dB of ``peak`` (half-open)."""
    on = np.flatnonzero(db >= peak + rel)
    if not len(on):
        return 0, 0
    return int(on[0]), int(on[-1]) + 1


def peak_level(db: np.ndarray) -> float:
    """The sentence peak: the loudest 30 ms (three frames averaged)."""
    if len(db) < 3:
        return float(np.max(db)) if len(db) else -150.0
    return float(np.max(np.convolve(db, np.ones(3) / 3, mode="valid")))


# --------------------------------------------------------------------------
# Loudness contour
# --------------------------------------------------------------------------


def phrase_edges(db: np.ndarray, frame_s: float = FRAME_S, lo: float = EDGE_LO, hi: float = EDGE_HI) -> dict:
    """Rise of the sentence from ``lo`` to ``hi`` dB under its peak, and its
    fall from ``hi`` back to ``lo``, seconds."""
    peak = peak_level(db)
    a, b = speech_bounds(db, peak, lo)
    if b <= a:
        return {"rise": None, "fall": None}
    up_lo = crossing(db, peak + lo, 0, 1)
    up_hi = crossing(db, peak + hi, 0, 1)
    dn_lo = crossing(db, peak + lo, len(db) - 1, -1)
    dn_hi = crossing(db, peak + hi, len(db) - 1, -1)
    rise = (up_hi - up_lo) * frame_s if up_lo is not None and up_hi is not None else None
    fall = (dn_lo - dn_hi) * frame_s if dn_lo is not None and dn_hi is not None else None
    return {"rise": rise, "fall": fall}


def dips(db: np.ndarray, frame_s: float = FRAME_S, rel: float = DIP_DB, dead: float = DEAD_DB) -> dict:
    """Stretches inside the sentence (between its first and last frame
    within ``rel`` of the peak) where the level is more than ``rel`` dB
    below the peak: how many per second of speech, how long, and how much
    of the sentence they take; the same for dead silence (``dead``)."""
    peak = peak_level(db)
    a, b = speech_bounds(db, peak, rel)
    inside = db[a:b]
    span = max(len(inside), 1) * frame_s
    low = [(s, e) for s, e in runs(inside < peak + rel)]
    lens = [(e - s) * frame_s for s, e in low]
    dead_runs = [(e - s) * frame_s for s, e in runs(inside < peak + dead)]
    return {
        "per_s": len(lens) / span,
        "mean_ms": 1000 * float(np.mean(lens)) if lens else 0.0,
        "max_ms": 1000 * float(np.max(lens)) if lens else 0.0,
        "share": float(np.sum(lens)) / span,
        "dead_share": float(np.sum(dead_runs)) / span,
        "dead_max_ms": 1000 * float(np.max(dead_runs)) if dead_runs else 0.0,
    }


def steps(x: np.ndarray, rel: float = -40.0) -> dict:
    """Steepness of the level: change over 10 ms of a fine envelope (10 ms
    window, 2.5 ms hop), in speech (either end within ``rel`` dB of the
    peak). ``p95_up``/``p95_down``: 95th percentile of rises and falls, dB
    per 10 ms; ``walls_per_s``: changes steeper than ``WALL_DB``."""
    db = level_db(x, FINE_WIN, FINE_HOP)
    peak = peak_level(level_db(x))
    lag = int(round(0.01 / FINE_S))
    d = db[lag:] - db[:-lag]
    loud = np.maximum(db[lag:], db[:-lag]) >= peak + rel
    a, b = speech_bounds(db, peak, EDGE_LO)
    span = max(b - a, 1) * FINE_S
    d = d[loud]
    if not len(d):
        return {"p95_up": None, "p95_down": None, "walls_per_s": None}
    ups, downs = d[d > 0], -d[d < 0]
    # A wall is one event, however many hops of it are steep.
    steep = np.zeros(len(db) - lag, bool)
    steep[loud] = np.abs(d) > WALL_DB
    return {
        "p95_up": float(np.percentile(ups, 95)) if len(ups) else 0.0,
        "p95_down": float(np.percentile(downs, 95)) if len(downs) else 0.0,
        "walls_per_s": len(runs(steep)) / span,
    }


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def segment_edges(x: np.ndarray, depth: float = SEGMENT_EDGE_DB) -> dict:
    """Rise and fall times of the syllables and segments inside the
    sentence: every climb of the (20 ms smoothed) fine envelope from a
    trough to the next crest by at least ``depth`` dB, timed from 10 % to
    90 % of the climb, and every such descent; medians in ms."""
    db = smooth(level_db(x, FINE_WIN, FINE_HOP), 8)
    peak = peak_level(level_db(x))
    db = np.maximum(db, peak + DEAD_DB)
    ext = turning_points(db, 3.0)
    rises, falls = [], []
    for p, q in zip(ext[:-1], ext[1:]):
        lo_k, hi_k = (p, q) if db[q] > db[p] else (q, p)
        swing = db[hi_k] - db[lo_k]
        if swing < depth or db[hi_k] < peak + EDGE_LO:
            continue
        seg = db[min(p, q) : max(p, q) + 1]
        if db[q] > db[p]:
            t10 = crossing(seg, db[p] + 0.1 * swing, 0, 1)
            t90 = crossing(seg, db[p] + 0.9 * swing, 0, 1)
            if t10 is not None and t90 is not None:
                rises.append((t90 - t10) * FINE_S)
        else:
            t90 = crossing(seg, db[q] + 0.9 * swing, len(seg) - 1, -1)
            t10 = crossing(seg, db[q] + 0.1 * swing, len(seg) - 1, -1)
            if t10 is not None and t90 is not None:
                falls.append((t10 - t90) * FINE_S)
    med = lambda v: 1000 * float(np.median(v)) if v else None  # noqa: E731
    return {"rise_ms": med(rises), "fall_ms": med(falls), "n": len(rises) + len(falls)}


def turning_points(y: np.ndarray, hysteresis: float) -> list[int]:
    """Alternating troughs and crests of ``y``, wiggles smaller than
    ``hysteresis`` ignored; the first and last points are included."""
    pts = [0]
    lo_i = hi_i = 0
    trend = 0
    for k in range(1, len(y)):
        if trend >= 0 and y[k] > y[hi_i]:
            hi_i = k
        if trend <= 0 and y[k] < y[lo_i]:
            lo_i = k
        if trend >= 0 and y[k] < y[hi_i] - hysteresis:
            if hi_i != pts[-1]:
                pts.append(hi_i)
            trend, lo_i = -1, k
        elif trend <= 0 and y[k] > y[lo_i] + hysteresis:
            if lo_i != pts[-1]:
                pts.append(lo_i)
            trend, hi_i = 1, k
    last = hi_i if trend > 0 else lo_i
    if last != pts[-1]:
        pts.append(last)
    if len(y) - 1 != pts[-1]:
        pts.append(len(y) - 1)
    return pts


MOD_BANDS = [(0.5, 2.0), (2.0, 8.0), (8.0, 32.0)]


def modulation(x: np.ndarray) -> dict:
    """Modulation spectrum of the amplitude envelope (10 ms frames, the
    sentence from its first to its last loud frame): the frequency of its
    peak between 1 and 16 Hz and the share of modulation power in
    0.5-2, 2-8 and 8-32 Hz (choppy speech has too much above 8 Hz)."""
    db = level_db(x)
    peak = peak_level(db)
    a, b = speech_bounds(db, peak)
    env = np.sqrt(10 ** (db[a:b] / 10))
    out = {"peak_hz": None, "low": None, "mid": None, "high": None, "spectrum": None}
    if len(env) < 50:
        return out
    env = env / env.mean() - 1.0
    n = 4096
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)), n)) ** 2
    f = np.fft.rfftfreq(n, FRAME_S)
    total = spec[(f >= 0.5) & (f < 32)].sum()
    shares = [spec[(f >= lo) & (f < hi)].sum() / total for lo, hi in MOD_BANDS]
    sm = smooth(spec, max(1, int(round(0.5 / (f[1] - f[0])))))
    sel = (f >= 1.0) & (f <= 16.0)
    out.update(peak_hz=float(f[sel][np.argmax(sm[sel])]), low=shares[0], mid=shares[1], high=shares[2])
    # Octave bands for averaging across sentences.
    edges = [0.5, 1, 2, 4, 8, 16, 32]
    out["spectrum"] = [float(spec[(f >= lo) & (f < hi)].sum() / total) for lo, hi in zip(edges[:-1], edges[1:])]
    return out


# --------------------------------------------------------------------------
# Log-mel and time warping
# --------------------------------------------------------------------------


def mel_filters(n_fft: int = 512, n_mels: int = N_MELS, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, len(f)))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb


def log_mel(x: np.ndarray) -> np.ndarray:
    """Frames x mel bands, dB relative to the loudest frame, floored at
    ``MEL_FLOOR_DB``, then each band's mean over the sentence removed (so
    a difference of overall timbre between two speakers counts less than
    how the spectrum moves)."""
    w = np.hanning(WIN + 2)[1:-1]
    spec = np.abs(np.fft.rfft(frames(x, WIN, HOP) * w, 512, axis=1)) ** 2
    m = 10 * np.log10(np.maximum(spec @ mel_filters().T, 1e-15))
    m = np.maximum(m - m.max(), MEL_FLOOR_DB)
    return m - m.mean(axis=0, keepdims=True)


def dtw(a: np.ndarray, b: np.ndarray, band: float = 0.25) -> tuple[float, list[tuple[int, int]], np.ndarray]:
    """Dynamic time warping of frame sequences ``a`` (n x d) and ``b``
    (m x d) with steps (1,0), (0,1), (1,1), inside a band of ``band`` of
    the longer length around the diagonal. Local cost: RMS difference per
    band (dB). Returns the mean cost along the path, the path, and the
    local costs on it."""
    n, m = len(a), len(b)
    sq = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    cost = np.sqrt(np.maximum(sq, 0.0) / a.shape[1])
    width = max(int(band * max(n, m)), abs(n - m) + 2)
    big = 1e18
    jj = np.arange(m)
    centre = np.arange(n) * (m - 1) / max(n - 1, 1)
    D = np.full((n, m), big)
    for i in range(n):
        c = np.where(np.abs(jj - centre[i]) <= width, cost[i], big)
        if i == 0:
            D[0] = np.cumsum(c)
            continue
        prev = D[i - 1]
        diag = np.concatenate([[big], prev[:-1]])
        best = c + np.minimum(prev, diag)
        # D[i, j] = min(best[j], D[i, j-1] + c[j]): a running minimum.
        C = np.cumsum(c)
        D[i] = C + np.minimum.accumulate(best - C)
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        cands = []
        if i > 0 and j > 0:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            cands.append((D[i - 1, j], i - 1, j))
        if j > 0:
            cands.append((D[i, j - 1], i, j - 1))
        _, i, j = min(cands)
        path.append((i, j))
    path.reverse()
    local = np.array([cost[p, q] for p, q in path])
    return float(local.mean()), path, local


# --------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------


def track_pitch(x: np.ndarray) -> np.ndarray:
    """Hz per 10 ms frame, 0 where unvoiced (``match_voice``'s tracker)."""
    import match_voice

    return match_voice.track_pitch(np.asarray(x, np.float64), SR, 55.0, 450.0).f0


def pitch_measures(f0: np.ndarray, vowel_spans: list[tuple[int, int]] | None = None) -> dict:
    """Pitch over a sentence (``f0`` per 10 ms frame, 0 = unvoiced):

    * ``median_hz``; ``range_st``: 5-95 % range, semitones;
    * ``declination``: slope of a line fitted to the voiced frames, st/s;
    * ``final_fall``: the last 150 ms of voice against the sentence median,
      st (negative falls);
    * ``syllable_step``: mean absolute change between the median pitches of
      successive vowels, st; ``in_vowel``: mean absolute change across a
      vowel (first to last voiced 30 ms), st;
    * ``micro``: frame-to-frame wobble, RMS of the difference between the
      pitch and its 50 ms running median inside voiced stretches, st."""
    v = f0 > 0
    out = dict.fromkeys(["median_hz", "range_st", "declination", "final_fall", "syllable_step", "in_vowel", "micro"])
    if v.sum() < 10:
        return out
    med = float(np.median(f0[v]))
    st = np.where(v, 12 * np.log2(np.where(v, f0, med) / med), np.nan)
    t = np.arange(len(f0)) * FRAME_S
    tv, sv = t[v], st[v]
    slope = float(np.polyfit(tv, sv, 1)[0]) if tv[-1] - tv[0] > 0.3 else None
    tail = sv[-15:]
    micro = []
    for s, e in runs(v):
        if e - s >= 7:
            seg = st[s:e]
            rm = np.array([np.median(seg[max(0, k - 2) : k + 3]) for k in range(len(seg))])
            micro.extend(seg - rm)
    steps_, inside = [], []
    if vowel_spans:
        meds = []
        for a, b in vowel_spans:
            seg = st[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 3:
                meds.append(float(np.median(seg)))
                inside.append(abs(float(np.mean(seg[-3:]) - np.mean(seg[:3]))))
        steps_ = [abs(q - p) for p, q in zip(meds[:-1], meds[1:])]
    out.update(
        median_hz=med,
        range_st=float(np.percentile(sv, 95) - np.percentile(sv, 5)),
        declination=slope,
        final_fall=float(np.mean(tail)),
        syllable_step=float(np.mean(steps_)) if steps_ else None,
        in_vowel=float(np.mean(inside)) if inside else None,
        micro=float(np.sqrt(np.mean(np.square(micro)))) if micro else None,
    )
    return out


# --------------------------------------------------------------------------
# Fricatives
# --------------------------------------------------------------------------


def noise_spectrum(x: np.ndarray, lo: float = 1000.0, hi: float = 8000.0, n_fft: int = 512) -> dict:
    """Shape of a stretch of noise (Welch spectrum, 32 ms frames): centroid
    and spread over ``lo..hi`` Hz, flatness (geometric over arithmetic mean
    power, 1 = white), the peak's frequency, its prominence over the band's
    median (dB) and the width of the band within 6 dB of the peak (Hz; a
    whistle is narrow)."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    w = np.hanning(n_fft + 2)[1:-1]
    hop = n_fft // 4
    fr = frames(x, n_fft, hop)[: max(1, 1 + (len(x) - n_fft) // hop)]
    p = np.mean(np.abs(np.fft.rfft(fr * w, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    sel = (f >= lo) & (f <= hi)
    fs, ps = f[sel], p[sel] + 1e-20
    centroid = float(np.sum(fs * ps) / np.sum(ps))
    spread = float(np.sqrt(np.sum((fs - centroid) ** 2 * ps) / np.sum(ps)))
    flat = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
    db = smooth(10 * np.log10(ps), 7)  # about 200 Hz
    k = int(np.argmax(db))
    above = db >= db[k] - 6.0
    a = k
    while a > 0 and above[a - 1]:
        a -= 1
    b = k
    while b < len(db) - 1 and above[b + 1]:
        b += 1
    return {
        "centroid": centroid,
        "spread": spread,
        "flatness": flat,
        "peak_hz": float(fs[k]),
        "prominence": float(db[k] - np.median(db)),
        "width6": float(fs[b] - fs[a] + (fs[1] - fs[0])),
    }


def hiss_edges(band_db: np.ndarray, centre: int, lo: int, hi: int, share: np.ndarray | None = None,
               frame_s: float = FINE_S) -> dict:  # fmt: skip
    """Around frame ``centre`` of a band envelope, inside ``lo..hi``: the
    hiss is the run within 10 dB of its peak; its rise is the time from
    15 dB under the peak to 3 dB under, before the peak, and its fall the
    same after. ``None`` where the envelope never gets 15 dB down inside
    the window (the hiss runs into a sound as loud). With ``share`` (the
    band's level over the whole signal's, dB) the peak is placed where the
    band dominates most, among frames within 20 dB of the loudest: a weak
    hiss beside a vowel is not mistaken for the vowel's own highs."""
    lo, hi = max(0, lo), min(len(band_db), hi)
    seg = band_db[lo:hi]
    if not len(seg):
        return {"rise_ms": None, "fall_ms": None, "dur_ms": None, "peak": None, "peak_at": None}
    c = int(np.clip(centre - lo, 0, len(seg) - 1))
    top = float(seg.max())
    if share is not None:
        sh = smooth(share[lo:hi], 4)
        k = int(np.argmax(np.where(seg >= top - 20.0, sh, -np.inf)))
    else:
        # The highest point of the run above (max - 10 dB) that holds the
        # centre, or the highest overall.
        k = int(np.argmax(seg))
        for s, e in runs(seg >= top - 10.0):
            if s <= c < e:
                k = s + int(np.argmax(seg[s:e]))
                break
    pk = float(seg[k])
    core = seg >= pk - 10.0
    a = k
    while a > 0 and core[a - 1]:
        a -= 1
    b = k
    while b < len(seg) - 1 and core[b + 1]:
        b += 1

    def time_between(hi_db: float, lo_db: float, step: int) -> float | None:
        # Walk from the peak outwards until the level is under lo_db.
        j = k
        t_hi = t_lo = None
        while 0 <= j < len(seg):
            if t_hi is None and seg[j] < pk + hi_db:
                t_hi = j
            if seg[j] < pk + lo_db:
                t_lo = j
                break
            j += step
        if t_hi is None or t_lo is None:
            return None
        return abs(t_lo - t_hi) * frame_s * 1000

    return {
        "rise_ms": time_between(-3.0, -15.0, -1),
        "fall_ms": time_between(-3.0, -15.0, 1),
        "dur_ms": (b - a + 1) * frame_s * 1000,
        "peak": pk,
        "peak_at": lo + k,
    }


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


@dataclass
class Seg:
    phone: str
    start: float  # seconds
    end: float
    spike: float  # seconds: middle of the frames the model emitted it on

    @property
    def cls(self) -> str:
        return phone_class(self.phone)


def align(pj, audio: np.ndarray, expected: list[str], lp=None) -> list[Seg]:
    """Force-align ``expected`` (scoring symbols) to ``audio``: one segment
    per canonical phoneme (an r-coloured pair the model writes as one token
    is one segment, named by its first phoneme), boundaries halfway between
    the spikes of neighbours; the first starts and the last ends half a
    spike-gap (at most 60 ms) out from their spikes."""
    import judge as J
    import probes as P

    if lp is None:
        lp = pj.log_probs([audio], batch_size=1)[0]
    ref, _ = P.canonical_map(list(expected))
    units = J.units_of(ref, True, pj.unit_ids)
    classes = sorted(set(units))
    small = J.class_matrix(lp, classes, pj.unit_ids, pj.blank).numpy()
    spans = P.ctc_viterbi(small, [classes.index(u) + 1 for u in units])
    f = ALIGN_FRAME_S
    mids = [((s + e + 1) / 2) * f for s, e in spans]
    bounds = [((spans[k][1] + 1 + spans[k + 1][0]) / 2) * f for k in range(len(spans) - 1)]
    segs = []
    for k, u in enumerate(units):
        a = bounds[k - 1] if k > 0 else max(0.0, spans[k][0] * f - 0.03)
        b = bounds[k] if k < len(bounds) else min(len(audio) / SR, (spans[k][1] + 1) * f + 0.03)
        segs.append(Seg(u[0], a, b, mids[k]))
    return segs


# --------------------------------------------------------------------------
# One utterance
# --------------------------------------------------------------------------


def analyse(audio: np.ndarray, segs: list[Seg], syllables: int) -> dict:
    """Every per-sentence measure of one utterance (see the module)."""
    x = np.asarray(audio, np.float64)
    db = level_db(x)
    peak = peak_level(db)
    a, b = speech_bounds(db, peak)
    res: dict = {"phrase": phrase_edges(db), "dips": dips(db), "steps": steps(x), "segments": segment_edges(x),
                 "modulation": modulation(x)}  # fmt: skip
    # Pauses and rate.
    pauses = [(s, e) for s, e in runs(db[a:b] < peak + PAUSE_DB) if (e - s) * FRAME_S >= PAUSE_S]
    speech_s = (b - a) * FRAME_S - sum((e - s) * FRAME_S for s, e in pauses)
    res["rate"] = {"syll_per_s": syllables / speech_s if speech_s > 0 else None, "pauses": len(pauses),
                   "speech_s": speech_s}  # fmt: skip
    # Pitch.
    f0 = track_pitch(x)
    to_frame = lambda t: int(round(t / FRAME_S))  # noqa: E731
    vowel_spans = [(to_frame(s.start), to_frame(s.end)) for s in segs if s.cls == "vowel"]
    res["pitch"] = pitch_measures(f0, vowel_spans)
    # Durations per class (the first and last segment have no left/right
    # neighbour to bound them, so they are left out).
    durs: dict[str, list[float]] = {c: [] for c in CLASSES if c != "silence"}
    for s in segs[1:-1]:
        durs[s.cls].append(1000 * (s.end - s.start))
    res["durations"] = {c: (float(np.mean(v)) if v else None) for c, v in durs.items()}
    res["durations_all"] = {c: v for c, v in durs.items()}
    # Fricatives.
    fr = []
    bands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, s in enumerate(segs):
        group = fric_group(s.phone)
        if group is None:
            continue
        if group not in bands:
            lo_hz, hi_hz = FRIC_GROUPS[group][1]
            band = band_level_db(x, lo_hz, hi_hz)
            bands[group] = (band, band - level_db(x, FINE_WIN * 2, FINE_HOP))
        band, share = bands[group]
        w0 = int((s.start - 0.04) / FINE_S)
        w1 = int((s.end + 0.04) / FINE_S)
        edges = hiss_edges(band, int(s.spike / FINE_S), w0, w1, share)
        if edges["peak_at"] is None:
            continue
        # The hiss itself: the run within 10 dB of its peak.
        c0 = int(edges["peak_at"] * FINE_HOP - edges["dur_ms"] / 2000 * SR)
        c1 = int(edges["peak_at"] * FINE_HOP + edges["dur_ms"] / 2000 * SR)
        core = x[max(0, c0) : max(c0 + 1, c1)]
        spec = noise_spectrum(core)
        # Level against the neighbouring vowel's peak (full band, 25 ms).
        fr_lvl = float(np.max(db[max(0, to_frame(s.start)) : max(to_frame(s.start) + 1, to_frame(s.end))]))
        vlev = []
        for nb in (segs[k - 1] if k > 0 else None, segs[k + 1] if k + 1 < len(segs) else None):
            if nb is not None and nb.cls == "vowel":
                vlev.append(float(np.max(db[to_frame(nb.start) : max(to_frame(nb.start) + 1, to_frame(nb.end))])))
        fr.append({
            "phone": s.phone, "group": group, "t": round(s.start, 3),
            "rise_ms": edges["rise_ms"], "fall_ms": edges["fall_ms"], "dur_ms": edges["dur_ms"],
            "vs_vowel_db": fr_lvl - max(vlev) if vlev else None, **spec,
        })  # fmt: skip
    res["fricatives"] = fr
    return res


def class_distances(path: list[tuple[int, int]], local: np.ndarray, human_db: np.ndarray, segs: list[Seg]) -> dict:
    """Mean warped distance by the class of the human's frame: its aligned
    segment's, or silence where the human is more than 35 dB under the
    peak outside a stop (or outside every segment)."""
    peak = peak_level(human_db)
    labels = np.full(len(human_db), "silence", dtype=object)
    for s in segs:
        labels[int(round(s.start / FRAME_S)) : int(round(s.end / FRAME_S))] = s.cls
    quiet = human_db < peak + PAUSE_DB
    labels[quiet & (labels != "stop")] = "silence"
    acc: dict[str, list[float]] = {c: [] for c in CLASSES}
    for (i, _), d in zip(path, local):
        acc[labels[min(i, len(labels) - 1)]].append(float(d))
    return {c: (float(np.mean(v)) if v else None) for c, v in acc.items()}


# --------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------


@dataclass
class Utt:
    id: str
    text: str
    expected: list[str]
    audio: np.ndarray
    segs: list[Seg] = field(default_factory=list)


LISTENER_TEXT = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."


def reference_items(listener: Path | None):
    """The LibriSpeech items and, when given, the listener's sentences."""
    import espeak_util as E
    import material

    items = material.load(["human"])
    if listener is not None:
        items.append(material.Item("listener_sentences", "listener", LISTENER_TEXT,
                                   tuple(E.phonemes(LISTENER_TEXT)), LISTENER_TEXT))  # fmt: skip
    return items


def system_audio(system: str, items, profile: Path | None, voice: str, listener: Path | None, jobs=None):
    import render

    if system == "human":
        out = []
        for it in items:
            path = listener if it.set == "listener" else render.ANCHOR_DIR / f"{it.id}.flac"
            out.append(render.load_audio(path))
        return out
    rendered = render.render(items, system, profile, voice, jobs)
    return [render.load_audio(r.path) for r in rendered]


def syllables(expected) -> int:
    return sum(1 for p in expected if p in VOWELS)


MEASURES = [
    # (key, label, unit, path in an analysis, aggregate)
    ("phrase.rise", "phrase rise, -30 to -6 dB", "ms", 1000),
    ("phrase.fall", "phrase fall, -6 to -30 dB", "ms", 1000),
    ("segments.rise_ms", "segment rise, 10-90 %", "ms", 1),
    ("segments.fall_ms", "segment fall, 90-10 %", "ms", 1),
    ("dips.per_s", "dips > 30 dB inside a sentence", "/s", 1),
    ("dips.mean_ms", "mean dip length", "ms", 1),
    ("dips.max_ms", "longest dip", "ms", 1),
    ("dips.share", "share of sentence in dips", "%", 100),
    ("dips.dead_share", "share in dead silence (< -60 dB)", "%", 100),
    ("dips.dead_max_ms", "longest dead silence", "ms", 1),
    ("steps.p95_up", "level rise, 95th pct", "dB/10ms", 1),
    ("steps.p95_down", "level fall, 95th pct", "dB/10ms", 1),
    ("steps.walls_per_s", f"walls (> {WALL_DB:.0f} dB in 10 ms)", "/s", 1),
    ("modulation.peak_hz", "envelope modulation peak", "Hz", 1),
    ("modulation.mid", "modulation share 2-8 Hz", "%", 100),
    ("modulation.high", "modulation share 8-32 Hz", "%", 100),
    ("pitch.median_hz", "median pitch", "Hz", 1),
    ("pitch.range_st", "pitch range 5-95 %", "st", 1),
    ("pitch.declination", "declination", "st/s", 1),
    ("pitch.final_fall", "final 150 ms vs median", "st", 1),
    ("pitch.syllable_step", "pitch step vowel to vowel", "st", 1),
    ("pitch.in_vowel", "pitch change across a vowel", "st", 1),
    ("pitch.micro", "micro-variation (vs 50 ms median)", "st", 1),
    ("rate.syll_per_s", "speaking rate", "syll/s", 1),
    ("durations.vowel", "vowel duration", "ms", 1),
    ("durations.fricative", "fricative duration", "ms", 1),
    ("durations.stop", "stop duration", "ms", 1),
    ("durations.nasal", "nasal duration", "ms", 1),
    ("durations.approximant", "approximant duration", "ms", 1),
]
FRIC_MEASURES = [
    ("rise_ms", "hiss rise, -15 to -3 dB", "ms"),
    ("fall_ms", "hiss fall, -3 to -15 dB", "ms"),
    ("dur_ms", "hiss duration (within 10 dB)", "ms"),
    ("vs_vowel_db", "level vs neighbouring vowel", "dB"),
    ("centroid", "spectral centroid 1-8 kHz", "Hz"),
    ("flatness", "spectral flatness 1-8 kHz", ""),
    ("prominence", "peak over band median", "dB"),
    ("width6", "width within 6 dB of peak", "Hz"),
]


#: The measures the tuner's naturalness guard reads (``distance``): every
#: per-sentence measure the person varies on (his dead silence is always 0,
#: so it has no spread and is left to the engine's gates), and the hiss of
#: /s z/ and /f v th dh/ (/sh zh/ comes three times in his sentences).
GUARD_KEYS = [k for k, _, _, _ in MEASURES if not k.startswith("dips.dead")] + [
    f"fric.{g}.{k}"
    for g in ("s z", "f v th dh")
    for k in ("rise_ms", "fall_ms", "dur_ms", "vs_vowel_db", "centroid", "flatness")
]
#: z is clipped here, so one measure far off cannot outweigh the rest.
GUARD_CLIP = 4.0
#: The warped log-mel distance has no person-to-person spread; it enters
#: the guard as dB over this.
GUARD_DTW_DB = 1.0


def distance(summary: dict, human: dict, clip: float | None = GUARD_CLIP) -> tuple[float, dict]:
    """How far a system's naturalness measures are from the person's: the
    mean of |z| (clipped at ``clip``, by default ``GUARD_CLIP``; ``None``
    leaves z as it is) over ``GUARD_KEYS``, with the warped log-mel distance
    (dB / ``GUARD_DTW_DB``) as one more term. 0 is the person's average
    sentence; lower is closer. Returns the distance and every term."""
    terms: dict[str, float] = {}
    for key in GUARD_KEYS:
        h, v = human.get(key) or {}, (summary.get(key) or {}).get("mean")
        if v is None or h.get("mean") is None or not h.get("sd"):
            continue
        z = (v - h["mean"]) / h["sd"]
        terms[key] = float(z if clip is None else np.clip(z, -clip, clip))
    dtw_mean = (summary.get("dtw.mean") or {}).get("mean")
    if dtw_mean is not None:
        terms["dtw.mean"] = float(dtw_mean / GUARD_DTW_DB)
    if not terms:
        return float("nan"), terms
    return float(np.mean([abs(v) for v in terms.values()])), terms


def _get(d: dict, path: str):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k)
    return d


def _stats(vals) -> dict:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "sd": None, "p10": None, "p90": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(len(v))}  # fmt: skip


def summarize(per_item: list[dict]) -> dict:
    """Mean and spread of every measure over the LibriSpeech sentences."""
    rows = [r for r in per_item if r["set"] == "human"]
    out: dict = {}
    for key, _, _, scale in MEASURES:
        st = _stats([_get(r["analysis"], key) for r in rows])
        out[key] = {k: (v * scale if isinstance(v, float) else v) for k, v in st.items()}
    for c in CLASSES:
        out[f"dtw.{c}"] = _stats([_get(r, f"dtw.{c}") for r in rows])
    out["dtw.mean"] = _stats([_get(r, "dtw.mean") for r in rows])
    fr = [f for r in rows for f in r["analysis"]["fricatives"]]
    for g in FRIC_GROUPS:
        for key, _, _ in FRIC_MEASURES:
            out[f"fric.{g}.{key}"] = _stats([f[key] for f in fr if f["group"] == g])
    spec = [r["analysis"]["modulation"]["spectrum"] for r in rows if r["analysis"]["modulation"]["spectrum"]]
    out["modulation.octaves"] = np.mean(spec, axis=0).tolist() if spec else None
    lis = [r for r in per_item if r["set"] == "listener"]
    if lis:
        r = lis[0]
        out["listener"] = {key: (_get(r["analysis"], key) * scale if _get(r["analysis"], key) is not None else None)
                           for key, _, _, scale in MEASURES}  # fmt: skip
        out["listener"]["dtw.mean"] = r.get("dtw", {}).get("mean")
        for g in FRIC_GROUPS:
            for key, _, _ in FRIC_MEASURES:
                out["listener"][f"fric.{g}.{key}"] = _stats(
                    [f[key] for f in r["analysis"]["fricatives"] if f["group"] == g])["mean"]  # fmt: skip
    return out


def human_reference(pj, listener: Path | None) -> dict:
    """The human recordings' audio and alignments, cached for the run."""
    items = reference_items(listener)
    audios = system_audio("human", items, None, "", listener)
    lps = pj.log_probs(audios, batch_size=2)
    utts = []
    for it, au, lp in zip(items, audios, lps):
        u = Utt(it.id, it.text, list(it.expected), au)
        u.segs = align(pj, au, u.expected, lp)
        utts.append(u)
    return {u.id: u for u in utts}


def compare(item_id: str, item_set: str, audio: np.ndarray, segs: list[Seg], syllables_: int,
            human_audio: np.ndarray, human_segs: list[Seg]) -> dict:  # fmt: skip
    """One sentence of a system against the person's: its analysis and the
    warped log-mel distance. Plain numpy, so it can run in a worker
    process (the tuner's guard does)."""
    rec = {"id": item_id, "set": item_set, "analysis": analyse(audio, segs, syllables_)}
    mean, path, local = dtw(log_mel(human_audio), log_mel(audio))
    rec["dtw"] = {"mean": mean, **class_distances(path, local, level_db(human_audio), human_segs)}
    return rec


def run_system(pj, system: str, humans: dict, profile: Path | None, voice: str, listener: Path | None,
               jobs=None) -> list[dict]:  # fmt: skip
    """Analyse one system's renders of every reference text."""
    items = reference_items(listener)
    audios = system_audio(system, items, profile, voice, listener, jobs)
    lps = pj.log_probs(audios, batch_size=2)
    out = []
    for it, au, lp in zip(items, audios, lps):
        segs = humans[it.id].segs if system == "human" else align(pj, au, list(it.expected), lp)
        rec = {"id": it.id, "set": it.set, "analysis": analyse(au, segs, syllables(it.expected))}
        if system != "human":
            h = humans[it.id]
            mean, path, local = dtw(log_mel(h.audio), log_mel(au))
            rec["dtw"] = {"mean": mean, **class_distances(path, local, level_db(h.audio), h.segs)}
        rec["segments"] = [[s.phone, round(s.start, 3), round(s.end, 3)] for s in segs]
        out.append(rec)
    return out


def run(systems: list[str], profile: Path | None, name: str | None, voice: str, listener: Path | None,
        jobs=None) -> None:  # fmt: skip
    import judge as J
    import render

    pj = J.PhonemeJudge()
    try:
        humans = human_reference(pj, listener)
        for system in ["human"] + [s for s in systems if s != "human"]:
            if system == "human":
                label = "human"
            elif name and len([x for x in systems if x != "human"]) == 1:
                label = name
            elif profile and system in render.ENGINE_SYSTEMS:
                label = f"{system}-{Path(profile).stem}"
            else:
                label = system
            prof = profile if system in render.ENGINE_SYSTEMS else None
            per_item = run_system(pj, system, humans, prof, voice, listener, jobs)
            res = {"name": label, "system": system, "profile": str(prof) if prof else None,
                   "summary": summarize(per_item), "items": per_item}  # fmt: skip
            J.OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = J.OUT_DIR / f"natural-{label}.json"
            path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
            print(summary_line(res))
    finally:
        pj.close()


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def summary_line(res: dict) -> str:
    s = res["summary"]
    g = lambda k: s.get(k, {}).get("mean")  # noqa: E731
    f = lambda v, fmt="{:.0f}": "-" if v is None else fmt.format(v)  # noqa: E731
    return (f"[natural] {res['name']}: dtw {f(g('dtw.mean'), '{:.2f}')} dB, rise/fall {f(g('phrase.rise'))}/"
            f"{f(g('phrase.fall'))} ms, dips {f(g('dips.per_s'), '{:.2f}')}/s (dead {f(g('dips.dead_share'), '{:.1f}')} %),"
            f" walls {f(g('steps.walls_per_s'), '{:.2f}')}/s, mod peak {f(g('modulation.peak_hz'), '{:.1f}')} Hz,"
            f" pitch range {f(g('pitch.range_st'), '{:.1f}')} st, final {f(g('pitch.final_fall'), '{:+.1f}')} st,"
            f" s rise {f(g('fric.s z.rise_ms'))} ms")  # fmt: skip


def load(name: str) -> dict | None:
    import judge as J

    p = J.OUT_DIR / f"natural-{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(v, unit: str) -> str:
    if v is None:
        return "-"
    if unit in ("%",):
        return f"{v:.1f}"
    if unit in ("st", "st/s", "dB", "/s", "dB/10ms", "syll/s") or abs(v) < 10:
        return f"{v:+.2f}" if unit in ("st/s",) or (unit == "st" and v < 0) else f"{v:.2f}"
    return f"{v:.0f}"


def report(names: list[str], out: Path) -> None:
    """Side-by-side Markdown of ``out/natural-<name>.json``; the first that
    is ``human`` gives the reference columns."""
    runs_ = [r for r in (load(n) for n in names) if r is not None]
    human = next((r for r in runs_ if r["system"] == "human"), None)
    others = [r for r in runs_ if r["system"] != "human"]
    hs = human["summary"] if human else {}
    lis = hs.get("listener") or {}
    lines = ["# Naturalness", "",
             "Generated by `judge.py natural-report`. Each system says the 16 LibriSpeech texts; values are the mean "
             "over sentences (fricatives: over every fricative of the group). **person** is the LibriSpeech reader "
             "(mean, and in brackets the 10th to 90th percentile across his sentences: what a person scores); "
             "**listener** is the listener's own two sentences (one recording, room noise included) when it was "
             "given. `z` columns: (system - person) / person's spread across sentences.", ""]  # fmt: skip
    head = "| measure | unit | person | listener | " + " | ".join(r["name"] for r in others) + " |"
    lines += [head, "|" + "---|" * (4 + len(others))]

    def row(key: str, label: str, unit: str) -> str:
        h = hs.get(key, {})
        ref = "-" if h.get("mean") is None else f"{_fmt(h['mean'], unit)} [{_fmt(h['p10'], unit)}..{_fmt(h['p90'], unit)}]"
        cells = [label, unit, ref, _fmt(lis.get(key), unit)]
        for r in others:
            v = r["summary"].get(key, {}).get("mean")
            z = ""
            if v is not None and h.get("sd"):
                z = f" (z {(v - h['mean']) / h['sd']:+.1f})"
            cells.append(_fmt(v, unit) + z)
        return "| " + " | ".join(cells) + " |"

    lines.append("| **spectrum in time** | | | | " + " | " * len(others))
    lines.append(row("dtw.mean", "log-mel distance to the person, warped", "dB"))
    for c in CLASSES:
        lines.append(row(f"dtw.{c}", f"... on the person's {c} frames", "dB"))
    section = {"phrase.rise": "loudness", "pitch.median_hz": "pitch", "rate.syll_per_s": "timing"}
    for key, label, unit, _ in MEASURES:
        if key in section:
            lines.append(f"| **{section[key]}** | | | | " + " | " * len(others))
        lines.append(row(key, label, unit))
    for g in FRIC_GROUPS:
        lines.append(f"| **fricatives /{g}/** | | | | " + " | " * len(others))
        for key, label, unit in FRIC_MEASURES:
            lines.append(row(f"fric.{g}.{key}", label, unit))
    if lis:
        lines += ["", "## The listener's two sentences", "",
                  "The same measures on \"The birch canoe slid on the smooth planks. Glue the sheet to the dark blue "
                  "background.\", said by the listener (one recording with room noise and breath, which lengthen "
                  "his phrase edges and fill his dips) and by each system; the log-mel distance is to his recording.",
                  "", "| measure | unit | listener | " + " | ".join(r["name"] for r in others) + " |",
                  "|" + "---|" * (3 + len(others))]  # fmt: skip
        keys = [("dtw.mean", "log-mel distance to the listener, warped", "dB")]
        keys += [(k, label, unit) for k, label, unit, _ in MEASURES]
        keys += [(f"fric.{g}.{k}", f"/{g}/ {label}", unit) for g in FRIC_GROUPS for k, label, unit in FRIC_MEASURES]
        for key, label, unit in keys:
            cells = [label, unit, _fmt(lis.get(key), unit)]
            cells += [_fmt((r["summary"].get("listener") or {}).get(key), unit) for r in others]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The log-mel distance has no person-to-person value: no second person says these sentences. "
              "Read it against eSpeak and the classic engine.", ""]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
