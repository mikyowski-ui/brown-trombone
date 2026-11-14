"""Prosody transfer: a person's timing, pitch and level as a schedule the
physical engine speaks with (``say --match-prosody``).

From a recording and its text:

* **phones** -- the engine's own phonemes for the text (the first line of
  ``bin/say``, read by ``arpabet.parse_engine_line``), force-aligned to the
  recording with the judge's wav2vec2 phoneme model (``natural.align``'s
  Viterbi CTC path; boundaries placed between the spikes of neighbours where
  the frame posteriors cross, not halfway), so that every engine phone gets
  a stretch of the recording. A pair the model writes as one token
  (``AA R`` = ``ɑːɹ``) is shared between its two phones. Every phone gets at
  least ``FLOOR``, taken from the neighbours that have the most to spare.
  Silences of ``PAUSE_MIN`` or more between two phones (longer beside a
  stop, whose closure is silent too) become pauses.
* **pitch** -- ``match_voice.track_pitch`` every 10 ms (autocorrelation with
  a Viterbi path and an octave guard), octave jumps against the running
  median folded back, voiced islands under 30 ms dropped, then a curve
  through the unvoiced stretches (linear in semitones) so the engine always
  has a target.
* **level** -- RMS in dB every 10 ms (``natural.level_db``); each phone's
  mean against the mean of the utterance's vowels.

All of it goes into one JSON schedule (``analyse``). ``render`` speaks it
with ``bin/say --match-prosody`` and, with ``--level``, maps the level onto
the lungs: the engine's rendering is measured phone by phone and the breath
of each sonorant moved (within ``LEVEL_RANGE`` dB) towards the person's
level relative to the utterance, twice. ``check`` force-aligns the rendering
again and reports the phone durations against the schedule (mean absolute
error, ms) and the pitch against the schedule's (RMS semitones on frames
voiced in both). ``roundtrip`` does all of it on the engine's own rendering
of a text, where the true phone times are known.

    uv run python prosody.py analyse speaker.wav --text "what they said" --out s.json
    uv run python prosody.py render s.json --out engine.wav --level [--pitch voice]
    uv run python prosody.py check s.json engine.wav
    uv run python prosody.py roundtrip "The birch canoe slid on the smooth planks."
    uv run python prosody.py demo --out-dir DIR [--listener my.wav]   # the owner's set

Model-free parts are tested in ``tests/test_prosody.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402

SR = 16_000
FRAME = 0.02  #: the phoneme model's frame, s
HOP = 0.01  #: pitch and level frames, s
FINE_HOP = 0.005  #: frames of the spectral change that places the boundaries, s
#: Shortest phone, s.
FLOOR = 0.025
#: A silence at least this long between two phones is a pause (s); beside a
#: stop or affricate, whose closure is silent too, ``PAUSE_MIN_STOP``.
PAUSE_MIN, PAUSE_MIN_STOP = 0.12, 0.20
#: Silence: this far under the utterance's peak level, dB; over a noise
#: floor, within ``NOISE_MARGIN`` of it, but never closer to the peak than
#: ``SILENCE_CAP``.
SILENCE_DB, NOISE_MARGIN, SILENCE_CAP = -35.0, 8.0, -20.0
#: A quieter stretch, unvoiced and this far under the peak (dB), at least
#: ``QUIET_MIN`` long between two sonorants, is a pause too (a breath, a
#: hesitation over the noise of the room).
QUIET_DB, QUIET_MIN = -24.0, 0.08
#: A pause shorter than this after the phones around it took their share is
#: not one, s.
PAUSE_KEEP = 0.05
#: How much of a silence before a phone it keeps (its closure, or its
#: onset), s.
LEAD_IN = {"voiceless_stop": 0.05, "voiced_stop": 0.03, "other": 0.01}
#: How much of a silence after a phone it keeps (its decay), s.
TAIL = 0.02
#: Boundary offset found on the engine's own renderings, where the true
#: phone times are known (``roundtrip``): the model's boundaries come this
#: much late, s.
BOUNDARY_OFFSET = 0.0
#: Where the phoneme model's spike sits in a phone, as a share of its
#: duration from its start (median over the engine's renderings of the
#: ``sentence_tune`` texts, true phone times known).
SPIKE_SHARE = {"vowel": 0.29, "diphthong": 0.16, "voiceless_stop": 0.70, "voiced_stop": 0.74, "affricate": 0.53,
               "voiceless_fricative": 0.44, "voiced_fricative": 0.57, "nasal": 0.59, "liquid": 0.49, "glide": 0.67,
               "aspirate": 0.62}  # fmt: skip
#: Width of the prior around the rules' boundary: this share of the time
#: between the two spikes, at least ``PRIOR_SIGMA_MIN`` seconds.
PRIOR_SIGMA, PRIOR_SIGMA_MIN = 0.3, 0.015
#: Weights a unit the model writes as one token is shared out by.
SHARE_WEIGHT = {"vowel": 1.5, "other": 1.0}
#: The speaker's pitch counts as moving fast from this many semitones a
#: second (``pitch_error`` reports those frames apart).
FAST_PITCH = 40.0
#: How far the level may be moved from the rules', dB either way.
LEVEL_RANGE = 4.0
#: Level of the engine's voice per doubling of its breath share, dB
#: (measured: ``python prosody.py slope``).
LEVEL_PER_DOUBLING = 9.0

FOUR_SENTENCES = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
    "She sells sea shells by the sea shore. The juice of lemons makes fine punch."
)

VOICELESS_STOPS = {"P", "T", "K", "CH"}
VOICED_STOPS = {"B", "D", "G", "JH"}
STOP_LIKE = VOICELESS_STOPS | VOICED_STOPS
SONORANTS = set(A.VOWELS + A.DIPHTHONGS + A.NASALS + A.APPROXIMANTS)
VOCALIC = set(A.VOWELS + A.DIPHTHONGS)


def say_bin() -> Path:
    import render

    return render.say_bin()


# --------------------------------------------------------------------------
# The engine's phonemes
# --------------------------------------------------------------------------


def engine_phones(text: str, voice: str = "deep") -> list[str]:
    """The phonemes the engine says for ``text`` (first line of ``say``)."""
    return engine_rule(text, voice)[0]


def engine_rule(text: str, voice: str = "deep") -> tuple[list[str], np.ndarray]:
    """The phonemes the engine says for ``text`` (first line of ``say``)
    and how long its rules make each, seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "t.json"
        res = subprocess.run(
            [str(say_bin()), "--engine", "physical", "--voice", voice, "--out", str(Path(tmp) / "x.wav"),
             "--phone-times", str(times), text],
            capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        spans = np.array(json.loads(times.read_text())["spans"], float)
    phones = A.parse_engine_line(res.stdout.splitlines()[0])
    if len(phones) != len(spans):
        raise RuntimeError(f"say's phonemes ({len(phones)}) and phone times ({len(spans)}) disagree for {text!r}")
    return phones, spans[:, 1] - spans[:, 0]


def units_with_owners(phones: list[str], merge: bool = True) -> tuple[list[tuple[str, ...]], list[list[int]]]:
    """The CTC units the judge aligns for ``phones`` (canonical form, with
    ``merge`` the pairs the model writes as one token merged, as
    ``judge.units_of``) and, per unit, the indices of the phones it stands
    for."""
    import probes as P

    canon, src = P.canonical_map(list(phones))
    units: list[tuple[str, ...]] = []
    owners: list[list[int]] = []
    i = 0
    while i < len(canon):
        pair = tuple(canon[i : i + 2])
        if merge and len(pair) == 2 and pair in A.MERGED_TOKENS:
            units.append(pair)
            owners.append(sorted(set(src[i] + src[i + 1])))
            i += 2
        else:
            units.append((canon[i],))
            owners.append(list(src[i]))
            i += 1
    return units, owners


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def change_curve(x: np.ndarray, hop: float = FINE_HOP) -> np.ndarray:
    """How fast the spectrum moves, every ``hop`` seconds: mean absolute
    difference over 40 mel bands (dB, floored 60 dB under the loudest
    frame) between the frames 10 ms before and 10 ms after."""
    import natural as N

    h = int(round(hop * SR))
    w = np.hanning(N.WIN + 2)[1:-1]
    spec = np.abs(np.fft.rfft(N.frames(np.asarray(x, np.float64), N.WIN, h) * w, 512, axis=1)) ** 2
    m = 10 * np.log10(np.maximum(spec @ N.mel_filters().T, 1e-15))
    m = np.maximum(m - m.max(), N.MEL_FLOOR_DB)
    d = int(round(0.01 / hop))
    c = np.zeros(len(m))
    if len(m) > 2 * d:
        c[d:-d] = np.mean(np.abs(m[2 * d :] - m[: -2 * d]), axis=1)
    return c


def phone_kind(p: str) -> str:
    """The class a phone's spike position is kept for (``SPIKE_SHARE``)."""
    if p in A.VOWELS:
        return "vowel"
    if p in A.DIPHTHONGS:
        return "diphthong"
    if p in ("P", "T", "K"):
        return "voiceless_stop"
    if p in ("B", "D", "G"):
        return "voiced_stop"
    if p in A.AFFRICATES:
        return "affricate"
    if p in ("F", "TH", "S", "SH"):
        return "voiceless_fricative"
    if p in A.FRICATIVES:
        return "voiced_fricative"
    if p in A.NASALS:
        return "nasal"
    if p in ("L", "R"):
        return "liquid"
    if p in ("W", "Y"):
        return "glide"
    return "aspirate"


def place_bounds(spans: list[tuple[int, int]], change: np.ndarray, units: list[tuple[str, ...]],
                 prior: list[float] | None = None, hop: float = FINE_HOP) -> list[float]:  # fmt: skip
    """Boundary between each pair of neighbouring units, seconds: where the
    spectrum moves fastest (``change``) between their spikes, weighted by a
    Gaussian around ``prior`` (where the rules' durations would put it; see
    ``boundary_prior``) when given. The model's spikes are a frame or two
    wide and sit anywhere in their phone (a vowel's near its start, a
    stop's at its release), so the halfway point between them can be off by
    half a vowel; the acoustic change is where one sound turns into the
    next. The search runs from the middle of the first spike (the end of it
    after a stop, whose release is its spike) to the start of the second
    (20 ms before it into a stop, whose closure comes before its release).
    Between two stops (``sheet to``, ``dark blue``) the closures run into
    one silence with nothing to mark where one gives way to the other: the
    prior, or halfway between the spikes."""
    out = []
    for k in range(len(spans) - 1):
        first, second = units[k][-1], units[k + 1][0]
        a_mid = (spans[k][0] + spans[k][1] + 1) / 2 * FRAME
        b_mid = (spans[k + 1][0] + spans[k + 1][1] + 1) / 2 * FRAME
        a = (spans[k][1] + 1) * FRAME if first in STOP_LIKE else a_mid
        b0 = spans[k + 1][0] * FRAME
        b = b0 - 0.02 if second in STOP_LIKE else b0
        guess = prior[k] if prior is not None else 0.5 * (a_mid + b_mid)
        if first in STOP_LIKE and second in STOP_LIKE:
            out.append(float(np.clip(guess, a_mid, b_mid)))
            continue
        lo, hi = int(math.ceil(a / hop)), min(len(change), int(math.floor(b / hop)) + 1)
        if hi - lo < 2:
            out.append(float(np.clip(guess, a_mid, b_mid)))
            continue
        seg = change[lo:hi]
        if prior is not None:
            t = np.arange(lo, hi) * hop
            sigma = max(PRIOR_SIGMA_MIN, PRIOR_SIGMA * (b_mid - a_mid))
            seg = seg * np.exp(-0.5 * ((t - guess) / sigma) ** 2)
        best = np.flatnonzero(seg >= seg.max() - 1e-12)
        out.append((lo + float(best.mean())) * hop)
    return out


def boundary_prior(spans: list[tuple[int, int]], owners: list[list[int]], phones: list[str],
                   rule: np.ndarray) -> list[float]:  # fmt: skip
    """Where each boundary between neighbouring units would fall between
    their spikes if the phones kept the proportions of the rules' durations
    (``rule``, one per phone): the part of the first unit after its spike
    against the part of the second before its own (``SPIKE_SHARE``)."""
    dur = [float(sum(rule[i] for i in own)) for own in owners]
    share = [SPIKE_SHARE[phone_kind(phones[own[0]])] * rule[own[0]] / max(d, 1e-6) for own, d in zip(owners, dur)]
    out = []
    for k in range(len(owners) - 1):
        a = (spans[k][0] + spans[k][1] + 1) / 2 * FRAME
        b = (spans[k + 1][0] + spans[k + 1][1] + 1) / 2 * FRAME
        rest, lead = (1 - share[k]) * dur[k], share[k + 1] * dur[k + 1]
        out.append(a + (b - a) * rest / max(rest + lead, 1e-6))
    return out


def share_unit(start: float, end: float, phones: list[str]) -> list[tuple[float, float]]:
    """Split one unit's stretch between the phones it stands for."""
    w = np.array([SHARE_WEIGHT["vowel" if p in VOCALIC else "other"] for p in phones], float)
    edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(phones))]


def head_of(p: str) -> float:
    if p in VOICELESS_STOPS:
        return LEAD_IN["voiceless_stop"]
    if p in VOICED_STOPS:
        return LEAD_IN["voiced_stop"]
    return LEAD_IN["other"]


def silent_runs(db: np.ndarray, hop: float = HOP, rel: float = SILENCE_DB) -> list[tuple[float, float]]:
    """Stretches of silence, as (start, end) seconds: under ``rel`` dB of
    the peak, or, in a recording with a noise floor above that, under
    ``NOISE_MARGIN`` dB over the floor (its 5th percentile level; at most
    ``SILENCE_CAP`` dB under the peak)."""
    import natural as N

    peak = N.peak_level(db)
    threshold = min(peak + SILENCE_CAP, max(peak + rel, float(np.percentile(db, 5)) + NOISE_MARGIN))
    return [(s * hop, e * hop) for s, e in N.runs(db < threshold)]


@dataclass
class Aligned:
    phones: list[str]
    start: np.ndarray  # s
    end: np.ndarray  # s
    spike: np.ndarray  # s, middle of the frames the model emitted the phone's unit on


def bounds_from_units(
    phones: list[str], owners: list[list[int]], spans: list[tuple[int, int]], bounds: list[float], length: float
) -> Aligned:
    """Phone stretches from unit spans and boundaries (seconds), contiguous;
    the first starts and the last ends 30 ms out from their spikes."""
    n_u = len(owners)
    ub = [max(0.0, spans[0][0] * FRAME - 0.03)] + [b + BOUNDARY_OFFSET for b in bounds]
    ub.append(min(length, (spans[-1][1] + 1) * FRAME + 0.03))
    start, end, spike = np.zeros(len(phones)), np.zeros(len(phones)), np.zeros(len(phones))
    for u in range(n_u):
        own = owners[u]
        for i, (a, b) in zip(own, share_unit(ub[u], ub[u + 1], [phones[i] for i in own])):
            start[i], end[i] = a, b
            spike[i] = (spans[u][0] + spans[u][1] + 1) / 2 * FRAME
    return Aligned(list(phones), start, end, spike)


def carve_pauses(al: Aligned, silences: list[tuple[float, float]], length: float,
                 quiet: list[tuple[float, float]] = ()) -> list[int]:  # fmt: skip
    """Turn silences between phones into pauses, in place: the phone before
    keeps ``TAIL`` of it, the one after its lead-in (its closure, for a
    stop). ``quiet`` stretches (unvoiced and ``QUIET_DB`` under the peak, a
    breath or a hesitation over the noise) become pauses too, from
    ``QUIET_MIN`` on, when both phones around them are sonorants, which
    are never that quiet. Also trims the first phone's start and the last
    phone's end to the speech. Returns the phones a pause follows."""
    n = len(al.phones)
    paused: list[int] = []

    def carve(r0: float, r1: float, shortest: float, sonorants_only: bool) -> None:
        if r1 - r0 < shortest:
            return
        inside = [k for k in range(n - 1) if r0 - 0.03 <= al.end[k] <= r1 + 0.03 and k not in paused]
        if not inside:
            return
        mid = 0.5 * (r0 + r1)
        k = min(inside, key=lambda q: abs(al.end[q] - mid))
        pair = {al.phones[k], al.phones[k + 1]}
        if sonorants_only and not pair <= SONORANTS:
            return
        if pair & STOP_LIKE and r1 - r0 < PAUSE_MIN_STOP:
            return
        end_k = max(al.start[k] + FLOOR, min(al.end[k], r0 + TAIL))
        start_n = min(al.end[k + 1] - FLOOR, max(al.start[k + 1], r1 - head_of(al.phones[k + 1])))
        if start_n - end_k < PAUSE_KEEP:
            return
        al.end[k], al.start[k + 1] = end_k, start_n
        paused.append(k)

    for r0, r1 in silences:
        if r0 <= al.start[0] + 1e-9 and r1 > al.start[0]:
            # Silence the first phone was given: it starts where the sound does.
            al.start[0] = max(al.start[0], min(r1 - head_of(al.phones[0]), al.end[0] - FLOOR))
            continue
        if r1 >= al.end[-1] - 1e-9 and r0 < al.end[-1]:
            al.end[-1] = min(al.end[-1], max(r0 + TAIL, al.start[-1] + FLOOR))
            continue
        carve(r0, r1, PAUSE_MIN, False)
    for r0, r1 in quiet:
        if r0 > al.start[0] and r1 < al.end[-1]:
            carve(r0, r1, QUIET_MIN, True)
    al.start[0] = max(0.0, al.start[0])
    al.end[-1] = min(length, al.end[-1])
    return sorted(paused)


def quiet_runs(db: np.ndarray, voiced: np.ndarray, hop: float = HOP) -> list[tuple[float, float]]:
    """Unvoiced stretches ``QUIET_DB`` under the peak, as (start, end) s."""
    import natural as N

    n = min(len(db), len(voiced))
    mask = (db[:n] < N.peak_level(db) + QUIET_DB) & ~np.asarray(voiced[:n], bool)
    return [(s * hop, e * hop) for s, e in N.runs(mask)]


def enforce_floor(al: Aligned, floor: float = FLOOR) -> None:
    """Give every phone at least ``floor``, in place, taken from the
    neighbours (and pauses) with the most to spare and from no one beyond
    need."""
    n = len(al.phones)
    for _ in range(3):
        for k in range(n):
            need = floor - (al.end[k] - al.start[k])
            if need <= 1e-9:
                continue
            # What the left and right side can give: a neighbour phone down to
            # the floor, a pause down to PAUSE_KEEP.
            if k > 0:
                gap = al.start[k] - al.end[k - 1]
                left = gap - PAUSE_KEEP if gap > 1e-9 else (al.end[k - 1] - al.start[k - 1]) - floor
            else:
                left = al.start[k]
            if k + 1 < n:
                gap = al.start[k + 1] - al.end[k]
                right = gap - PAUSE_KEEP if gap > 1e-9 else (al.end[k + 1] - al.start[k + 1]) - floor
            else:
                right = 0.0
            left, right = max(left, 0.0), max(right, 0.0)
            if left + right <= 1e-9:
                continue
            take = min(need, left + right)
            tl, tr = take * left / (left + right), take * right / (left + right)
            gap_l = k > 0 and al.start[k] - al.end[k - 1] > 1e-9
            gap_r = k + 1 < n and al.start[k + 1] - al.end[k] > 1e-9
            al.start[k] -= tl
            if k > 0 and not gap_l:
                al.end[k - 1] = al.start[k]
            al.end[k] += tr
            if k + 1 < n and not gap_r:
                al.start[k + 1] = al.end[k]


def ctc_score(lp, units: list[tuple[str, ...]], pj) -> float:
    """CTC log P(units | audio), nats."""
    import torch

    import judge as J

    classes = sorted(set(units))
    small = J.class_matrix(lp, classes, pj.unit_ids, pj.blank)
    target = torch.tensor([[classes.index(u) + 1 for u in units]], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(small[:, None, :], target, torch.tensor([small.shape[0]]),
                                        torch.tensor([len(units)]), blank=0, reduction="sum")  # fmt: skip
    return -float(loss)


def align(pj, audio: np.ndarray, phones: list[str], lp=None, db: np.ndarray | None = None,
          rule_durations: np.ndarray | None = None,
          voiced: np.ndarray | None = None) -> tuple[Aligned, list[int]]:  # fmt: skip
    """Force-align ``phones`` (the engine's) to ``audio``: every phone a
    stretch, pauses carved out of the silences; with ``rule_durations``
    (the engine's for the same phones) the boundaries lean towards their
    proportions (``boundary_prior``); with ``voiced`` (per 10 ms frame)
    quiet unvoiced stretches between sonorants are pauses too. Returns the
    alignment and the phones a pause follows."""
    import judge as J
    import natural as N
    import probes as P

    if lp is None:
        lp = pj.log_probs([audio], batch_size=1)[0]
    # The pairs the model may write as one token (``AA R`` = ``ɑːɹ``):
    # merged or not, whichever it finds likelier.
    options = [units_with_owners(phones, m) for m in (True, False)]
    if options[0][0] == options[1][0]:
        options = options[:1]
    scores = [ctc_score(lp, u, pj) for u, _ in options]
    units, owners = options[int(np.argmax(scores))]
    classes = sorted(set(units))
    small = J.class_matrix(lp, classes, pj.unit_ids, pj.blank).numpy()
    labels = [classes.index(u) + 1 for u in units]
    spans = P.ctc_viterbi(small, labels)
    prior = None if rule_durations is None else boundary_prior(spans, owners, phones, rule_durations)
    bounds = place_bounds(spans, change_curve(audio), units, prior)
    length = len(audio) / SR
    al = bounds_from_units(phones, owners, spans, bounds, length)
    if db is None:
        db = N.level_db(np.asarray(audio, np.float64))
    quiet = quiet_runs(db, voiced) if voiced is not None else []
    paused = carve_pauses(al, silent_runs(db), length, quiet)
    enforce_floor(al)
    return al, paused


# --------------------------------------------------------------------------
# Pitch and level
# --------------------------------------------------------------------------


def clean_pitch(f0: np.ndarray, window: int = 15, island: int = 3) -> np.ndarray:
    """Octave errors folded back and short voiced islands dropped (0 =
    unvoiced): a frame an octave (within 3 semitones) off the running median
    of its neighbours' pitch is moved by the octave, one more than 8
    semitones off otherwise is dropped."""
    import natural as N

    f = np.asarray(f0, float).copy()
    for _ in range(2):
        v = f > 0
        if v.sum() < 3:
            return np.zeros_like(f)
        st = np.where(v, 12 * np.log2(np.where(v, f, 1.0)), np.nan)
        for i in np.flatnonzero(v):
            lo, hi = max(0, i - window), min(len(f), i + window + 1)
            near = st[lo:hi]
            near = near[np.isfinite(near)]
            if len(near) < 3:
                continue
            d = st[i] - float(np.median(near))
            if abs(d - 12) < 3:
                f[i] /= 2
            elif abs(d + 12) < 3:
                f[i] *= 2
            elif abs(d) > 8:
                f[i] = 0.0
    for s, e in N.runs(f > 0):
        if e - s < island:
            f[s:e] = 0.0
    return f


def fill_pitch(f0: np.ndarray, median: float | None = None) -> np.ndarray:
    """A value everywhere: linear in semitones through unvoiced stretches,
    held at the ends, then smoothed lightly (1-2-1 over 30 ms)."""
    f = np.asarray(f0, float)
    v = f > 0
    if not v.any():
        return np.full(len(f), median or 100.0)
    st = 12 * np.log2(np.where(v, f, 1.0))
    idx = np.arange(len(f))
    filled = np.interp(idx, idx[v], st[v])
    sm = filled.copy()
    sm[1:-1] = 0.25 * filled[:-2] + 0.5 * filled[1:-1] + 0.25 * filled[2:]
    return 2 ** (sm / 12)


def phone_levels(db: np.ndarray, start: np.ndarray, end: np.ndarray, phones: list[str], hop: float = HOP) -> np.ndarray:
    """Each phone's mean level (power mean of the frames centred in it), dB
    against the mean over the utterance's vowels."""
    t = np.arange(len(db)) * hop
    lv = np.full(len(phones), np.nan)
    for k, (a, b) in enumerate(zip(start, end)):
        m = (t >= a) & (t < b)
        if not m.any():
            m = np.zeros(len(db), bool)
            m[min(len(db) - 1, int(round(0.5 * (a + b) / hop)))] = True
        lv[k] = 10 * np.log10(np.mean(10 ** (db[m] / 10)))
    vow = [k for k, p in enumerate(phones) if p in VOCALIC and np.isfinite(lv[k])]
    ref = 10 * np.log10(np.mean(10 ** (lv[vow] / 10))) if vow else float(np.nanmean(lv))
    return lv - ref


def analyse_audio(pj, audio: np.ndarray, text: str, phones: list[str], lp=None, source: str = "",
                  rule: np.ndarray | None = None) -> dict:  # fmt: skip
    """The schedule for ``audio`` saying ``text`` (``phones``: the engine's;
    ``rule``: its rule durations, for the aligner's prior)."""
    import match_voice as M
    import natural as N

    x = np.asarray(audio, np.float64)
    db = N.level_db(x)
    raw = M.track_pitch(x, SR, 55.0, 450.0).f0
    f0 = clean_pitch(raw)
    v = f0 > 0
    al, paused = align(pj, x, phones, lp, db, rule, v)
    median = float(np.median(f0[v])) if v.any() else 100.0
    track = fill_pitch(f0, median)
    lv = phone_levels(db, al.start, al.end, al.phones)
    t = np.arange(len(track)) * HOP
    phones_out = []
    for k, p in enumerate(al.phones):
        a, b = float(al.start[k]), float(al.end[k])
        pts = [float(np.interp(a + q * (b - a), t, track)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
        m = (t >= a) & (t < b)
        phones_out.append({
            "phone": p, "start": round(a, 4), "end": round(b, 4), "level_db": round(float(lv[k]), 2),
            "breath": 1.0, "pitch": [round(h, 1) for h in pts],
            "voiced": round(float(v[m].mean()), 2) if m.any() else 0.0,
        })  # fmt: skip
    pauses = [{"after": k, "start": phones_out[k]["end"], "end": phones_out[k + 1]["start"]} for k in paused]
    return {
        "format": "synthie-prosody-1",
        "text": text,
        "source": source,
        "phones": phones_out,
        "pauses": pauses,
        "median_hz": round(median, 2),
        "pitch": {"start": 0.0, "hop": HOP, "hz": [round(float(h), 2) for h in track],
                  "voiced": [int(x) for x in v]},  # fmt: skip
    }


# --------------------------------------------------------------------------
# Rendering and checking
# --------------------------------------------------------------------------


def render(schedule: dict | Path, out: Path, pitch: str = "speaker", voice: str = "deep",
           follow: bool = True, kick: bool = True) -> dict:  # fmt: skip
    """Speak a schedule (or, with ``follow=False``, its text by the rules)
    with ``bin/say``; ``kick=False`` leaves out the engine's pitch kick
    after voiceless consonants. Returns the phone spans of the output and
    what ``say`` printed."""
    if isinstance(schedule, dict):
        path = Path(out).with_suffix(".schedule.json")
        path.write_text(json.dumps(schedule), encoding="utf-8")
        sched = schedule
    else:
        path, sched = Path(schedule), json.loads(Path(schedule).read_text(encoding="utf-8"))
    times = Path(out).with_suffix(".times.json")
    cmd = [str(say_bin()), "--engine", "physical", "--voice", voice, "--out", str(out), "--phone-times", str(times)]
    if follow:
        cmd += ["--match-prosody", str(path), "--prosody-pitch", pitch, "--prosody-kick", "on" if kick else "off"]
    res = subprocess.run(cmd + [sched["text"]], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
    spans = np.array(json.loads(times.read_text())["spans"], float)
    times.unlink()
    if isinstance(schedule, dict):
        path.unlink()
    followed = next((ln for ln in res.stdout.splitlines() if ln.startswith("followed:")), "")
    shift = 0.0
    if "contour moved" in followed:
        shift = float(followed.split("contour moved")[1].split("st")[0])
    return {"spans": spans, "phones": A.parse_engine_line(res.stdout.splitlines()[0]), "followed": followed,
            "shift": shift}  # fmt: skip


def to_recording(tau: np.ndarray, spans: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The engine's clock onto the recording's, phone onto phone and
    linearly through the pauses between (as ``Utterance::follow``)."""
    tau = np.atleast_1d(np.asarray(tau, float))
    # knots: every phone's start and end on both clocks
    e_knots = np.ravel(np.column_stack([spans[:, 0], spans[:, 1]]))
    h_knots = np.ravel(np.column_stack([start, end]))
    out = np.interp(tau, e_knots, h_knots)
    out = np.where(tau < e_knots[0], h_knots[0] - (e_knots[0] - tau), out)
    return np.where(tau > e_knots[-1], h_knots[-1] + (tau - e_knots[-1]), out)


def sched_arrays(sched: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    ph = sched["phones"]
    return [p["phone"] for p in ph], np.array([p["start"] for p in ph]), np.array([p["end"] for p in ph])


def pitch_error(audio: np.ndarray, spans: np.ndarray, sched: dict, shift: float = 0.0) -> dict:
    """The rendering's pitch against the schedule's, on frames voiced in
    both (the schedule's own voiced frames, not its curve through the
    unvoiced ones): RMS and mean semitones, and the share of frames."""
    import match_voice as M

    f_e = clean_pitch(M.track_pitch(np.asarray(audio, np.float64), SR, 55.0, 450.0).f0)
    hz = np.array(sched["pitch"]["hz"], float)
    voiced = np.array(sched["pitch"]["voiced"], bool)
    _, start, end = sched_arrays(sched)
    tau = np.arange(len(f_e)) * HOP
    inside = (tau >= spans[0, 0]) & (tau <= spans[-1, 1])
    t = to_recording(tau, spans, start, end)
    idx = np.clip(np.round(t / sched["pitch"]["hop"]).astype(int), 0, len(hz) - 1)
    both = inside & (f_e > 0) & voiced[idx]
    if both.sum() < 5:
        return {"rms_st": None, "mean_st": None, "frames": int(both.sum())}
    d = 12 * np.log2(f_e[both] / (hz[idx[both]] * 2 ** (shift / 12)))
    # where the speaker's pitch moves fast (FAST_PITCH semitones a second or more)
    slope = np.abs(np.gradient(12 * np.log2(hz), sched["pitch"]["hop"]))[idx[both]]
    fast = slope >= FAST_PITCH
    rms = lambda x: float(np.sqrt(np.mean(x**2))) if len(x) else None  # noqa: E731
    return {"rms_st": rms(d), "mean_st": float(np.mean(d)), "frames": int(both.sum()),
            "share": float(both.sum() / max(1, (inside & voiced[idx]).sum())),
            "rms_fast_st": rms(d[fast]), "rms_slow_st": rms(d[~fast]), "fast_share": float(fast.mean())}  # fmt: skip


def duration_error(a_start: np.ndarray, a_end: np.ndarray, b_start: np.ndarray, b_end: np.ndarray) -> dict:
    """Phone durations against each other, interior phones (the first and
    last have no neighbour to bound them): mean absolute error, ms, and
    the share within 20 ms."""
    da, db = (a_end - a_start)[1:-1], (b_end - b_start)[1:-1]
    err = np.abs(da - db) * 1000
    return {"mae_ms": float(err.mean()), "within_20ms": float((err <= 20).mean()), "n": int(len(err)),
            "max_ms": float(err.max())}  # fmt: skip


def boundary_error(a_start: np.ndarray, b_start: np.ndarray) -> dict:
    """Phone onsets against each other after removing the mean offset, ms."""
    d = (a_start - b_start)[1:]
    return {"mean_ms": float(1000 * d.mean()), "mae_ms": float(1000 * np.abs(d - d.mean()).mean())}


def level_step(sched: dict, audio: np.ndarray, spans: np.ndarray) -> tuple[dict, dict]:
    """One step of moving each sonorant's breath towards the person's level
    (relative to the utterance's vowels), within ``LEVEL_RANGE`` dB of the
    rules'; obstruents take their neighbours'. Returns the new schedule and
    the level error before this step (RMS dB over sonorants)."""
    import natural as N

    phones, _, _ = sched_arrays(sched)
    db = N.level_db(np.asarray(audio, np.float64))
    eng = phone_levels(db, spans[:, 0], spans[:, 1], phones)
    hum = np.array([p["level_db"] for p in sched["phones"]])
    son = np.array([p in SONORANTS for p in phones])
    err = hum - eng
    new = json.loads(json.dumps(sched))
    breath = np.array([p.get("breath", 1.0) for p in sched["phones"]])
    db_now = LEVEL_PER_DOUBLING * np.log2(breath)
    db_new = np.clip(db_now + err, -LEVEL_RANGE, LEVEL_RANGE)
    # obstruents: the mean of the sonorants beside them
    for k in np.flatnonzero(~son):
        near = [db_new[j] for j in (k - 1, k + 1) if 0 <= j < len(phones) and son[j]]
        db_new[k] = float(np.mean(near)) if near else 0.0
    for k, p in enumerate(new["phones"]):
        p["breath"] = round(float(2 ** (db_new[k] / LEVEL_PER_DOUBLING)), 4)
    stats = {"rms_db": float(np.sqrt(np.mean(err[son] ** 2))), "mean_abs_db": float(np.mean(np.abs(err[son]))),
             "at_limit": int(np.sum(np.abs(db_new[son]) >= LEVEL_RANGE - 1e-6))}  # fmt: skip
    return new, stats


def transfer(pj, sched: dict, out: Path, pitch: str = "speaker", level: bool = True, voice: str = "deep") -> dict:
    """Render a schedule, with the level loop when ``level``; returns the
    final schedule, the level errors and the render's spans."""
    import render as R

    levels = []
    r = render(sched, out, pitch, voice)
    if level:
        for _ in range(2):
            sched2, st = level_step(sched, R.load_audio(out), r["spans"])
            levels.append(st)
            sched = sched2
            r = render(sched, out, pitch, voice)
        _, st = level_step(sched, R.load_audio(out), r["spans"])
        levels.append(st)
    return {"schedule": sched, "levels": levels, **r}


def check(pj, sched: dict, audio: np.ndarray, spans: np.ndarray, shift: float = 0.0,
          rule: np.ndarray | None = None) -> dict:  # fmt: skip
    """Duration test (the rendering aligned again against the schedule, as
    the recording was: with the rules' durations as the prior) and pitch
    test (its F0 against the schedule's)."""
    phones, start, end = sched_arrays(sched)
    al, _ = align(pj, np.asarray(audio, np.float64), phones, rule_durations=rule)
    return {
        "realigned_vs_schedule": duration_error(al.start, al.end, start, end),
        "followed_vs_schedule": duration_error(spans[:, 0], spans[:, 1], start, end),
        "realigned_vs_followed": duration_error(al.start, al.end, spans[:, 0], spans[:, 1]),
        "pitch": pitch_error(audio, spans, sched, shift),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _pj():
    import judge as J

    return J.PhonemeJudge()


def cmd_analyse(args) -> int:
    import render as R

    pj = _pj()
    try:
        phones, rule = engine_rule(args.text, args.voice)
        sched = analyse_audio(pj, R.load_audio(args.wav), args.text, phones, source=Path(args.wav).name, rule=rule)
    finally:
        pj.close()
    Path(args.out).write_text(json.dumps(sched, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(sched['phones'])} phones, {len(sched['pauses'])} pauses, median {sched['median_hz']:.0f} Hz")
    return 0


def cmd_render(args) -> int:
    sched = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    res = transfer(None, sched, Path(args.out), args.pitch, args.level, args.voice)
    print(res["followed"])
    for st in res["levels"]:
        print(f"level vs the person: RMS {st['rms_db']:.1f} dB over sonorants, {st['at_limit']} at the limit")
    if args.level:
        Path(args.out).with_suffix(".schedule.json").write_text(json.dumps(res["schedule"], indent=1))
    return 0


def cmd_check(args) -> int:
    import render as R

    sched = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    pj = _pj()
    try:
        # the phone spans of the rendering: render again to get them (deterministic)
        with tempfile.TemporaryDirectory() as tmp:
            r = render(sched, Path(tmp) / "x.wav", args.pitch, args.voice)
        rule = engine_rule(sched["text"], args.voice)[1]
        res = check(pj, sched, R.load_audio(args.wav), r["spans"], r["shift"], rule)
    finally:
        pj.close()
    print(json.dumps(res, indent=1))
    return 0


def roundtrip(pj, text: str, out_dir: Path, voice: str = "deep") -> dict:
    """The engine's own rendering of ``text`` as the recording: its schedule
    must give it back."""
    import render as R

    out_dir.mkdir(parents=True, exist_ok=True)
    orig = out_dir / "roundtrip-rule.wav"
    phones = engine_phones(text, voice)
    r0 = render({"text": text, "phones": []}, orig, follow=False, voice=voice)
    audio0 = R.load_audio(orig)
    # No prior here: the rules' durations are this recording's true ones.
    sched = analyse_audio(pj, audio0, text, phones, source=orig.name)
    _, start, end = sched_arrays(sched)
    truth = r0["spans"]
    res = {"aligner_vs_truth": duration_error(start, end, truth[:, 0], truth[:, 1]),
           "aligner_onsets": boundary_error(start, truth[:, 0])}  # fmt: skip
    tr = transfer(pj, sched, out_dir / "roundtrip-follow.wav", "speaker", True, voice)
    (out_dir / "roundtrip.json").write_text(json.dumps(tr["schedule"], indent=1), encoding="utf-8")
    audio1 = R.load_audio(out_dir / "roundtrip-follow.wav")
    res["transfer_vs_truth"] = duration_error(tr["spans"][:, 0], tr["spans"][:, 1], truth[:, 0], truth[:, 1])
    res.update(check(pj, tr["schedule"], audio1, tr["spans"]))
    # pitch of the copy against the original's, phone onto phone
    import match_voice as M

    f_a = clean_pitch(M.track_pitch(np.asarray(audio0, np.float64), SR, 55.0, 450.0).f0)
    f_b = clean_pitch(M.track_pitch(np.asarray(audio1, np.float64), SR, 55.0, 450.0).f0)
    tau = np.arange(len(f_b)) * HOP
    t = to_recording(tau, tr["spans"], truth[:, 0], truth[:, 1])
    idx = np.clip(np.round(t / HOP).astype(int), 0, len(f_a) - 1)
    both = (f_b > 0) & (f_a[idx] > 0)
    d = 12 * np.log2(f_b[both] / f_a[idx[both]])
    res["pitch_vs_original"] = {"rms_st": float(np.sqrt(np.mean(d**2))), "frames": int(both.sum())}
    res["levels"] = tr["levels"]
    res["followed"] = tr["followed"]
    return res


def transfer_file(pj, wav: Path | str, text: str, out_dir: Path, name: str, pitch: str = "speaker",
                  voice: str = "deep", level: bool = True, audio: np.ndarray | None = None) -> dict:  # fmt: skip
    """Any recording and its text: the schedule (``<name>.json``), the engine
    speaking it (``<name>-transfer.wav``) and by its rules
    (``<name>-rule.wav``), and the checks of the transfer."""
    import render as R

    out_dir.mkdir(parents=True, exist_ok=True)
    if audio is None:
        audio = R.load_audio(wav)
    phones, rule = engine_rule(text, voice)
    sched = analyse_audio(pj, audio, text, phones, source=Path(wav).name, rule=rule)
    tr = transfer(pj, sched, out_dir / f"{name}-transfer.wav", pitch, level, voice)
    (out_dir / f"{name}.json").write_text(json.dumps(tr["schedule"], indent=1), encoding="utf-8")
    render({"text": text}, out_dir / f"{name}-rule.wav", follow=False, voice=voice)
    res = check(pj, tr["schedule"], R.load_audio(out_dir / f"{name}-transfer.wav"), tr["spans"], tr["shift"], rule)
    res.update(levels=tr["levels"], followed=tr["followed"], shift=tr["shift"], pauses=len(sched["pauses"]),
               median_hz=sched["median_hz"])  # fmt: skip
    return res


def cmd_transfer(args) -> int:
    pj = _pj()
    try:
        name = args.name or Path(args.wav).stem
        res = transfer_file(pj, args.wav, args.text, Path(args.out_dir), name, args.pitch, args.voice, not args.no_level)
    finally:
        pj.close()
    print(json.dumps(res, indent=1, default=float))
    return 0


def cmd_roundtrip(args) -> int:
    pj = _pj()
    try:
        res = roundtrip(pj, args.text, Path(args.out_dir), args.voice)
    finally:
        pj.close()
    print(json.dumps(res, indent=1))
    return 0


def level_slope(text: str, voice: str = "deep", share: float = 1.41) -> float:
    """The engine's level per doubling of its breath share, dB: the second
    half of the phones given ``share``, the first half 1, against all 1
    (``say`` normalises every file, so only levels within one file are
    compared)."""
    import natural as N
    import render as R

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = render({"text": text}, tmp / "a.wav", follow=False, voice=voice)
        phones, sp = base["phones"], base["spans"]
        half = len(phones) // 2
        vow = np.array([p in VOCALIC for p in phones])
        diffs = []
        for b in (1.0, share):
            sched = {"text": text, "median_hz": 105.0, "pitch": {"start": 0.0, "hop": HOP, "hz": [105.0] * 3000},
                     "phones": [{"phone": p, "start": float(a), "end": float(e), "breath": (b if k >= half else 1.0)}
                                for k, (p, (a, e)) in enumerate(zip(phones, sp))]}  # fmt: skip
            r = render(sched, tmp / "b.wav", voice=voice)
            db = N.level_db(R.load_audio(tmp / "b.wav").astype(np.float64))
            lv = phone_levels(db, r["spans"][:, 0], r["spans"][:, 1], phones)
            k = np.arange(len(phones))
            diffs.append(float(np.mean(lv[vow & (k >= half + 2)]) - np.mean(lv[vow & (k < half - 2)])))
    return (diffs[1] - diffs[0]) / math.log2(share)


def cmd_slope(args) -> int:
    print(f"{level_slope(args.text, args.voice):.2f} dB per doubling of the breath")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", help="a recording and its text -> schedule JSON")
    a.add_argument("wav")
    a.add_argument("--text", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--voice", default="deep")
    r = sub.add_parser("render", help="speak a schedule")
    r.add_argument("schedule")
    r.add_argument("--out", required=True)
    r.add_argument("--pitch", choices=["speaker", "voice"], default="speaker")
    r.add_argument("--level", action="store_true", help="follow the level too (moves the breath, two passes)")
    r.add_argument("--voice", default="deep")
    c = sub.add_parser("check", help="align a rendering again against its schedule")
    c.add_argument("schedule")
    c.add_argument("wav")
    c.add_argument("--pitch", choices=["speaker", "voice"], default="speaker")
    c.add_argument("--voice", default="deep")
    t = sub.add_parser("roundtrip", help="schedule of the engine's own rendering, spoken again")
    t.add_argument("text")
    t.add_argument("--out-dir", default=str(HERE / "out" / "prosody"))
    t.add_argument("--voice", default="deep")
    tf = sub.add_parser("transfer", help="a recording and its text -> schedule, the engine saying it both ways, checks")
    tf.add_argument("wav")
    tf.add_argument("--text", required=True)
    tf.add_argument("--out-dir", default=str(HERE / "out" / "prosody"))
    tf.add_argument("--name")
    tf.add_argument("--pitch", choices=["speaker", "voice"], default="speaker")
    tf.add_argument("--voice", default="deep")
    tf.add_argument("--no-level", action="store_true")
    sl = sub.add_parser("slope", help="the engine's level per doubling of its breath")
    sl.add_argument("--text", default=FOUR_SENTENCES)
    sl.add_argument("--voice", default="deep")
    args = ap.parse_args(argv)
    cmds = {"analyse": cmd_analyse, "render": cmd_render, "check": cmd_check, "roundtrip": cmd_roundtrip,
            "slope": cmd_slope, "transfer": cmd_transfer}  # fmt: skip
    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
