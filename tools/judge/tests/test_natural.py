"""Model-free tests of ``natural.py`` on synthetic signals: every measure
reads what was built into the signal (no GPU, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import natural as N  # noqa: E402

SR = N.SR
RNG = np.random.default_rng(7)


def tone(seconds: float, hz: float = 200.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * hz * t)


def ramp_db(n: int, lo_db: float, hi_db: float) -> np.ndarray:
    """Gain rising from ``lo_db`` to ``hi_db`` (dB) linearly in dB."""
    return 10 ** (np.linspace(lo_db, hi_db, n) / 20)


def test_phrase_edges_read_a_built_in_ramp():
    # 0.3 s rising 40 dB, 0.5 s steady, 0.6 s falling 40 dB, in silence.
    up, hold, down = int(0.3 * SR), int(0.5 * SR), int(0.6 * SR)
    env = np.concatenate([ramp_db(up, -40, 0), np.ones(hold), ramp_db(down, 0, -40)])
    x = np.concatenate([np.zeros(SR // 5), tone(len(env) / SR) * env, np.zeros(SR // 5)])
    e = N.phrase_edges(N.level_db(x))
    # -30 to -6 dB is 24 of the 40 dB: 0.18 s of the rise, 0.36 s of the fall
    # (the 25 ms window blurs a little).
    assert abs(e["rise"] - 0.18) < 0.03, e
    assert abs(e["fall"] - 0.36) < 0.04, e


def test_dips_count_gaps_inside_the_sentence_and_dead_silence():
    a, gap, b = tone(0.4), np.zeros(int(0.12 * SR)), tone(0.4)
    quiet = 10 ** (-40 / 20) * tone(0.08)  # a 40 dB dip that is not silent
    x = np.concatenate([np.zeros(SR // 10), a, gap, b, quiet, a, np.zeros(SR // 10)])
    d = N.dips(N.level_db(x))
    assert d["per_s"] > 0 and abs(d["max_ms"] - 120) < 30, d
    # The 25 ms window takes some of the silence's edges.
    assert d["dead_share"] > 0 and 80 <= d["dead_max_ms"] <= 130, d
    # Two dips: the silent gap and the quiet stretch.
    span = (len(x) - 2 * SR // 10) / SR
    assert abs(d["per_s"] * span - 2) < 0.5, d
    smooth = np.concatenate([np.zeros(SR // 10), tone(1.0), np.zeros(SR // 10)])
    assert N.dips(N.level_db(smooth))["per_s"] == 0


def test_walls_are_steep_steps_and_ramps_are_not():
    hard = np.concatenate([np.zeros(SR // 5), tone(0.3), np.zeros(SR // 5), tone(0.3), np.zeros(SR // 5)])
    n = int(0.3 * SR)
    # Out of silence and back into it, from 60 dB down (steps that far under
    # the peak are not counted).
    soft_env = np.concatenate([ramp_db(n // 2, -60, 0), ramp_db(n // 2, 0, -60)])
    soft = np.concatenate([np.zeros(SR // 5), tone(0.3) * soft_env] * 2 + [np.zeros(SR // 5)])
    assert N.steps(hard)["walls_per_s"] > 2
    assert N.steps(soft)["walls_per_s"] == 0
    assert N.steps(soft)["p95_up"] < N.steps(hard)["p95_up"]


def test_segment_edges_time_each_rise_and_fall():
    # Syllables rising over 40 ms and falling over 80 ms (10-90 % of 30 dB).
    syl = []
    for _ in range(4):
        up, down = int(0.05 * SR), int(0.1 * SR)
        env = np.concatenate([ramp_db(up, -30, 0), np.ones(int(0.1 * SR)), ramp_db(down, 0, -30)])
        syl.append(tone(len(env) / SR) * env)
        syl.append(10 ** (-30 / 20) * tone(0.08))
    e = N.segment_edges(np.concatenate(syl))
    assert e["rise_ms"] is not None and 25 < e["rise_ms"] < 60, e
    assert e["fall_ms"] is not None and e["fall_ms"] > e["rise_ms"], e


def test_modulation_peaks_at_the_rate_built_in():
    t = np.arange(3 * SR) / SR
    noise = RNG.standard_normal(len(t))
    for hz in (3.0, 6.0):
        x = noise * (1.05 + np.sin(2 * np.pi * hz * t))
        m = N.modulation(x)
        assert abs(m["peak_hz"] - hz) < 0.6, (hz, m["peak_hz"])
    assert N.modulation(noise * (1.05 + np.sin(2 * np.pi * 12 * t)))["high"] > 0.5


def test_pitch_measures_read_a_falling_glide():
    # 1.5 s from 150 to 100 Hz: 7 semitones down, a final fall below the
    # median, steady declination; no micro-variation.
    n = int(1.5 * SR)
    f = np.linspace(150, 100, n)
    x = 0.5 * np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))  # harmonics for the tracker
    f0 = N.track_pitch(x)
    p = N.pitch_measures(f0)
    assert 115 < p["median_hz"] < 130, p
    assert p["declination"] < -3.5, p  # about -4.7 st/s
    assert p["final_fall"] < -2.5, p
    assert p["micro"] < 0.1, p
    # The same tone held flat has no range, no fall.
    flat = N.pitch_measures(N.track_pitch(0.5 * np.sign(np.sin(2 * np.pi * 120 * np.arange(n) / SR))))
    assert flat["range_st"] < 0.3 and abs(flat["final_fall"]) < 0.3, flat


def test_noise_spectrum_tells_broad_noise_from_a_whistle():
    white = RNG.standard_normal(SR // 5)
    # A narrow resonance at 3 kHz excited by the same noise.
    from scipy.signal import lfilter

    r, w = 0.995, 2 * np.pi * 3000 / SR
    whistle = lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], white)
    b, n = N.noise_spectrum(white), N.noise_spectrum(whistle)
    assert b["flatness"] > 0.5 and n["flatness"] < 0.1, (b["flatness"], n["flatness"])
    assert b["width6"] > 2000 and n["width6"] < 400, (b["width6"], n["width6"])
    assert abs(n["peak_hz"] - 3000) < 150 and n["prominence"] > 20
    assert 4000 < b["centroid"] < 5000  # white over 1-8 kHz: 4.5 kHz


def test_hiss_edges_read_a_ramped_burst():
    # Noise rising 20 dB over 30 ms, held 100 ms, falling over 15 ms.
    up, hold, down = int(0.03 * SR), int(0.1 * SR), int(0.015 * SR)
    env = np.concatenate([ramp_db(up, -20, 0), np.ones(hold), ramp_db(down, 0, -20)])
    x = np.concatenate([np.zeros(SR // 10), RNG.standard_normal(len(env)) * env, np.zeros(SR // 10)])
    band = N.band_level_db(x, 3000, 8000)
    c = int((0.1 + 0.08) / N.FINE_S)
    e = N.hiss_edges(band, c, c - 60, c + 60)
    # -15 to -3 dB is 12 of the 20 dB: 18 ms of the rise, 9 of the fall,
    # widened by the 20 ms analysis window.
    assert 12 < e["rise_ms"] < 40, e
    assert e["fall_ms"] < e["rise_ms"], e
    assert 90 < e["dur_ms"] < 150, e


def test_dtw_finds_a_stretched_copy_closer_than_another_sound():
    a = np.concatenate([tone(0.2, 200), RNG.standard_normal(SR // 5) * 0.3, tone(0.2, 500)])
    stretched = np.concatenate([tone(0.3, 200), RNG.standard_normal(SR // 4) * 0.3, tone(0.15, 500)])
    other = np.concatenate([tone(0.2, 900), tone(0.2, 300), RNG.standard_normal(SR // 5) * 0.3])
    same, path, _ = N.dtw(N.log_mel(a), N.log_mel(a))
    near, _, _ = N.dtw(N.log_mel(a), N.log_mel(stretched))
    far, _, _ = N.dtw(N.log_mel(a), N.log_mel(other))
    assert same < 1e-6 and path[0] == (0, 0)
    assert near < far, (near, far)


def test_turning_points_alternate():
    y = np.array([0, 5, 1, 1.5, 0.8, 6, 6, 2, 9, 0], float)
    pts = N.turning_points(y, 3.0)
    vals = [y[k] for k in pts]
    diffs = np.sign(np.diff(vals))
    assert all(diffs[i] != diffs[i + 1] for i in range(len(diffs) - 1)), (pts, vals)


def test_distance_is_mean_clipped_z_with_the_warped_distance():
    human = {"pitch.range_st": {"mean": 10.0, "sd": 2.0}, "rate.syll_per_s": {"mean": 4.5, "sd": 0.5},
             "steps.walls_per_s": {"mean": 1.0, "sd": 0.5}}  # fmt: skip
    same = {k: {"mean": v["mean"]} for k, v in human.items()}
    d, terms = N.distance(same, human)
    assert d == 0.0 and set(terms) == set(human)
    off = dict(same, **{"pitch.range_st": {"mean": 6.0}, "steps.walls_per_s": {"mean": 9.0}, "dtw.mean": {"mean": 3.0}})
    d, terms = N.distance(off, human)
    assert terms["pitch.range_st"] == -2.0
    assert terms["steps.walls_per_s"] == N.GUARD_CLIP  # 16 sd, clipped
    assert terms["dtw.mean"] == 3.0 / N.GUARD_DTW_DB
    assert abs(d - (2.0 + N.GUARD_CLIP + 0.0 + 3.0) / 4) < 1e-12
