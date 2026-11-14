"""Model-free tests of ``prosody.py``: the alignment's bookkeeping, pauses,
floors, pitch cleaning, levels and the clock mapping on built signals, and
(when ``bin/say`` is there) that the engine follows a schedule."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import prosody as Pr  # noqa: E402


def test_units_merge_what_the_model_writes_as_one_token():
    units, owners = Pr.units_with_owners(["D", "AA", "R", "K"])
    assert units == [("D",), ("AA", "R"), ("K",)]
    assert owners == [[0], [1, 2], [3]]
    units, owners = Pr.units_with_owners(["D", "AA", "R", "K"], merge=False)
    assert units == [("D",), ("AA",), ("R",), ("K",)]
    # a canonical rewrite (AX R -> ER) keeps both phones on the one unit
    units, owners = Pr.units_with_owners(["B", "AX", "R", "D"], merge=False)
    assert units == [("B",), ("ER",), ("D",)] and owners[1] == [1, 2]


def test_a_unit_is_shared_with_the_vowel_weighted():
    (a0, a1), (b0, b1) = Pr.share_unit(1.0, 1.25, ["AA", "R"])
    assert a0 == 1.0 and b1 == 1.25 and a1 == b0
    assert abs((a1 - a0) / (b1 - b0) - 1.5) < 1e-9


def test_boundaries_go_where_the_spectrum_changes():
    # spikes at frames 10 and 20 (0.2 s and 0.4 s); the change peaks at 0.33 s
    hop = Pr.FINE_HOP
    change = np.zeros(200)
    change[int(0.33 / hop)] = 5.0
    b = Pr.place_bounds([(10, 10), (20, 20)], change, [("AA",), ("N",)])
    assert abs(b[0] - 0.33) < 1e-9
    # between two stops: halfway between the spikes, whatever the spectrum does
    b = Pr.place_bounds([(10, 10), (20, 20)], change, [("T",), ("T",)])
    assert abs(b[0] - 0.31) < 1e-9


def test_the_prior_wins_over_a_far_change():
    hop = Pr.FINE_HOP
    change = np.zeros(200)
    change[int(0.23 / hop)] = 1.0
    change[int(0.36 / hop)] = 0.8
    assert abs(Pr.place_bounds([(10, 10), (20, 20)], change, [("AA",), ("N",)])[0] - 0.23) < 1e-9
    b = Pr.place_bounds([(10, 10), (20, 20)], change, [("AA",), ("N",)], prior=[0.37])
    assert abs(b[0] - 0.36) < 1e-9


def test_boundary_prior_follows_the_rule_proportions():
    # a vowel (spike at 29 % of it) then a nasal (spike at 59 %)
    spans = [(10, 10), (30, 30)]
    prior = Pr.boundary_prior(spans, [[0], [1]], ["AA", "N"], np.array([0.2, 0.1]))
    a, b = 10.5 * Pr.FRAME, 30.5 * Pr.FRAME
    rest, lead = 0.71 * 0.2, 0.59 * 0.1
    assert abs(prior[0] - (a + (b - a) * rest / (rest + lead))) < 1e-9


def aligned(phones, bounds):
    b = np.array(bounds, float)
    return Pr.Aligned(list(phones), b[:-1].copy(), b[1:].copy(), 0.5 * (b[:-1] + b[1:]))


def test_a_silence_between_words_becomes_a_pause():
    al = aligned(["AA", "N", "S", "IY"], [0.1, 0.3, 0.6, 0.7, 0.9])
    paused = Pr.carve_pauses(al, [(0.0, 0.08), (0.4, 0.58)], 1.0)
    assert paused == [1]
    assert abs(al.end[1] - (0.4 + Pr.TAIL)) < 1e-9
    assert al.start[2] == 0.6  # it started after the silence already


def test_a_closure_is_not_a_pause():
    al = aligned(["AA", "T", "IY"], [0.1, 0.3, 0.45, 0.7])
    assert Pr.carve_pauses(al, [(0.28, 0.43)], 1.0) == []
    # a long one is, and the stop keeps its closure
    al = aligned(["AA", "T", "IY"], [0.1, 0.3, 0.7, 0.9])
    assert Pr.carve_pauses(al, [(0.32, 0.62)], 1.0) == [0]
    assert abs(al.start[1] - (0.62 - Pr.LEAD_IN["voiceless_stop"])) < 1e-9


def test_a_quiet_breath_between_sonorants_is_a_pause_but_not_beside_a_fricative():
    al = aligned(["IY", "R", "S", "AA"], [0.1, 0.4, 0.7, 0.8, 1.0])
    assert Pr.carve_pauses(al, [], 1.1, quiet=[(0.3, 0.42)]) == [0]
    assert abs(al.end[0] - (0.3 + Pr.TAIL)) < 1e-9
    al = aligned(["IY", "R", "S", "AA"], [0.1, 0.4, 0.7, 0.8, 1.0])
    assert Pr.carve_pauses(al, [], 1.1, quiet=[(0.62, 0.75)]) == []
    db = np.full(100, -10.0)
    db[30:45] = -40.0
    voiced = np.ones(100, bool)
    voiced[28:46] = False
    assert Pr.quiet_runs(db, voiced) == [(0.3, 0.45)]


def test_every_phone_gets_its_floor_from_who_can_spare_it():
    al = aligned(["S", "AX", "N", "D"], [0.0, 0.2, 0.21, 0.24, 0.4])
    Pr.enforce_floor(al)
    d = al.end - al.start
    assert np.all(d >= Pr.FLOOR - 1e-9), d
    assert np.all(np.abs(al.end[:-1] - al.start[1:]) < 1e-12)  # still contiguous
    assert abs(d.sum() - 0.4) < 1e-9
    # the 30 ms nasal had only 5 ms to spare: the 200 ms /s/ gave most of it
    assert d[0] < 0.2 - 0.01 and d[2] >= Pr.FLOOR - 1e-9


def test_octave_errors_are_folded_back_and_islands_dropped():
    f = np.full(60, 120.0)
    f[20:23] = 240.0  # an octave up
    f[40] = 60.0  # one down
    f[50:52] = 0.0
    f[55] = 0.0
    f[57:60] = 0.0
    f[56] = 300.0  # a lone voiced frame
    out = Pr.clean_pitch(f)
    assert np.allclose(out[:50], 120.0)
    assert out[56] == 0.0


def test_the_pitch_curve_runs_through_unvoiced_stretches_in_semitones():
    f = np.zeros(21)
    f[:5], f[16:] = 100.0, 400.0
    out = Pr.fill_pitch(f)
    assert np.all(out > 0)
    # halfway through the gap: two octaves up by half, one octave
    assert abs(out[10] / 200.0 - 1) < 0.02


def test_phone_levels_are_against_the_vowels():
    db = np.concatenate([np.full(10, -20.0), np.full(10, -30.0), np.full(10, -26.0)])
    lv = Pr.phone_levels(db, np.array([0.0, 0.1, 0.2]), np.array([0.1, 0.2, 0.3]), ["AA", "S", "IY"])
    ref = 10 * np.log10(np.mean(10 ** (np.array([-20.0, -26.0]) / 10)))
    assert np.allclose(lv, np.array([-20.0, -30.0, -26.0]) - ref)


def test_the_engine_clock_maps_onto_the_recording_phone_by_phone():
    spans = np.array([[0.1, 0.2], [0.2, 0.4], [0.5, 0.6]])
    start, end = np.array([1.0, 1.05, 1.5]), np.array([1.05, 1.25, 1.7])
    t = Pr.to_recording(np.array([0.15, 0.3, 0.45, 0.55, 0.0, 0.7]), spans, start, end)
    assert np.allclose(t, [1.025, 1.15, 1.375, 1.6, 0.9, 1.8])


def test_duration_error_leaves_out_the_ends():
    a = np.array([0.0, 0.1, 0.2, 0.3])
    e = Pr.duration_error(a, a + 0.1, a, a + np.array([0.5, 0.11, 0.08, 0.5]))
    assert abs(e["mae_ms"] - 15.0) < 1e-9 and e["n"] == 2


@pytest.mark.skipif(not (HERE / "bin" / "say").exists(), reason="needs bin/say")
def test_the_engine_follows_a_schedule():
    text = "She sells sea shells."
    phones, rule = Pr.engine_rule(text)
    # every phone 30 % longer than the rules have it, a pause after "sells"
    t, out = 0.3, []
    for k, (p, d) in enumerate(zip(phones, rule)):
        out.append({"phone": p, "start": t, "end": t + 1.3 * d, "breath": 1.0})
        t += 1.3 * d + (0.25 if k == 5 else 0.0)
    sched = {"text": text, "phones": out, "median_hz": 140.0,
             "pitch": {"start": 0.0, "hop": 0.01, "hz": list(np.linspace(150, 110, 400))}}  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        r = Pr.render(sched, Path(tmp) / "x.wav")
        assert (Path(tmp) / "x.wav").exists()
    spans = r["spans"]
    assert r["phones"] == phones
    d = spans[:, 1] - spans[:, 0]
    assert np.allclose(d, 1.3 * rule, atol=2e-4), (d, 1.3 * rule)
    assert abs((spans[6, 0] - spans[5, 1]) - 0.25) < 2e-4
    assert "0 below the engine's floor" in r["followed"]
    json.dumps(sched)
