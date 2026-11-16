"""``match_voice.py``: analysis on known signals, and the self-consistency gate.

The gate: the engine renders a sentence with a hidden profile; ``match_voice.py``
is run on that recording; the engine then says the same sentence with the
profile it wrote, and the two recordings must re-analyse to within 5 % on
median pitch, 3 % on formant scale and 3 dB on H1*-H2*. The table of every
case (and how close the recovered settings are to the hidden ones) is
printed (``pytest -s`` shows it) and written to ``$TMPDIR/match_voice_gate.json``.

Needs ``bin/say`` (skipped without it) and takes about a minute.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import match_voice as M  # noqa: E402

TEXT = "The boy was there when the sun rose. A rod is used to catch pink salmon. The source of the huge river is the clear spring."

#: Hidden voices: pitch 90..300 Hz, tract length 0.7..1.25, tenseness 0.35..0.85.
HIDDEN = [
    {"pitch": 90.0, "tract_length": 1.25, "tenseness": 0.6},
    {"pitch": 120.0, "tract_length": 1.0, "tenseness": 0.85},
    {"pitch": 160.0, "tract_length": 1.1, "tenseness": 0.35},
    {"pitch": 210.0, "tract_length": 0.85, "tenseness": 0.45},
    {"pitch": 250.0, "tract_length": 0.8, "tenseness": 0.7, "expressiveness": 0.9, "rate": 1.15},
    {"pitch": 300.0, "tract_length": 0.7, "tenseness": 0.55, "rate": 0.9},
    {"pitch": 280.0, "tract_length": 1.0, "tenseness": 0.75},
    {"pitch": 100.0, "tract_length": 0.75, "tenseness": 0.4, "expressiveness": 0.2},
]

needs_say = pytest.mark.skipif(not M.say_bin().exists(), reason="bin/say is missing")


def _tone(f0: float, seconds: float = 1.5, sr: int = 16_000, h2_db: float = -6.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * f0 * t) + 10 ** (h2_db / 20) * np.sin(2 * np.pi * 2 * f0 * t + 0.3)


@pytest.mark.parametrize("f0", [80.0, 140.0, 230.0, 380.0])
def test_pitch_tracker_on_tones(f0):
    p = M.track_pitch(_tone(f0) + 0.4 * _tone(3 * f0, h2_db=-40))
    v = p.f0 > 0
    assert v.mean() > 0.8
    assert abs(np.median(p.f0[v]) / f0 - 1) < 0.01


def test_pitch_tracker_ignores_noise():
    rng = np.random.default_rng(1)
    p = M.track_pitch(rng.normal(size=16_000) * 0.1)
    assert (p.f0 > 0).mean() < 0.1


def test_h1h2_on_two_harmonics():
    # Two harmonics 6 dB apart, no formants to correct for: H1-H2 = 6 dB.
    x = _tone(150.0, h2_db=-6.0)
    f0 = np.full(1 + (len(x) - 1) // 160, 150.0)
    f0[:3] = f0[-3:] = 0
    forms = np.full((len(f0), 5), np.nan)
    h12, _, _, _ = M.harmonic_measures(x, f0, forms)
    assert abs(np.median(h12) - 6.0) < 0.5


def test_profile_keys_and_ranges():
    vals = {"pitch": 900.0, "tract_length": 0.1, "tenseness": 2.0, "expressiveness": -1.0, "rate": 1.0,
            "posterior_gap": 0.3, "lung_pressure": 30.0}  # fmt: skip
    prof = M.profile_of(vals)
    allowed = {"persona", "pitch", "tract_length", "tenseness", "expressiveness", "rate", "vibrato",
               "vibrato_rate", "jitter", "nasality", "engine", "tuning"}  # fmt: skip
    assert set(prof) <= allowed
    assert 0.5 <= prof["tract_length"] <= 1.6
    assert 0.0 <= prof["tenseness"] <= 1.0
    assert 4.0 <= prof["tuning"]["lung_pressure"] <= 16.0
    parts = json.loads((HERE / "params.json").read_text())
    assert set(prof["tuning"]) <= {n for fields in parts.values() for n in fields}
    # A profile the engine wrote, its tuning by part, reads back the same.
    nested = {**prof, "tuning": {"anatomy": prof["tuning"]}}
    assert M.values_of_profile(nested) == M.values_of_profile(prof)


def test_unit_mapping_round_trip():
    vals = {n: p[1] for n, p in zip(M.NAMES, M.PARAMS)}
    back = M.values_of(M.u_of(vals))
    for n in M.NAMES:
        assert abs(back[n] - vals[n]) <= 1e-3 * max(1.0, abs(vals[n]))


@needs_say
def test_profile_is_accepted_by_say(tmp_path):
    prof = M.profile_of(M.values_of(M.u_of({"pitch": 180.0, "tract_length": 0.9})))
    path = tmp_path / "p.json"
    path.write_text(json.dumps(prof))
    res = subprocess.run([str(M.say_bin()), "--engine", "physical", "--profile", str(path), "--dump-profile"],
                         capture_output=True, text=True, check=True)  # fmt: skip
    eff = json.loads(res.stdout)
    assert abs(eff["pitch"] - 180.0) < 0.1 and abs(eff["tract_length"] - 0.9) < 1e-3


def _run_case(i: int, hidden: dict, workdir: Path) -> dict:
    wav = M.render(TEXT, {"engine": "Physical", **hidden}, workdir / f"hidden{i}.wav")
    out, report = workdir / f"matched{i}.json", workdir / f"report{i}.json"
    cmd = [sys.executable, str(HERE / "match_voice.py"), str(wav), "--text", TEXT, "--transcript",
           "--out", str(out), "--report", str(report), "--quiet"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    assert res.returncode == 0, res.stderr[-2000:]
    rep = json.loads(report.read_text())
    got = M.values_of_profile(json.loads(out.read_text()))
    return {"hidden": hidden, "recovered": got, "errors": rep["errors"], "seconds": rep["seconds"],
            "rounds": len(rep["rounds"])}  # fmt: skip


@needs_say
def test_self_consistency_gate():
    with tempfile.TemporaryDirectory(prefix="match-test-") as wd:
        with ThreadPoolExecutor(max_workers=4) as pool:
            cases = list(pool.map(lambda a: _run_case(a[0], a[1], Path(wd)), enumerate(HIDDEN)))
    lines = [
        "hidden (pitch / tract / tenseness)  ->  recovered                 "
        "pitch %   formant %  H1*-H2* dB   range st  rate %   s"
    ]
    failures = []
    for c in cases:
        h, g, e = c["hidden"], c["recovered"], c["errors"]
        lines.append(
            f"{h['pitch']:5.0f} {h['tract_length']:5.2f} {h['tenseness']:5.2f}"
            f"  ->  {g['pitch']:6.1f} {g['tract_length']:6.3f} {g['tenseness']:6.3f}   "
            f"{e['pitch_pct']:+7.2f}  {e['formant_pct']:+8.2f}  {e['h1h2_db']:+9.2f}  {e['range_st']:+8.2f}"
            f"  {e['rate_pct']:+6.1f}  {c['seconds']:4.0f}"
        )
        for k, tol in M.TOL.items():
            if abs(e[k]) > tol:
                failures.append(f"{h}: {k} {e[k]:+.2f} (tolerance {tol})")
    print("\n" + "\n".join(lines))
    Path(tempfile.gettempdir(), "match_voice_gate.json").write_text(json.dumps(cases, indent=2))
    assert not failures, "\n".join(failures)
