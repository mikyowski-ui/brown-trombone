"""Model-free tests of ``speaker_voice.py`` and of the voice parts of a
prosody schedule: per-phone measures on built frames, the tenseness and
formant steps, the breath map, and (when ``bin/say`` is there) that the
engine takes a schedule's voice, tenseness and formants."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))

import prosody as Pr  # noqa: E402
import speaker_voice as SV  # noqa: E402


def frames(n=100, f0=120.0):
    """Built per-frame measures: voiced throughout, F1 500 F2 1500."""
    forms = np.full((n, 5), np.nan)
    forms[:, 0], forms[:, 1] = 500.0, 1500.0
    return {"f0": np.full(n, f0), "forms": forms, "h12": np.full(n, 6.0), "tilt": np.full(n, -10.0),
            "db": np.zeros(n)}  # fmt: skip


def test_a_vowel_nucleus_is_measured_and_a_wild_one_rejected():
    fr = frames()
    fr["h12"][20:30] = 10.0  # the AA's middle frames are breathier
    fr["forms"][40:50, 1] = 900.0 * 1.4 ** (np.arange(10) % 4)  # the IY's F2 jumps about
    start, end = np.array([0.18, 0.38, 0.6]), np.array([0.32, 0.52, 0.7])
    out = SV.phone_voice(fr, start, end, ["AA", "IY", "S"])
    assert out["h12"][0] == pytest.approx(10.0) and out["h12"][1] == pytest.approx(6.0)
    assert out["f1"][0] == pytest.approx(500.0) and out["f2"][0] == pytest.approx(1500.0)
    assert np.isnan(out["f2"][1])  # no two frames agree
    assert np.isnan(out["f1"][2])  # not a vowel
    # an F2 where no F2 can be is not taken either
    fr = frames()
    fr["forms"][:, 1] = 3500.0
    assert np.isnan(SV.phone_voice(fr, start, end, ["AA", "IY", "S"])["f2"][0])


def test_deviations_are_against_the_mean_and_smoothed_by_neighbours():
    h = np.array([4.0, 8.0, np.nan, 6.0, 30.0])
    n = np.array([10, 10, 0, 10, 10])
    d = SV.deviations(h, n, ["AA", "N", "S", "IY", "T"], smooth=False)
    assert np.allclose(d[[0, 1, 3]], [-2.0, 2.0, 0.0]) and np.isnan(d[2]) and np.isnan(d[4])
    sm = SV.deviations(h, n, ["AA", "N", "S", "IY", "T"])
    assert sm[0] == pytest.approx(0.0) and sm[1] == pytest.approx(0.0) and sm[3] == pytest.approx(0.0)


def schedule(phones):
    return {"phones": [{"phone": p, "start": 0.1 * k, "end": 0.1 * k + 0.1} for k, p in enumerate(phones)]}


def test_a_phone_breathier_in_the_engine_than_in_the_speaker_is_made_tenser():
    phones = ["AA", "N", "IY", "S", "AE"]
    eng = {"h12": np.array([10.0, 6.0, 2.0, np.nan, 6.0]), "frames": np.array([20, 20, 20, 0, 20])}
    spk = {"h12": np.array([6.0, 6.0, 6.0, np.nan, 6.0]), "frames": np.array([20, 20, 20, 0, 20])}
    new, fit = SV.tenseness_step(schedule(phones), eng, spk)
    t = [p["tenseness"] for p in new["phones"]]
    assert t[0] > 1.0 and t[2] < 1.0 and t[3] == 1.0
    assert max(abs(x - 1) for x in t) <= SV.TENSENESS_STEP + 1e-9
    assert fit["n"] == 4
    # repeated steps stay inside the range
    for _ in range(10):
        new, _ = SV.tenseness_step(new, eng, spk)
    t = [p["tenseness"] for p in new["phones"]]
    assert min(t) >= SV.TENSENESS_RANGE[0] - 1e-9 and max(t) <= SV.TENSENESS_RANGE[1] + 1e-9


def test_a_vowel_takes_the_median_step_of_its_tokens_within_its_range():
    phones = ["AA", "T", "AA", "AA", "IY", "UW", "UW", "UW", "UW"]
    nan = np.nan
    # Over the formant scales: the speaker's /a/ has F1 10 % higher than the
    # engine's, her /u/ the engine's, one /a/ token's F2 is a tracking error
    # (+50 %), and the /i/'s F2 is off by more than a vowel can be.
    eng = {"f1": np.array([700.0, nan, 700.0, 700.0, 300.0, 320, 320, 320, 320]),
           "f2": np.array([1200.0, nan, 1200.0, nan, 2300.0, 900, 900, 900, 900])}  # fmt: skip
    spk = {"f1": np.array([770.0, nan, 770.0, 770.0, 300.0, 320, 320, 320, 320]) * 1.1,
           "f2": np.array([1200.0, nan, 1800.0, 1200.0, 2300.0 * 1.4, 900, 900, 900, 900]) * 1.1}  # fmt: skip
    new = SV.formant_step(schedule(phones), eng, spk, s_eng=1.0, s_spk=1.1)
    f = [p.get("formants") for p in new["phones"]]
    assert f[1] is None and f[4] is None and f[5] is None
    for k in (0, 2, 3):
        assert f[k] == pytest.approx([1.1, 1.0], abs=1e-3)
    err = SV.formant_error(eng, spk, 1.0, 1.1)
    assert err["f1_pct"] == pytest.approx(100 * (1 - 1 / 1.1) * 3 / 8, rel=1e-3)
    # An overall bias (every F1 of hers 12 % lower) is not followed: only
    # how her vowels stand against each other.
    spk["f1"] = spk["f1"] * 0.88
    new = SV.formant_step(schedule(phones), eng, spk, s_eng=1.0, s_spk=1.1)
    assert [p.get("formants") for p in new["phones"]][5] is None
    assert SV.formant_error(eng, spk, 1.0, 1.1)["f1_pattern_pct"] < 5


def test_the_breath_map_reaches_the_new_range_and_turns_round():
    assert Pr.breath_db(1.0) == pytest.approx(0.0)
    lo, hi = Pr.LEVEL_RANGE
    assert lo <= -10 and hi >= 5  # wider than the ±4 dB of round 8
    for db in np.linspace(lo, hi, 13):
        assert Pr.breath_db(Pr.db_breath(db)) == pytest.approx(db, abs=1e-6)
    assert 0.5 <= Pr.db_breath(lo) <= Pr.db_breath(hi) <= 1.5
    assert np.all(np.diff(Pr.breath_db(np.linspace(0.3, 1.8, 50))) > 0)


def test_a_recordings_formant_scale_falls_back_on_the_spacing_when_the_two_part():
    import match_voice as M

    def a(scale, lpc):
        return M.Analysis(f0_median=100.0, f0_p10=90.0, f0_p90=110.0, range_st=3.0, formants=[500.0] * 5, dF=1000.0,
                          order=22, h1h2=5.0, h1h2_raw=4.0, alpha=-10.0, cpp=2.0, hnr=8.0, syll_rate=4.0, voiced_s=2.0,
                          speech_s=3.0, scale=scale, scale_lpc=lpc)  # fmt: skip

    assert M.steady_scale(a(1.09, 1.07)).scale == pytest.approx(1.09)  # they agree: the density's
    assert M.steady_scale(a(0.69, 1.015)).scale == pytest.approx(1.015)  # they part: the spacing's
    assert M.steady_scale(a(float("nan"), 1.02)).scale == pytest.approx(1.02)
    assert SV.BASELINE_FIXED["lung_pressure"] >= 8.0 and SV.BASELINE_BOUNDS["tenseness"][0] >= 0.45


def test_the_five_are_engine_against_speaker():
    rec = {"f0_median": 200.0, "range_st": 8.0, "scale": 1.1, "h1h2": 10.0, "hnr": 9.0, "speech_s": 4.0}
    syn = {"f0_median": 210.0, "range_st": 7.0, "scale": 1.1 * 1.02, "h1h2": 9.0, "hnr": 10.0, "speech_s": 4.4}
    f = SV.five(rec, syn)
    assert f["pitch_pct"] == pytest.approx(5.0) and f["range_st"] == pytest.approx(-1.0)
    assert f["formant_scale_pct"] == pytest.approx(2.0) and f["h1h2_db"] == pytest.approx(-1.0)
    assert f["pace_pct"] == pytest.approx(100 * (4.0 / 4.4 - 1))
    assert SV.voice_of({"engine": "Physical", "pitch": 1, "rate": 1.2, "expressiveness": 1, "tuning": {}}) == {
        "engine": "Physical", "pitch": 1, "tuning": {}}  # fmt: skip


@pytest.mark.skipif(not (HERE / "bin" / "say").exists(), reason="needs bin/say")
def test_the_engine_takes_a_schedules_voice_tenseness_and_vowels():
    text = "She sells sea shells."
    phones, rule = Pr.engine_rule(text)
    t, out = 0.3, []
    for p, d in zip(phones, rule):
        out.append({"phone": p, "start": t, "end": t + d, "breath": 1.0})
        t += d
    sched = {"text": text, "phones": out, "median_hz": 200.0,
             "pitch": {"start": 0.0, "hop": 0.01, "hz": [200.0] * 400}}  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plain = Pr.render(sched, tmp / "a.wav")
        voiced = dict(sched, voice={"engine": "Physical", "tract_length": 0.85, "tenseness": 0.55})
        r = Pr.render(voiced, tmp / "b.wav")
        off = Pr.render(voiced, tmp / "c.wav", off=("voice",))
        assert (tmp / "a.wav").read_bytes() == (tmp / "c.wav").read_bytes()
        assert (tmp / "a.wav").read_bytes() != (tmp / "b.wav").read_bytes()
        shaped = json.loads(json.dumps(sched))
        for p in shaped["phones"]:
            if p["phone"] in Pr.VOCALIC:
                p["tenseness"], p["formants"] = 1.1, [1.08, 0.95]
        r = Pr.render(shaped, tmp / "d.wav")
        assert "0 vowels shaped" not in r["followed"], r["followed"]
        r = Pr.render(shaped, tmp / "e.wav", off=("tenseness", "formants"))
        assert "0 vowels shaped" in r["followed"]
        assert (tmp / "a.wav").read_bytes() == (tmp / "e.wav").read_bytes()
    _ = plain, off
