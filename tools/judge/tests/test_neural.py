"""Model-free tests of ``neural.py`` (tiny-voca's prosody as a schedule): the
frame clock, the phoneme mapping, the alignment and the time it shares out,
the pauses, the level, the schedule's shape and, when ``bin/say`` is there,
that the engine follows a schedule of this shape. Neither the model nor
onnxruntime is needed."""

from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402
import neural as N  # noqa: E402
import prosody as Pr  # noqa: E402


def test_importing_the_bridge_needs_no_onnx_runtime():
    assert "onnxruntime" not in sys.modules and "onnx" not in sys.modules


def test_the_model_frame_is_the_engine_control_block():
    assert N.frame_seconds() == pytest.approx(256 / 24000)
    assert N.frame_seconds() == pytest.approx(N.ENGINE_BLOCK / N.ENGINE_RATE)
    assert N.engine_blocks(N.frame_seconds()) == pytest.approx(1.0)
    assert N.engine_blocks(1.0) == pytest.approx(93.75)


def test_the_front_end_fingerprint_is_the_one_the_model_records():
    # config.json of n4ze3m/tiny-voca: the token convention 4 and the 76 symbols
    assert len(N.SYMBOLS) == 76 and len(set(N.SYMBOLS)) == 76
    assert N.front_end_fingerprint(4, list(N.SYMBOLS)) == (
        "c729f235e10daa1a08ce7711169c1cb93fb552b3c24eecb16527513d7b69d4da")
    assert N.front_end_fingerprint(3, list(N.SYMBOLS)) != N.front_end_fingerprint(4, list(N.SYMBOLS))


def test_every_phone_of_the_model_folds_to_the_engine_alphabet():
    for sym in N.SYMBOLS[17:]:
        parts = N.fold(sym)
        assert 1 <= len(parts) <= 2, sym
        assert all(p in A.ENGINE_PHONEMES or p == A.FLAP for p in parts), (sym, parts)
    assert N.fold("ɾ") == ("DX",) and N.fold("ʔ") == ("T",)
    assert N.fold("ɚ") == ("ER",) and N.fold("ɐ") == ("AX",) and N.fold("ᵻ") == ("IH",)
    assert N.fold("ɑːɹ") == ("AA", "R") and N.fold("oːɹ") == ("AO", "R") and N.fold("aɪɚ") == ("AY", "ER")
    assert N.fold("əl") == ("AX", "L") and N.fold("n̩") == ("AX", "N")
    assert N.NOT_PHONES == frozenset(N.SYMBOLS[:17])


def _run(symbols, durations, f0=None):
    n = int(sum(durations))
    return {"symbols": symbols, "durations": np.array(durations), "f0": np.full(n, 100.0) if f0 is None else f0,
            "voiced": np.ones(n, bool), "energy": np.zeros(n)}  # fmt: skip


def test_tokens_are_laid_out_on_the_frame_clock():
    fr = N.frame_seconds()
    mp = N.lay_out([_run(["<bos>", "k", "ɑːɹ", "<sp>", "t", ".", "<eos>"], [3, 2, 5, 0, 2, 4, 1])])
    assert mp.symbols == ["K", "AA", "R", "T"]
    assert mp.start[0] == pytest.approx(3 * fr) and mp.end[0] == pytest.approx(5 * fr)
    # the unit of two phones is shared out with the vowel weighted 1.5 to 1
    a, r = mp.end[1] - mp.start[1], mp.end[2] - mp.start[2]
    assert a / r == pytest.approx(N.VOWEL_WEIGHT) and mp.end[2] == pytest.approx(10 * fr)
    assert mp.start[3] == pytest.approx(10 * fr) and mp.end[3] == pytest.approx(12 * fr)
    assert len(mp.f0) == 17 and mp.pieces == [(pytest.approx(3 * fr), pytest.approx(12 * fr))]
    # a second sentence goes on after the first one's frames
    mp2 = N.lay_out([_run(["<bos>", "k", "<eos>"], [1, 2, 1]), _run(["<bos>", "t", "<eos>"], [2, 3, 1])])
    assert mp2.start[1] == pytest.approx((4 + 2) * fr) and len(mp2.f0) == 10
    assert len(mp2.pieces) == 2


def test_energy_is_the_log_rms_in_db():
    assert N.energy_db(0.0) == pytest.approx(20 * np.log10(np.exp(-4.0)))
    assert N.energy_db(1.0) - N.energy_db(0.0) == pytest.approx(2.0 * 20 / np.log(10))


def test_alignment_matches_like_sounds_and_leaves_out_the_rest():
    ops = N.align_phones(["DH", "AX", "K", "AE", "T"], ["DH", "AH", "K", "AE", "T"])
    assert ops == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    # the flap matches the engine's T
    assert N.align_phones(["B", "EH", "DX", "ER"], ["B", "EH", "T", "ER"]) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    # a phone only the engine says, and one only the model says
    assert N.align_phones(["S", "IY"], ["S", "T", "IY"]) == [(0, 0), (None, 1), (1, 2)]
    assert N.align_phones(["S", "T", "IY"], ["S", "IY"]) == [(0, 0), (1, None), (2, 1)]
    # a vowel is not matched with a stop: both are left out instead
    assert sorted(N.align_phones(["P"], ["AA"]), key=str) == sorted([(0, None), (None, 0)], key=str)


def _mp(symbols, edges):
    mp = N.ModelPhones()
    mp.symbols = list(symbols)
    mp.start = [float(a) for a, _ in edges]
    mp.end = [float(b) for _, b in edges]
    return mp


def test_an_engine_phone_the_model_lacks_takes_its_time_from_its_neighbours():
    mp = _mp(["S", "IY"], [(0.0, 0.1), (0.1, 0.3)])
    pl = N.place(mp, ["S", "T", "IY"], np.array([0.1, 0.05, 0.2]))
    assert pl.added == ["T"] and pl.matched == 2
    assert pl.start[0] == 0.0 and pl.end[-1] == pytest.approx(0.3)
    assert np.allclose(pl.end[:-1], pl.start[1:])
    # at the model's tempo (1.0 here), no more than half of the neighbours
    assert pl.end[1] - pl.start[1] == pytest.approx(0.05)
    assert pl.end[0] - pl.start[0] == pytest.approx(0.1 - 0.05 / 3)


def test_a_phone_only_the_model_says_goes_to_the_one_before():
    mp = _mp(["S", "T", "IY"], [(0.0, 0.1), (0.1, 0.15), (0.15, 0.3)])
    pl = N.place(mp, ["S", "IY"], np.array([0.1, 0.2]))
    assert pl.dropped == ["T"]
    assert pl.end[0] == pytest.approx(0.15) and pl.start[1] == pytest.approx(0.15)


def test_an_added_phone_in_a_pause_takes_it_from_the_pause():
    mp = _mp(["AA", "AA"], [(0.0, 0.2), (0.5, 0.7)])
    pl = N.place(mp, ["AA", "HH", "AA"], np.array([0.2, 0.05, 0.2]))
    assert pl.end[0] == pytest.approx(0.2) and pl.end[1] == pytest.approx(0.5)
    assert pl.end[1] - pl.start[1] == pytest.approx(0.05)


def test_short_pauses_are_closed_and_long_ones_kept():
    start, end = np.array([0.0, 0.13, 0.5]), np.array([0.1, 0.3, 0.6])
    assert N.close_short_pauses(start, end) == 1
    assert end[0] == pytest.approx(0.115) and start[1] == pytest.approx(0.115)
    assert start[2] - end[1] == pytest.approx(0.2)


def test_sentences_are_split_and_given_a_final_mark():
    assert N.sentences("One two. Three four! Five") == ["One two.", "Three four!", "Five."]
    assert N.sentences("Dr. Smith went home. He slept.") == ["Dr. Smith went home.", "He slept."]
    assert N.sentences('He said "go." Then he went.') == ['He said "go."', "Then he went."]
    assert N.sentences("   ") == []


def test_breath_shares_follow_the_vowels_and_the_consonants_lie_between():
    fr = N.frame_seconds()
    db = np.concatenate([np.full(10, -20.0), np.full(10, -30.0), np.full(10, -40.0)])
    phones = ["AA", "S", "AA"]
    start, end = np.array([0, 10, 20]) * fr, np.array([10, 20, 30]) * fr
    lv, br = N.breath_shares(phones, start, end, db, fr)
    # against the vowels' (power) mean
    assert lv[0] - lv[2] == pytest.approx(20.0, abs=0.01) and lv[0] > 0 > lv[2]
    assert br[0] > 1.0 > br[2]
    # the consonant takes the vowels' line at its middle, not its own level
    assert br[1] == pytest.approx(float(Pr.db_breath(0.5 * (lv[0] + lv[2]))), abs=1e-6)


def _schedule_inputs():
    symbols = ["<bos>", "ð", "ə", "<sp>", "k", "æ", "t", ",", "<sp>", "s", "æ", "t", ".", "<eos>"]
    durations = [20, 4, 6, 0, 5, 12, 1, 25, 0, 9, 14, 6, 0, 3]
    n = sum(durations)
    run = _run(symbols, durations, np.linspace(130, 90, n))
    run["voiced"] = np.array([symbols[k] in ("ð", "ə", "æ") for k in np.repeat(np.arange(len(symbols)), durations)])
    run["energy"] = np.linspace(1.0, -1.0, n)
    engine = ["DH", "AX", "K", "AE", "T", "S", "AE", "T"]
    return N.lay_out([run]), engine, np.array([0.05, 0.06, 0.07, 0.12, 0.08, 0.1, 0.13, 0.08])


def _schedule(breath=False):
    fr = N.frame_seconds()
    symbols = ["<bos>", "ð", "ə", "<sp>", "k", "æ", "t", ",", "<sp>", "s", "æ", "t", ".", "<eos>"]
    durations = [20, 4, 6, 0, 5, 12, 1, 25, 0, 9, 14, 6, 0, 3]
    n = sum(durations)
    f0 = np.linspace(130, 90, n)
    run = _run(symbols, durations, f0)
    run["voiced"] = np.array([symbols[k] in ("ð", "ə", "æ") for k in np.repeat(np.arange(len(symbols)), durations)])
    run["energy"] = np.linspace(1.0, -1.0, n)
    mp = N.lay_out([run])
    engine = ["DH", "AX", "K", "AE", "T", "S", "AE", "T"]
    rule = np.array([0.05, 0.06, 0.07, 0.12, 0.08, 0.1, 0.13, 0.08])
    return N.build_schedule("The cat, sat.", mp, engine, rule, breath, {"voice": "M2"}), fr, n


def test_the_schedule_has_the_shape_say_reads():
    (sched, rep), fr, n = _schedule()
    assert sched["format"] == "synthie-prosody-1" and sched["text"] == "The cat, sat."
    ph = sched["phones"]
    assert [p["phone"] for p in ph] == ["DH", "AX", "K", "AE", "T", "S", "AE", "T"]
    starts, ends = np.array([p["start"] for p in ph]), np.array([p["end"] for p in ph])
    assert np.all(ends > starts) and np.all(starts[1:] >= ends[:-1] - 1e-9)
    assert np.all(ends - starts >= Pr.FLOOR - 1e-4)
    # the T of one frame was below the floor before it was given its floor
    assert rep["below_floor"] and rep["below_floor"][0][0] == "T"
    # the comma's pause is kept, on the model's clock
    assert len(sched["pauses"]) == 1 and sched["pauses"][0]["after"] == 4
    assert sched["pauses"][0]["end"] - sched["pauses"][0]["start"] == pytest.approx(25 * fr, abs=0.03)
    # the frames of the model, and those the floor on consonants added
    assert rep["floored"] >= 1 and rep["floor_added_s"] >= 0
    assert sched["pitch"]["hop"] == pytest.approx(fr)
    assert len(sched["pitch"]["hz"]) == n + round(rep["floor_added_s"] / fr)
    assert all(h > 0 for h in sched["pitch"]["hz"])
    assert 90 <= sched["median_hz"] <= 130
    assert all(p["breath"] == 1.0 for p in ph)
    (sched2, _), _, _ = _schedule(breath=True)
    assert sched2["phones"][1]["breath"] > sched2["phones"][-2]["breath"]


# --------------------------------------------------------------------------
# The floor on consonants and the pitch spikes
# --------------------------------------------------------------------------


def test_the_floor_holds_stops_affricates_sibilants_and_clusters():
    phones = ["S", "T", "R", "AA", "M", "AA", "N", "F", "AA", "L", "JH", "AA", "W", "AA", "V", "N"]
    start = np.arange(len(phones)) * 0.1
    end = start + 0.1
    # a pause after the F: the V N after it stay a cluster, the F and the AA do not
    start[8:] += 0.2
    end[8:] += 0.2
    held = N.floored(phones, start, end)
    assert held == [True, True, True, False, False, False, True, True, False, True, True, False, False, False,
                    True, True]  # fmt: skip
    # a lone nasal, glide or weak fricative between vowels is not held
    lone = N.floored(["AA", "M", "AA", "W", "AA", "F", "AA"], np.arange(7) * 0.1, np.arange(7) * 0.1 + 0.1)
    assert not any(lone)
    # nor one beside a consonant across a pause
    assert N.floored(["AA", "N", "F", "AA"], np.array([0, 0.1, 0.4, 0.5]), np.array([0.1, 0.2, 0.5, 0.6])) == [
        False, False, False, False]  # fmt: skip


def test_a_short_stop_is_lengthened_from_the_vowels_beside_it_and_the_total_is_kept():
    phones = ["AA", "T", "IY"]
    start, end = np.array([0.0, 0.2, 0.22]), np.array([0.2, 0.22, 0.42])
    rule = np.array([0.15, 0.08, 0.15])
    s, e, n, added = N.floor_consonants(phones, start, end, rule, share=0.85, keep=0.6)
    assert n == 1 and added == pytest.approx(0.0, abs=1e-12)
    assert e[1] - s[1] == pytest.approx(0.85 * 0.08)
    # the vowels gave the 48 ms in proportion to what they had above 60 % of their rule durations (equal here)
    assert e[0] - s[0] == pytest.approx(0.2 - 0.024) and e[2] - s[2] == pytest.approx(0.2 - 0.024)
    assert s[0] == 0.0 and e[-1] == pytest.approx(0.42)
    assert np.allclose(e[:-1], s[1:])


def test_what_the_vowels_cannot_give_is_added_and_the_pauses_keep_their_length():
    # the vowel before has 10 ms above 60 % of its rule duration, the one after is across a pause
    phones = ["AA", "K", "IY", "S"]
    start, end = np.array([0.0, 0.1, 0.4, 0.6]), np.array([0.1, 0.12, 0.6, 0.7])
    rule = np.array([0.15, 0.1, 0.15, 0.1])
    s, e, n, added = N.floor_consonants(phones, start, end, rule, share=0.85, keep=0.6)
    assert n == 1
    assert e[0] - s[0] == pytest.approx(0.09)
    assert e[1] - s[1] == pytest.approx(0.085)
    assert added == pytest.approx(0.085 - 0.02 - 0.01)
    # the pause after the K keeps its 0.28 s, the vowel after it its length; the phones after move later
    assert s[2] - e[1] == pytest.approx(0.28) and e[2] - s[2] == pytest.approx(0.2)
    assert e[-1] == pytest.approx(0.7 + added)


def test_a_vowel_gives_no_more_than_down_to_its_keep():
    phones = ["AH", "P", "AH"]
    start, end = np.array([0.0, 0.07, 0.08]), np.array([0.07, 0.08, 0.15])
    rule = np.array([0.1, 0.075, 0.1])
    s, e, _, added = N.floor_consonants(phones, start, end, rule, share=0.85, keep=0.6)
    d = e - s
    assert d[0] == pytest.approx(0.06) and d[2] == pytest.approx(0.06)
    assert d[1] == pytest.approx(0.85 * 0.075)
    assert added == pytest.approx(0.85 * 0.075 - 0.01 - 0.02)


def test_with_words_only_the_vowels_of_the_consonants_own_word_give_it_time():
    # "birch canoe": ER CH | K AX: the K may take from the AX of its word, not the ER of "birch"
    phones = ["ER", "CH", "K", "AX"]
    start, end = np.array([0.0, 0.16, 0.29, 0.34]), np.array([0.16, 0.29, 0.34, 0.425])
    rule = np.array([0.139, 0.116, 0.099, 0.1])
    s, e, _, added = N.floor_consonants(phones, start, end, rule, words=[0, 0, 1, 1])
    d = e - s
    assert d[0] == pytest.approx(0.16) and d[2] == pytest.approx(0.85 * 0.099)
    assert d[3] == pytest.approx(0.06)  # the AX gave down to 60 % of its rule duration
    assert added == pytest.approx(0.85 * 0.099 - 0.05 - 0.025)
    s, e, _, added = N.floor_consonants(phones, start, end, rule)
    assert (e - s)[0] < 0.16 and added == pytest.approx(0.0, abs=1e-12)


def test_the_model_phones_are_numbered_by_word_and_the_engine_phones_take_theirs():
    symbols = ["<bos>", "k", "æ", "t", "<sp>", "s", "æ", "t", ".", "<eos>"]
    mp = N.lay_out([_run(symbols, [2, 3, 6, 3, 1, 4, 6, 3, 2, 1])])
    assert mp.word[:3] == [mp.word[0]] * 3 and mp.word[3:] == [mp.word[3]] * 3 and mp.word[3] > mp.word[0]
    # an engine phone the model has no token for takes the word of the phone before it
    pl = N.place(mp, ["K", "AE", "T", "S", "T", "AE", "T"], np.full(7, 0.08))
    assert pl.added == ["T"] and pl.word[:3] == [mp.word[0]] * 3 and pl.word[3:] == [mp.word[3]] * 4


def test_a_long_enough_consonant_or_a_floor_of_zero_changes_nothing():
    phones = ["AA", "T", "S", "IY", "N"]
    start = np.array([0.0, 0.15, 0.22, 0.32, 0.5])
    end = np.array([0.15, 0.22, 0.32, 0.5, 0.55])
    rule = np.array([0.15, 0.08, 0.1, 0.15, 0.1])
    s, e, n, added = N.floor_consonants(phones, start, end, rule)
    assert n == 0 and added == 0.0 and np.array_equal(s, start) and np.array_equal(e, end)
    short = end.copy()
    short[1] = 0.17
    s, e, n, _ = N.floor_consonants(phones, start, short, rule, share=0.0)
    assert n == 0 and np.array_equal(e, short)


def test_the_tracks_stay_with_their_phones_when_the_floor_moves_them():
    fr = N.frame_seconds()
    mp = N.ModelPhones()
    m = 40
    # pitch 100 Hz in the first phone, 200 Hz in the second, a sentence piece over both
    mp.f0 = np.where(np.arange(m) < 20, 100.0, 200.0)
    mp.voiced = np.arange(m) < 30
    mp.energy_db = np.where(np.arange(m) < 20, -20.0, -30.0)
    mp.pieces = [(0.0, 40 * fr)]
    start, end = np.array([0.0, 20 * fr]), np.array([20 * fr, 40 * fr])
    new_start, new_end = np.array([0.0, 10 * fr]), np.array([10 * fr, 44 * fr])
    out = N.warp_tracks(mp, start, end, new_start, new_end, fr)
    assert len(out.f0) == m + 4 and len(out.voiced) == m + 4 and len(out.energy_db) == m + 4
    assert np.all(out.f0[:10] < 150) and np.all(out.f0[11:] > 150)
    assert out.energy_db[5] == pytest.approx(-20.0) and out.energy_db[30] == pytest.approx(-30.0)
    # the voicing stops halfway into the second phone, as before
    assert out.voiced[:27].all() and not out.voiced[27:].any()
    assert out.pieces == [(0.0, pytest.approx(44 * fr))]


def test_a_single_frame_pitch_spike_is_replaced_by_the_median_of_three():
    f0 = 100.0 * 2 ** (np.arange(20) * 0.2 / 12)  # a rise of 0.2 semitones a frame
    voiced = np.ones(20, bool)
    voiced[12:14] = False
    spiky = f0.copy()
    spiky[5] *= 2 ** (4 / 12)  # 4 semitones up, one frame
    spiky[11] *= 2 ** (-3 / 12)  # the last voiced frame before a gap, 3 semitones down
    out, n = N.smooth_spikes(spiky, voiced, spike=2.0)
    assert n == 2
    assert out[5] == pytest.approx(f0[6], rel=1e-9)  # the median of 4.8, 1.0 + 4, 1.2 semitones
    assert out[11] == pytest.approx(f0[9], rel=1e-9)  # of the frames 9 and 10 inside and itself, the middle
    others = [k for k in range(20) if k not in (5, 11)]
    assert np.allclose(out[others], spiky[others])


def test_a_smooth_contour_and_the_unvoiced_frames_pass_the_smoother_unchanged():
    f0 = 120.0 * 2 ** (np.sin(np.arange(60) / 6.0) * 3 / 12)  # +-3 semitones, at most 0.5 a frame
    voiced = np.ones(60, bool)
    voiced[20:30] = False
    f0[25] = 400.0  # unvoiced: not the model's pitch
    voiced[40] = True
    voiced[39] = voiced[41] = False  # a single voiced frame: no neighbours to judge it by
    f0[40] = 300.0
    out, n = N.smooth_spikes(f0, voiced, spike=2.0)
    assert n == 0 and np.array_equal(out, f0)
    spiky = f0.copy()
    spiky[10] *= 2.0
    out, n = N.smooth_spikes(spiky, voiced, spike=0.0)
    assert n == 0 and np.array_equal(out, spiky)


def _schedule_long_enough(floor):
    """The fixture of `_schedule` with every consonant at least 0.85 of its rule duration."""
    symbols = ["<bos>", "ð", "ə", "<sp>", "k", "æ", "t", ",", "<sp>", "s", "æ", "t", ".", "<eos>"]
    durations = [20, 5, 6, 0, 7, 12, 8, 25, 0, 9, 14, 8, 0, 3]
    n = sum(durations)
    run = _run(symbols, durations, np.linspace(130, 90, n))
    run["voiced"] = np.array([symbols[k] in ("ð", "ə", "æ") for k in np.repeat(np.arange(len(symbols)), durations)])
    run["energy"] = np.linspace(1.0, -1.0, n)
    engine = ["DH", "AX", "K", "AE", "T", "S", "AE", "T"]
    rule = np.array([0.05, 0.06, 0.07, 0.12, 0.08, 0.1, 0.13, 0.08])
    return N.build_schedule("The cat, sat.", N.lay_out([run]), engine, rule, floor=floor)


def test_a_schedule_without_short_consonants_passes_through_unchanged():
    (on, rep_on), (off, rep_off) = _schedule_long_enough(N.CONSONANT_FLOOR), _schedule_long_enough(0.0)
    assert rep_on["floored"] == 0 and rep_on["floor_added_s"] == 0.0 and rep_on["spikes"] == 0
    assert on == off


def test_the_floor_in_the_schedule_takes_from_the_vowels_and_keeps_the_pitch_with_its_phones():
    (sched, rep), fr, n = _schedule()
    (bare, _) = N.build_schedule("The cat, sat.", *_schedule_inputs(), floor=0.0)
    on = {k: p for k, p in enumerate(sched["phones"])}
    off = {k: p for k, p in enumerate(bare["phones"])}
    d = lambda p: p["end"] - p["start"]  # noqa: E731
    # the K (5 frames, 53 ms of 70) is given 85 % of its rule duration, the vowel after it gives the time
    assert d(on[2]) == pytest.approx(0.85 * 0.07, abs=1e-4) and d(off[2]) < 0.85 * 0.07
    assert d(on[3]) < d(off[3])
    # each phone keeps its pitch: the K's middle sounds where it did
    assert on[2]["pitch"][2] == pytest.approx(off[2]["pitch"][2], abs=1.5)



def test_the_pitch_is_bridged_through_unvoiced_frames():
    (sched, _), fr, n = _schedule()
    hz, v = np.array(sched["pitch"]["hz"]), np.array(sched["pitch"]["voiced"], bool)
    assert not v.all() and v.any()
    # a line through the gap: no jump where the voicing stops
    assert np.max(np.abs(np.diff(12 * np.log2(hz)))) < 0.5


def test_dizione_is_taken_out_of_its_tarball_and_nothing_else():
    root = f"dizione-{N.DIZIONE_COMMIT}/"
    files = {root + "src/dizione/__init__.py": b"VERSION = 4\n", root + "src/dizione/g2p.py": b"x = 1\n",
             root + "src/dizione/data/units.txt": b"m metre\n", root + "LICENSE": b"Apache\n",
             root + "scripts/build.sh": b"rm -rf /\n", root + "src/dizione/built/x.npz": b"..",
             "elsewhere/src/dizione/evil.py": b"import os\n"}  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "d.tar.gz"
        with tarfile.open(tgz, "w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        out = Path(tmp) / "out"
        N._unpack_dizione(tgz, out)
        got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        assert got == ["LICENSE", "__init__.py", "data/units.txt", "g2p.py"]
        code, tables = N.dizione_hashes(out)
        assert len(code) == 64 and len(tables) == 64 and code != tables


def test_say_follows_a_schedule_of_this_shape():
    say = HERE / "bin" / "say"
    if not say.exists():
        pytest.skip("bin/say not built")
    (sched, _), _, _ = _schedule()
    with tempfile.TemporaryDirectory() as tmp:
        phones = N.engine_phones("The cat, sat.", say)[0]
        if phones != [p["phone"] for p in sched["phones"]]:
            pytest.skip(f"the engine says {phones} for the text")
        r = Pr.render(sched, Path(tmp) / "x.wav", pitch="voice")
    assert "followed: 8 phones" in r["followed"]
    spans = r["spans"]
    want = np.array([p["end"] - p["start"] for p in sched["phones"]])
    assert np.median(np.abs((spans[:, 1] - spans[:, 0]) - want)) < 0.02


def test_tags_are_taken_out_of_the_model_text_and_phonemes_refused():
    given = "I was so tired. [sigh] Well, um, [clear-throat] go home."
    text, tags = N.split_tags(given)
    assert text == "I was so tired. Well, um, go home." and tags == [given.index("[sigh"), given.index("[clear")]
    assert N.split_tags("No tags, uh, here.") == ("No tags, uh, here.", [])
    with pytest.raises(ValueError, match="bracketed phonemes"):
        N.split_tags("Say [HH AX L OW1] now.")


def test_vocal_effort_tags_are_out_of_the_model_text_and_hold_no_pause():
    given = "Wait. [whisper] I heard it. [shout] Run! [sigh] [tired] Oh. [normal] Gone."
    text, tags = N.split_tags(given)
    # the model reads the words; only the vocalization keeps a place for its pause
    assert text == "Wait. I heard it. Run! Oh. Gone."
    assert tags == [given.index("[sigh")]
    for name in N.EFFORTS:
        assert N.split_tags(f"One [{name}] two.") == ("One two.", [])


def test_a_pause_held_open_moves_what_follows_and_the_tracks_with_it():
    fr = N.frame_seconds()
    mp = N.ModelPhones()
    mp.f0, mp.voiced, mp.energy_db = np.full(30, 120.0), np.ones(30, bool), np.zeros(30)
    start, end = np.array([0, 10, 20]) * fr, np.array([10, 20, 30]) * fr
    s2, e2, mp2 = N.open_gaps(mp, start, end, {1: 5.5 * fr})
    assert np.allclose(s2[:2], start[:2]) and np.allclose(e2[:2], end[:2])
    assert s2[2] - e2[1] == pytest.approx(6 * fr)  # whole frames, at least what was asked
    assert len(mp2.f0) == 36 and (~mp2.voiced).sum() == 6 and np.all(mp2.f0 == 120.0)
    assert np.all(mp2.energy_db[~mp2.voiced] < -90)
    # a pause already long enough is left alone, and so is the last phone
    s3, _, mp3 = N.open_gaps(mp, start, end, {1: 0.0, 2: 1.0})
    assert np.allclose(s3, start) and len(mp3.f0) == 30


def test_sentence_ends_are_found_and_their_pauses_can_be_lengthened():
    fr = N.frame_seconds()
    runs = [_run(["<bos>", "s", "iː", ".", "<eos>"], [2, 8, 10, 0, 3]), _run(["<bos>", "t", "uː", ".", "<eos>"], [25, 5, 12, 0, 2])]
    mp = N.lay_out(runs)
    engine, rule = ["S", "IY", "T", "UW"], np.array([0.1, 0.12, 0.06, 0.15])
    (s0, _) = N.build_schedule("See. Two.", mp, engine, rule)
    gap0 = s0["phones"][2]["start"] - s0["phones"][1]["end"]
    assert gap0 == pytest.approx(28 * fr, abs=1e-3)
    assert N.sentence_ends(mp, [p["start"] for p in s0["phones"]], [p["end"] for p in s0["phones"]]) == [1]
    (s1, rep) = N.build_schedule("See. Two.", mp, engine, rule, sentence_pause=0.5)
    gap1 = s1["phones"][2]["start"] - s1["phones"][1]["end"]
    assert 0.5 <= gap1 < 0.5 + fr + 1e-3 and rep["held_open_s"] == pytest.approx(gap1 - gap0, abs=1e-3)
    assert len(s1["pitch"]["hz"]) == len(s0["pitch"]["hz"]) + round((gap1 - gap0) / fr)


def test_a_vocalization_keeps_the_rules_pause():
    say = HERE / "bin" / "say"
    if not say.exists():
        pytest.skip("bin/say not built")
    text = "I was so tired. [sigh] Well, go home."
    model_text, tags = N.split_tags(text)
    phones, spans = N.engine_spans(text, say, "physical")
    gaps = N.vocal_gaps(text, tags, phones, spans, say)
    n = len(N.engine_spans("I was so tired.", say)[0])
    assert list(gaps) == [n - 1] and gaps[n - 1] > 1.0  # the sigh, in the rules' pause
