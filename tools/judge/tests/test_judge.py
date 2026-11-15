"""Model-free parts of ``judge.py`` and ``tune.py`` (no GPU, no downloads)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import judge as J  # noqa: E402
import tune as T  # noqa: E402


def test_word_hits():
    assert J.word_hits("the cat sat".split(), "the cat sat".split()) == (3, 0)
    assert J.word_hits("the cat sat".split(), "a cat sat down".split()) == (2, 2)
    assert J.word_hits(["bat"], []) == (0, 1)


def test_fast_material_is_the_four_listening_sentences():
    items = J.bench_items()
    assert [i.set for i in items] == ["bench"] * 4 + ["bench_clip"]
    # The clip is the four sentences one after another, text and phonemes.
    assert items[-1].ortho == " ".join(i.ortho for i in items[:4])
    assert items[-1].expected == sum((i.expected for i in items[:4]), ())
    # Every expected phoneme is one the scoring alphabet knows.
    for i in items:
        assert set(i.expected) <= set(J.A.SCORING_SYMBOLS), i.id
    # The three that are also tuning sentences agree with the material files.
    known = {i.ortho: i.expected for i in J.material.load(["sentence_tune"])}
    assert sum(known.get(i.ortho) == i.expected for i in items[:4]) == 3


def test_params_file_matches_brief():
    params = T.load_params(HERE / "params.json")
    # Every field of the engine's `Tuning`, in its passes (a: how sounds are
    # made, b: how they follow one another, c: vowel lengths).
    assert sum(p["pass"] == "a" for p in params.values()) == 58
    assert sum(p["pass"] == "b" for p in params.values()) == 53
    assert sum(p["pass"] == "c" for p in params.values()) == 11
    assert len(params) == 122
    # The twelve the classic engine reads as well are not tuned by default.
    shared = {n for n, p in params.items() if p.get("classic")}
    assert len(shared) == 12 and {"vowel_length", "stop_length", "aspiration"} <= shared


def test_params_file_agrees_with_the_engine():
    """Defaults and passes as the engine has them, and the tuner's bounds
    inside the engine's own ranges (needs ``bin/say``)."""
    import subprocess

    import pytest
    import render

    if not render.say_bin().exists():
        pytest.skip("no bin/say")
    eng = json.loads(subprocess.run([str(render.say_bin()), "--dump-parameters"], capture_output=True,
                                    text=True, check=True).stdout)  # fmt: skip
    params = T.load_params(HERE / "params.json")
    assert list(params) == list(eng)
    for n, p in params.items():
        e = eng[n]
        assert abs(p["default"] - e["default"]) <= 1e-6 * max(1.0, abs(e["default"])), n
        assert p["pass"] == e["pass"], n
        assert e["lo"] - 1e-9 <= p["lo"] and p["hi"] <= e["hi"] + 1e-9, n


def test_tuner_never_scores_held_out_sets():
    class A:
        pass_ = "bc"
        with_classic = False

    params = T.load_params(HERE / "params.json")
    names = T.tunable(params, A, {"lead": 0.03})
    assert "lead" not in names and "vowel_hold" in names and "body_rate" in names
    assert "vowel_length" not in names and "abduction_time" not in names
    assert set(T.TUNE_SETS).isdisjoint(T.HELDOUT_SETS)
    import pytest

    with pytest.raises(SystemExit, match="held-out"):
        T.main(["--sets", "sentence_tune,sentence_heldout", "--pass", "a", "--out", "/nonexistent/x.json"])


def test_gate_folder_names_every_gate_binary(tmp_path):
    for b in T.GATE_BINARIES:
        (tmp_path / b).write_text("")
    assert [p.name for p in T.gate_binaries([tmp_path])] == [
        "physical_gates", "natural_gate", "sibilant_gate", "vowel_glide_gate"]  # fmt: skip
    # Round 4's winner failed a gate the loop skipped: only the slowest one
    # and determinism are skipped now.
    assert not any("in_any_voice" in g for g in T.GATE_SKIP)
    (tmp_path / "vowel_glide_gate").unlink()
    import pytest

    with pytest.raises(SystemExit, match="vowel_glide_gate"):
        T.gate_binaries([tmp_path])


def _texts(walls, rises, n=16):
    """Per-text naturalness records as ``natural.compare`` returns them,
    with only the level steps filled in."""
    return [{"id": f"t{k:02d}", "set": "human", "analysis": {
        "steps": {"walls_per_s": float(walls[k]), "p95_up": float(rises[k]), "p95_down": 1.0},
        "fricatives": [], "modulation": {"spectrum": None}}} for k in range(n)]  # fmt: skip


def test_paired_worse_tells_a_steady_rise_from_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(3.4, 0.8, 16)
    # Every text a little worse: beyond noise, though the change is small
    # beside the spread across texts.
    m, lo = T.paired_worse(base + 0.15, base)
    assert abs(m - 0.15) < 1e-9 and lo > 0
    # Better and worse by turns, nothing on average: within noise.
    m, lo = T.paired_worse(base + rng.normal(0, 0.3, 16), base)
    assert lo <= 0
    # Better: never barred.
    assert T.paired_worse(base - 0.5, base)[1] < 0


def test_natural_hold_bars_steeper_rises_and_more_walls():
    rng = np.random.default_rng(1)
    walls, rises = rng.normal(3.4, 0.8, 16), rng.normal(9.0, 1.0, 16)
    human = {"steps.walls_per_s": {"mean": 1.1, "sd": 0.5}, "steps.p95_up": {"mean": 6.0, "sd": 1.0}}
    base = _texts(walls, rises)
    assert T.natural_barred(_texts(walls, rises), base, human)[0] == []
    # Round 4's winner: walls 3.4 -> 4.0 a second, rises steeper in every
    # sentence; far beyond 4 sd of the person, where a clipped z saw nothing.
    bad, diffs = T.natural_barred(_texts(walls + 0.6, rises + 0.4), base, human)
    assert "natural:steps.walls_per_s" in bad and "natural:steps.p95_up" in bad
    assert abs(diffs["steps.walls_per_s"][0] - 0.6) < 1e-9
    # Fewer walls and gentler rises are welcome.
    assert T.natural_barred(_texts(walls - 0.6, rises - 0.4), base, human)[0] == []


def test_natural_excess_charges_every_measure_moved_away_unclipped():
    z0 = {"a": 5.0, "b": -1.0, "c": 0.5}
    assert T.natural_excess(z0, z0) == 0.0
    # 5 -> 6 sd counts in full (a clip at 4 hid it); moving closer to the
    # person earns nothing back on another measure.
    assert abs(T.natural_excess({"a": 6.0, "b": -0.5, "c": 0.5}, z0) - 1.0 / 3) < 1e-12
    assert abs(T.natural_excess({"a": -7.0, "b": -1.0, "c": 0.5}, z0) - 2.0 / 3) < 1e-12


def test_whisper_words_enter_the_objective():
    class A:
        ll_weight, prior_weight, word_weight = 0.1, 0.1, 0.5

    params = T.load_params(HERE / "params.json")
    ev = object.__new__(T.Evaluator)
    ev.space = T.Space(params, ["lead"], {})
    ev.args, ev.sets = A, ["sentence_tune", "carrier_tune"]

    def item(s, edits, hits=None):
        it = J.ItemScore("x", s, ["AA", "B"], ["AA", "B"], [], edits, -1.0)
        if hits is not None:
            it.word_hits, it.word_n = hits, 4
        return it

    scores = [item("sentence_tune", 1, 3), item("sentence_tune", 0, 1), item("carrier_tune", 0)]
    r = ev.objective(scores, ev.space.u0)
    assert abs(r["word_acc"] - 0.5) < 1e-12 and r["words"] == {"sentence_tune": 0.5}
    base = (0.25 + 0.0) / 2 + 0.1 * 1.0
    assert abs(r["objective"] - (base + 0.5 * 0.5)) < 1e-12
    A.word_weight = 0.0
    assert abs(ev.objective(scores, ev.space.u0)["objective"] - base) < 1e-12


def test_final_check_ranks_passing_candidates_best_first():
    rec = lambda j, u, g=None: {"objective": j, "u": [u], "gates_failed": g or [], "eval": u}  # noqa: E731
    rs = [rec(0.5, 0.1), rec(0.3, 0.2, ["pinch"]), rec(0.4, 0.3), rec(0.4, 0.3), rec(0.45, 0.4, ["natural:distance"]),
          rec(0.6, 0.5)]  # fmt: skip
    assert [r["eval"] for r in T.ranked(rs, 2)] == [0.3, 0.1]
    assert [r["eval"] for r in T.ranked(rs, 9)] == [0.3, 0.1, 0.5]


def test_exclude_names_parameters_of_the_engine():
    import pytest

    with pytest.raises(SystemExit, match="unknown"):
        T.main(["--pass", "a", "--exclude", "no_such_thing", "--out", "/nonexistent/x.json"])


def test_space_round_trip_and_penalty():
    params = T.load_params(HERE / "params.json")
    names = [n for n, p in params.items() if p["pass"] == "a"]
    space = T.Space(params, names, {"lead": 0.04})
    vals = space.values(space.u0)
    assert vals["lead"] == 0.04
    for n in names:
        assert abs(vals[n] - params[n]["default"]) < 1e-12
    assert space.penalty(space.u0) == 0.0
    hi = space.values(np.ones(len(names)) * 5)  # clipped to the box
    assert all(abs(hi[n] - params[n]["hi"]) < 1e-12 for n in names)


def test_profile_shape(tmp_path):
    p = tmp_path / "x.json"
    T.write_profile(p, {"lead": 0.0351234567, "jet_efficiency": 0.05})
    assert json.loads(p.read_text()) == {"tuning": {"lead": 0.0351235, "jet_efficiency": 0.05}}


def test_inserted_segments_count_only_what_was_not_asked_for():
    assert J.inserted_segments(["S", "L", "IH", "D"], ["S", "L", "IH", "D"]) == 0
    assert J.inserted_segments(["S", "L", "IH", "D"], ["S", "AA", "L", "IH", "D"]) == 1
    # A substitution is not an insertion; a deletion neither.
    assert J.inserted_segments(["S", "L", "IH", "D"], ["S", "R", "IH"]) == 0
    assert J.inserted_segments(["L"], ["L", "AA"]) == 1


def test_pooled_sums_give_the_same_distance_as_summarize():
    import natural as N

    rng = np.random.default_rng(2)
    rows = _texts(rng.normal(3, 1, 16), rng.normal(14, 2, 16))
    for k, r in enumerate(rows):
        r["dtw"] = {"mean": float(rng.normal(15, 1))}
        r["analysis"]["rate"] = {"syll_per_s": float(rng.normal(4, 0.5))}
        # Hisses pooled over every token of a group, two or none per text.
        r["analysis"]["fricatives"] = [
            {"group": "s z", "rise_ms": float(rng.normal(25, 5)), "fall_ms": None, "dur_ms": 80.0, "vs_vowel_db": -9.0,
             "centroid": 5000.0 + k, "flatness": 0.3, "prominence": 6.0, "width6": 900.0} for _ in range(k % 3)
        ]  # fmt: skip
    human = {"steps.walls_per_s": {"mean": 1.1, "sd": 0.5}, "steps.p95_up": {"mean": 12.0, "sd": 1.5},
             "rate.syll_per_s": {"mean": 4.5, "sd": 0.4}, "fric.s z.rise_ms": {"mean": 31.0, "sd": 8.0},
             "fric.s z.centroid": {"mean": 5800.0, "sd": 600.0}}  # fmt: skip
    S, C, hm, hs, _ = T.text_sums(rows, human)
    want = N.distance(N.summarize(rows), human, clip=None)[0]
    assert abs(float(T.sums_distance(S.sum(0), C.sum(0), hm, hs)) - want) < 1e-9
    sub = [rows[i] for i in (0, 0, 3, 5, 7)]
    S, C, hm, hs, _ = T.text_sums(sub, human)
    assert abs(float(T.sums_distance(S.sum(0), C.sum(0), hm, hs)) - N.distance(N.summarize(sub), human, clip=None)[0]) < 1e-9


def test_box_narrows_a_parameter_to_its_free_side():
    import pytest

    params = T.load_params(HERE / "params.json")
    space = T.Space(params, ["lead", "closure_floor", "body_rate"], {})
    lo, hi = space.box({"lead": [0.03, 0.036], "closure_floor": [0.022, 0.04]})
    p = params["lead"]
    assert lo[0] == pytest.approx((0.03 - p["lo"]) / (p["hi"] - p["lo"]))
    assert hi[0] == pytest.approx((0.036 - p["lo"]) / (p["hi"] - p["lo"]))
    assert hi[1] == 1.0 and (lo[2], hi[2]) == (0.0, 1.0)
    with pytest.raises(SystemExit, match="default"):
        space.box({"lead": [0.04, 0.05]})
