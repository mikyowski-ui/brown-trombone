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


def test_params_file_matches_brief():
    params = T.load_params(HERE / "params.json")
    # Every field of the engine's `Tuning`, in its passes (a: how sounds are
    # made, b: how they follow one another, c: vowel lengths).
    assert sum(p["pass"] == "a" for p in params.values()) == 31
    assert sum(p["pass"] == "b" for p in params.values()) == 36
    assert sum(p["pass"] == "c" for p in params.values()) == 4
    assert len(params) == 71
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
