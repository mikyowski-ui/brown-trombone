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
    # made, b: how they follow one another, c: vowel hold).
    assert sum(p["pass"] == "a" for p in params.values()) == 29
    assert sum(p["pass"] == "b" for p in params.values()) == 33
    assert sum(p["pass"] == "c" for p in params.values()) == 1
    assert len(params) == 63
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


def test_gate_folder_names_both_gate_binaries(tmp_path):
    for b in T.GATE_BINARIES:
        (tmp_path / b).write_text("")
    assert [p.name for p in T.gate_binaries([tmp_path])] == ["physical_gates", "natural_gate"]


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
