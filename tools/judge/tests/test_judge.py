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
    assert sum(p["pass"] == "a" for p in params.values()) == 14
    assert sum(p["pass"] == "b" for p in params.values()) == 16
    # Pass c: how long the physical engine holds its vowels (not tuned by
    # tune.py).
    assert sum(p["pass"] == "c" for p in params.values()) == 1
    assert len(params) == 31


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
