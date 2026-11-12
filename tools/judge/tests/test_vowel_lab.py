"""Model-free parts of the vowel lab and the vowel tuner."""

import vowel_lab as L
import vowel_tune as V


def test_targets_are_read_from_the_engine_source():
    t = L.targets()
    for k in V.KEYS:
        assert k in t and len(t[k]) == 3
        f1, f2, f3 = t[k]
        assert 200 < f1 < f2 < f3 < 4000, k
    # Diphthongs have a start and an end.
    assert all(k + "<" in t and k + ">" in t for k in ["EY", "AY", "OW", "AW", "OY"])


def test_splits_do_not_overlap():
    assert not set(L.TUNE_SETS) & set(L.HELDOUT_SETS)


def test_tuner_keeps_bounds_and_order():
    pub = {k: v for k, v in L.targets().items() if k in V.KEYS}
    assert V.allowed(pub, pub)
    assert not V.allowed(V.moved(pub, "IH", 0, 1.2), pub)  # past the F1 bound
    # IH's F1 may not rise above EH's.
    p = {k: list(v) for k, v in pub.items()}
    p["IH"][0] = p["EH"][0] * 1.01
    assert not V.allowed(p, pub)


def test_profile_carries_the_vowels():
    pub = {k: v for k, v in L.targets().items() if k in V.KEYS}
    prof = V.profile({"engine": "Physical"}, pub)
    assert prof["engine"] == "Physical" and set(prof["vowels"]) == set(V.KEYS)
