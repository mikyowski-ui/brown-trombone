"""say --export-lipsync: the mouth that speaks a text, beside its WAV (model-free; needs bin/say)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SAY = Path(__file__).resolve().parents[1] / "bin" / "say"
CHANNELS = ["jaw_open", "lip_close", "lip_funnel", "lip_pucker", "lip_press",
            "tongue_up", "tongue_back", "velum_open", "larynx_height", "glottal_open"]  # fmt: skip


def _say(tmp_path, *args, text="Papa. Mama. We fish for food."):
    if not SAY.exists():
        pytest.skip("bin/say not built")
    wav = tmp_path / "lip.wav"
    r = subprocess.run([str(SAY), "--engine", "physical", "--voice", "deep", *args, "--out", str(wav), text],
                       capture_output=True, text=True)  # fmt: skip
    if "--export-lipsync" in args and "visemes.json" not in r.stdout:
        pytest.skip("bin/say is older than --export-lipsync")
    assert r.returncode == 0, r.stderr
    return wav


def test_the_frames_are_aligned_to_the_wav(tmp_path):
    wav = _say(tmp_path, "--export-lipsync")
    x, sr = sf.read(str(wav), dtype="float32")
    d = json.loads((tmp_path / "lip.visemes.json").read_text())
    assert d["format"] == "synthie-visemes-1" and d["audio"] == "lip.wav"
    assert d["sample_rate"] == sr == 48000 and d["frame_samples"] == 512
    assert d["frame_rate"] == pytest.approx(93.75)
    assert d["channels"] == CHANNELS
    assert d["samples"] == len(x)
    f = np.array(d["frames"])
    # one frame for every 512 samples of the file, the first at its first
    assert f.shape == (-(-len(x) // 512), 10)
    assert f.min() >= 0.0 and f.max() <= 1.0
    # sealed lips are a quiet or a hummed stretch of the file, never an open vowel: where the lips
    # read sealed through three frames with the nose shut, the sound is far below the file's peak
    close, velum = f[:, 1], f[:, 7]
    level = np.array([np.abs(x[k * 512:(k + 1) * 512]).max() for k in range(len(f))])
    held = [k for k in range(1, len(f) - 1) if close[k - 1:k + 2].min() >= 0.98 and velum[k] < 0.05]
    assert len(held) >= 4
    assert max(level[k] for k in held) < 0.25 * level.max()
    # the two /m/ of "mama" are sealed with the nose open
    assert ((close >= 0.98) & (velum > 0.9)).sum() >= 6
    # and the jaw opens somewhere, the lips pucker for "we" and "food", the lip meets the teeth for the /f/
    assert f[:, 0].max() > 0.2 and f[:, 3].max() > 0.5 and f[:, 4].max() > 0.6


def test_exporting_the_frames_leaves_the_sound_alone(tmp_path):
    a = _say(tmp_path, "--export-lipsync").read_bytes()
    (tmp_path / "lip.visemes.json").unlink()
    b = _say(tmp_path).read_bytes()
    assert a == b
    assert not (tmp_path / "lip.visemes.json").exists()


def test_stereo_has_the_same_frames(tmp_path):
    _say(tmp_path, "--export-lipsync")
    mono = json.loads((tmp_path / "lip.visemes.json").read_text())
    _say(tmp_path, "--export-lipsync", "--stereo")
    stereo = json.loads((tmp_path / "lip.visemes.json").read_text())
    assert stereo["samples"] == mono["samples"]
    assert stereo["frames"] == mono["frames"]
