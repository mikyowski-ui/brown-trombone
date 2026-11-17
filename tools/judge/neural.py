"""Neural prosody: tiny-voca's timing, pitch and level for any text, as a
schedule the physical engine speaks with (``say --match-prosody``).

tiny-voca (Apache-2.0, https://github.com/n4ze3m/tiny-voca, weights at
https://huggingface.co/n4ze3m/tiny-voca) is a 4,537,291-parameter
FastSpeech-2-like English TTS model with two voices, F2 and M2. Its ONNX
graph takes dizione phoneme ids and stress and returns the waveform (24 kHz)
and each token's duration in frames of 256 samples (10.67 ms, the engine's
control block of 512 samples at 48 kHz). The pitch, voicing and energy it
predicts per frame are not outputs of the graph: this module taps them
inside it (``TAPS``) and cuts the graph off there (``prosody.onnx``, without the
acoustic transformer and the waveform head), which runs about four times
faster than the whole graph.

From the text:

* **phones** -- dizione's phonemes (IPA, one token per phone, the r-coloured
  vowels and syllabic consonants as one token) folded to the engine's
  ARPAbet (``arpabet.fold_token``) and aligned to the engine's own phones for
  the text (``align_phones``: edit distance, a cheap substitution between
  sounds of one kind). A run of engine phones the model has no token for
  takes their rule durations at the model's tempo from the phones either
  side; a model phone the engine does not say gives its time to the engine
  phone before it. Every phone gets at least ``prosody.FLOOR``.
* **pauses** -- the frames the model gives punctuation, word gaps and the
  utterance edges, where they are at least ``KEEP_PAUSE`` long; shorter ones
  go to the phones either side. The model reads one sentence per call; the
  sentences are joined on one clock, at least ``SENTENCE_PAUSE`` apart so that
  the engine's breath between them is drawn. Its creak at the end of a
  statement (``ProsodyConfig::final_fry``) is the engine's own and stays.
* **tags and fillers** -- a vocalization tag (``[sigh]``, ``[laugh]`` ...) is
  taken out of the text the model reads (it would read the word) and its
  pause, the vocalization's own length, held open at the rules' length
  (``vocal_gaps``); a filler ("um", "uh") is a word to the model and phones to
  the engine and keeps the model's timing. Bracketed phonemes are refused.
* **pitch** -- the model's F0 every frame; through the frames it calls
  unvoiced a line in semitones (``prosody.fill_pitch``).
* **level** -- the model's energy (log RMS of its recordings) as dB, each
  phone's mean against the mean of the vowels (``level_db`` in the schedule).
  With ``--breath energy`` each vowel's becomes its breath share
  (``prosody.db_breath``, within ``prosody.LEVEL_RANGE``) and the consonants take
  the line between the vowels either side. Not by default (``BREATH``): the
  rules' own level already follows the model's (measured on the 14
  sentences of the fast check, 123 vowels: correlation 0.68, 2.6 dB RMS
  apart, spread 2.9 dB against the model's 3.5), so the shares laid on top
  of it doubled the swing (5.0 dB, 3.1 dB RMS from the model's) and the
  phonemes were heard worse in every comparison made (held-out sentences
  18.0 against 27.3 %, 15.5 against 24.9 % at the speed 0.8).
* **speed** -- the model's own tempo is 23 % faster than the engine's rules
  (M2; 18 % for F2; vowels 30 % shorter), past what the engine's
  articulators keep up with; ``SPEED`` slows it to about the rules' pace.

    python prosody.py neural "Any text at all." --out s.json [--voice M2] [--speed 1.0]
    bin/say --engine physical --voice deep --match-prosody s.json "Any text at all."

Needs the ``neural`` dependency group (onnxruntime, onnx) and fetches the model
and dizione's source at pinned revisions into ``.cache/tiny-voca`` on first use
(``--model-dir`` or ``TINY_VOCA_DIR`` point elsewhere). Model-free parts are
tested in ``tests/test_neural.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402

# --------------------------------------------------------------------------
# The model and where it comes from
# --------------------------------------------------------------------------

#: The model's repository and the revision fetched (16 Nov 2026).
MODEL_REPO = "n4ze3m/tiny-voca"
MODEL_REVISION = "d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14"
#: The files fetched, with their SHA-256.
MODEL_FILES = {
    "config.json": "27bcbf322fc428581f0e959be1c141dd7751c61f4fed5eded326e779b215f69c",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "onnx/model.onnx": "fe6d918791a15b82e37465981d7f27a070e46b46635e5a3482fc8dff87a1ae46",
    "phonemizer/align.json": "864763dbd558a864950266b2c30627f1cad203e75062ee842ee63c653e823b75",
    "phonemizer/g2p.json": "c518fd839ea0109c0b2b4f778c17506bf0d7653557b8b8c97a4c8dd9883d4432",
    "phonemizer/g2p.npz": "eb3fd519ff00e72d850928d2fccb968adc36134bca2f3bda09de7e794c2ed01b",
    "phonemizer/homographs.json": "9c3a96fcdf79fa27d1b3686dc6a507c37e4f754966928608ee9ddb617ba39192",
    "phonemizer/lexicon.tsv": "3fc930345c6b3cabc533981c4b212fa309eec83913fcdbd42919338270d64b06",
    "phonemizer/LICENSE.cmudict": "bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a",
    "phonemizer/LICENSE.homograph-data": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}  # fmt: skip
#: dizione (Apache-2.0, https://github.com/n4ze3m/dizione), the model's
#: phonemizer, NumPy only. Its source at this commit is fetched and imported
#: from the cache, checked by the hashes of its code and tables (as
#: ``dizione.identity`` computes them).
DIZIONE_REPO = "n4ze3m/dizione"
DIZIONE_COMMIT = "8cd889e4de7be6bd1dd2e6891344a0729fa14dd4"
DIZIONE_CODE = "102db3045b4a24979e3619000e5817d202afa5f9902e929908b2e43d331f2734"
DIZIONE_TABLES = "fc40bd3e65ad2c5af834409c4c7f377fe04be144f4bc61737471324ff73c7d29"

#: Inside the graph (the torch.onnx export of ``tiny_voca.export.Exportable``):
#: the F0 in Hz after the pitch range, the shift and the 50 to 600 Hz clamp;
#: the voiced flag (vowel frames forced voiced); the normalised log RMS
#: energy.
TAPS = {"f0": "/Clip_7_output_0", "voiced": "/Cast_6_output_0", "energy": "/Gather_9_output_0"}
INPUTS = ("tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range")
#: The model's sample rate and frame (its config).
SAMPLE_RATE, HOP = 24_000, 256
#: Its energy is normalised as (ln RMS - mean) / scale.
ENERGY_MEAN, ENERGY_SCALE = -4.0, 2.0
#: The engine's control block: 512 samples at 48 kHz, the same 10.67 ms.
ENGINE_RATE, ENGINE_BLOCK = 48_000, 512
#: Most tokens the model reads in one call.
MAX_TOKENS = 512
#: A pause the model makes shorter than this goes to the phones either side,
#: s: the engine would make a phrase break of it.
KEEP_PAUSE = 0.05
#: Threads the model runs on (the model card's speed is for four).
THREADS = 4
#: The model's speaking rate asked for (1 = as it predicts, larger is
#: faster): at 1 its phones last 0.77 of the engine's rule durations (M2;
#: F2 0.82; median of the 14 sentences of the fast check), and the engine,
#: whose articulators are driven at most 1.43 times their speed to keep up
#: (``follow_speed_max``), was heard worse on the held-out sentences (phoneme
#: error 18.0 % at 1, 21.2 % at 0.9, 15.5 % at 0.8, the rules 15.5 %).
SPEED = 0.8
#: The least pause between two sentences, s. The model leaves 0.27 to
#: 0.32 s between sentences (the four test sentences, M2); the engine's
#: breath between them (``ProsodyConfig::breath_intake``, on since wave 3) is
#: kept in a followed pause only where 0.22 s are left before it
#: (``Utterance::follow_with``), so with the model's pauses it fell out (the
#: pauses as quiet as with ``--no-breath``, -58 to -70 dB against -37 to -46
#: with the breath). Held at 0.5 s the breath is drawn as by the rules
#: (whose pause is 0.55 s); Whisper and the phonemes heard the clip the same
#: (28 of 31 words; 9.1 against 8.1 % phoneme error, one phoneme).
SENTENCE_PAUSE = 0.5
#: Where the breath shares come from: ``"rules"`` (a share of 1, the rules'
#: level) or ``"energy"`` (the model's level, see the module).
BREATH = "rules"


def frame_seconds(hop: int = HOP, rate: int = SAMPLE_RATE) -> float:
    """One frame of the model, s (256 / 24000 = 10.67 ms)."""
    return hop / rate


def engine_blocks(seconds: float) -> float:
    """``seconds`` in the engine's control blocks (512 samples at 48 kHz)."""
    return seconds * ENGINE_RATE / ENGINE_BLOCK


def front_end_fingerprint(version: int, symbols: list[str]) -> str:
    """The fingerprint tiny-voca records for its front end
    (``tiny_voca.text.frontend_fingerprint`` with dizione): the token
    convention's version and the symbol inventory, nothing of dizione's
    build, so any dizione of version 4 with the same 76 symbols fits."""
    ident = {"frontend_version": version, "phonemizer": "dizione", "symbols": list(symbols)}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_dir() -> Path:
    return Path(os.environ.get("TINY_VOCA_DIR") or HERE / ".cache" / "tiny-voca")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _get(url: str, dest: Path) -> None:
    """An anonymous download (no token, a generic client)."""
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    tmp.replace(dest)


def dizione_hashes(pkg: Path) -> tuple[str, str]:
    """The hashes of a dizione package's code and tables, as
    ``dizione.identity`` computes them."""
    code, tables = hashlib.sha256(), hashlib.sha256()
    for p in sorted(pkg.glob("*.py")):
        code.update(p.name.encode())
        code.update(p.read_bytes())
    for p in sorted((pkg / "data").glob("*.txt")):
        tables.update(p.name.encode())
        tables.update(p.read_bytes())
    return code.hexdigest(), tables.hexdigest()


def _unpack_dizione(tgz: Path, out: Path) -> None:
    """dizione's package (its .py files and data tables) and licence out of
    the repository's tarball, nothing else."""
    root = f"dizione-{DIZIONE_COMMIT}/"
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            rel = m.name[len(root) :] if m.name.startswith(root) else ""
            keep = rel == "LICENSE" or re.fullmatch(r"src/dizione/(data/)?[a-z0-9_]+\.(py|txt)", rel)
            if not (m.isfile() and keep):
                continue
            dest = out / ("LICENSE" if rel == "LICENSE" else rel[len("src/dizione/") :])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())


def fetch(directory: Path | None = None, log=print) -> Path:
    """The model's files and dizione's source in ``directory``, fetched at the
    pinned revisions where missing, and checked."""
    d = Path(directory or model_dir())
    for name, sha in MODEL_FILES.items():
        p = d / name
        if p.exists() and _sha256(p) == sha:
            continue
        log(f"fetching {MODEL_REPO}/{name}")
        _get(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}", p)
        if _sha256(p) != sha:
            p.unlink()
            raise RuntimeError(f"{name}: not the file pinned (its SHA-256 differs)")
    pkg = d / "dizione"
    if not pkg.is_dir() or dizione_hashes(pkg) != (DIZIONE_CODE, DIZIONE_TABLES):
        log(f"fetching {DIZIONE_REPO} at {DIZIONE_COMMIT[:8]}")
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            tgz = Path(tmp) / "dizione.tar.gz"
            _get(f"https://codeload.github.com/{DIZIONE_REPO}/tar.gz/{DIZIONE_COMMIT}", tgz)
            out = Path(tmp) / "dizione"
            _unpack_dizione(tgz, out)
            if dizione_hashes(out) != (DIZIONE_CODE, DIZIONE_TABLES):
                raise RuntimeError("dizione: not the source pinned (its code or tables differ)")
            if pkg.exists():
                for p in sorted(pkg.rglob("*"), reverse=True):
                    p.unlink() if p.is_file() else p.rmdir()
                pkg.rmdir()
            out.replace(pkg)
    return d


def load_dizione(directory: Path):
    """dizione from the cache, reading the model's phonemizer data."""
    os.environ["DIZIONE_DATA"] = str(Path(directory) / "phonemizer")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import dizione

    return dizione


class TinyVoca:
    """The model's prosody: durations per token and F0, voicing and energy
    per frame, from the graph cut at ``TAPS``."""

    def __init__(self, directory: Path | None = None, threads: int = THREADS):
        import onnxruntime as ort

        self.dir = fetch(directory, log=lambda m: print(m, file=sys.stderr))
        self.config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.dizione = load_dizione(self.dir)
        have = front_end_fingerprint(self.dizione.VERSION, list(self.dizione.symbols.SYMBOLS))
        if have != self.config["phonemizer"]["fingerprint"] or tuple(self.dizione.symbols.SYMBOLS) != SYMBOLS:
            raise RuntimeError("dizione's tokens are not the ones the model was trained on")
        if self.config["sample_rate"] != SAMPLE_RATE or self.config["hop"] != HOP:
            raise RuntimeError("the model's frame is not 256 samples at 24 kHz")
        self.voices = list(self.config["voices"])
        cut = self.dir / "prosody.onnx"
        if not cut.exists():
            import onnx.utils

            onnx.utils.extract_model(str(self.dir / "onnx" / "model.onnx"), str(cut), list(INPUTS),
                                     ["durations", *TAPS.values()])  # fmt: skip
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(cut), so, providers=["CPUExecutionProvider"])

    def phonemize(self, text: str):
        return self.dizione.phonemize(text)

    def run(self, ph, voice: str = "M2", speed: float = 1.0, pitch_shift_st: float = 0.0,
            pitch_range: float = 1.0) -> dict:  # fmt: skip
        """One sentence's phonemes (``phonemize``) through the model."""
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4")
        one = lambda v: np.array([v], np.float32)  # noqa: E731
        feed = {
            "tokens": np.array([ph.ids], np.int64), "stress": np.array([ph.stress], np.int64),
            "voice": np.array([self.voices.index(voice)], np.int64), "speed": one(speed),
            "pitch_shift": one(pitch_shift_st * math.log(2) / 12), "pitch_range": one(pitch_range),
        }  # fmt: skip
        dur, f0, voiced, energy = self.session.run(["durations", *TAPS.values()], feed)
        return {"symbols": list(ph.symbols), "stress": list(ph.stress), "durations": dur[0].astype(int),
                "f0": f0[0].astype(float), "voiced": voiced[0] > 0.5, "energy": energy[0].astype(float)}  # fmt: skip


# --------------------------------------------------------------------------
# Text into sentences (one model call each)
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?…])([\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
_NOT_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no)|\b[A-Z])\.$", re.IGNORECASE)
_MARKS = tuple(".!?…,;:")


def sentences(text: str) -> list[str]:
    """The sentences of ``text`` (not after a title or an initial), each ending
    with a mark: without one the model lets the last word break off."""
    pieces = _SENTENCE_END.split(re.sub(r"\s+", " ", text).strip())
    # the split keeps the closing quotes as pieces of their own
    joined = [pieces[k] + (pieces[k + 1] if k + 1 < len(pieces) else "") for k in range(0, len(pieces), 2)]
    out, cur = [], ""
    for piece in joined:
        cur = f"{cur} {piece}" if cur else piece
        if not _NOT_AN_END.search(cur):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    out = [s for s in out if any(c.isalnum() for c in s)]
    return [s if s.rstrip("\"')]")[-1:] in _MARKS else s + "." for s in out]


# --------------------------------------------------------------------------
# The model's tokens on one clock
# --------------------------------------------------------------------------


@dataclass
class ModelPhones:
    """The model's phones of a text on one clock: each phone token folded
    to scoring symbols (one or two), each with its stretch, s; and the frame
    tracks."""

    symbols: list[str] = field(default_factory=list)  #: one scoring symbol per sub-phone
    token: list[str] = field(default_factory=list)  #: the model token it comes from
    start: list[float] = field(default_factory=list)
    end: list[float] = field(default_factory=list)
    f0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voiced: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    energy_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pieces: list[tuple[float, float]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


#: The model's 76 tokens (its ``vocab``, dizione's ``SYMBOLS``, in id order): 5
#: special ones, 12 marks, 26 consonants and 33 vowels.
SYMBOLS = (
    "<pad>", "<bos>", "<eos>", "<sp>", "<unk>", ",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…",
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "ɹ", "j", "w",
    "tʃ", "dʒ", "ɾ", "ʔ",
    "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː", "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ",
    "iə", "oː", "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩",
)  # fmt: skip
#: The tokens that are no phone: the utterance edges, the word gap, the
#: unknown word and the punctuation.
NOT_PHONES = frozenset(SYMBOLS[:17])

#: How a token the model writes for two phones shares its time out: the
#: vowel's part weighs this much against a consonant's 1 (as
#: ``prosody.SHARE_WEIGHT``).
VOWEL_WEIGHT = 1.5


def fold(token: str) -> tuple[str, ...]:
    """A dizione phone token -> the scoring symbols it stands for (the flap
    stays ``DX``, which matches the engine's T or D)."""
    return A.fold_token(token)


def energy_db(energy) -> np.ndarray:
    """The model's normalised energy as dB (20 log10 RMS)."""
    return (np.asarray(energy, float) * ENERGY_SCALE + ENERGY_MEAN) * 20.0 / math.log(10.0)


def lay_out(runs: list[dict], frame: float | None = None) -> ModelPhones:
    """The model's output for each sentence (``TinyVoca.run``) joined on one
    clock: every phone token's stretch from the frames before it, split into
    its scoring symbols (the vowel weighted ``VOWEL_WEIGHT``)."""
    frame = frame or frame_seconds()
    mp = ModelPhones()
    f0, voiced, energy = [], [], []
    t0 = 0
    for run in runs:
        d = np.asarray(run["durations"], int)
        ends = t0 + np.cumsum(d)
        starts = ends - d
        first = len(mp.start)
        for sym, a, b, n in zip(run["symbols"], starts, ends, d):
            parts = () if sym in NOT_PHONES else fold(sym)
            if n <= 0 or not parts:
                continue
            if A.OTHER in parts:
                mp.unknown.append(sym)
            w = np.array([VOWEL_WEIGHT if p in A.VOCALIC else 1.0 for p in parts])
            edges = a + np.concatenate([[0.0], np.cumsum(w)]) / w.sum() * (b - a)
            for p, e0, e1 in zip(parts, edges[:-1], edges[1:]):
                mp.symbols.append(p)
                mp.token.append(sym)
                mp.start.append(float(e0) * frame)
                mp.end.append(float(e1) * frame)
        if len(mp.start) > first:
            mp.pieces.append((mp.start[first], mp.end[-1]))
        f0.append(np.asarray(run["f0"], float))
        voiced.append(np.asarray(run["voiced"], bool))
        energy.append(energy_db(run["energy"]))
        t0 = int(ends[-1]) if len(ends) else t0
    if f0:
        mp.f0, mp.voiced, mp.energy_db = np.concatenate(f0), np.concatenate(voiced), np.concatenate(energy)
    return mp


# --------------------------------------------------------------------------
# The model's phones onto the engine's
# --------------------------------------------------------------------------

_KIND: dict[str, str] = {}
for _k, _group in (("V", A.VOWELS + A.DIPHTHONGS), ("S", A.STOPS + A.AFFRICATES + [A.FLAP]),
                   ("F", A.FRICATIVES + A.ASPIRATE), ("N", A.NASALS), ("A", A.APPROXIMANTS)):  # fmt: skip
    for _p in _group:
        _KIND[_p] = _k
#: Cost of a phone one side has and the other not.
GAP_COST = 0.6
#: Cost of one sound for another of the same kind (two vowels, two stops or
#: affricates, two fricatives, two nasals, two approximants), and of any
#: other pair: dearer than leaving one out on each side.
NEAR_COST, FAR_COST = 0.45, 1.3


def sub_cost(a: str, b: str) -> float:
    if A.same(a, b):
        return 0.0
    ka, kb = _KIND.get(a), _KIND.get(b)
    return NEAR_COST if ka is not None and ka == kb else FAR_COST


def align_phones(model: list[str], engine: list[str]) -> list[tuple[int | None, int | None]]:
    """The least-cost alignment of the model's scoring symbols to the
    engine's phones: pairs ``(i, j)``, ``(i, None)`` for a model phone the engine
    does not say, ``(None, j)`` for an engine phone the model has no token for."""
    n, m = len(model), len(engine)
    cost = np.array([[sub_cost(a, b) for b in engine] for a in model], float).reshape(n, m)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1) * GAP_COST
    d[0, :] = np.arange(m + 1) * GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + cost[i - 1, j - 1], d[i - 1, j] + GAP_COST, d[i, j - 1] + GAP_COST)
    ops: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(d[i, j] - (d[i - 1, j - 1] + cost[i - 1, j - 1])) < 1e-9:
            ops.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(d[i, j] - (d[i - 1, j] + GAP_COST)) < 1e-9:
            ops.append((i - 1, None))
            i -= 1
        else:
            ops.append((None, j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class Placed:
    """The engine's phones on the model's clock, and what the mapping did."""

    start: np.ndarray
    end: np.ndarray
    matched: int = 0
    substituted: list[tuple[str, str]] = field(default_factory=list)  #: (model's, engine's)
    dropped: list[str] = field(default_factory=list)  #: model phones the engine does not say
    added: list[str] = field(default_factory=list)  #: engine phones the model has no token for


def place(mp: ModelPhones, engine: list[str], rule) -> Placed:
    """Each engine phone's stretch on the model's clock (see the module): a
    matched phone the model's; a model phone the engine does not say goes to
    the engine phone it follows without a pause (else to the next); a run of
    engine phones the model has none for gets their rule durations at the
    model's tempo, taken from the phones either side (no more than half of
    them), or out of the pause it stands in (down to ``KEEP_PAUSE``)."""
    ops = align_phones(mp.symbols, engine)
    n = len(engine)
    rule = np.asarray(rule, float)
    own: list[list[int]] = [[] for _ in range(n)]
    res = Placed(np.full(n, np.nan), np.full(n, np.nan))
    for i, j in ops:
        if i is not None and j is not None:
            own[j].append(i)
            res.matched += 1
            if not A.same(mp.symbols[i], engine[j]):
                res.substituted.append((mp.symbols[i], engine[j]))
    for k, (i, j) in enumerate(ops):
        if i is None or j is not None:
            continue
        res.dropped.append(mp.symbols[i])
        before = next((jj for ii, jj in reversed(ops[:k]) if jj is not None and ii is not None), None)
        after = next((jj for ii, jj in ops[k + 1 :] if jj is not None and ii is not None), None)
        if before is not None and abs(max(mp.end[q] for q in own[before]) - mp.start[i]) < 1e-9:
            own[before].append(i)
        elif after is not None:
            own[after].append(i)
        elif before is not None:
            own[before].append(i)
    for j in range(n):
        if own[j]:
            res.start[j] = min(mp.start[i] for i in own[j])
            res.end[j] = max(mp.end[i] for i in own[j])
    have = ~np.isnan(res.start)
    if not have.any():
        raise ValueError("no phone of the model's matched one of the engine's")
    tempo = float(np.sum(res.end[have] - res.start[have]) / max(1e-9, float(np.sum(rule[have]))))
    j = 0
    while j < n:
        if have[j]:
            j += 1
            continue
        b = j
        while b < n and not have[b]:
            b += 1
        run = list(range(j, b))
        res.added += [engine[q] for q in run]
        left, right = (j - 1 if j > 0 else None), (b if b < n else None)
        need = tempo * float(np.sum(rule[run]))
        if left is not None and right is not None and res.start[right] - res.end[left] > 1e-9:
            # in a pause: its end, keeping KEEP_PAUSE of it, the rest from the next phone
            gap = res.start[right] - res.end[left]
            g = min(need, max(0.0, gap - KEEP_PAUSE))
            r = min(need - g, 0.5 * (res.end[right] - res.start[right]))
            a0, a1 = res.start[right] - g, res.start[right] + r
            res.start[right] = a1
        else:
            dl = res.end[left] - res.start[left] if left is not None else 0.0
            dr = res.end[right] - res.start[right] if right is not None else 0.0
            need = min(need, 0.5 * (dl + dr))
            tl = need * dl / (dl + dr) if dl + dr > 0 else 0.0
            t = res.end[left] if left is not None else res.start[right]
            a0, a1 = t - tl, t - tl + need
            if left is not None:
                res.end[left] = a0
            if right is not None:
                res.start[right] = a1
        w = rule[run] / max(1e-9, float(np.sum(rule[run])))
        edges = a0 + np.concatenate([[0.0], np.cumsum(w)]) * (a1 - a0)
        res.start[run], res.end[run] = edges[:-1], edges[1:]
        j = b
    return res


def close_short_pauses(start: np.ndarray, end: np.ndarray, keep: float = KEEP_PAUSE) -> int:
    """Pauses shorter than ``keep`` shared out between the phones either side,
    in place. Returns how many."""
    closed = 0
    for k in range(len(start) - 1):
        gap = start[k + 1] - end[k]
        if 1e-9 < gap < keep:
            mid = end[k] + 0.5 * gap
            end[k], start[k + 1] = mid, mid
            closed += 1
    return closed


# --------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------


def pitch_track(mp: ModelPhones) -> np.ndarray:
    """The model's F0 with its unvoiced frames bridged (``prosody.fill_pitch``)."""
    import prosody as Pr

    med = float(np.median(mp.f0[mp.voiced])) if mp.voiced.any() else 100.0
    return Pr.fill_pitch(np.where(mp.voiced, mp.f0, 0.0), med)


def breath_shares(phones: list[str], start, end, db, hop: float) -> tuple[np.ndarray, np.ndarray]:
    """Each phone's level (dB against the mean of the vowels,
    ``prosody.phone_levels``) and breath share: a vowel's from its own level
    (within ``prosody.LEVEL_RANGE``), a consonant's from the line through the
    vowels' levels at its middle (its own level is the consonant's, not the
    lungs')."""
    import prosody as Pr

    start, end = np.asarray(start, float), np.asarray(end, float)
    lv = Pr.phone_levels(np.asarray(db, float), start, end, phones, hop)
    mids = 0.5 * (start + end)
    vow = np.array([p in Pr.VOCALIC and bool(np.isfinite(v)) for p, v in zip(phones, lv)])
    if not vow.any():
        return lv, np.ones(len(phones))
    target = np.interp(mids, mids[vow], lv[vow])
    return lv, np.asarray(Pr.db_breath(np.clip(target, *Pr.LEVEL_RANGE)), float)


def sentence_ends(mp: ModelPhones, start, end) -> list[int]:
    """The engine phones each sentence but the last ends on: the one whose
    end is nearest the end of the model's sentence."""
    ends = np.asarray(end[:-1], float)
    return [int(np.argmin(np.abs(ends - pe))) for _, pe in mp.pieces[:-1]]


def open_gaps(mp: ModelPhones, start, end, want: dict[int, float], frame: float | None = None):
    """The pause after each phone ``k`` of ``want`` made at least ``want[k]`` seconds
    long, in whole frames: the phones after it move later and the frame
    tracks get unvoiced frames there (the pitch held, the level silent).
    Returns the new starts, ends and model tracks (a copy of ``mp``)."""
    frame = frame or frame_seconds()
    start, end = np.array(start, float), np.array(end, float)
    f0, voiced, db = list(mp.f0), list(mp.voiced), list(mp.energy_db)
    for k in sorted(want, reverse=True):
        if not 0 <= k < len(start) - 1:
            continue
        have = start[k + 1] - end[k]
        if want[k] <= have + 1e-9:
            continue
        n = int(math.ceil((want[k] - have) / frame - 1e-9))
        at = min(len(f0), max(0, int(round(0.5 * (end[k] + start[k + 1]) / frame))))
        hold = f0[at - 1] if at > 0 else (f0[0] if f0 else 100.0)
        f0[at:at] = [hold] * n
        voiced[at:at] = [False] * n
        db[at:at] = [-100.0] * n
        start[k + 1 :] += n * frame
        end[k + 1 :] += n * frame
    out = ModelPhones(mp.symbols, mp.token, mp.start, mp.end, np.array(f0, float), np.array(voiced, bool),
                      np.array(db, float), mp.pieces, mp.unknown)  # fmt: skip
    return start, end, out


def build_schedule(text: str, mp: ModelPhones, engine: list[str], rule, breath: bool = False,
                   meta: dict | None = None, gaps: dict[int, float] | None = None,
                   sentence_pause: float = 0.0) -> tuple[dict, dict]:  # fmt: skip
    """The schedule (``prosody.analyse_audio``'s format) of the engine's phones on
    the model's timing, pitch and level, and a report of the mapping;
    ``gaps`` (phone -> seconds) are pauses held open at least that long (a
    vocalization's, see ``vocal_gaps``), ``sentence_pause`` the least pause
    between two sentences (0: the model's)."""
    import prosody as Pr

    frame = frame_seconds()
    pl = place(mp, engine, rule)
    start, end = pl.start.copy(), pl.end.copy()
    closed = close_short_pauses(start, end)
    short = [(engine[k], round(float(end[k] - start[k]), 4)) for k in range(len(engine))
             if end[k] - start[k] < Pr.FLOOR - 1e-9]  # fmt: skip
    al = Pr.Aligned(list(engine), start, end, 0.5 * (start + end))
    Pr.enforce_floor(al)
    start, end = al.start, al.end
    want = dict(gaps or {})
    if sentence_pause > 0:
        for k in sentence_ends(mp, start, end):
            want[k] = max(want.get(k, 0.0), sentence_pause)
    opened = 0.0
    if want:
        total = float(end[-1])
        start, end, mp = open_gaps(mp, start, end, want, frame)
        opened = float(end[-1]) - total
    track = pitch_track(mp)
    t = np.arange(len(track)) * frame
    lv, br = breath_shares(engine, start, end, mp.energy_db, frame)
    phones_out = []
    for k, p in enumerate(engine):
        a, b = float(start[k]), float(end[k])
        m = (t >= a) & (t < b)
        phones_out.append({
            "phone": p, "start": round(a, 4), "end": round(b, 4),
            "level_db": round(float(lv[k]), 2) if np.isfinite(lv[k]) else 0.0,
            "breath": round(float(br[k]), 4) if breath else 1.0,
            "pitch": [round(float(np.interp(a + q * (b - a), t, track)), 1) for q in (0.1, 0.3, 0.5, 0.7, 0.9)],
            "voiced": round(float(mp.voiced[m].mean()), 2) if m.any() else 0.0,
        })  # fmt: skip
    pauses = [{"after": k, "start": phones_out[k]["end"], "end": phones_out[k + 1]["start"]}
              for k in range(len(engine) - 1) if phones_out[k + 1]["start"] - phones_out[k]["end"] > 1e-6]  # fmt: skip
    median = float(np.median(mp.f0[mp.voiced])) if mp.voiced.any() else 100.0
    report = {
        "phones": len(engine), "model_phones": len(mp.symbols), "matched": pl.matched,
        "substituted": pl.substituted, "dropped": pl.dropped, "added": pl.added,
        "below_floor": short, "pauses_closed": closed, "pauses": len(pauses), "unknown": mp.unknown,
        "held_open_s": round(opened, 4),
    }  # fmt: skip
    sched = {
        "format": "synthie-prosody-1",
        "text": text,
        "source": "tiny-voca",
        "text_source": "given",
        "pieces": [[round(a, 3), round(b, 3)] for a, b in mp.pieces],
        "phones": phones_out,
        "pauses": pauses,
        "median_hz": round(median, 2),
        "pitch": {"start": 0.0, "hop": frame, "hz": [round(float(h), 2) for h in track],
                  "voiced": [int(v) for v in mp.voiced]},  # fmt: skip
        "neural": {**(meta or {}), "mapping": {k: v for k, v in report.items() if k != "phones"}},
    }
    return sched, report


def engine_phones(text: str, say: Path | None = None) -> tuple[list[str], np.ndarray]:
    """The engine's phones for ``text`` and their rule durations (see
    ``engine_spans``)."""
    phones, spans = engine_spans(text, say)
    return phones, spans[:, 1] - spans[:, 0]


def engine_spans(text: str, say: Path | None = None, engine: str = "classic") -> tuple[list[str], np.ndarray]:
    """The engine's phones for ``text`` and their rule spans (s), from
    ``say --phone-times``, by default on the classic engine: its phones are the
    physical engine's (the same on all 25 sentences of the fast check and
    the tuning set, measured) and it renders in milliseconds; the durations
    only share time out where the model and the engine disagree. The
    physical engine's own spans (a render) are needed where the pauses
    matter: the classic engine leaves 0.18 s for a vocalization."""
    import render

    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "t.json"
        subprocess.run([str(say or render.say_bin()), "--engine", engine, "--voice", "deep", "--out",
                        str(Path(tmp) / "x.wav"), "--phone-times", str(times), text],
                       capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)  # fmt: skip
        return read_spans(times)


def read_phones(path: Path) -> tuple[list[str], np.ndarray]:
    """The engine's phones and their rule durations (see ``read_spans``)."""
    phones, spans = read_spans(path)
    return phones, spans[:, 1] - spans[:, 0]


def read_spans(path: Path) -> tuple[list[str], np.ndarray]:
    """The engine's phones and their rule spans as ``say --phone-times`` (and
    ``say --neural``) write them: ``{"phones": [...], "spans": [[a, b], ...]}``."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    spans = np.array(rec["spans"], float).reshape(-1, 2)
    if len(spans) != len(rec["phones"]):
        raise ValueError(f"{path}: {len(rec['phones'])} phones and {len(spans)} spans")
    return list(rec["phones"]), spans


#: The engine's tags for vocalizations (``vocal::Vocalization::name``): written
#: ``[sigh]`` in the text.
TAGS = ("sigh", "laugh", "chuckle", "clear-throat", "cough")
_BRACKET = re.compile(r"\[([^\]]*)\]?")


def split_tags(text: str) -> tuple[str, list[int]]:
    """The text the model reads (the vocalization tags taken out: it would
    read ``[sigh]`` as the word) and where each tag stood (its offset in
    ``text``). Bracketed phonemes (``[HH AX L OW1]``, the judge's probes) are
    refused: the model reads words."""
    tags, out, last = [], [], 0
    for m in _BRACKET.finditer(text):
        if m.group(1).strip() not in TAGS:
            raise ValueError(f"neural prosody reads words, not bracketed phonemes: {m.group(0)!r}")
        out.append(text[last : m.start()])
        tags.append(m.start())
        last = m.end()
    out.append(text[last:])
    return re.sub(r"\s+", " ", " ".join(out)).strip(), tags


def vocal_gaps(text: str, tags: list[int], engine: list[str], spans, say: Path | None = None) -> dict[int, float]:
    """For each tag, the engine phone it follows and the rules' pause there
    (the vocalization's own length and the pauses around it): the schedule
    holds that pause open, or the vocalization would be squeezed into the
    model's (``Utterance::follow`` gives a pause the speaker's length). The
    phone is found by the engine's phones for the text before the tag,
    which must be the first ones of the whole text's."""
    spans = np.asarray(spans, float)
    out: dict[int, float] = {}
    for pos in tags:
        prefix = text[:pos]
        if not any(c.isalnum() for c in _BRACKET.sub(" ", prefix)):
            continue
        n = len(engine_spans(prefix, say)[0])
        if engine_spans(prefix, say)[0] != engine[:n]:
            raise ValueError(f"the engine's phones before the tag at {pos} are not the text's first ones")
        if 0 < n < len(engine):
            out[n - 1] = max(out.get(n - 1, 0.0), float(spans[n, 0] - spans[n - 1, 1]))
    return out


def neural_schedule(model: TinyVoca, text: str, voice: str = "M2", speed: float = SPEED, pitch_shift_st: float = 0.0,
                    pitch_range: float = 1.0, breath: bool = BREATH == "energy",
                    phones: tuple[list[str], np.ndarray] | None = None, say: Path | None = None,
                    sentence_pause: float = SENTENCE_PAUSE) -> tuple[dict, dict]:  # fmt: skip
    """The schedule for ``text`` with the model's prosody, and a report (the
    mapping, the time the model took). ``phones``: the engine's phones and
    their rule spans, if known (``say --neural`` passes them). Fillers ("um")
    are words to the model and phones to the engine, and pass through with
    the model's timing; vocalization tags are taken out of the model's text
    and their pauses held open at the rules' length (``vocal_gaps``)."""
    t0 = time.perf_counter()
    model_text, tags = split_tags(text)
    runs = []
    for s in sentences(model_text):
        ph = model.phonemize(s)
        if len(ph.ids) > MAX_TOKENS:
            raise ValueError(f"a sentence of more than {MAX_TOKENS} tokens: {s[:40]!r}")
        runs.append(model.run(ph, voice, speed, pitch_shift_st, pitch_range))
    t_model = time.perf_counter() - t0
    mp = lay_out(runs)
    engine, spans = phones if phones is not None else engine_spans(text, say, "physical" if tags else "classic")
    rule = spans[:, 1] - spans[:, 0]
    gaps = vocal_gaps(text, tags, engine, spans, say) if tags else {}
    meta = {"model": f"{MODEL_REPO}@{MODEL_REVISION[:8]}", "dizione": DIZIONE_COMMIT[:8], "voice": voice,
            "speed": speed, "pitch_shift_st": pitch_shift_st, "pitch_range": pitch_range,
            "frame_s": frame_seconds(), "frames": int(len(mp.f0)), "sentences": len(runs),
            "tokens": [" ".join(r["symbols"]) for r in runs], "sentence_pause": sentence_pause,
            "vocal_gaps": {str(k): round(v, 4) for k, v in gaps.items()}}  # fmt: skip
    sched, report = build_schedule(text, mp, engine, rule, breath, meta, gaps, sentence_pause)
    v = mp.f0[mp.voiced]
    report.update({
        "model_s": t_model, "speech_s": len(mp.f0) * frame_seconds(), "total_s": time.perf_counter() - t0,
        "f0": {"median": float(np.median(v)) if len(v) else 0.0, "min": float(v.min()) if len(v) else 0.0,
               "max": float(v.max()) if len(v) else 0.0},
    })  # fmt: skip
    return sched, report


def cmd_neural(args) -> int:
    model = TinyVoca(Path(args.model_dir) if args.model_dir else None, args.threads)
    phones = read_spans(Path(args.phones)) if args.phones else None
    sched, rep = neural_schedule(model, args.text, args.voice, args.speed, args.pitch_shift, args.pitch_range,
                                 args.breath == "energy", phones, Path(args.say) if args.say else None,
                                 args.sentence_pause)  # fmt: skip
    Path(args.out).write_text(json.dumps(sched, indent=1, ensure_ascii=False), encoding="utf-8")
    f0 = rep["f0"]
    print(f"{args.out}: {rep['phones']} phones, {rep['pauses']} pauses, {rep['speech_s']:.2f} s of speech; the model "
          f"took {1000 * rep['model_s']:.0f} ms ({rep['speech_s'] / max(rep['model_s'], 1e-9):.0f}x real time), "
          f"the bridge {1000 * rep['total_s']:.0f} ms")  # fmt: skip
    print(f"  mapping: {rep['matched']} matched ({len(rep['substituted'])} as another sound), "
          f"{len(rep['dropped'])} model phones dropped, {len(rep['added'])} engine phones added, "
          f"{len(rep['below_floor'])} below the floor, {rep['pauses_closed']} short pauses closed, "
          f"{rep['held_open_s']:.2f} s of pauses held open")  # fmt: skip
    for a, b in rep["substituted"]:
        print(f"    the model's {a} for the engine's {b}")
    if rep["dropped"]:
        print(f"    dropped: {' '.join(rep['dropped'])}")
    if rep["added"]:
        print(f"    added: {' '.join(rep['added'])}")
    print(f"  pitch: median {f0['median']:.0f} Hz, {f0['min']:.0f} to {f0['max']:.0f} Hz")
    return 0


def add_parser(sub) -> None:
    """``prosody.py neural``."""
    n = sub.add_parser("neural", help="any text -> schedule JSON with tiny-voca's timing, pitch and level")
    n.add_argument("text")
    n.add_argument("--out", required=True)
    n.add_argument("--voice", choices=["F2", "M2"], default="M2", help="the model's voice (default: %(default)s)")
    n.add_argument("--speed", type=float, default=SPEED,
                   help="the model's speaking rate, 0.25 to 4, larger is faster (default: %(default)s, about the rules' pace)")
    n.add_argument("--pitch-shift", type=float, default=0.0, help="semitones added to the model's pitch")
    n.add_argument("--pitch-range", type=float, default=1.0, help="the model's pitch movement around its mean, scaled")
    n.add_argument("--breath", choices=["rules", "energy"], default=BREATH,
                   help="breath shares from the rules (1) or the model's energy (default: %(default)s)")
    n.add_argument("--sentence-pause", type=float, default=SENTENCE_PAUSE,
                   help="the least pause between two sentences, s, room for the engine's breath "
                   "(default: %(default)s; 0 keeps the model's own)")
    n.add_argument("--phones", help="the engine's phones and rule spans as JSON (as say --phone-times writes them)")
    n.add_argument("--say", help="the say binary for the engine's phones (default: bin/say, or SAY_BIN)")
    n.add_argument("--model-dir", help="where the model is (default: TINY_VOCA_DIR, else .cache/tiny-voca)")
    n.add_argument("--threads", type=int, default=THREADS)
