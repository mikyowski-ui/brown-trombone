"""Render material through a "system" into WAV files.

Systems:

* ``classic`` / ``physical`` -- the engine, via the prebuilt ``bin/say``
  (override with the ``SAY_BIN`` environment variable).  ``--profile`` is
  passed to the binary only when a profile is given.
* ``espeak``  -- eSpeak NG speaking the same phonemes (bracketed ARPAbet is
  turned into eSpeak phoneme input) or the same sentence.
* ``human``   -- the anchor recordings fetched by ``anchors.py``; nothing is
  rendered.

``say`` is run once per item, in parallel across cores.  Renders are cached
under ``out/render/`` by a hash of everything that decides the sound: the
binary, engine, voice, profile content and text.

    uv run python render.py --system physical --sets vcv,pair
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import arpabet as A  # noqa: E402
import material  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
RENDER_DIR = OUT_DIR / "render"
ANCHOR_DIR = HERE / ".cache" / "anchors"
SYSTEMS = ["classic", "physical", "espeak", "human"]
ENGINE_SYSTEMS = ["classic", "physical"]
DEFAULT_VOICE = "deep"
TARGET_SR = 16_000


@dataclass
class Rendered:
    item: material.Item
    path: Path
    #: Phonemes the engine says it speaks (first stdout line of ``say``),
    #: one per phoneme; ``None`` for systems that are not the engine.
    engine_phones: list[str] | None = None
    cached: bool = False


def say_bin() -> Path:
    return Path(os.environ.get("SAY_BIN") or HERE / "bin" / "say").resolve()


@lru_cache(maxsize=8)
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def file_digest(path: Path) -> str:
    st = path.stat()
    return _file_digest(str(path), st.st_mtime_ns, st.st_size)


def profile_digest(profile: Path | None) -> str:
    """Hash of a profile's *content* (key order and spacing ignored)."""
    if profile is None:
        return "none"
    data = json.loads(Path(profile).read_text(encoding="utf-8"))
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def say_command(
    engine: str, text: str, out: Path, voice: str = DEFAULT_VOICE, profile: Path | None = None
) -> list[str]:
    cmd = [str(say_bin()), "--engine", engine, "--voice", voice]
    if profile is not None:  # the flag is only passed when a profile is given
        cmd += ["--profile", str(profile)]
    return cmd + ["--out", str(out), text]


def _run_say(engine: str, text: str, out: Path, voice: str, profile: Path | None) -> str:
    res = subprocess.run(say_command(engine, text, out, voice, profile), capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"say failed ({res.returncode}) for {text!r}: {' | '.join(tail)}")
    lines = res.stdout.splitlines()
    return lines[0].strip() if lines else ""


def _render_one(
    item: material.Item,
    system: str,
    voice: str,
    profile: Path | None,
    dest: Path,
    key_base: str,
    use_cache: bool,
) -> Rendered:
    if system == "human":
        path = ANCHOR_DIR / f"{item.id}.flac"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `uv run python anchors.py fetch`")
        return Rendered(item, path, None, True)

    key = hashlib.sha1(f"{key_base}|{item.text}".encode()).hexdigest()[:12]
    wav = dest / f"{item.id}-{key}.wav"
    meta = wav.with_suffix(".txt")
    is_engine = system in ENGINE_SYSTEMS
    if use_cache and wav.exists() and (meta.exists() or not is_engine):
        line = meta.read_text(encoding="utf-8") if is_engine else None
        return Rendered(item, wav, A.parse_engine_line(line) if line is not None else None, True)

    tmp = wav.with_name(f".{wav.stem}.{os.getpid()}.tmp.wav")
    if is_engine:
        line = _run_say(system, item.text, tmp, voice, profile)
        meta.write_text(line, encoding="utf-8")
        os.replace(tmp, wav)
        return Rendered(item, wav, A.parse_engine_line(line), False)
    import espeak_util as E

    E.synth(item.text, tmp)
    os.replace(tmp, wav)
    return Rendered(item, wav, None, False)


def render(
    items: list[material.Item],
    system: str,
    profile: Path | str | None = None,
    voice: str = DEFAULT_VOICE,
    jobs: int | None = None,
    cache: bool = True,
    out_dir: Path | None = None,
) -> list[Rendered]:
    """Render ``items`` and return them in the same order.

    ``cache=False`` always re-renders (the tuner uses it with a throwaway
    ``out_dir``, so 300 candidates do not fill the disk).
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; try one of {', '.join(SYSTEMS)}")
    profile = Path(profile).resolve() if profile is not None else None
    if profile is not None and system not in ENGINE_SYSTEMS:
        raise ValueError("--profile only applies to the engine systems")
    if system in ENGINE_SYSTEMS:
        binary = say_bin()
        if not binary.exists():
            raise FileNotFoundError(f"renderer not found: {binary} (set SAY_BIN)")
        pd = profile_digest(profile)
        key_base = f"{file_digest(binary)}|{system}|{voice}|{pd}"
        sub = system if profile is None else f"{system}-{pd[:8]}"
    else:
        key_base, sub = system, system
    dest = Path(out_dir) if out_dir is not None else RENDER_DIR / sub
    if system != "human":
        dest.mkdir(parents=True, exist_ok=True)
    jobs = jobs or os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda it: _render_one(it, system, voice, profile, dest, key_base, cache), items))


def load_audio(path: Path | str, sr: int = TARGET_SR) -> np.ndarray:
    """Mono float32 audio at ``sr`` Hz."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if rate != sr:
        g = gcd(int(rate), sr)
        x = resample_poly(x, sr // g, int(rate) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def scratch_dir(prefix: str = "cand-") -> Path:
    """A throwaway render directory under ``out/tmp``."""
    base = OUT_DIR / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--profile", type=Path, default=None, help="voice profile JSON (engine systems only)")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--sets", default=None, help="comma-separated sets (default: all that apply)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    sets = args.sets.split(",") if args.sets else (["human"] if args.system == "human" else material.SYNTH_SETS)
    items = material.load(sets)
    t0 = time.perf_counter()
    done = render(items, args.system, args.profile, args.voice, args.jobs, cache=not args.no_cache)
    dt = time.perf_counter() - t0
    fresh = sum(1 for r in done if not r.cached)
    print(f"{len(done)} items ({fresh} rendered, {len(done) - fresh} cached) in {dt:.2f} s -> {done[0].path.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
