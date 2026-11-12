"""Vowel lab: how well the phoneme model hears each vowel, under a profile.

    uv run python vowel_lab.py eval --split tune                     # the engine as built
    uv run python vowel_lab.py eval --split heldout --profile p.json --name p
    uv run python vowel_lab.py eval --split tune --system espeak

Scores every vowel slot of the chosen material (``--split tune``: the
tuning split, ``heldout``: the held-out split, or ``--sets a,b``) and
prints, per vowel, the slot count, the vowel error rate and the mean
margin (CTC log P(intended) - log P(best other vowel), nats; see
``probes.py``), pooled over the split and per set.  In sentences every
vowel is a slot; in the carrier phrases only the vowel of the CVC.

The splits:

* **tune** -- ``sentence_tune`` (Harvard list 1), ``harvard_tune`` (lists 3
  and 4) and ``carrier_tune`` (every stressed vowel in four CVC frames in
  two carrier phrases, AX in three words).  The vowel tuner optimises on
  these.
* **heldout** -- ``sentence_heldout`` (list 2), ``harvard_heldout`` (lists 5
  and 6), ``carrier_heldout`` (other frames, other carriers) and ``human``
  (the 16 LibriSpeech transcripts).  Never tuned on.

A profile may carry a ``"vowels"`` object of F1/F2/F3 targets besides the
voice's own keys (see ``examples/say.rs``).  Renders go to a scratch
folder that is deleted afterwards; the GPU lock (``JUDGE_LOCK``, default
the scratchpad's ``judge.lock``) is held only while the model judges.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402
import judge as J  # noqa: E402
import material  # noqa: E402
import probes as P  # noqa: E402
import render  # noqa: E402

OUT_DIR = HERE / "out"
TUNE_SETS = ["sentence_tune", "harvard_tune", "carrier_tune"]
HELDOUT_SETS = ["sentence_heldout", "harvard_heldout", "carrier_heldout", "human"]
SPLITS = {"tune": TUNE_SETS, "heldout": HELDOUT_SETS}
#: The vowels in report order (AX is scored together with AH, as the
#: probes do).
VOWELS = material.VOWEL_TARGETS
LOCK = Path(os.environ.get("JUDGE_LOCK", "/tmp/claude-1000/-home-carlos/236d0868-08d3-484c-b5ac-f6e709946f58/scratchpad/judge.lock"))


@contextlib.contextmanager
def gpu_lock():
    """Hold the shared GPU lock for one batch of judging."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_profile(profile: dict, folder: Path) -> Path:
    """``profile`` as a JSON file named by its content."""
    text = json.dumps(profile, sort_keys=True)
    path = folder / f"p-{hashlib.sha1(text.encode()).hexdigest()[:12]}.json"
    path.write_text(text, encoding="utf-8")
    return path


class Lab:
    """Renders material under profiles and judges its vowel slots; the
    phoneme model is loaded once, on first use."""

    def __init__(self, jobs: int | None = None):
        self.pj: J.PhonemeJudge | None = None
        self.jobs = jobs
        self.items = {s: material.load([s]) for s in material.ALL_SETS if (material.MATERIAL_DIR / f"{s}.tsv").exists()}

    def close(self) -> None:
        if self.pj is not None:
            self.pj.close()
            self.pj = None

    def render_many(self, sets: list[str], profiles: list[dict | None], system: str = "physical"):
        """Render ``sets`` once per profile, all at once; returns
        ``(scratch folder, [[(set, items, rendered)] per profile])``."""
        scratch = render.scratch_dir("lab-")
        jobs = []
        for k, prof in enumerate(profiles):
            sub = scratch / f"c{k:03d}"
            sub.mkdir()
            path = write_profile(prof, sub) if prof is not None and system in render.ENGINE_SYSTEMS else None
            for s in sets:
                if system == "human" and s != "human":
                    continue
                jobs.append((k, s, path, sub))
        out: list[list] = [[] for _ in profiles]
        all_items = []
        for k, s, path, sub in jobs:
            for it in self.items[s]:
                all_items.append((k, s, it, path, sub))
        # One pool for everything: a whole batch of candidates renders in parallel.
        from concurrent.futures import ThreadPoolExecutor

        def one(job):
            k, s, it, path, sub = job
            if system == "human":
                return render.Rendered(it, render.ANCHOR_DIR / f"{it.id}.flac", None, True)
            r = render.render([it], system, path, jobs=1, cache=False, out_dir=sub)
            return r[0]

        with ThreadPoolExecutor(max_workers=self.jobs or os.cpu_count() or 4) as pool:
            done = list(pool.map(one, all_items))
        groups: dict[tuple[int, str], list] = defaultdict(list)
        for (k, s, it, _, _), r in zip(all_items, done):
            groups[(k, s)].append(r)
        for (k, s), rs in groups.items():
            out[k].append((s, [r.item for r in rs], rs))
        return scratch, out

    def slots(self, sets: list[str], profiles: list[dict | None], system: str = "physical", batch: int = 8):
        """Every vowel slot of ``sets`` under each profile: a list (per
        profile) of dicts with ``set item target margin correct heard
        token frame frontend_ok``."""
        scratch, rendered = self.render_many(sets, profiles, system)
        try:
            audios = [[[render.load_audio(r.path) for r in rs] for _, _, rs in per] for per in rendered]
            with gpu_lock():
                if self.pj is None:
                    self.pj = J.PhonemeJudge()
                results = []
                for per, auds in zip(rendered, audios):
                    rows = []
                    for (s, items, rs), au in zip(per, auds):
                        said = [r.engine_phones for r in rs] if system in render.ENGINE_SYSTEMS else None
                        sentence = s in material.SENTENCE_SETS
                        res = P.score_slots(self.pj, items, au, "vowels", sentence=sentence, batch_size=batch, intended=said)
                        for ir in res:
                            for sl in ir.slots:
                                rows.append(
                                    {
                                        "set": s,
                                        "item": sl.item,
                                        "slot": sl.slot,
                                        "target": sl.target,
                                        "margin": sl.margin,
                                        "correct": P.same_vowel(sl.heard, sl.target),
                                        "heard": sl.heard,
                                        "token": sl.token,
                                        "frame": sl.meta.get("frame", ""),
                                        "frontend_ok": sl.meta.get("frontend_ok", True),
                                    }
                                )
                    results.append(rows)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return results


def summarize(rows: list[dict], frontend_only: bool = False) -> dict:
    """Pooled and per-vowel error rate and mean margin."""
    rows = [r for r in rows if r["frontend_ok"] or not frontend_only]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["target"]].append(r)

    def stats(rs):
        if not rs:
            return {"n": 0, "err": None, "margin": None}
        return {
            "n": len(rs),
            "err": sum(not r["correct"] for r in rs) / len(rs),
            "margin": sum(r["margin"] for r in rs) / len(rs),
        }

    out = {"all": stats(rows), "vowels": {v: stats(by[v]) for v in sorted(by, key=lambda v: VOWELS.index(v) if v in VOWELS else 99)}}
    sets = sorted({r["set"] for r in rows})
    out["sets"] = {s: stats([r for r in rows if r["set"] == s]) for s in sets}
    return out


def heard_table(rows: list[dict], top: int = 4) -> dict[str, str]:
    """Per vowel: what was heard instead, most often first."""
    by: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r["correct"]:
            by[r["target"]][r["token"] or "-"] += 1
    return {v: " ".join(f"{t}:{n}" for t, n in sorted(c.items(), key=lambda x: -x[1])[:top]) for v, c in by.items()}


def fmt(x, pct=False):
    if x is None:
        return "  –  "
    return f"{100 * x:5.1f}" if pct else f"{x:+6.2f}"


def print_summary(name: str, rows: list[dict]) -> None:
    s = summarize(rows)
    f = summarize(rows, frontend_only=True)
    heard = heard_table(rows)
    print(f"== {name}: vowel error {fmt(s['all']['err'], True)} % on {s['all']['n']} slots, margin {fmt(s['all']['margin'])}"
          f" (front end right: {fmt(f['all']['err'], True)} % on {f['all']['n']})")  # fmt: skip
    for st, v in s["sets"].items():
        print(f"   {st:17} {v['n']:4d} slots  error {fmt(v['err'], True)} %  margin {fmt(v['margin'])}")
    for v, st in s["vowels"].items():
        print(f"   {v:3} {st['n']:4d}  error {fmt(st['err'], True)} %  margin {fmt(st['margin'])}   {heard.get(v, '')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="vowel error and margins under one profile")
    ev.add_argument("--split", choices=list(SPLITS), default=None)
    ev.add_argument("--sets", default=None, help="comma-separated sets (instead of --split)")
    ev.add_argument("--profile", type=Path, default=None)
    ev.add_argument("--system", default="physical", choices=render.SYSTEMS)
    ev.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    sets = args.sets.split(",") if args.sets else SPLITS[args.split or "tune"]
    prof = json.loads(args.profile.read_text()) if args.profile else None
    if prof is None and args.system == "physical":
        prof = {"engine": "Physical"}
    lab = Lab()
    t0 = time.perf_counter()
    rows = lab.slots(sets, [prof], args.system)[0]
    lab.close()
    name = args.name or (args.profile.stem if args.profile else args.system)
    print_summary(f"{name} on {','.join(sets)} ({time.perf_counter() - t0:.0f} s)", rows)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"lab-{name}-{args.split or 'sets'}.json"
    path.write_text(json.dumps({"sets": sets, "profile": prof, "rows": rows}, indent=0) + "\n")
    print(f"-> {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
