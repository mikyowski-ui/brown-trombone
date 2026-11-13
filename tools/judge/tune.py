"""CMA-ES tuner for the physical engine's tuning parameters.

    uv run python tune.py --self-test
    uv run python tune.py --pass a --gates bin --budget 300 --seed 1 --out out/profile-a.json
    uv run python tune.py --pass bc --lock out/profile-a.json --gates bin \\
        --budget 300 --seed 1 --out out/profile-b.json

Search space: the parameters of the chosen passes in ``params.json`` (``a``,
``b``, ``c`` or several letters, minus any fixed by ``--lock``), each mapped
to 0..1 over its lo..hi range, so CMA-ES sees a unit box (hard bounds) and
starts at the defaults.

Objective (lower is better), averaged over the chosen material sets (by
default the tuning split ``sentence_tune``, ``harvard_tune``,
``carrier_tune``; the held-out sets are refused)::

    J = mean(PER) + ll_weight * mean(-CTC log-likelihood per phoneme)
        + word_weight * (1 - Whisper word accuracy on the tuning sentences)
        + prior_weight * mean(((x - default) / (hi - lo))^2)
        + natural_weight * mean_k max(0, |z_k| - |z_k default|)

where ``z_k`` are the naturalness measures of the candidate saying the 16
LibriSpeech texts against the person's (``natural.distance`` unclipped:
loudness, pitch, timing and hiss), so a candidate pays for every measure it
moves further from the person, however far off the measure already is.  On
top of that the candidate is barred, like a gate failure, when its level
walls per second, its level rises (95th percentile) or its naturalness
distance are worse than the defaults' beyond noise (a paired bootstrap over
the 16 texts: ``NATURAL_HOLD``).  Only the naturalness measures of those
texts are read, never their phoneme or word scores.

Every candidate is rendered by ``bin/say --engine physical --profile ...``
(all items of a whole CMA-ES generation in parallel across cores, into a
throwaway folder) and judged by the wav2vec2 phoneme model on the GPU, loaded
once.  Whisper is not used inside the loop; at the end the best profile is
scored by ``judge.py`` with Whisper on the same sets.  With
``--word-weight`` above 0 Whisper also judges every candidate's tuning
sentences inside the loop (loaded on the GPU next to the phoneme model).

With ``--final-check K`` the K best candidates that passed the loop's gates
and guard are run, best first, through every gate binary with nothing
skipped; the first that passes them all becomes ``--out``.

Every evaluation is appended to ``out/tune-<out stem>.jsonl`` (parameters,
objective, per-set PER and log-likelihood).  ``--resume`` replays that log:
CMA-ES with the same seed asks the same candidates, and the ones already in
the log are not evaluated again.  The best profile so far is rewritten to
``--out`` after every generation, in the shape the engine accepts:
``{"tuning": {name: value, ...}}`` (tuned + locked parameters).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import judge as J  # noqa: E402
import material  # noqa: E402
import render  # noqa: E402

OUT_DIR = HERE / "out"
SYSTEM = "physical"

# Physics gates as a constraint. A recognizer can be pleased by sounds no
# person makes (a hissing /f/ as loud as /s/, a voice slow to start), so a
# candidate must also pass the engine's own gate tests: `--gates` names the
# compiled test binary of `crates/synthie-trombone/tests/physical_gates.rs`,
# which reads the profile from the environment variable below. The slow
# gates and those no tuning value can affect are skipped inside the loop;
# run the whole suite on the final profile. `--gates bin` runs both
# `physical_gates` and `natural_gate` (a copy of each in `bin/`).
GATE_BINARIES = ["physical_gates", "natural_gate", "sibilant_gate", "vowel_glide_gate"]
GATE_ENV = "SYNTHIE_TROMBONE_PROFILE"
# Skipped in the loop: the slowest gate (about 70 s; `--final-check` runs
# it) and determinism, which no tuning value changes. Round 4 skipped more,
# and its best candidate failed one of them
# (`fricatives_..._in_any_voice`).
GATE_SKIP = ["no_held_note", "same_seed"]
# Added to the objective of a candidate that fails gates: a wall higher than
# any phoneme error rate, plus a slope by the number of gates failed so the
# search can find its way back.
GATE_WALL = 1.0
GATE_SLOPE = 0.05

#: Naturalness held to the defaults (higher is worse for each): a candidate
#: worse than the defaults on one of these beyond noise is barred like a
#: gate failure. Round 4's winner made the level walls 3.4 -> 4.0 a second
#: and its rises steeper in 16 of 16 sentences, unseen by a guard that
#: clipped each z at 4.
NATURAL_HOLD = ["steps.walls_per_s", "steps.p95_up"]
#: Resamples of the paired bootstrap over the naturalness texts.
NATURAL_BOOT = 400


#: Material the tuner never scores on: the held-out split.
HELDOUT_SETS = ["sentence_heldout", "harvard_heldout", "carrier_heldout", "human"]
TUNE_SETS = ["sentence_tune", "harvard_tune", "carrier_tune"]


def gate_binaries(paths: list[Path]) -> list[Path]:
    """The gate test binaries: a directory stands for every one of
    ``GATE_BINARIES`` in it."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found = [p / b for b in GATE_BINARIES if (p / b).exists()]
            if len(found) != len(GATE_BINARIES):
                raise SystemExit(f"{p}: needs {', '.join(GATE_BINARIES)}")
            out += found
        elif p.exists():
            out.append(p)
        else:
            raise SystemExit(f"gate binary not found: {p}")
    return [p.resolve() for p in out]


def run_gates(binaries: Path | list[Path], profile: Path, threads: int = 4, skip: list[str] | None = None) -> list[str]:
    """Names of the gate tests `profile` fails (empty when it passes)."""
    failed: list[str] = []
    for binary in [binaries] if isinstance(binaries, Path) else binaries:
        cmd = [str(binary), "--test-threads", str(threads)]
        for name in GATE_SKIP + (skip or []):
            cmd += ["--skip", name]
        env = dict(os.environ, **{GATE_ENV: str(profile.resolve())})
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        bad = [
            line.split()[1]
            for line in proc.stdout.splitlines()
            if line.startswith("test ") and line.rstrip().endswith("FAILED")
        ]
        if proc.returncode != 0 and not bad:
            bad = [f"{binary.name}-run-error"]
        failed += bad
    return failed


def per_text(per_item: list[dict], key: str) -> np.ndarray:
    """One naturalness measure per text (``natural.analyse`` key path), in
    text order; NaN where a text has none."""
    import natural as N

    vals = []
    for r in sorted(per_item, key=lambda r: r["id"]):
        v = N._get(r["analysis"], key)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals, float)


def paired_worse(cand: np.ndarray, base: np.ndarray, n_boot: int = NATURAL_BOOT, seed: int = 0) -> tuple[float, float]:
    """Mean of ``cand - base`` over the texts both have, and the lower end of
    its 95 % paired bootstrap interval: worse beyond noise when that is
    above 0 (for a measure where higher is worse)."""
    d = np.asarray(cand, float) - np.asarray(base, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5))


def distance_worse(cand: list[dict], base: list[dict], human: dict, n_boot: int = NATURAL_BOOT // 2,
                   seed: int = 0) -> tuple[float, float]:
    """The naturalness distance (``natural.distance``, unclipped) of the
    candidate minus the defaults', and the lower end of its 95 % paired
    bootstrap interval over the texts."""
    import natural as N

    cand = sorted(cand, key=lambda r: r["id"])
    base = sorted(base, key=lambda r: r["id"])
    dist = lambda rows: N.distance(N.summarize(rows), human, clip=None)[0]  # noqa: E731
    d0 = dist(cand) - dist(base)
    rng = np.random.default_rng(seed)
    n = len(cand)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs.append(dist([cand[i] for i in idx]) - dist([base[i] for i in idx]))
    return float(d0), float(np.percentile(bs, 2.5))


def natural_excess(z: dict, z0: dict) -> float:
    """Mean over the measures of how much further from the person a
    candidate is than the defaults: max(0, |z| - |z0|), unclipped; a
    measure only the defaults have counts 0, one only the candidate has
    counts its |z|."""
    keys = set(z) | set(z0)
    if not keys:
        return 0.0
    return float(np.mean([max(0.0, abs(z.get(k, 0.0)) - abs(z0.get(k, 0.0))) if k in z else 0.0 for k in keys]))


def natural_barred(cand: list[dict], base: list[dict], human: dict, n_boot: int = NATURAL_BOOT) -> tuple[list[str], dict]:
    """The ``NATURAL_HOLD`` measures (and the distance) on which ``cand`` is
    worse than ``base`` beyond noise, with every paired difference."""
    bad, diffs = [], {}
    for key in NATURAL_HOLD:
        m, lo = paired_worse(per_text(cand, key), per_text(base, key), n_boot)
        diffs[key] = [m, lo]
        if lo > 0:
            bad.append(f"natural:{key}")
    m, lo = distance_worse(cand, base, human, max(50, n_boot // 2))
    diffs["distance"] = [m, lo]
    if lo > 0:
        bad.append("natural:distance")
    return bad, diffs


@contextlib.contextmanager
def gpu_lock():
    """The shared GPU lock, held for one batch of judging (see ``vowel_lab``)."""
    import vowel_lab

    with vowel_lab.gpu_lock():
        yield


# --------------------------------------------------------------------------
# Parameters and profiles
# --------------------------------------------------------------------------


def load_params(path: Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # also accept [{"name": ..., ...}, ...]
        data = {d["name"]: d for d in data}
    for name, p in data.items():
        for k in ("lo", "hi", "default", "pass"):
            if k not in p:
                raise ValueError(f"{path}: {name} lacks {k!r}")
        if not p["lo"] < p["hi"] or not p["lo"] <= p["default"] <= p["hi"]:
            raise ValueError(f"{path}: {name} has a bad range {p}")
    return data


def tunable(params: dict[str, dict], args, lock: dict[str, float]) -> list[str]:
    """The parameters of the chosen passes, minus the locked ones and,
    unless ``--with-classic``, those the classic engine reads too
    (``"classic": true``): the classic engine is pinned bit for bit, so a
    tuned value of those could not become a default."""
    return [
        n
        for n, p in params.items()
        if p["pass"] in args.pass_ and n not in lock and (args.with_classic or not p.get("classic"))
    ]


def load_lock(path: Path | None, params: dict[str, dict]) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tuning = data.get("tuning", data)
    unknown = [k for k in tuning if k not in params]
    if unknown:
        raise ValueError(f"{path}: unknown parameters {unknown}")
    return {k: float(v) for k, v in tuning.items()}


def write_profile(path: Path, values: dict[str, float]) -> None:
    """Exactly ``{"tuning": {...}}``; atomic so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({"tuning": {k: float(f"{v:.6g}") for k, v in values.items()}}, indent=2) + "\n")
    os.replace(tmp, path)


class Space:
    """Unit box <-> parameter values."""

    def __init__(self, params: dict[str, dict], names: list[str], fixed: dict[str, float]):
        self.params, self.names, self.fixed = params, names, fixed
        self.lo = np.array([params[n]["lo"] for n in names], float)
        self.hi = np.array([params[n]["hi"] for n in names], float)
        self.default = np.array([params[n]["default"] for n in names], float)
        self.u0 = (self.default - self.lo) / (self.hi - self.lo)

    def values(self, u) -> dict[str, float]:
        x = self.lo + np.clip(np.asarray(u, float), 0, 1) * (self.hi - self.lo)
        out = dict(self.fixed)
        out.update({n: float(v) for n, v in zip(self.names, x)})
        return out

    def penalty(self, u) -> float:
        return float(np.mean((np.clip(np.asarray(u, float), 0, 1) - self.u0) ** 2))


def ukey(u) -> str:
    return ",".join(f"{float(v):.10f}" for v in u)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class Evaluator:
    def __init__(self, items: list[material.Item], space: Space, args):
        self.items, self.space, self.args = items, space, args
        order = J.SET_ORDER + [s for s in material.PROBE_SETS if s not in J.SET_ORDER]
        self.sets = sorted({it.set for it in items}, key=order.index)
        binary = render.say_bin()
        if not binary.exists():
            raise SystemExit(f"renderer not found: {binary} (set SAY_BIN)")
        #: The naturalness guard's texts and the person's analyses of them.
        self.nat_items: list[material.Item] = []
        self.humans: dict = {}
        #: The starting point's naturalness: its per-text analyses and its
        #: z per measure (set after evaluation 0, ``set_base``).
        self.nat0_items: list[dict] | None = None
        self.nat0_z: dict | None = None
        self.pool: ProcessPoolExecutor | None = None
        self.words = None
        with gpu_lock():
            self.judge = J.PhonemeJudge()
            if getattr(args, "word_weight", 0) > 0:
                self.words = J.WordJudge()
            if args.natural_weight > 0:
                import natural as N

                self.nat_items = N.reference_items(None)
                self.humans = N.human_reference(self.judge, None)
                self.human_summary = N.summarize(N.run_system(self.judge, "human", self.humans, None, "", None))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        if self.words is not None:
            self.words.close()
        self.judge.close()

    def set_base(self, rec: dict) -> None:
        """Hold every later candidate to the naturalness of ``rec`` (the
        starting point's record, with its ``natural_items``)."""
        self.nat0_items = rec.get("natural_items")
        self.nat0_z = rec.get("natural_z")

    def render_population(self, profiles: list[Path], scratch: Path) -> list[list[render.Rendered]]:
        """Every item (then every naturalness text) of every candidate."""
        every = self.items + self.nat_items
        jobs = []
        for k, prof in enumerate(profiles):
            dest = scratch / f"c{k:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            for it in every:
                jobs.append((k, it, prof, dest))
        out: list[list] = [[None] * len(every) for _ in profiles]
        index = {(it.set, it.id): n for n, it in enumerate(every)}

        def one(job):
            k, it, prof, dest = job
            return k, index[(it.set, it.id)], render._render_one(it, SYSTEM, self.args.voice, prof, dest, "", False)

        with ThreadPoolExecutor(max_workers=self.args.jobs or os.cpu_count() or 4) as pool:
            for k, n, r in pool.map(one, jobs):
                out[k][n] = r
        return out

    def objective(self, scores: list[J.ItemScore], u) -> dict:
        per, ll = {}, {}
        for s in self.sets:
            d = J.summarize_set([x for x in scores if x.set == s])
            per[s], ll[s] = d["per"], d["ll"]
        mper = float(np.mean(list(per.values())))
        mnll = float(np.mean([-v for v in ll.values()]))
        pen = self.space.penalty(u)
        obj = mper + self.args.ll_weight * mnll + self.args.prior_weight * pen
        out = {"objective": obj, "per": per, "ll": ll, "mean_per": mper, "mean_nll": mnll, "penalty": pen}
        wsc = [x for x in scores if x.word_n]
        if wsc:
            # Pooled over the sentences, as the scoreboard counts words.
            acc = sum(x.word_hits or 0 for x in wsc) / max(1, sum(x.word_n or 0 for x in wsc))
            out["word_acc"] = acc
            out["words"] = {s: J.summarize_set([x for x in wsc if x.set == s]).get("word_acc")
                            for s in self.sets if any(x.set == s for x in wsc)}  # fmt: skip
            out["objective"] += self.args.word_weight * (1.0 - acc)
        return out

    def natural_jobs(self, rendered: list[render.Rendered]) -> list:
        """Align one candidate's renders of the LibriSpeech texts (needs the
        model) and return the per-sentence comparisons to run on the CPU
        (``natural.compare``)."""
        import natural as N

        audios = [render.load_audio(r.path) for r in rendered]
        lps = self.judge.log_probs(audios, batch_size=2)
        jobs = []
        for it, au, lp in zip(self.nat_items, audios, lps):
            segs = N.align(self.judge, au, list(it.expected), lp)
            h = self.humans[it.id]
            jobs.append((it.id, it.set, au, segs, N.syllables(it.expected), h.audio, h.segs))
        return jobs

    def naturalness(self, per_item: list[dict]) -> dict:
        """The naturalness distance of one candidate (``natural.distance``,
        clipped as the reports print it, and unclipped) and its z per
        measure, unclipped."""
        import natural as N

        summary = N.summarize(per_item)
        d, _ = N.distance(summary, self.human_summary)
        du, zs = N.distance(summary, self.human_summary, clip=None)
        return {"distance": d, "distance_unclipped": du, "z": zs}

    def evaluate(self, us: list) -> list[dict]:
        scratch = render.scratch_dir("tune-")
        try:
            profiles = []
            for k, u in enumerate(us):
                p = scratch / f"profile-{k:02d}.json"
                write_profile(p, self.space.values(u))
                profiles.append(p)
            t0 = time.perf_counter()
            rendered = self.render_population(profiles, scratch)
            t1 = time.perf_counter()
            # The gates run on the CPU while the GPU judges.
            gates = None
            if self.args.gates:
                pool = ThreadPoolExecutor(max_workers=self.args.gate_jobs)
                gates = [
                    pool.submit(run_gates, self.args.gates, p, self.args.gate_threads, self.args.gate_skip)
                    for p in profiles
                ]
            out = []
            n = len(self.items)
            nat_jobs = []
            with gpu_lock():
                for u, rs in zip(us, rendered):
                    scores = J.score_phonemes(self.judge, rs[:n], batch_size=self.args.batch)
                    if self.words is not None:
                        J.score_words(self.words, rs[:n], scores, batch_size=self.args.batch)
                    out.append(self.objective(scores, u))
                    if self.nat_items:
                        nat_jobs.append(self.natural_jobs(rs[n:]))
            # The naturalness measures are plain numpy: worker processes,
            # with the GPU lock let go.
            if nat_jobs:
                import natural as N

                if self.pool is None:
                    ctx = multiprocessing.get_context("spawn")  # the parent holds CUDA: no fork
                    self.pool = ProcessPoolExecutor(max_workers=self.args.natural_jobs, mp_context=ctx)
                futs = [[self.pool.submit(N.compare, *job) for job in jobs] for jobs in nat_jobs]
                for r, fs in zip(out, futs):
                    items = [f.result() for f in fs]
                    nat = self.naturalness(items)
                    r["natural"] = nat["distance"]
                    r["natural_unclipped"] = nat["distance_unclipped"]
                    r["natural_z"] = nat["z"]
                    r["natural_items"] = items
                    if self.nat0_z is not None:
                        r["natural_penalty"] = natural_excess(nat["z"], self.nat0_z)
                        r["objective"] += self.args.natural_weight * r["natural_penalty"]
                    if self.nat0_items is not None and self.args.natural_hold:
                        bad, diffs = natural_barred(items, self.nat0_items, self.human_summary)
                        r["natural_vs_default"] = diffs
                        r["natural_barred"] = bad
            t2 = time.perf_counter()
            for k, r in enumerate(out):
                failed = (gates[k].result() if gates is not None else []) + r.get("natural_barred", [])
                if gates is not None or "natural_barred" in r:
                    r["gates_failed"] = failed
                if failed:
                    r["objective"] += GATE_WALL + GATE_SLOPE * len(failed)
            if gates is not None:
                pool.shutdown()
            t3 = time.perf_counter()
            for r in out:
                r["render_s"] = (t1 - t0) / len(us)
                r["judge_s"] = (t2 - t1) / len(us)
                r["gate_s"] = (t3 - t2) / len(us)
            return out
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# Log
# --------------------------------------------------------------------------


class Log:
    def __init__(self, path: Path, meta: dict, resume: bool):
        self.path = path
        self.cache: dict[str, dict] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            old = next((x for x in lines if x.get("kind") == "meta"), None)
            keys = ("params", "sets", "items", "seed", "popsize", "sigma0", "ll_weight", "prior_weight", "lock",
                    "natural_weight", "natural_hold", "word_weight", "binary")
            if old is not None:
                diff = [k for k in keys if old.get(k) != meta.get(k)]
                if diff:
                    raise SystemExit(f"--resume: {path} was written with different settings ({', '.join(diff)})")
            for x in lines:
                if x.get("kind") == "eval":
                    self.cache[ukey(x["u"])] = x
            print(f"[resume] {len(self.cache)} evaluations in {path.relative_to(HERE)}")
        else:
            if path.exists():
                bak = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
                os.replace(path, bak)
                print(f"[log] previous log kept as {bak.relative_to(HERE)}")
            self.append({"kind": "meta", **meta})

    def append(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def select_items(sets: list[str], subset: int | None) -> list[material.Item]:
    items = material.load(sets)
    if subset:
        # every k-th item of each set, so a small subset still spans it
        picked = []
        for s in sets:
            its = [it for it in items if it.set == s]
            step = max(1, len(its) // subset)
            picked += its[::step][:subset]
        items = picked
    return items


def run(args) -> dict:
    import cma

    params = load_params(args.params)
    lock = load_lock(args.lock, params)
    names = tunable(params, args, lock)
    excluded = [n for n in names if n in args.exclude]
    names = [n for n in names if n not in args.exclude]
    unknown = [n for n in args.exclude if n not in params]
    if unknown:
        raise SystemExit(f"--exclude: unknown parameters {', '.join(unknown)}")
    if excluded:
        print(f"[tune] left at their defaults: {', '.join(excluded)}")
    shadowed = [n for n, p in params.items() if p["pass"] in args.pass_ and n in lock]
    if shadowed:
        print(f"[warn] locked, so not tuned: {', '.join(shadowed)}")
    if not names:
        raise SystemExit(f"no parameters to tune in pass {args.pass_}")
    space = Space(params, names, lock)
    sets = args.sets.split(",")
    heldout = [x for x in sets if x in HELDOUT_SETS]
    if heldout:
        raise SystemExit(f"held-out sets are never tuned on: {', '.join(heldout)}")
    items = select_items(sets, args.subset)
    out = Path(args.out).resolve()
    log_path = OUT_DIR / f"tune-{out.stem}.jsonl"
    n = len(names)
    popsize = args.popsize or 4 + int(3 * math.log(n))
    meta = {
        "params": names,
        "sets": sets,
        "items": [it.id for it in items],
        "seed": args.seed,
        "popsize": popsize,
        "sigma0": args.sigma,
        "ll_weight": args.ll_weight,
        "prior_weight": args.prior_weight,
        "lock": lock,
        "natural_weight": args.natural_weight,
        "natural_hold": args.natural_hold,
        "word_weight": args.word_weight,
        "budget": args.budget,
        "binary": render.file_digest(render.say_bin()),
        "voice": args.voice,
        "gates": [str(g) for g in args.gates] if args.gates else None,
        "gate_skip": args.gate_skip,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log = Log(log_path, meta, args.resume)
    print(
        f"[tune] pass {args.pass_}: {n} parameters, {len(items)} items ({','.join(sets)}), "
        f"budget {args.budget}, popsize {popsize}, seed {args.seed} -> {out.relative_to(HERE) if out.is_relative_to(HERE) else out}"
    )

    ev = Evaluator(items, space, args)
    best = {"objective": math.inf}
    n_eval = 0
    t_start = time.perf_counter()
    fresh_seconds: list[float] = []

    def run_batch(us: list, gen: int) -> list[dict]:
        nonlocal n_eval
        results: list[dict | None] = [log.cache.get(ukey(u)) for u in us]
        todo = [k for k, r in enumerate(results) if r is None]
        if todo:
            t0 = time.perf_counter()
            fresh = ev.evaluate([us[k] for k in todo])
            dt = (time.perf_counter() - t0) / len(todo)
            fresh_seconds.extend([dt] * len(todo))
            for k, r in zip(todo, fresh):
                if gen != -1:
                    # Only the starting point keeps its per-text analyses
                    # (every later candidate is held to them).
                    r.pop("natural_items", None)
                rec = {
                    "kind": "eval",
                    "eval": None,
                    "gen": gen,
                    "u": [float(v) for v in us[k]],
                    "values": space.values(us[k]),
                    **r,
                    "seconds": dt,
                }
                results[k] = rec
        for k, r in enumerate(results):
            assert r is not None
            if k in todo:
                r["eval"] = n_eval
                log.append(r)
                log.cache[ukey(r["u"])] = r
            n_eval += 1
        return results  # type: ignore[return-value]

    # Baseline: the defaults (with the lock applied), evaluation 0.
    base = run_batch([space.u0.copy()], gen=-1)[0]
    ev.set_base(base)
    best = dict(base)
    write_profile(out, base["values"])
    print(f"[base] J {base['objective']:.4f}  PER {fmt_sets(base['per'])}  ll {fmt_sets(base['ll'], pct=False)}"
          f"  natural {base.get('natural', float('nan')):.3f}  gates failed {base.get('gates_failed')}")

    es = cma.CMAEvolutionStrategy(
        list(space.u0),
        args.sigma,
        {"bounds": [0, 1], "seed": args.seed, "popsize": popsize, "verbose": -9, "maxfevals": args.budget},
    )
    gen = 0
    while not es.stop() and n_eval - 1 < args.budget:
        us = es.ask()
        us = [np.clip(u, 0, 1) for u in us]
        res = run_batch(us, gen)
        es.tell(us, [r["objective"] for r in res])
        gbest = min(res, key=lambda r: r["objective"])
        if gbest["objective"] < best["objective"]:
            best = dict(gbest)
            write_profile(out, best["values"])
        el = time.perf_counter() - t_start
        barred = sum(1 for r in res if r.get("gates_failed"))
        print(
            f"[gen {gen:3d}] evals {n_eval:4d}  barred by gates {barred:2d}  gen best J {gbest['objective']:.4f}  "
            f"best J {best['objective']:.4f} (PER {fmt_sets(best['per'])}, natural "
            f"{best.get('natural', float('nan')):.3f})  {el:6.0f} s",
            flush=True,
        )
        gen += 1
    ev.close()
    del ev

    if args.final_check:
        chosen = final_check(log, args, out)
        if chosen is not None:
            best = chosen

    summary = {
        "evaluations": n_eval,
        "fresh_evaluations": len(fresh_seconds),
        "seconds_per_eval": float(np.mean(fresh_seconds)) if fresh_seconds else None,
        "wall_seconds": time.perf_counter() - t_start,
        "baseline": {k: base[k] for k in ("objective", "per", "ll")},
        "best": {k: best.get(k) for k in ("objective", "per", "ll", "eval", "natural", "gates_failed", "word_acc",
                                          "final_gates")},
        "profile": str(out),
        "log": str(log_path),
    }
    log.append({"kind": "summary", **summary})
    print(
        f"[done] {n_eval} evaluations ({len(fresh_seconds)} fresh), "
        f"{summary['seconds_per_eval'] or 0:.2f} s per fresh evaluation; "
        f"J {base['objective']:.4f} -> {best['objective']:.4f} (eval {best['eval']})"
    )
    print(f"       best profile -> {out}")
    print(f"       log          -> {log_path}")

    if args.final_whisper:
        print("[final] scoring the best profile with phonemes + Whisper on the tuning sets")
        with gpu_lock():
            J.score([SYSTEM], sets, profile=out, name=f"{SYSTEM}-{out.stem}", whisper=True, voice=args.voice)
    return summary


def ranked(records: list[dict], k: int) -> list[dict]:
    """The ``k`` best evaluations that passed the loop's gates and guard,
    best first, one per distinct point."""
    seen, out = set(), []
    for r in sorted(records, key=lambda r: r["objective"]):
        key = ukey(r["u"])
        if r.get("gates_failed") or key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) == k:
            break
    return out


def final_check(log: "Log", args, out: Path) -> dict | None:
    """Run the best ``--final-check`` candidates through every gate binary
    with nothing skipped (the loop skips ``GATE_SKIP``), best first; the
    first that passes them all is written to ``out``. Returns its record
    (``None`` when none passes, and ``out`` is left as the loop's best)."""
    if not args.gates:
        raise SystemExit("--final-check needs --gates")
    records = [r for r in log.cache.values() if r.get("kind") == "eval"]
    for r in ranked(records, args.final_check):
        tmp = out.with_name(f".{out.stem}-check.json")
        write_profile(tmp, r["values"])
        global GATE_SKIP
        saved, GATE_SKIP = GATE_SKIP, []
        try:
            failed = run_gates(args.gates, tmp, threads=args.gate_threads * args.gate_jobs)
        finally:
            GATE_SKIP = saved
            tmp.unlink(missing_ok=True)
        print(f"[final-check] eval {r['eval']} J {r['objective']:.4f}: "
              f"{'passes every gate' if not failed else 'fails ' + ', '.join(failed)}", flush=True)
        log.append({"kind": "final_check", "eval": r["eval"], "failed": failed})
        if not failed:
            write_profile(out, r["values"])
            return dict(r, final_gates=[])
    print("[final-check] no candidate passes every gate; --out keeps the loop's best")
    return None


def fmt_sets(d: dict, pct: bool = True) -> str:
    return " ".join(f"{s}:{100 * v:.1f}" if pct else f"{s}:{v:.2f}" for s, v in d.items())


# --------------------------------------------------------------------------
# Sensitivity scan
# --------------------------------------------------------------------------


def scan_points(n: int) -> list[float]:
    """Positions in a parameter's unit range, the default left out: both
    ends and evenly between."""
    return [k / (n - 1) for k in range(n)]


def scan(args) -> dict:
    """One parameter at a time across its range, the others at their
    defaults (plus the lock): the objective, its parts, the naturalness
    distance and the gates failed at every point. Logged to
    ``out/scan-<out stem>.jsonl`` (``--resume`` reuses it); a table is
    printed at the end."""
    params = load_params(args.params)
    lock = load_lock(args.lock, params)
    names = tunable(params, args, lock)
    space = Space(params, names, lock)
    sets = args.sets.split(",")
    heldout = [x for x in sets if x in HELDOUT_SETS]
    if heldout:
        raise SystemExit(f"held-out sets are never tuned on: {', '.join(heldout)}")
    items = select_items(sets, args.subset)
    out = Path(args.out).resolve()
    log_path = OUT_DIR / f"scan-{out.stem}.jsonl"
    meta = {"params": names, "sets": sets, "items": [it.id for it in items], "seed": None, "popsize": None,
            "sigma0": None, "ll_weight": args.ll_weight, "prior_weight": args.prior_weight, "lock": lock,
            "natural_weight": args.natural_weight, "natural_hold": args.natural_hold,
            "word_weight": args.word_weight, "binary": render.file_digest(render.say_bin()),
            "gates": [str(g) for g in args.gates] if args.gates else None, "points": args.scan,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    log = Log(log_path, meta, args.resume)
    ev = Evaluator(items, space, args)
    todo: list[tuple[str, float, np.ndarray]] = [("default", float("nan"), space.u0.copy())]
    for k, name in enumerate(names):
        for t in scan_points(args.scan):
            u = space.u0.copy()
            u[k] = t
            if abs(t - space.u0[k]) > 1e-9:
                todo.append((name, float(space.lo[k] + t * (space.hi[k] - space.lo[k])), u))
    rows = []
    chunk = args.popsize or 12
    for c in range(0, len(todo), chunk):
        part = todo[c : c + chunk]
        cached = [log.cache.get(ukey(u)) for _, _, u in part]
        fresh_k = [k for k, r in enumerate(cached) if r is None]
        if fresh_k:
            res = ev.evaluate([part[k][2] for k in fresh_k])
            for k, r in zip(fresh_k, res):
                name, value, u = part[k]
                if name != "default":
                    r.pop("natural_items", None)
                rec = {"kind": "eval", "param": name, "value": value, "u": [float(v) for v in u], **r}
                log.append(rec)
                log.cache[ukey(u)] = rec
                cached[k] = rec
        if c == 0:
            ev.set_base(cached[0])
        rows += cached
        print(f"[scan] {min(c + chunk, len(todo))}/{len(todo)}", flush=True)
    ev.close()
    base = rows[0]
    table = []
    for name in names:
        pts = [r for r in rows if r.get("param") == name]
        table.append((name, pts))
    print(f"\n[scan] default: objective {base['objective']:.4f}  mean PER {100 * base['mean_per']:.2f}  "
          f"-ll {base['mean_nll']:.3f}  natural {base.get('natural', float('nan')):.3f}")
    print(f"{'parameter':26s} {'value':>9s} {'dJ':>8s} {'dPER':>6s} {'d-ll':>6s} {'dnat':>6s}  gates failed")
    for name, pts in table:
        for r in pts:
            dj = r["mean_per"] + args.ll_weight * r["mean_nll"] - base["mean_per"] - args.ll_weight * base["mean_nll"]
            dn = (r.get("natural") or 0) - (base.get("natural") or 0)
            print(f"{name:26s} {r['value']:9.4g} {dj:+8.4f} {100 * (r['mean_per'] - base['mean_per']):+6.2f} "
                  f"{r['mean_nll'] - base['mean_nll']:+6.3f} {dn:+6.3f}  {','.join(r.get('gates_failed') or [])}")
    return {"rows": rows}


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def self_test(args) -> int:
    """A few evaluations on a tiny subset; checks the loop, the log, resume
    and that the engine accepts the written profile."""
    args.sets = "vcv,pair"
    args.subset = args.subset or 6
    args.budget = 6
    args.popsize = 3
    args.lock = None
    args.pass_ = "a"
    args.with_classic = False
    args.final_whisper = False
    args.out = str(OUT_DIR / "selftest-profile.json")
    args.resume = False
    s1 = run(args)
    assert s1["fresh_evaluations"] == 7, s1  # baseline + 2 generations of 3
    assert math.isfinite(s1["best"]["objective"]), s1

    out = Path(args.out)
    prof = json.loads(out.read_text())
    params = load_params(args.params)
    assert list(prof) == ["tuning"], prof
    assert set(prof["tuning"]) <= set(params), prof
    for k, v in prof["tuning"].items():
        assert params[k]["lo"] <= v <= params[k]["hi"], (k, v)
    res = subprocess.run(
        [str(render.say_bin()), "--engine", SYSTEM, "--profile", str(out), "--dump-profile"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    dumped = json.loads(res.stdout)["tuning"]
    for k, v in prof["tuning"].items():
        assert abs(dumped[k] - v) <= 1e-5 * max(1.0, abs(v)), (k, v, dumped[k])

    log = [json.loads(x) for x in (OUT_DIR / f"tune-{out.stem}.jsonl").read_text().splitlines()]
    evals = [x for x in log if x["kind"] == "eval"]
    assert len(evals) == 7 and all("values" in e and "objective" in e for e in evals)

    print("[self-test] resuming from the log: nothing should be re-evaluated")
    args.resume = True
    s2 = run(args)
    assert s2["fresh_evaluations"] == 0, s2
    assert abs(s2["best"]["objective"] - s1["best"]["objective"]) < 1e-12, (s1, s2)
    print(f"[self-test] OK: {s1['seconds_per_eval']:.2f} s per evaluation on {args.subset} items per set")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=Path, default=HERE / "params.json")
    ap.add_argument("--pass", dest="pass_", default="a", help="pass letters to tune together: a, b, c or e.g. bc")
    ap.add_argument("--lock", type=Path, default=None, help="profile whose tuning values stay fixed")
    ap.add_argument("--sets", default=",".join(TUNE_SETS), help="comma-separated material sets to score on "
                    "(default: the tuning split; held-out sets are refused)")
    ap.add_argument("--budget", type=int, default=300, help="CMA-ES evaluations (the baseline is extra)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="best profile (default: out/profile-<pass>.json)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse evaluations from the run's log")
    ap.add_argument("--popsize", type=int, default=None, help="default 4 + 3 ln(n)")
    ap.add_argument("--sigma", type=float, default=0.25, help="initial step, in units of each parameter's range")
    ap.add_argument("--ll-weight", type=float, default=0.1, help="weight of -log-likelihood (nats/phoneme)")
    ap.add_argument("--prior-weight", type=float, default=0.1, help="weight of the distance-from-default penalty")
    ap.add_argument("--with-classic", action="store_true",
                    help="also tune the parameters the classic engine reads (marked classic in params.json)")
    ap.add_argument("--natural-weight", type=float, default=0.1,
                    help="weight of the naturalness measures' moves away from the person beyond the defaults' "
                    "(0: no guard)")
    ap.add_argument("--no-natural-hold", dest="natural_hold", action="store_false",
                    help="do not bar candidates whose level walls, level rises or naturalness distance are worse "
                    "than the defaults' beyond noise")
    ap.add_argument("--word-weight", type=float, default=0.0,
                    help="weight of Whisper's word error on the tuning sentences (0: Whisper not used in the loop)")
    ap.add_argument("--exclude", type=lambda x: [n for n in x.split(",") if n], default=[],
                    help="parameters of the chosen passes left at their defaults (comma-separated), e.g. those "
                    "the sensitivity scan found pinned by gates")
    ap.add_argument("--final-check", type=int, default=0, metavar="K",
                    help="run the K best candidates through every gate with nothing skipped; the first that "
                    "passes becomes --out")
    ap.add_argument("--subset", type=int, default=None, help="use only N items per set (quick runs)")
    ap.add_argument("--jobs", type=int, default=None, help="render threads (default: all cores)")
    ap.add_argument("--batch", type=int, default=8, help="GPU batch size")
    ap.add_argument("--voice", default=render.DEFAULT_VOICE)
    ap.add_argument(
        "--gates",
        type=Path,
        nargs="+",
        default=None,
        help="compiled gate test binaries, or a folder with physical_gates and natural_gate; "
        "candidates that fail a gate are rejected",
    )
    ap.add_argument("--natural-jobs", type=int, default=4, help="worker processes for the naturalness guard")
    ap.add_argument("--gate-jobs", type=int, default=4, help="candidates gated at once")
    ap.add_argument("--gate-threads", type=int, default=4, help="test threads per gate run")
    ap.add_argument("--gate-skip", type=lambda x: [n for n in x.split(",") if n], default=[],
                    help="more gates to skip in the loop (comma-separated name filters), e.g. slow ones the "
                    "sensitivity scan never saw fail; run the whole suite on the result")
    ap.add_argument("--no-final-whisper", dest="final_whisper", action="store_false")
    ap.add_argument("--scan", type=int, default=None, metavar="N",
                    help="sensitivity scan instead of tuning: each parameter alone at N points across its range")
    args = ap.parse_args(argv)
    if not args.pass_ or set(args.pass_) - set("abc"):
        ap.error("--pass takes the letters a, b and c")
    if args.gates:
        args.gates = gate_binaries(args.gates)
    if args.self_test:
        return self_test(args)
    args.out = args.out or str(OUT_DIR / f"profile-{args.pass_}.json")
    if args.scan:
        scan(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
