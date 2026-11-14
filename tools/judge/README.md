# Intelligibility judge and tuner

An offline judge that scores how intelligible synthesized speech is, a
measure of how natural it sounds against a person, and a CMA-ES tuner that
searches the physical engine's tuning parameters (`params.json`, all 71
fields of `Tuning`) to improve that score. Everything runs locally once the models
are downloaded.

## Setup

```sh
cd tools/judge
uv sync                                   # Python 3.12, CUDA 12.6 PyTorch, transformers, whisper, cma
uv run python anchors.py fetch            # 16 LibriSpeech recordings -> .cache/anchors, material/human.tsv
```

`espeak-ng` must be on `PATH` (reference phonemes for plain text, and the
`espeak` system). The engine is called through the prebuilt `bin/say` (the
`say` example of `crates/synthie-trombone`); set `SAY_BIN` to use another
build. The models download on first use into `.cache/` (Hugging Face
`facebook/wav2vec2-lv-60-espeak-cv-ft`, about 1.3 GB, and Whisper `small.en`,
about 0.5 GB). Once the phoneme model is there, `judge.py` sets
`HF_HUB_OFFLINE=1`, so no network is touched. A GPU is used when one is
available (`JUDGE_DEVICE=cpu` forces CPU). The phoneme model and Whisper are
loaded one after the other, never together.

```sh
uv run pytest                             # normalizer + model-free tests
```

## Scoring

```sh
uv run python judge.py score human espeak classic physical     # the anchors
uv run python judge.py score physical --profile out/profile-b.json --name physical-b
uv run python judge.py score physical --sets sentence_heldout --profile out/profile-b.json --name physical-b-heldout
uv run python judge.py score physical --fast                   # quick check between changes (about 15 s)
uv run python judge.py report                                  # rebuild report/judge.md
uv run python judge.py items physical --set vcv                # what was heard, item by item
```

Each scored system writes `out/score-<name>.json` (per set, per item,
confusions) and the comparison table `report/judge.md` is rebuilt from every
`out/score-*.json`. Delete a score file to drop it from the report.

`score --fast` is the check to run between changes: the ten held-out
sentences (`sentence_heldout`) and the four listening sentences ("The birch
canoe ...", each alone and all four as the one clip), every clip in one
pass through each model. It prints phoneme error and Whisper words for the
three groups with the change since the last fast run of the same name, and
what Whisper wrote for the listening sentences; it writes
`out/fast-<name>.json` and the four-sentence clip `out/fast-<name>.wav`,
and leaves the scoreboard alone. It takes about 15 seconds, nearly all of
it loading the two models. Ten sentences give an interval of about ±5
points, so it shows a break or a large gain, not a small one.

What is measured, per material set:

| metric | how |
|---|---|
| PER | wav2vec2 CTC phoneme recognizer, greedy decoding, eSpeak IPA tokens folded to the engine's ARPAbet by `arpabet.py`, aligned against the expected phonemes; edits / reference phonemes over the whole set. The flap `DX` matches `T` and `D`. |
| CTC log-likelihood | log P(reference phonemes given the audio), nats per reference phoneme, with every model token that folds to a phoneme accepted for it, and r-coloured pairs (`AA R` = `ɑːɹ`) tried merged and split. Higher is better; smoother than PER. |
| confusions | substitutions, deletions (`X→-`) and insertions (`-→Y`) from the alignments, overall and per set. |
| word accuracy | Whisper `small.en`, reference words found in the transcript after Whisper's English text normalizer (also stored as WER). VCVs have no written form, so they have no word score. |
| PER vs intended | engine only: the same alignment against the phonemes the engine says it spoke (first line of `say`'s output), so the engine's own text-to-phoneme mistakes on plain-text sentences are left out. |

Material sets (`material/*.tsv`, built by `material.py build`):

| set | items | content |
|---|---:|---|
| `vcv` | 104 | every consonant in `AA C AA`, `IY C IY`, `UW C UW`; every vowel in `hVd` and `bVt` |
| `pair` | 44 | 22 minimal pairs (B/P, S/SH, R/L, IY/IH ...) |
| `word` | 42 | common words that cover every phoneme |
| `sentence_tune` | 10 | Harvard sentences, lists 1 and 2: tuning |
| `sentence_heldout` | 10 | Harvard sentences: held out, never tuned on |
| `human` | 16 | LibriSpeech dev-clean sentences. The `human` system plays the recordings; every other system synthesizes the transcript, so all systems are compared on the same text (shown as `librispeech` in the report). |
| `vowel` | 96 | probe set (not in the default scoreboard): every vowel alone and in five CVC frames |
| `voicing` | 110 | probe set: P/B T/D K/G S/Z F/V minimal pairs (initial, medial, final) and VCVs |
| `harvard_tune` / `harvard_heldout` | 20 / 20 | vowel tuner: Harvard lists 3 and 4 (tuning) and 5 and 6 (held out); every vowel slot scored |
| `carrier_tune` / `carrier_heldout` | 126 / 126 | vowel tuner: every stressed vowel in four CVC frames inside two carrier phrases, AX in three words; tuning and held-out splits share no frame and no carrier; only the CVC's vowel is scored |

The probe sets have a sixth column, `meta` (`key=value;...`): the scored
slot, the intended phoneme, the frame, the minimal-pair twin and so on.

`vcv`, `pair` and `word` are given to the engine as bracketed ARPAbet
(`[B AE1 T]`), so the expected phonemes are exactly what it was told to say.
Sentences are plain text, and the expected phonemes come from eSpeak NG
(`espeak-ng -v en-us --ipa`, folded).

### Probes: vowels, voicing, consonant confusions

Targeted measurements for the vowel, voicing and consonant-place stages
(`probes.py`, run through `judge.py`):

```sh
uv run python judge.py vowels physical --profile some.json    # -> physical-some
uv run python judge.py voicing espeak
uv run python judge.py confusions classic --name classic-x
uv run python judge.py vowels physical --set human          # the same slots inside the LibriSpeech sentences
uv run python judge.py vowels human                         # human ceiling, in-sentence
uv run python judge.py vowels human --cut                   # human words cut out of the recordings
uv run python judge.py probe-report r4-physical r4-classic --refs r4-espeak,human-cut \
    --scores r4-physical --out report/probes.md             # one page, side by side
uv run python frontend_check.py                             # engine text front end vs eSpeak
```

Each takes a system (`espeak`, `classic`, `physical`, and `human` for the
references), `--profile`, `--name` (default: the system, or
`system-<profile stem>`; `-insent` / `-cut` is appended for `--set human` /
`--cut`), honours `SAY_BIN`, uses the render cache (`--fresh` renders into a
scratch folder instead, for timing), writes `out/<probe>-<name>.json` (every
item and slot) and `report/<probe>-<name>.md`, and prints one summary line.

| probe | material | metrics |
|---|---|---|
| `vowels` | `vowel`: 96 items, the 16 vowels (IY IH EH AE AH AA AO UH UW ER AX EY AY OW AW OY) alone and in hVd, bVd, bVt, sVt, dVd (AX: alone and unstressed in five words); 50 real words, the rest labelled nonsense or isolated | **vowel error rate**: share of items whose vowel slot, after aligning the transcription to the reference, is not the intended vowel (AH = AX: the model writes the same vowel ʌ stressed and ə unstressed; a second rate also merges AA/AO, the cot-caught merger); **PER** on the set; a **confusion table** (intended x heard in the slot, with the model tokens); the **margin** (below) per vowel and frame |
| `voicing` | `voicing`: 110 items, P/B T/D K/G S/Z F/V as real minimal pairs in initial, medial and final position (medial T/D as "a tie"/"a die": American English flaps "latter"/"ladder") plus VCV syllables, both members of every pair | **forced choice**: CTC log P(intended) - log P(minimal-pair twin); accuracy = share where the intended member wins, and the mean margin; **free recognition**: the consonant heard in the slot has the intended voicing (a flap counts half for T/D; a sonorant, vowel or deletion is wrong). Broken down by pair, position, pair x position, intended consonant and intended voicing. Every pair and position has as many voiced as voiceless items, so voicing everything scores 50 %. The headline is P/B T/D K/G S/Z; F/V is reported separately. |
| `confusions` | `vcv` (the consonant of every `V C V`) | consonant accuracy; full consonant x consonant table; tables by place (bilabial, labiodental, dental, alveolar, postalveolar, palatal, velar, glottal), manner (stop, nasal, fricative, affricate, approximant) and voicing; heard-side columns for vowel, other and deletion |

**The margin** (`vowels`, and the forced choice of `voicing`) is computed
from the phoneme model's frame posteriors: the CTC log-probability of the
reference string minus that of the best competing string, where a competitor
is the reference with the slot replaced (by any other vowel; by the voicing
twin), every model token that folds to a phoneme pooled for it, as for the
CTC log-likelihood. In nats; positive means the intended phoneme wins; it
moves smoothly before the transcription flips, so a tuner can use it. Capped
at ±30 nats; 0 when the audio is too short to carry either string.

```python
import probes
pj = probes.ProbeJudge()                                  # loads the phoneme model once
r = pj.vowel("x.wav", ["B", "IY1", "D"])                  # slot: the first vowel by default
r["margin"], r["rival"], r["heard"], r["correct"]
rs = pj.vowels([a1, a2], [ref1, ref2], [slot1, slot2])    # wav paths or 16 kHz float arrays, batched
pj.voicing("y.wav", ["P", "AE", "T"], 0)["margin"]        # vs B AE T
```

**Human and in-sentence references.** `--set human` scores the same slots in
the 16 LibriSpeech sentences (every vowel, with stressed/unstressed from
eSpeak; every P B T D K G S Z F V against its twin; every consonant), for
the recordings (`human`) or any synthesizer saying the transcripts. `--cut`
(human only) force-aligns each recording to its reference (Viterbi CTC
path) and cuts out every word (eSpeak's word boundaries, halfway between the
spikes of neighbouring words, 10 ms fades, 0.15 s of silence each side);
each word is scored as an isolated item, the human counterpart of the
synthetic material. For the engine reading sentences, slots where its own
phonemes (the first line of `say`) differ from the reference are flagged
`frontend_ok: false`, and every in-sentence summary is repeated without them
("front end right"), so the engine's text-to-phoneme mistakes can be told
from its sound. Cutting three-phoneme excerpts was tried first and was
unusable (the excerpts spill into their neighbours and the model answers in
other languages).

`frontend_check.py` compares the engine's own text-to-phoneme output (the
first line of `say`) with eSpeak's for every word of the material, ignores
notation-only differences (AX/AH/IH, IY/IH, AA/AO, flap) and a short list of
eSpeak errors and US variants, and writes `report/frontend.md`.

#### What the probes can and cannot tell

- The judge is harsh on isolated items, also with real speech: stressed
  vowels in words cut out of the LibriSpeech recordings get about 35 % vowel
  error (about 4.5 % for the same vowels heard in their sentences), and eSpeak
  saying the `vowel` set gets 57 %. On isolated items the model drifts to
  non-English tokens (`a`, `aː`, `ɑ` without length, even Mandarin finals),
  which `arpabet.py` folds to AA, so "heard as AA" on isolated items is partly
  the judge: `report/vowels-*.md` lists the model tokens, and
  `heard_aa_tokens` in the JSON counts them.
- So read the isolated-item numbers against eSpeak's and the cut-word human
  scores, not against zero; the in-sentence numbers (`--set human`) are
  closer to what a listener hears, but mix in the engine's text front end.
  `report/round4-baseline.md` gives the ceilings.

### The gate

The human anchors must score **under 15 % PER**. If they do not, the
normalizer or the decoder is wrong and no other number from the judge can be
trusted: fix the judge, do not move the threshold. `report/judge.md` prints
PASS or FAIL at the top.

## Vowels: the lab and the tuner

```sh
uv run python vowel_lab.py eval --split tune                     # vowel error and margins, per vowel
uv run python vowel_lab.py eval --split heldout --profile p.json --name p
uv run python vowel_tune.py --base base.json --out out/vowels-tuned.json   # coordinate search on the tuning split
```

`vowel_lab.py` scores every vowel slot of a split under a profile (renders
in a scratch folder, GPU lock held only while judging): **tune** =
`sentence_tune`, `harvard_tune`, `carrier_tune`; **heldout** =
`sentence_heldout`, `harvard_heldout`, `carrier_heldout`, `human` (the
held-out split is never tuned on). A profile may carry a `"vowels"` object
of F1/F2/F3 targets (adult man, Hz) keyed `IH`, `AY<`, `AY>`, `IH>`; `bin/say`
solves their shapes through the babble map at start-up, so a candidate
needs no rebuild. `vowel_tune.py` moves F1 and F2 of every vowel shape in
steps, within 12 % (F1) and 10 % (F2) of the published value, keeping the
vowels' order in F1 and F2, with a mild pull back to the published value;
the objective is the mean clipped margin over the tuning split. Bake a
result into `crates/synthie-trombone/src/vowel_targets.rs` and regenerate
the shapes (`cargo run -p synthie-trombone --example vowel_size --release`).

### Naturalness

The recognizer hears whether the words come through, not whether they sound
like a person. `natural.py` (`judge.py natural`) compares a system saying the
16 LibriSpeech texts (and, with `--listener`, the listener's own two
sentences) with the recordings of people saying them:

```sh
uv run python judge.py natural physical espeak classic --listener ~/my-sentences.wav
SAY_BIN=other/say uv run python judge.py natural physical --name physical-new
uv run python judge.py natural-report human espeak classic physical physical-new   # -> report/natural.md
```

| group | measures |
|---|---|
| spectrum in time | log-mel distance (40 bands, per-band mean removed) to the person after dynamic time warping, overall and by the class of the person's segment |
| loudness | phrase rise and fall (-30 to -6 dB of the peak), segment rise and fall (10-90 % of each climb of 15 dB or more), dips more than 30 dB down inside a sentence (per second, length, share; dead silence under -60 dB), level steps (95th percentile, walls of 20 dB in 10 ms), modulation spectrum of the envelope (peak, share in 2-8 and 8-32 Hz) |
| pitch | median, 5-95 % range, declination, last 150 ms against the median, step from vowel to vowel, change across a vowel, micro-variation |
| fricatives | per /s z/, /sh zh/, /f v th dh/: hiss rise and fall (-15 to -3 dB), duration within 10 dB, level against the neighbouring vowel, centroid, flatness, peak prominence and width (1-8 kHz) |
| timing | speaking rate (syllables per second of speech), mean duration per class |

Segments come from forced alignment with the phoneme model (boundaries
halfway between neighbouring spikes). The person's column gives his mean and
the 10th to 90th percentile across his sentences; each system's value has a
`z` against that spread. There is no second person saying the same
sentences, so the warped distance has no person-to-person value. Each run
writes `out/natural-<name>.json`; `tests/test_natural.py` checks every
measure on synthetic signals.

## Tuning

**Retired after Round 5.** The CMA-ES search below is no longer run. In Rounds 4 and 5 every tuned
profile fitted the tuning sentences and was no better, or worse, on the held-out ones (Round 5: held-out
phoneme error +2.9 [1.1, 4.8] and +1.6 [0.1, 3.0] for the two passes), so the engine is now improved by
changes to its mechanisms, scored directly with `judge.py`. `tune.py` and its tests stay for the gate
runner, the one-parameter scans and the record; the rejected profiles are in the git history
(last present in commit b531112).

```sh
uv run python tune.py --self-test                    # 7 evaluations on 12 items, then a resume check
uv run python tune.py --pass a --scan 4 --gates bin --out out/scan-a.json        # sensitivity scan
uv run python tune.py --pass a --gates bin --budget 300 --seed 1 --out out/profile-a.json
uv run python tune.py --pass bc --lock out/profile-a.json --gates bin \
    --budget 300 --seed 1 --out out/profile-b.json
uv run python tune.py --pass a --lock out/lock-lax.json --gates bin --budget 200 --sigma 0.08 \
    --word-weight 0.3 --final-check 6 --gate-jobs 7 --gate-threads 2 --out out/profile-r5a.json \
    --exclude lung_pressure,adduction_time,attack_hold,jet_efficiency,teeth_gain,abduction_time,postalveolar_lips,frication_widening
                                                     # round 5's pass a (below); out/lock-lax.json holds lax_length 0.62
uv run python judge.py score physical --sets sentence_heldout,harvard_heldout,carrier_heldout \
    --profile out/profile-b.json --name physical-b-heldout
```

`tune.py` is not run under the GPU lock: it takes the lock itself for each
batch of judging and lets it go while it renders and runs the gates.

- Searches the parameters of the chosen passes (`--pass`: `a` how sounds
  are made, 31 parameters; `b` how they follow one another, 36; `c` the
  vowel lengths, 4; letters combine, `bc`), minus those fixed by `--lock`,
  those named by `--exclude` (left at their defaults, e.g. the ones the
  scan found pinned by gates) and the twelve the classic engine reads too (`"classic": true` in
  `params.json`: the classic engine is pinned bit for bit, so a tuned value
  of those could never become a default; `--with-classic` tunes them
  anyway). Each parameter is mapped to 0..1 over its `lo..hi` range (hard
  bounds, inside the engine's own range) and the search starts at the
  defaults. A test checks `params.json` against the engine
  (`say --dump-parameters`).
- Material: the tuning split, `sentence_tune`, `harvard_tune`,
  `carrier_tune` (`--sets`); the held-out sets are refused.
- Objective (lower is better), averaged over the sets:
  `mean PER + 0.1 * mean(-log-likelihood) + w * (1 - Whisper word accuracy)
  + 0.1 * mean(((x - default)/(hi - lo))^2) + 0.1 * mean_k max(0, |z_k| - |z_k default|)`
  (`--ll-weight`, `--word-weight`, default 0, `--prior-weight`,
  `--natural-weight`).
- **Whisper in the loop** (`--word-weight` above 0): Whisper `small.en` is
  loaded on the GPU next to the phoneme model and transcribes every
  candidate's tuning sentences (`sentence_tune`, `harvard_tune`; the carrier
  phrases have no written form); the word accuracy is pooled over them.
  It adds about 5 s a candidate.
- **The naturalness guard**: each candidate also says the 16 LibriSpeech
  texts, and every one of its loudness, pitch, timing and hiss measures
  gets a z against the person's (`natural.distance` with `clip=None`; the
  warped log-mel distance in dB is one more term). A candidate pays for
  each measure it moves further from the person than the defaults are,
  unclipped and term by term, so a measure already far off cannot get worse
  for free and a gain on one measure does not pay for a loss on another.
  On top of that, a candidate is **barred** like a gate failure
  (`natural:<measure>` in the log) when its level walls per second, its
  level rises (`steps.p95_up`) or its unclipped naturalness distance are
  worse than the defaults' beyond noise: the lower end of the 95 % paired
  bootstrap interval over the 16 texts above 0 (`NATURAL_HOLD`,
  `--no-natural-hold` turns it off). Round 4's guard clipped each z at 4,
  and its winner made the walls 3.4 to 4.0 a second and the rises steeper
  in 16 of 16 sentences unseen; the hold bars exactly that. Only the
  naturalness measures of those texts are read, never their phoneme or
  word scores; the decision below still leaves them out of the held-out
  pool.
- **Gates** (`--gates bin`: `bin/physical_gates`, `bin/natural_gate`,
  `bin/sibilant_gate` and `bin/vowel_glide_gate`, copies of the compiled
  test binaries, which read the candidate from `SYNTHIE_TROMBONE_PROFILE`):
  a candidate that fails any of them gets 1 plus 0.05 per failed gate
  added, so it can never win. Only the slowest gate (`no_held_note`, about
  70 s) and determinism (`same_seed`) are skipped in the loop (round 4
  skipped more, and its best candidate failed one of those). They run on
  the CPU while the GPU judges. `--final-check K` runs the K best
  candidates that passed, best first, through every gate binary with
  nothing skipped and writes the first that passes to `--out`; the other
  test binaries (`listen_gate`, `mimic_gate`, `inverse_gate`,
  `vowel_size_gate`, ...) only test the built-in tuning, so a result is
  made the default and `cargo test --workspace --release` run on that.
- Evaluation 0 is the starting point (defaults plus the lock), then CMA-ES
  generations (`--popsize`, default `4 + 3 ln n`; `--sigma` 0.25). The last
  generation may overrun `--budget` by less than one population.
- All items of a generation render in parallel (`--jobs`, default all cores)
  into `out/tmp/` (deleted afterwards); the phoneme model stays loaded on the
  GPU and judges in batches (`--batch 8`); the naturalness measures run in
  worker processes (`--natural-jobs 4`), the gates `--gate-jobs` at a time
  with `--gate-threads` each.
- Every evaluation is appended to `out/tune-<out stem>.jsonl` (unit-box
  point, parameter values, objective, per-set PER and log-likelihood, the
  naturalness distance and its terms, the gates failed, timings).
  `--resume` replays it: same seed, same candidates, and logged ones are not
  evaluated again. Without `--resume` an existing log is kept as `.bak`.
- The best profile so far is rewritten to `--out` after every generation, as
  `{"tuning": {...}}` with the tuned and locked parameters, the only shape
  `bin/say --profile` accepts.
- At the end the best profile is scored by
  `judge.py` (phonemes + Whisper) on the tuning sets, as
  `out/score-physical-<out stem>.json` (`--no-final-whisper` skips this).
- `--subset N` uses N items per set for quick experiments.
- `--scan N` runs a sensitivity scan instead: every parameter alone at N
  points across its range (both ends included), the rest at their defaults,
  with the gates; it logs to `out/scan-<out stem>.jsonl` and prints, per
  point, the change in objective, PER, log-likelihood and naturalness
  distance and the gates that failed.

Measured on the RTX 3070 / 16 threads machine (shared with other work):
a pass over the tuning split (156 items, plus the 16 naturalness texts and
both gate binaries per candidate, the slow gates the scan never saw fail
skipped with `--gate-skip`) took **25 s per evaluation**, a 300-evaluation
pass about two hours; the scan of 51 parameters at 4 points, 198
evaluations, about 1.5 hours.

**What the round-4 scan found.** The defaults sit close to many gate
bounds: of the 51 parameters the tuner may move, 12 fail a gate at every
scanned point away from the default (`lung_pressure`, `adduction_time`,
`attack_hold`, `jet_efficiency`, `teeth_gain`, `abduction_time`,
`postalveolar_lips`, `frication_ramp`, `fortis_closure`,
`articulation_rate`, `closing_ratio`, `vowel_hold`), mostly on the pinch
recovery (12 ms), the onset pitch kick, the regulator, voiced fricatives,
clicks and the level of /s/ and /ʃ/. `frication_widening` does nothing
while `frication_ramp` is 0. Points next to the default move the mean
phoneme error by about a point either way, which is the noise floor of a
single scan point. The parameters that move it most and pass the gates are
the breath noise (`glottal_efficiency`, less is better), the velar port
(`nasal_port`, less is better) and the breath over a phrase (`final_drop`).

**The round-4 pass** (pass a, the 19 parameters not pinned, the rest of
pass a locked at the defaults, `--sigma 0.15`, seed 1): J 0.566 to 0.471
in 301 evaluations, about 8 of every 12 candidates barred by a gate. Two
lessons for the next one:

- The best candidate (evaluation 274) failed a gate the loop skips by
  default (`fricatives_keep_their_level_and_no_vowel_hisses_in_any_voice`);
  the best that passes the whole suite was evaluation 138. Check the whole suite on the top
  few, not only the winner.
- The guard clips each z at 4, so a measure already more than 4 spreads
  from the person can get worse without cost: the candidate's level walls
  went from 3.4 to 4.0 a second (the person: 1.1) and its level rises got
  steeper in every sentence, while its distance improved. It was kept out
  of the defaults for that. A guard that charges any measure for moving
  further from the person, clipped or not, would have caught it.

Held out (paired bootstrap over items, against the defaults): phoneme
error -3.5 points [-5.3, -1.7] over `sentence_heldout`, `harvard_heldout`
and `carrier_heldout`; Whisper +7.0 points [2.2, 12.1] on the held-out
sentences, +5.1 [-1.7, 12.2] on the tuning ones.

**Round 5.** Pass a (21 parameters with the lax lengths locked at 0.62,
the scan's pinned ones excluded, `--sigma 0.08 --word-weight 0.3`, 209
evaluations, 53 s each with Whisper and the four gate binaries) fitted the
tuning split (pooled phoneme error -3.8 [-5.4, -2.2], Whisper +8.1) and held
out was worse (+2.9 [1.1, 4.8]). Pass
b+c ran from the defaults on the sentences only (`--sets
sentence_tune,harvard_tune --sigma 0.05 --prior-weight 1.0`), after a
gates-only scan of each parameter at one and two steps of 5 % either side
(`vowel_hold` pinned, ten free on one side only), with `--box` holding each
to its free side; held out it was worse too (+1.6 [0.1, 3.0]). Neither
became the default. The
improvement on the tuning split is about the size of the sets' noise, so a
bigger tuning split would help more than a longer search.

## Files

| file | what |
|---|---|
| `arpabet.py` | ARPAbet <-> eSpeak IPA normalizer: forward map, fold of all 392 model tokens, canonical rules, alignment. `python arpabet.py --dump` rewrites `data/fold_table.tsv`. |
| `espeak_util.py` | eSpeak NG: reference phonemes for text; ARPAbet -> eSpeak mnemonics for the `espeak` system. |
| `material.py` | builds and loads `material/*.tsv`. |
| `anchors.py` | fetches the LibriSpeech anchors (`fetch`). |
| `render.py` | renders a material set through a system (`classic`, `physical`, `espeak`, `human`); renders are cached under `out/render/` by binary, voice, profile and text. |
| `judge.py` | the judge: `score`, `report`, `items`, and the probes `vowels`, `voicing`, `confusions`, `probe-report`. |
| `natural.py` | the naturalness measure against the human recordings (`judge.py natural`, `natural-report`). |
| `prosody.py` | prosody transfer: a recording and its text as a schedule of phone times, pitch and level for `say --match-prosody`; rendering, checks, round trip (below). |
| `speaker_voice.py` | the speaker's voice for a schedule: the baseline profile (`match_voice` while the engine follows the schedule), per-phone H1*-H2*, tilt and vowel formants, and the tenseness and formant steps of the transfer loop. |
| `probes.py` | the probes: slot alignment, CTC forced-choice margins, `ProbeJudge` (the tuner API), cut tokens from the anchors, summaries and Markdown. |
| `frontend_check.py` | the engine's text-to-phoneme output vs eSpeak's, per word. |
| `tune.py` | the CMA-ES tuner and the sensitivity scan. |
| `vowel_lab.py` | vowel error and margins per vowel under a profile, on the tuning or held-out split. |
| `vowel_tune.py` | coordinate search over the vowel targets on the tuning split. |
| `params.json` | the tuning parameters: `lo`, `hi`, `default`, `pass` (a, b, c, as in the engine), `classic` for the twelve the classic engine reads too. |
| `data/w2v_vocab.json` | the phoneme model's tokenizer vocabulary (bundled; checked against the model at load). |
| `data/fold_table.tsv` | every model token and the phonemes it folds to, for review (a test checks it is current). |
| `tests/` | pytest: normalizer (every symbol maps, round trips, diphthongs and affricates as single model tokens, eSpeak agreement) and model-free parts of the judge and tuner. |
| `report/judge.md` | generated comparison of every scored system. |
| `out/` | scores, tuner logs, profiles, render cache (not committed). |
| `.cache/` | models and anchor audio (not committed). |
| `bin/say`, `bin/physical_gates`, `bin/natural_gate`, `bin/sibilant_gate`, `bin/vowel_glide_gate` | prebuilt renderer and gate test binaries (not committed; `cargo build --release --example say -p synthie-trombone`, `cargo test --release --no-run -p synthie-trombone --test physical_gates --test natural_gate --test sibilant_gate --test vowel_glide_gate`, then copy them here, without the hash in the name). |

## Caveats

- The phoneme model is a multilingual CommonVoice model. It transcribes the
  recorded human speech well (about 6 % PER), but it is harsh on synthetic
  speech in isolation: eSpeak's own voice scores about 50 % PER on the
  `vcv`/`pair`/`word` sets while Whisper understands most of its words, and
  around 15 to 25 % on sentences. Treat PER on isolated items as a relative
  measure (better or worse than before), not as "the fraction a listener
  would mishear". Decoding variants (pooling token variants, restricting to
  English tokens, padding with silence) were tried and changed the scores by
  only a few points, so greedy decoding was kept.
- The 16 human anchors come from the small `hf-internal-testing/librispeech_asr_dummy`
  subset, which is a single LibriSpeech speaker (1272, male). The gate
  checks the normalizer and the decoder, not robustness across speakers.
- Rendering is deterministic (same profile, same audio), so the objective has
  no noise from the engine; with small sets it can still overfit them, which is
  why the held-out sentences exist.

## Matching a voice

`match_voice.py` listens to a recording of someone speaking and writes a
voice profile (`bin/say --profile`) that makes the physical engine sound
like that speaker: same pitch and pitch range, same body size (formant
scale), same voice quality, same pace.

```sh
uv run python match_voice.py speaker.wav                        # -> speaker.profile.json
uv run python match_voice.py speaker.flac --text "what they said" --transcript --out voice.json
uv run python match_voice.py speaker.mp3 --analyse              # measurements only
uv run python match_voice.py --calibrate                        # rebuild data/voice_calibration.json (about 2 min)
uv run pytest tests/test_match_voice.py -s                      # includes the self-consistency gate (about 1 min)
```

Any format `soundfile` reads (wav, flac, ogg), else ffmpeg (mp3, m4a, ...);
mono or stereo, any rate, a few seconds to a minute of speech. `--text` is
what the engine says while it is compared with the recording (default: four
Harvard sentences); give the transcript when you have it, and `--transcript`
when it is exactly what the recording says (speaking rate is then matched by
speech time instead of by syllable rate). A match takes 5 to 30 s (2 to 8
renders of the comparison text). It prints the recording's measurements,
the same measurements of the engine speaking with the new profile, and the
differences; `--report r.json` keeps everything, round by round.

The profile sets `pitch`, `tract_length`, `tenseness`, `expressiveness`,
`rate` and `tuning.posterior_gap` / `tuning.lung_pressure` (plus
`"engine": "Physical"`), all inside the engine's ranges.

### What is measured

| quantity | how |
|---|---|
| median pitch, range | autocorrelation pitch (Boersma: normalised autocorrelation, 4 candidates per 10 ms frame, Viterbi path with octave-jump cost, frames > 0.8 octave from the median dropped); median and 10th..90th percentile in semitones |
| formant scale | LPC (order 20/22/26 by the speaker's formant spacing) on pitch-smoothed spectra of the steady voiced frames; the long-term distribution of formant candidates over log frequency is lined up with the default voice's saying the same text. The scale factor that lines them up is the formant scale (1/tract length for the engine, within 2 %). Where the alignment is ambiguous (several scales about 25 % apart fit almost equally well, as for human speech), the ratio of formant spacings `dF` (uniform tube fitted to median F1..F4) chooses |
| H1*-H2* | levels of the first two harmonics (48 ms frames), corrected for the first two formants (Iseli and Alwan), median over voiced frames; the uncorrected H1-H2 is printed too |
| tilt | alpha ratio: energy 1..5 kHz over 0..1 kHz of the voiced frames |
| breathiness | harmonics-to-noise ratio from the autocorrelation peak (median), and cepstral peak prominence |
| pace | syllable nuclei (envelope peaks in voiced speech) per second of speech; speech time; pauses of 0.25 s or more are left out of both |

### How it is calibrated

`data/voice_calibration.json` holds the measurements above taken on the
engine itself, saying the default comparison text at 129 settings (a Latin
hypercube over pitch 75..340 Hz, tract length 0.64..1.4, tenseness
0.25..0.92, expressiveness 0..2, rate 0.65..1.6, posterior gap 0.05..0.5 and
lung pressure 5..13, plus the default voice, whose analysis is the reference
for the formant scale). At start-up a quadratic model of each measurement is
fitted to it (robustly: samples the analysis got badly wrong are dropped).
Matching inverts the model for a first guess, then renders, re-analyses and
corrects: pitch, tract length and rate by ratio, expressiveness by the range
error over its slope, and tenseness, posterior gap and lung pressure together
by a Broyden step on H1*-H2*, HNR and tilt. It stops when everything is
close, or after 8 rounds, and writes the best round. Rebuild the calibration
when `bin/say` changes (the tool notes it when the binary differs).

### The gate

The engine speaking with the produced profile must re-analyse to within
5 % of the recording on median pitch, 3 % on formant scale and 3 dB on
H1*-H2*. `tests/test_match_voice.py` checks this on eight hidden engine voices
(pitch 90..300 Hz, tract length 0.7..1.25, tenseness 0.35..0.85) and prints
the errors and the recovered settings.

Recalibrated for round 4's engine (folds that close on every cycle, a
brighter voice): on the eight hidden voices the re-analysis lands within
1.0 % on pitch, 0.6 % on formant scale and 0.7 dB on H1*-H2* (with last
round's calibration, which still passes on this engine: 1.1 %, 0.3 %,
0.6 dB). The settings themselves are not always recovered: three of the
eight come back with a tract length 16 to 38 % off the hidden one (two
with the old calibration), measuring alike, and the hidden tenseness only
below about 220 Hz. On six LibriSpeech recordings of one reader four pass
the gate; one is matched 5.7 % low in pitch, and in one the formant scale
reads 0.63 and the tract goes to the end of its range (1.5).

### Limits

- The engine cannot make every voice. With an adult tract (length near 1)
  its H1*-H2* does not go below about 5 dB (eSpeak's male voices measure
  about -1 dB, and fail the H1*-H2* gate by about 7 dB); its pitch range tops
  out at 9 to 10 semitones at expressiveness 2 (the LibriSpeech reader
  already needs 1.4 to 2); its HNR is lower than very clean voices'.
- At high pitch (above about 220 Hz) different tenseness / lung pressure /
  posterior gap combinations measure almost alike, so the voice quality is
  matched but the settings are not unique (the hidden tenseness is not
  recovered there).
- The formant scale of human speech is less certain than the engine's: on 16
  utterances of one LibriSpeech speaker the guide (`dF`) varies by about 5 %
  and the chosen scale by more (tract length 0.78..0.97 across five of
  them). Long tracts at high pitch (above 250 Hz with tract length above 1.2)
  and near-whispers (tenseness below about 0.3 with a wide posterior gap) are
  measured unreliably, as are recordings with little voiced speech.
- Not measured, so left at the defaults: jitter, vibrato, nasality, the
  persona dials, and every timing/articulation tuning parameter except
  `rate`. Accent, articulation and intonation patterns are not copied; only
  their overall pitch range is.
- The scale and the rate depend on what is said; give `--text` (and
  `--transcript`) when you know it.

## Prosody transfer

`prosody.py` measures a person's timing, pitch and level on a recording of
a text and writes them as a schedule the physical engine speaks with
(`bin/say --match-prosody`, `speech::Utterance::follow` in the crate):
the engine's own articulation, coarticulation and laryngeal gestures, with
the person's durations, pauses and pitch contour.

```sh
uv run python prosody.py transfer speaker.wav --text "what they said" --out-dir out/prosody --name x
    # -> x.json (schedule), x-transfer.wav (the engine with it), x-rule.wav (by its rules), checks printed
uv run python prosody.py transfer speaker.mp3 --out-dir out/prosody --name x  # no text: Whisper hears it (x.txt)
uv run python prosody.py transfer speaker.wav --off voice,formants ...        # leave parts of the voice out
uv run python prosody.py analyse speaker.wav --text "..." --out s.json      # the schedule only
uv run python prosody.py render s.json --out e.wav --level [--pitch voice]  # speak it (level loop: --level)
uv run python prosody.py check s.json e.wav                                # durations and pitch against it
uv run python prosody.py roundtrip "The birch canoe slid on the smooth planks."
uv run python prosody.py slope                                             # dB per doubling of the breath
bin/say --engine physical --voice deep --match-prosody s.json [--prosody-pitch voice] \
    [--prosody-kick off] [--prosody-voice off] [--prosody-tenseness off] [--prosody-formants off] \
    [--prosody-speed off] [--phone-times t.json] --out e.wav "the same text"
```

**The text.** `--text`, else a `.txt` beside the recording, else what
the judge's cached Whisper hears (`judge.WordJudge`, never a download);
a heard text is written beside the *schedule* (`x.txt`) and the output
says it was heard, not given. A recording longer than 14 s is cut in its
longest silences (`split_pieces`) into pieces the aligner takes one at a
time, each with its share of the text (Whisper's of the piece, or the
given text's words lined up with what Whisper heard in it,
`distribute_words`), and the pieces' schedules are joined. The phones
aligned are always the engine's for the text, so a number, an
abbreviation or a name the engine reads otherwise than the speaker is
aligned as the engine will say it; every phone carries `align_lp`, the
best log-probability the phoneme model gives its unit on the frames the
forced path put it on, and those under -4.5 nats are listed as poorly
placed (`poorly_placed`; 2 % of the anchors' phones, mostly weak forms and
/r/; in both reference recordings the /r/ of "dark", which they do not
say, and the /k g/ of "background", which they say as one).

**The schedule** (JSON, `"format": "synthie-prosody-1"`): the text; one
entry per phone of the engine's own phonemes for it (the first line of
`say`, read by `arpabet.parse_engine_line`), with `start` and `end` on the
recording (s), `level_db` (mean level against the mean of the utterance's
vowels), `breath` (a share of the rules' lung pressure, 1 unless the level
loop set it), the pitch at five points inside and the voiced share; the
`pauses`; `median_hz` over voiced frames; and `pitch`, Hz every 10 ms with
a value everywhere (a curve through the unvoiced stretches) and the
`voiced` flags. A gap of 20 ms or more between two phones is a pause.
Since round 9 also: per phone `tenseness` (a share of the rules'),
`formants` (for a vowel, F1 and F2 as shares of the voice's own vowel),
`align_lp`, `word` and the recording's own `speaker` measures (H1*-H2*,
tilt, F1, F2); `voice`, a profile `say` lays over the voice (tract length,
tenseness, posterior gap, lung pressure, pitch); `baseline`, the
recording's analysis; `text_source`, `heard`, `pieces` and
`poorly_placed`.

**Alignment.** The phone sequence is the engine's, so every aligned stretch
is a phone the engine says; where the person's pronunciation differs (a
weak form, a dropped sound) the phone still gets a stretch. The judge's
phoneme model gives the Viterbi CTC path (`probes.ctc_viterbi`, as
`natural.align`; the pairs the model writes as one token are merged or not,
whichever it finds likelier, and shared by weight). Its spikes are a frame
or two wide and sit anywhere in their phone (a vowel's at 29 % of it, a
stop's at 70 % to 74 %, at the release; `SPIKE_SHARE`, measured on the
engine's own renderings), so halfway between spikes is off by up to half a
vowel (52 ms mean duration error on the engine's own audio). Each boundary
goes where the spectrum moves fastest between the two spikes (40 mel bands,
5 ms), weighted by a prior where the engine's rule durations for the same
phones would put it; between two stops (whose closures run into one
silence) the prior alone. Silences of 120 ms or more between two phones
(200 ms beside a stop, whose closure is silent too; over a noise floor,
8 dB above it) and quiet unvoiced stretches of 80 ms or more between two
sonorants become pauses; every phone gets at least 25 ms, taken from the
neighbours with time to spare. On the engine's own renderings, where the
true times are known: 16 ms mean absolute duration error and 10 ms onset
error without the prior (the four test sentences); with the prior, on
audio that follows a person's timing, 13 ms (LibriSpeech schedules) and
18 ms (the owner's two sentences).

**Pitch** is `match_voice.track_pitch` (10 ms), octave jumps against the
running median folded back, voiced islands under 30 ms dropped, a curve
linear in semitones through the unvoiced stretches, smoothed 1-2-1.
`--prosody-pitch speaker` (default) keeps the speaker's Hz,
`--prosody-pitch voice` moves the contour by the semitones between the
speaker's median and the voice's own rule median.

**Level.** `render --level` speaks the schedule, measures each phone's
level against the utterance's vowels, and moves the breath of each sonorant
towards the person's by the difference (through the engine's measured
level against breath, `BREATH_DB`), within -10.5 to +5.2 dB of the rules'
(`LEVEL_RANGE`; ±4 dB until round 9), twice; obstruents take their
neighbours'. The range is as far as the engine follows: a sweep of the
breath share over the four sentences (deep, bright and child voices)
gives -11 dB at a share of 0.5 with 96 to 99 % of the vowel frames still
voiced (at 0.4 the folds drop out of 9 to 16 %), +5.4 dB at 1.5, the most
the lungs give, with H1-H2 within 0.7 dB of the rules' (no pressing).

**The speaker's voice** (`speaker_voice.py`, round 9). `transfer` first
matches the engine's voice to the recording while the engine follows the
schedule (`match_voice.match` with every attempt spoken from the schedule,
so the pitch and pace are already the speaker's): median pitch, formant
scale (tract length), H1*-H2* (tenseness), HNR and tilt (posterior gap),
the lung pressure held at 8 cmH2O (free, the matcher took it to its
floor, where with a quiet phone's half breath the folds stop), the
recording's formant scale taken from the formant spacing where the
candidate-density estimate parts from it by more than 6 %
(`match_voice.steady_scale`: the density estimate ranged from 0.69 to
1.30 over the reader's 16 sentences, the spacing from 0.99 to 1.18), and
the tenseness no lower than 0.45 (sample 2's H1*-H2* took it to 0.29,
where that voice's voiced stops between vowels were voiced through 62 %
of their closures, at 0.45 through 96 %). A match that ends more than
8 % off in formant scale or 15 % in pitch is not used. Then the loop of
three renderings moves, besides the breath, each sonorant's tenseness
share by its H1*-H2* against the engine's mean, compared with the
speaker's against hers (`tenseness_step`; 5 dB per unit of share, at
most 0.06 a step, within 0.9 to 1.15: round 8 measured that 12 % less
clicks in the child voice; the obstruents keep the rules' tenseness), and
each vowel's F1 and F2 shares by the median over its tokens of the
speaker's formants against the engine's, each over its voice's formant
scale, after the speaker's overall bias is taken out (`formant_step`;
within ±15 %; a step over 30 % is a tracking error). The tilt is measured
but does not drive the tenseness: in this engine a looser voice is
brighter (alpha -8.1 dB at tenseness 0.65, -6.6 at 0.5), where a breathier
speaker is darker. The rendering closest to the speaker is kept
(`transfer_score`). In the crate, `Utterance::follow_with` gives a vowel
with formant shares a shape solved through the babble map for the voice's
formants there times the shares (from the voice's own shape, rounding
and all), held by the vowel and by the consonants made on its body, and
moves the tongue body up to 1.6 times faster through a phone shorter
than the rules make it; phones are made no shorter than 70 % of their
rule length (`Tuning::follow_shortest`).

Measured in round 9 on the 16 anchors (every part left out in turn from
the final schedules; F1/F2 errors are against the reader's, each over its
voice's formant scale, median over the vowels measured in both):

| | timing only | hybrid | without the voice | without tenseness | without vowels |
|---|---:|---:|---:|---:|---:|
| formant scale against his, |%| | 4.8 | 1.9 | 4.8 | 1.4 | 1.4 |
| H1*-H2* against his, |dB| | 1.4 | 0.6 | 1.6 | 0.7 | 0.6 |
| HNR against his, |dB| (the engine is noisier) | 2.7 | 3.0 | 2.9 | 2.9 | 3.0 |
| H1*-H2* contour, r with his (RMS) | 0.22 (3.1 dB) | 0.49 (2.7 dB) | 0.53 | 0.22 (3.1 dB) | 0.51 |
| vowel F1 / F2 error, median | 27 / 22 % | 17 / 16 % | 27 / 21 % | 17 / 16 % | 19 / 19 % |
| level against his, RMS over sonorants | 3.0 dB | 2.6 dB | 2.8 dB | 2.6 dB | 2.7 dB |

The level loop: 4.7 dB before it, 2.5 dB after (93 of 455 sonorants at
the limit); with the round 8 range 3.0 dB (218 at the limit).

**Checks** (`check`; `transfer` prints them): the rendering aligned again
against the schedule, interior phones, mean absolute duration error; how
exactly the engine followed (its own phone times, `--phone-times`, against
the schedule); the rendering's pitch against the schedule's on frames
voiced in both (RMS semitones, apart where the speaker's pitch moves 40
st/s or faster). `roundtrip` takes the engine's own rendering of a text as
the recording and speaks its schedule again.

Measured in round 8 (the four test sentences round trip; the 16
LibriSpeech anchors and the owner's two recorded sentences transferred):

| | round trip | anchors | owner |
|---|---:|---:|---:|
| engine's own phone times vs schedule | 0.0 ms | 0.0 ms (1 phone of 736 at its floor) | 0.0 ms |
| rendering aligned again vs schedule | 16 ms | 13 ms | 18 ms |
| pitch vs schedule, RMS (fast-moving / rest) | 1.04 st (1.07 / 1.04) | 1.17 st (1.43 / 1.06) | 1.20 st (1.43 / 1.14) |
| mean pitch error | +0.4 st | +0.3 st | +0.2 st |
| level vs the person's, RMS over sonorants, before / after the loop | 0.9 / 0.5 dB | 4.9 / 3.3 dB | 4.2 / 2.3 dB |

The engine's pitch kick after voiceless consonants is kept (it adds to the
speaker's own rise there): without it (`--prosody-kick off`) the anchors'
pitch error is 1.12 st instead of 1.18. The level follows only partly:
195 of the anchors' 455 sonorants end at the ±4 dB limit (vowels 39 %,
nasals and liquids about 45 %, glides 16 of 20), about as many lowered as
raised: the person's level moves more from syllable to syllable than a
few dB of breath can give, and a sonorant's level against the vowels is
partly the tract's. The level loop keeps whichever of its three
renderings comes closest.

Limits: the model's 20 ms frames limit the boundaries to about a frame; a
phone the person did not say gets the 25 ms floor; only pauses at word
boundaries become phrase breaks (a pause inside a word is shared between
the phones either side); a voiceless stop's closure keeps its floor (22 ms
plus 8 ms), so a faster stop than that is drawn out.
