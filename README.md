# synthie-trombone

This is [Pink Trombone](https://dood.al/pinktrombone/), Neil Thapen's
physical model of the human voice, rebuilt in Rust: a glottal source pushes
air through a 44-cell waveguide of the vocal tract, you squeeze the tract
somewhere, and out comes something between a vowel and a very confused
trombone. You can play it in your terminal, have it read your text aloud, or
give the characters in your game a voice.

**Play it in your browser: https://sunsided.github.io/synthie-trombone/**

![Singing an "ah" with the critter voice, a finger on the tongue tip](docs/singing.png)

![The bright voice mid-sentence, about to close the lips for the "m" in "am"](docs/speaking.png)

## Trying it out

The browser version runs the same Rust engine through WebAssembly in an
AudioWorklet; `task web:dev` starts it locally with hot reload (it needs
`wasm-pack` and Node with corepack). Everything below is about the
terminal app and the library.

You'll need a Rust toolchain (1.88 or newer for the library) and, on Linux,
the ALSA headers for audio output (`libasound2-dev` on Debian and Ubuntu).
The project uses [go-task](https://taskfile.dev/); `task` on its own lists
everything it can do.

```sh
task run                                   # start the terminal app
task install                               # install it as ~/.cargo/bin/synthie-trombone
task say -- --voice bright "Hello there!"  # speak a line into say.wav
task voices -- "Welcome, traveller!"       # the same line in every voice, in voices/*.wav
```

Without `task`, run `cargo run --release -p synthie-trombone-cli` for the app
and `cargo run --release -p synthie-trombone --example say -- "…"` for a WAV
file. Stick to release builds: the tract model takes 96,000 steps per second
of audio, and a debug build can't keep up with the sound card.

## Playing the terminal app

The app starts in play mode, with the piano on your home row. Hold a key to
sing, change the vowel with the number keys, and work everything else with
the mouse: the tract, the tongue, the velum, and every meter in the panels.

| Key | Action |
|---|---|
| `a w s e d f t g y h u j k o l p ;` | sing, piano layout; `z` / `x` change the octave |
| `Space` or `Esc` | stop singing (and speaking) |
| `1` … `0` | vowels IY IH EH AE AA AH AO UH UW ER |
| arrow keys | move the tongue body |
| `v` / `V` | next / previous voice |
| `'`, `PgUp` / `PgDn`, `Del` | pick a persona dial, turn it, back to neutral |
| `n`, `b`, `i` | nasal (velum), lip pop, drone |
| `[` `]`, `-` `=` | tenseness, vibrato |
| `m`, `r`, `c` | pitch wobble, reverb amount, clear constrictions |
| `\` | switch between the classic and the physical engine |
| `/` | Formant Lock: steer the vowel by F1 and F2 on a vowel chart (arrow keys then nudge F2 / F1) |
| `{` `}`, `<` `>` | lips back / forward by 0.2 cm (back past rest they follow the rounding again), larynx lower / higher by 0.2 cm (physical engine; heard with `lip_horn` and `larynx_travel` on) |
| `` ` `` | mimic on or off (start the app with `--mimic`; see below) |
| `F5` | start or stop recording a take (see below) |
| `F6` | speech through the gesture score or the old path (A/B them by ear; see the instrument panel below) |
| `Tab` | type a line, or paste one; `Enter` speaks it at the current note's pitch |
| `,` `.` | speech rate |
| `q` | quit |

| Mouse | Action |
|---|---|
| left-drag in the mouth or throat | a finger on the wall: the row of the midline shuts the tract (a stop), the row beside it leaves a gap that hisses (a fricative), rows further out only shape it |
| left-drag the tongue handle `◆`, or Shift-drag anywhere | move the tongue body (right-drag also works where the terminal passes it on) |
| click a vowel mark (`i e a o u`) | that vowel |
| click in the nose; drag in the gap under it | open or shut the velum; set how far it hangs open |
| click or drag a meter | tenseness, tongue, velum, vibrato, reverb, loudness (lung pressure, physical engine), and the four persona dials |
| click `[-]` `[+]`, `wobble`, `drone`, `engine` | octave and the toggles |
| click `[/ vowels]` under the scope; left-drag on the vowel chart | Formant Lock on or off; steer the vowel by its formants, or click a vowel mark |

Turning a persona dial or a meter while nothing is sounding plays the
current note for as long as you keep turning (and 0.4 s after), so you
hear what the change does; while a note, the drone, speech or mimic is
sounding, the change is heard in that instead. The tongue handle stays
inside the mouth and follows the voice's size, like the vowel marks.

**Formant Lock** (`/`, or `[/ vowels]` under the scope) lets you find
vowels by how they sound rather than by where the tongue goes. The scope
gives way to a vowel chart, drawn the way phoneticians draw it: F2 across,
high on the left, and F1 down, low at the top. The shaded region is what
this voice's tract can reach (it moves with the persona's size and the
voice's tract length), the IPA letters mark where this voice makes the
English vowels, and `◆` is where the voice is now, with its F1 and F2 in the
panel's title. Press or drag inside the chart and the physical engine's
babble map ([`inverse`](crates/synthie-trombone/src/inverse.rs)) finds the
tongue, lips and narrowing that give those formants; the tract view above
shows them move. A point out of reach is held at the nearest one in reach,
clicking a letter takes that vowel, and the arrow keys nudge F2 (left
raises it, as on the chart) and F1 (up lowers it) by 2 %. Formant Lock
belongs to the physical engine, so turning it on switches to it, and
switching to the classic engine turns it off; turning it off brings the
scope back and leaves the tract as it is.

**The instrument panel**, beside the tract view from 80 columns on, shows
what the voice is doing inside:

```
╭ ◇ instrument ────────────────────╮
│ F1  348 F2  837 F3 1212 Hz       │  the tract's own resonances (a nasal /a/)
│ tract resonances, 17 ms late     │
│ ΔP 10.1 sub 10.4 lungs  8.0 cmH2O│  across the folds, below them, asked for
│ shut ██████···········  35% folds│  closed quotient, and what is sounding
│ gap  0.005‥0.191 cm²             │  smallest and largest glottal area
│ horn 1.2  larynx -0.6 cm         │  lips out, larynx above rest ("off")
│ tiers: score · controls          │  cyan names: the score; white: controls
│ glottis ██████████████████████┃ 1│  one row per tier of the gesture score:
│ larynx  ████████████████┃·· 1 h-0.6│  the bar is where it is now, ┃ its
│ velum   ██████████████████████┃ 1│  target, the number its gestures;
│ body    █████┃················· 1│  a tier nothing drives is dimmed;
│ tip     ┃······················ ·│  h the larynx's height, p the lips'
│ lips    ┃················· · p1.2│  protrusion, cm
╰──────────────────────────────────╯
```

- **F1 F2 F3** are the peaks of the transfer function of the tract's shape
  at that moment (nose included), measured as `examples/replay.rs --trace`
  measures them: the resonances of the shape, not an analysis of the
  sound; `—` where it shows fewer than three peaks below 4.2 kHz (a closed
  mouth, and the classic engine's /u/). A measurement takes about 3 ms, so
  it is made on a thread of its own from a copy of the voice the audio
  thread sends at most 30 times a second, and only when the shape has
  changed (a reading holds while the shape stays). The panel says how late
  the reading came: from the moment the shape was taken to the frame that
  showed it, typically 15 to 35 ms (at most a probe interval, 33 ms, more
  while the tract keeps moving). With
  Formant Lock on, `●` marks the same F1 and F2 on the vowel chart, beside
  the `◆` the lock steers.
- **ΔP** is the pressure below the folds less the pressure just above them,
  cmH2O, averaged over the last 50 ms; **sub** the pressure below them
  (the lungs push harder than asked on a high or pressed note); **lungs**
  the pressure the voice asks for. **shut** is the share of the time the
  folds' vibrating edges are closed while the voice sounds, and whether the
  folds, the pulse source or nothing is sounding. The classic engine has no
  air: it shows `classic: no air` and its LF pulse.
- **horn** and **larynx** are how far the lips are pushed forward and the
  larynx stands above its rest, cm (`Trombone::tract_lengths`); `off`
  while `AnatomyConfig::lip_horn` or `larynx_travel` is `0` and the voice
  does not hear them (the tract's state still moves). The tract view draws
  the horn past the lips (yellow), a lowered larynx as the tube carried on
  below the glottis (orange), both faint while off, and the epilaryngeal
  tube's walls in green as far as `epilarynx_cells` reaches.
- **The tiers** are the gesture score's (glottal adduction, larynx and
  pitch, velum, tongue body, tongue tip, lips and jaw), on the score's
  scales (pitch 50 to 400 Hz, logarithmic; the tip's and the lips'
  closeness, 1 shut). A tier whose name is cyan is driven by the score's
  gestures and shows `Trombone::tier_activity`; a white one is read off the
  controls the voice has (the voicing gate, the pitch, the velum, the
  tongue body's height, the constrictions of the tip and the lips), which
  is what drives it when it is played by hand or when speech takes the old
  path.
- **`F6`** switches speech between the gesture score and the old path
  (`ArticulationConfig::gesture_score`, physical engine; the panel's title
  says `score` while it is on), so the two can be heard side by side on the
  same text: type it and press `Enter`, then `F6` and `↑` `Enter` (or, in
  play mode, `Enter`, which says the last line again). While the score is on,
  the lip pop (`b`) is played by hand as a gesture on the lips' tier (held
  while the key is, ended on release) instead of as a constriction; a take
  logs it and replays it to the sample.

Below 80 columns the panel gives way to a summary on the tract view's lower
edge (`F1 … F2 … F3 … · ΔP … · shut … folds`).

**Typing and pasting text.** `Tab` opens the text line; `Enter` speaks it,
however many sentences it holds: each gets a pause after it and its own
intonation, and `Esc` stops all of it at once (a second `Esc`, or `Tab`,
goes back to play mode). The line holds up to 4,000 characters and scrolls
sideways to keep the cursor in view. Paste with your terminal's paste key
(`Ctrl+Shift+V` in most, `Cmd+V` on a Mac) or `Shift`+middle-click for the
selection (the app takes mouse clicks itself, so a plain middle-click goes
to the app rather than to the terminal's paste). A paste goes in at the
cursor (from play mode it switches to typing) and is never spoken until you
press `Enter`: line breaks and tabs become spaces, curly quotes, dashes and
`…` become what the speech front end reads, and accents come off letters it
would otherwise skip (`café` is said as `cafe`). The terminal must support
bracketed paste, as all current ones do; in one that does not, the app
still turns line breaks inside a burst of pasted keys into spaces.

| Key (type mode) | Action |
|---|---|
| `←` `→`, `Home` `End` (`Ctrl+A` `Ctrl+E`) | move the cursor |
| `Ctrl+←` `Ctrl+→` (`Alt+B` `Alt+F`) | move by a word |
| `Backspace`, `Delete` | delete a character |
| `Ctrl+W` (`Alt+Backspace`, `Ctrl+Backspace`), `Ctrl+K` | delete the word before the cursor, everything after it |
| `Ctrl+U` | clear the line |
| `↑` `↓` | earlier lines |

Notes stop when you let go of the key in terminals that report key releases
(kitty, WezTerm, foot, Ghostty). In other terminals, and in ones that say
they report releases and then send none, a key held down plays until you
let go (the app takes the end of the key's repeats, 0.2 s without one, for
the release), and a note tapped shorter than the keyboard's repeat delay
holds until you press Space or play another. `--key-log FILE` writes every
key event as the terminal reports it, for finding out what a terminal sends.

**Recording takes.** `F5` starts recording a take and `F5` again stops it
(in either mode; starting one returns to play mode, so the keys sound);
meanwhile the header shows `● REC take-007 0:03`. A take
is two files in `takes/` under the current directory, or in the directory
given with `--record DIR`, numbered past the highest take already there so
that nothing is overwritten:

- `take-007.wav`: the voice as the engine made it, before the reverb (mono,
  32-bit float, at the sound card's rate);
- `take-007.jsonl`: a log, one JSON object per line. The first holds the
  state of every control at the take's first sample (sound model, voice
  settings, preset and persona, tongue, each constriction, velum,
  tenseness, lung pressure, pitch and voicing, drone, wobble, vibrato,
  reverb, mimic, the vowel shape), each further line one command with the
  sample the audio thread applied it before (`t`; `s` is the same in
  seconds), and the last one the take's length:

  ```json
  {"t":43200,"s":0.9,"cmd":"tongue","diameter":2.05,"index":23.5}
  {"t":52800,"s":1.1,"cmd":"speak","id":2,"prosody":{…},"text":"a big red cat"}
  ```

  Typed speech is logged with its text, and in mimic mode every gesture
  copied from the microphone. If the disk ever falls behind, the end line
  and the status line say how much was lost.

A take starts from silence: when recording starts, speech in progress stops
and the voice is rebuilt from its controls, the same way a replay builds it,
so start recording before you play. Replay a take offline with

```sh
cargo run -p synthie-trombone --example replay --release -- takes/take-007.jsonl \
    --out replay.wav --trace trace.tsv
```

It plays the commands at their samples through the same code as the app
and says whether the result is identical to the take's WAV file (it is,
sample for sample). `--trace` writes a tab-separated row every 5 ms: the
controls' targets, what the tract actually does (its narrowest points, each
constriction, the velum and all 44 diameters), voicing, pitch, level, and
F1–F3 of the tract at that moment.

## Voices

You pick a voice with `v` in the app, `--voice` on the command line, or a
`Preset` in code:

| Preset | Character |
|---|---|
| `deep` | 105 Hz, the model's native tract |
| `bright` | 210 Hz, tract 0.86, livelier intonation |
| `child` | 290 Hz, tract 0.72 |
| `elder` | breathy and slow, with a tremor and a shaky pitch |
| `giant` | 62 Hz, tract 1.4, slow |
| `critter` | 360 Hz, tract 0.6, fast, nasal and very excitable |
| `robot` | monotone, no jitter, pressed |
| `whisper` | breath only, no voicing |

Pitch and vocal tract length vary independently from person to person:
raising only the pitch sounds like the same speaker reaching higher, while a
shorter tract shifts every resonance and sounds like a smaller speaker. The
tract values above are relative to the model's native tract of about 17 cm.
The presets combine both with tenseness, jitter, vibrato, nasality, speaking
rate and how much the intonation moves, and all of these are public fields
of `VoiceSettings`, so your own voice starts from a preset and changes what
you like. With the `serde` feature, the settings can live in your game's
data files.

### Personas and profiles

Describing a voice by its parameters is slow. A `Persona` describes it the
way you would describe a person, with four dials that each move several
parameters together, as they move together in people:

| Dial | From … to | What moves |
|---|---|---|
| `size` | 0.6 (small body) … 1.4 (large) | tract length, and pitch faster than length |
| `effort` | 0 (murmur) … 1 (calling out) | lung pressure, firmer folds, higher and livelier pitch, quicker articulators |
| `breathiness` | 0 (pressed) … 1 (breathy) | how much air escapes past the folds |
| `clarity` | 0 (casual) … 1 (careful) | earlier and faster articulators, longer sounds, firmer closures |

```sh
cargo run --release -p synthie-trombone-cli -- --engine physical --voice bright --size 0.8 --effort 0.7
cargo run --release -p synthie-trombone --example say -- --engine physical --breathiness 0.7 "Hello there."
```

```rust
use synthie_trombone::{Persona, Preset};

let voice = Preset::Bright.settings().with_persona(&Persona {
    size: 0.8,
    effort: 0.7,
    ..Persona::NEUTRAL
});
```

A body of another size is not the same tube scaled: the walls, the side
cavities and the lips do not shrink with it, and unchanged vowel shapes
drift by up to 9 % in the first formant. So each vowel has a shape for
every tract length (a small generated table, `examples/vowel_size.rs`),
which keeps F1 and F2 within 3 % of the native vowel's, scaled, from 0.6
to 1.4. The exception is /u/ (and /w/) in a long tract: it is already as
close as a vowel may be, and its F1 sits up to 6 % high at 1.4.

Underneath sits `Tuning`: a set of numbers, each with a default and a
range a real speaker could plausibly occupy (`Tuning::FIELDS`;
`say --dump-parameters` prints them), in three parts by what each number
physically is. `AnatomyConfig` (`tuning.anatomy`, 46 numbers) is the body,
its tissues and its sources; `ArticulationConfig` (`tuning.articulation`,
76) is how sounds are made and joined; `ProsodyConfig` (`tuning.prosody`,
54) is the rhythm and melody of a phrase. The classic engine reads twelve
of them (marked *classic*: the lengths of sounds, stress, the gaps and the
breath after a burst); the physical engine reads the rest as well.

| Part | Topic | Fields |
|---|---|---|
| anatomy | the breath | `lung_pressure` |
| anatomy | the folds | `attack_depth`, `attack_hold`, `modal_gap`, `posterior_gap`, `voiced_stop_slack`, `cycle_jitter`, `pitch_flutter` |
| anatomy | the voice's source tilt and softness | `voice_tilt`, `tilt_corner`, `tilt_top`, `tilt_top_corner` (a second section, off), `soft_corner` |
| anatomy | breath noise | `glottal_efficiency`, `voiced_breath_corner`, `voiced_breath_gain` |
| anatomy | jets and hiss | `jet_efficiency`, `teeth_gain`, `hiss_corner`, `hiss_floor`, `teeth_corner`, `labial_edge`, `dental_edge` |
| anatomy | the nose and the walls | `nasal_port`, `cavity_expansion` |
| anatomy | the tract's length (wave 2A; on by default, the anatomical tract's; each `0` off) | `lip_horn`, `larynx_travel` |
| anatomy | the epilarynx (wave 2B; the anatomical tract's four cells flaring over four by default, `epilarynx_effort` `0` off) | `epilarynx_cells`, `epilarynx_diameter`, `epilarynx_flare`, `epilarynx_flare_cells`, `epilarynx_effort` |
| anatomy | the tongue body from measured modes (wave 2, `0` off; `say --tongue-model mri`) | `tongue_model` |
| anatomy | the head the ports radiate round (wave 3C; read in stereo, and by the mono mixdown, on by default; `head_mixdown` `0` the plain sum) | `head_radius`, `head_mixdown`, `chest_reflection`, `chest_distance`, `chest_corner` |
| anatomy | creak (wave 3; heard only where creak is asked for) | `creak_coupling`, `creak_press` |
| anatomy | non-verbal sounds (wave 3B; heard only where one is asked for) | `cough_pressure`, `laugh_pressure` |
| anatomy | the ventricular valve (c3-valve; heard only in a cough or a throat clearing) | `valve_open`, `flutter_rate` |
| articulation | the glottis in consonants | `adduction_time`, `abduction_time`, `spread_closing`, `abducted_area`, `voiceless_spread`, `voiced_fricative_spread`, `voiced_sibilant_spread`, `onset_kick`, `kick_time` |
| articulation | gaps and places | `fricative_gap` and `approximant_gap` (*classic*), `lateral_gap`, `lateral_channel`, `sibilant_teeth`, `postalveolar_lips`, `frication_widening`, `nasal_cavity`, `alveolar_velum`, `closure_overshoot` |
| articulation | leads and hand-overs | `lead`, `release_lead`, `velum_lead`, `labial_lead`, `labial_closing`, `sibilant_exit`, `tap_lead`, `lateral_release`, `liquid_transition`, `frication_ramp` |
| articulation | closures and releases | `closure_floor`, `fortis_closure`, `voicing_tail`, `aspiration` (*classic*), `vot_labial`, `vot_alveolar`, `vot_velar`, `unstressed_vot`, `devoicing_lead`, `tap_length` |
| articulation | vowels gliding into their neighbours | `lax_nucleus`, `ae_nucleus`, `fronted_nucleus`, `dark_l_share` |
| articulation | following a speaker's schedule | `follow_breath_low`, `follow_breath_high`, `follow_tenseness_low`, `follow_tenseness_high`, `follow_speed_max`, `follow_shortest`, `follow_shortest_light`, `follow_formant_range` |
| articulation | articulator speeds | `dorsum_rate`, `nasal_cavity_rate`, `closing_ratio`, `release_speed`, `velum_release`; the tiers' springs `articulation_rate` (`tongue_tip.stiffness`), `lips_rate` (`lips_and_jaw.stiffness`), `body_rate` (`tongue_body.stiffness`), `body_damping` (`tongue_body.damping`), `larynx_rate` (`larynx_and_pitch.stiffness`), `larynx_damping` (`larynx_and_pitch.damping`) |
| articulation | an /n/ into a fronted vowel, the coloured schwa (wave 1, "canoe"; each `0` or `1` off) | `nasal_tip_release`, `fronted_after_n`, `nasal_cavity_held`, `schwa_colour` |
| articulation | who lowers the larynx (wave 2A; `rounded_larynx` 0.5 by default, `voiced_stop_larynx` `0` off) | `voiced_stop_larynx`, `rounded_larynx` |
| articulation | breathing in (wave 3; heard only where a breath is asked for) | `breath_opening` |
| articulation | non-verbal sounds (wave 3B; heard only where one is asked for) | `laugh_rate`, `laugh_voiced` |
| articulation | the ventricular valve (c3-valve; heard only in a cough or a throat clearing) | `valve_closing`, `valve_opening` |
| prosody | the rules' lengths of sounds (*classic*) | `vowel_length`, `diphthong_length`, `stop_length`, `nasal_length`, `fricative_length`, `voiced_fricative_length`, `approximant_length`, `aspirate_length`; and `voiced_sibilant_length` |
| prosody | lengths by stress and position | `stress_stretch` (*classic*), `vowel_hold`, `lax_length`, `unstressed_lax_length`, `ih_length`, `lax_floor`, `prefortis_clip`, `prelenis_stretch`, `final_coda_stretch` |
| prosody | rhythm (round 11, off) | `stressed_onset_length`, `cluster_length`, `function_consonant_length`, `schwa_length`, `diphthong_hold`, `rhythm_pace` |
| prosody | intonation | `accent_size`, `accent_decay`, `accent_peak`, `declination`, `final_fall` |
| prosody | the breath over a phrase, and pauses | `phrase_onset`, `onset_pressure`, `sonorant_onset`, `phrase_decline`, `final_drop`, `vowel_decline`, `unaccented_level`, `sentence_pause` |
| prosody | tenseness and softness by syllable | `light_laxness`, `sonorant_slack`, `final_slack`, `sonorant_softness`, `unstressed_softness`, `final_softness` (the slacks off, `0`, since round 8, the softnesses since round 9) |
| prosody | breaths and creak (wave 3; on by default, `breath_intake` and `final_fry` `0` off) | `breath_intake`, `breath_first`, `breath_pause`, `breath_time`, `final_fry`, `fry_time` |
| prosody | hesitation fillers (wave 3B; on by default, `fillers` `0` off) | `fillers`, `filler_length`, `filler_pause`, `filler_pitch` |

`ArticulationConfig` also holds a `TierDynamics` (stiffness, damping,
dominance) for each of the six tiers of the gesture score
(`glottal_adduction`, `larynx_and_pitch`, `velum`, `tongue_body`,
`tongue_tip`, `lips_and_jaw`). The tongue body's, the tip's, the lips'
and the larynx's springs are read today, under the names above; the
tiers' other numbers wait for the gesture engine and are not in profiles
until it reads them.

A voice with its persona and tuning can be kept as a JSON profile (feature
`profile`); every key is optional and a misspelt one is an error. The
tuning's fields go flat, by their own names:

```json
{
  "persona": { "size": 0.9, "clarity": 0.7 },
  "pitch": 120.0,
  "tuning": { "lead": 0.035, "fricative_length": 0.11 }
}
```

or by part, as `--dump-profile` writes them (`"tuning": {"articulation":
{"lead": 0.035}, "prosody": {"fricative_length": 0.11}}`), or both mixed;
a field given both flat and in its part is an error, so which one counts
never depends on the order of the keys.

```sh
cargo run --release -p synthie-trombone --example say -- --engine physical --profile my-voice.json "Hello there."
cargo run --release -p synthie-trombone --example say -- --dump-profile     # the voice in use, as a profile
cargo run --release -p synthie-trombone-cli -- --engine physical --profile my-voice.json
```

A profile read by the `say` example may also carry a `"vowels"` object of
F1, F2 and F3 targets (adult man, Hz) keyed by vowel (`IH`, and `AY<` /
`AY>` for the start and end of a diphthong): `say` solves their shapes
through the babble map when it starts, so other vowel targets can be heard
without regenerating the vowel table. The library's own profiles do not
read it.

```json
{ "engine": "Physical", "vowels": { "IH": [410, 2050, 2700], "AY>": [400, 2100, 2800] } }
```

### Morphing between bodies

A `MorphTarget` is another body for the same voice: four numbers, each
against the voice it is laid on, that say nothing about how the voice
speaks (its pitch contour, timing and effort stay its own), so the same
sentence, by the rules or with a neural prosody (`--neural M2` or `F2`),
can be spoken by a smaller, a larger, a younger or a more mature body.

| Number | Range | What it is |
|---|---|---|
| `tract_scale` | 0.75 … 1.25 | the vocal tract's length against the voice's own; the folds scale with it |
| `larynx_depth` | −0.3 … 0.8 cm | how much lower the larynx rests (added to `anatomy.larynx_depth`) |
| `fold_thickness` | 0.6 … 1.6 | the folds' size against the body's (multiplies `anatomy.fold_thickness`) |
| `epilarynx_flare` | 0.6 … 1.6 | the width of the lower pharynx above the tube over the folds (multiplies `anatomy.epilarynx_flare`) |

```sh
say --engine physical --voice bright --morph-size 0.9 "Hello there."    # the tract's scale
say --engine physical --voice bright --morph-age 1 "Hello there."       # the same speaker, mature
say --engine physical --voice deep --morph-sweep 0.75,1.25 "The body grows while it speaks."
say --engine physical --voice bright --profile profiles/adult-female.json --neural F2 "Hello there."
```

```rust
use synthie_trombone::{MorphTarget, Preset, Talker};

// A body from the start …
let grown = Preset::Bright.settings().with_morph(&MorphTarget::matured(1.0));
let mut talker = Talker::new(48_000.0, grown);
talker.say("Stay a while.");
// … or one the voice glides into while it sounds.
talker.set_morph(MorphTarget::sized(1.2));
let mut block = [0.0f32; 512];
talker.render(&mut block);
```

Laid over the settings (`VoiceSettings::with_morph`, `--morph-size`,
`--morph-age`, `--morph TRACT,LARYNX,FOLDS,FLARE`; the terminal app takes
the first two) a morph is simply part of the voice. Given to a voice that
is sounding (`Trombone::set_morph`, `Talker::set_morph`,
`VoiceStream::morph`, `say --morph-sweep`) the body glides there, at most
`set_morph_speed` a second (by default the whole range in half a second).
Changing the tract's length used to start the physical engine over (a few
milliseconds of silence, the folds starting again). A glide resets
nothing: the tube's clock is retuned a little every control block, the
waves stay in their cells as the cells grow, the nose and the sinuses keep
theirs, the folds keep their motion, and each vowel spoken from then on
takes its shape for the body as it is. Measured on held /ɑ i u æ/ of the
deep and the bright voice, there and back between 0.75 and 1.25 in 1.25 s
(`examples/morph.rs`, `tests/morph_gate.rs`):

- no click: the roughest 10 ms (largest sample step over its own level) is
  0.8 to 1.2 times the steady vowel's;
- no stall: the glottal flow's swing never falls under 0.86 of the swing
  around it, and the pitch stays within 6 % of the pitch asked for;
- the formants follow the tube: /ɑ/'s F1 and F2 go from 1004 and 1788 Hz
  at 0.75 through 780 and 1360 to 638 and 1096 Hz at 1.25;
- with all four numbers moving at once the same holds (swing 0.96);
- in speech the roughest moments are the stops' own bursts, which differ
  as much between bodies that do not change (4.7 to 8.1) as through a
  glide (5.0 to 10.3); a sentence without bursts is as smooth morphing as
  not.

**The grown woman's body for `F2`.** With tiny-voca's `F2` prosody the
bright voice sounded young or thin. Measured, the bright voice's F1 and F2
are already a woman's (they are solved to Hillenbrand's values at 0.86 of
the man's tract); what was young was the pitch (`F2`'s own median is
280 Hz), a third formant some 5 % above women's, and a sound with too
little below 1 kHz (1 to 2.5 kHz stood 9 dB under 0 to 1 kHz; a man
reading stands at 16, the deep voice at 14). `MorphTarget::matured(1.0)`
(`--morph-age 1`) is what was calibrated against that: the tract 5 %
longer with the larynx resting 0.3 cm lower (7 % of length together, every
formant 7 % lower: a larger woman), the lower pharynx a quarter wider, the
folds 5 % smaller against the body (they stay as they were while the
tract grows). `profiles/adult-female.json` is that body on the bright
voice at a grown woman's pitch (185 Hz, a median of 214 Hz with `F2`'s
melody). What each number did, single runs with `--neural F2`:

| Bright voice, `F2` prosody | Clip words | 10 tuning | 10 held-out | 1–2.5 kHz against 0–1 kHz | Median pitch |
|---|---|---|---|---|---|
| `F2`'s own pitch (the old "F2-bright") | 27/31 | 74.1 % | 68.8 % | −8.9 dB | 280 Hz |
| pitch moved to the voice's 210 Hz | 24/31 | 80.2 % | | −9.9 dB | 243 Hz |
| tract 1.05, larynx 0.3, at 185 Hz | 27/31 | 81.5 % | 76.2 % | −11.3 dB | 214 Hz |
| + lower pharynx 1.3 times as wide | 27/31 | 81.5 % | 71.2 % | −12.0 dB | 214 Hz |
| + folds 0.95, pharynx 1.25 (`matured(1)`, the profile) | 27/31 | 80.2 % | 73.8 % | −12.3 dB | 214 Hz |
| folds 0.8 instead (tract 1.07, at 210 Hz) | 25/31 | 64.2 % | | −11.3 dB | 247 Hz |

The lower pitch and the wider pharynx each darken the sound without
costing words; thinner folds darken it most (at 0.9 and under the folds
close fully at 210 Hz, where the bright voice's do not) but cost words
below 0.9, so they are only 5 % thinner. Whether it sounds like a grown
woman is for the ear: `25-s1-f2-adult.wav` against `25-s1-f2-bright.wav`.

### Lip-sync from the articulators

The sound is made by articulators that really move, so a character's
mouth does not have to be guessed from the audio afterwards. An
`ArticulatorFrame` reads them off the voice as ten channels, each 0 to 1:

| Channel | What it is |
|---|---|
| `jaw_open` | how far the jaw is lowered (the engine's own jaw: /æ/ 0.93, /ɑ/ 0.70, /ɛ/ 0.60, /i/ 0.16, /u/ 0) |
| `lip_close` | how far the lips are brought together; exactly 1 while they are sealed (/p b m/) |
| `lip_funnel` | lips forward and open (/ʃ/ 0.21, /r/ 0.28): the lip horn's length by how open the lips are |
| `lip_pucker` | lips forward and narrow (/w/ 0.94, /u/ 0.68) |
| `lip_press` | the lower lip at the upper teeth (/f v/ 0.87; `lip_close` stays under 0.98) |
| `tongue_up` | the tip or blade raised: 1 where it shuts the mouth (/t d n/), /s/ 0.90, /l/ 0.65; a vowel's front counts 0.6 of itself |
| `tongue_back` | the tongue body from front (/i/ 0) to back (/ɑ/ 0.77, /ɔ/ 1) |
| `velum_open` | the way into the nose (/m n ŋ/ 1) |
| `larynx_height` | 0.5 at rest, lower for rounded vowels and mature bodies |
| `glottal_open` | the folds held apart: near 0 in voice, about 0.5 in a voiceless consonant, 1 breathing |

```sh
say --engine physical --export-lipsync --out line.wav "Papa, we fish for food."   # line.wav and line.visemes.json
cargo run --release -p synthie-trombone --example lipsync            # every phoneme's frame
```

```rust
use synthie_trombone::{ArticulatorFrame, Engine, Preset, Talker};

let mut npc = Talker::new(48_000.0, Preset::Deep.settings().with_engine(Engine::Physical));
// Baked: the samples and a frame for every 512 of them.
let (samples, frames) = npc.bake_with_articulators("Papa, we fish for food.");
assert_eq!(frames.len(), samples.len().div_ceil(512));
// Live: read the voice whenever a video frame is drawn.
let now: ArticulatorFrame = npc.voice_mut().articulators();
```

- **Any moment, no cost:** `Trombone::articulators()` reads the tract as
  it stands; it allocates nothing and leaves the sound alone.
- **A stream:** a `VoicePlayer` sends a frame every 10.67 ms (512 samples
  at 48 kHz; every 470.4 at 44.1 kHz) through a lock-free ring of 64, each
  with the sample it was taken at; the controller takes them with
  `VoiceStream::poll_frame()`. Nothing allocates or waits in the player; a
  controller that looks away for more than two thirds of a second loses
  the oldest.
- **Baked:** `Talker::bake_with_articulators` (or `record_articulators` and
  `take_articulators` around any rendering), and `say --export-lipsync`,
  which writes `NAME.visemes.json` beside the WAV: `{"format":
  "synthie-visemes-1", "sample_rate": 48000, "frame_samples": 512,
  "channels": [...], "frames": [[...], ...]}`, frame `k` at sample
  `k · 512` of the file. The file's sound is byte for byte the same with
  and without the export.
- **The terminal app** draws the mouth from the front beside the side
  view in a terminal 100 columns wide or more: the lips' outline drawn in
  as they round and shut as they seal, the teeth, the tongue tip, and the
  viseme in a word (lips sealed, lip to teeth, pucker, funnel, tip shut,
  tip up, open, half open, close).

Measured (`tests/lipsync_gate.rs`, `tools/judge/tests/test_lipsync.py`):
through "Papa. Mama. Baby boy. Bob may buy more. My mom, by a bear." on
the deep and the bright voice, at every moment the mouth acoustically
shuts (the lips' lumen under the engine's shut area, sample by sample)
the first frame at or after it reads `lip_close` at 0.98 or more, and the
frame two before it does not: the track is within one frame of the sound.
The jaw is the engine's own, which stays nearly shut for /ʌ ɔ o/ (0.17,
0.14, 0.07): what opens the mouth there is not in this model.

### Vocal effort

An `Effort` is a regime of the body, not a way of pronouncing: the same
sentence, by the rules or with a neural prosody, whispered, shouted,
spoken by someone exhausted or by someone whose voice shakes. The words,
their timing and their melody stay as they are; what changes is the
lungs' pressure, how the folds are held, the tube above them, the pitch
the effort lifts or lets sink, the jaw, the articulators' speed and a
tremor.

| State | What the body does |
|---|---|
| `whisper` | the folds held a slit apart behind an open chink (0.25 cm²) so that they never meet; the sound is the jet through them, with the lows that excite the formants; lungs 1.25 times the voice's |
| `shout` (`projected`) | lungs 16 cmH2O (2 times), tenseness +0.2, the epilaryngeal tube's area 0.6 of itself, pitch +4 semitones, the jaw 1.3 times as open; no creak, deeper breaths |
| `exhausted` (`tired`) | lungs 6 cmH2O (0.75 times), tenseness −0.1, a fifth of the folds' tension given up, pitch −1.5 semitones, articulators at 0.8 of their speed; a deep breath first, statements ending in 200 ms of creak, 12 % slower |
| `nervous` (`sob`) | pitch (0.05) and lungs (0.17) trembling at 7 Hz, the voice's attack deeper, three times as long held and twice as uneven, pitch +1.5 semitones; a breath first, 6 % faster |

```sh
say --engine physical --effort shout "Run!"                       # a state for the whole text
say --engine physical --neural M2 --effort exhausted "I can not go on."
say --engine physical "Wait. [whisper] I heard something. [shout] Run! [normal] It is gone."
```

```rust
use synthie_trombone::{Effort, Engine, Preset, Talker};

let voice = Preset::Deep.settings().with_engine(Engine::Physical);
let mut npc = Talker::new(48_000.0, voice.with_effort(Effort::Whisper));
npc.say("Did you hear that? [shout] Run!");
// … or while it sounds (it glides in over a tenth of a second):
npc.set_effort(Effort::Nervous);
```

The state is a number of the voice's tuning (`prosody.effort`: 0 normal,
1 whisper, 2 shout, 3 exhausted, 4 nervous; `VoiceSettings::with_effort`,
`Tuning::with_effort`), so profiles, talkers and streams carry it, the
voice takes the regime when it is given its settings, and speech is
compiled for it (pace, the breath before the utterance, the creak). A
voice that is sounding glides into a state with `Trombone::set_effort`,
`Talker::set_effort` or `VoiceStream::set_effort`; in text a tag
(`[whisper]`, `[shout]`, `[exhausted]`, `[nervous]`, `[normal]`, and the
other names) switches from the next sound on, with no pause of its own;
the next utterance starts again in the state it was compiled for. What
the speech, the score or the caller ask of the voice meanwhile (pitch,
tenseness, slack, lung pressure) is kept as asked and laid under the
regime. `--effort` with a number is still the persona's dial.

Measured on "We were away a year ago. Mary may win more money.", the deep
voice (`examples/effort.rs`, `tests/effort_gate.rs`):

| State | Level | Pitch | 2.5–5 kHz against 0–1 kHz | H1−H2 of the flow | Lungs (median) | Seconds | Jaw |
|---|---|---|---|---|---|---|---|
| normal | −25.6 dB | 116 Hz | −24.7 dB | 13.8 dB | 6.2 cmH2O | 4.10 | 0.60 |
| whisper | −39.4 dB | no voice (the glottis never under 0.03 cm²) | 0.0 dB | | 7.7 | 4.10 | 0.60 |
| shout | −16.0 dB | 145 Hz | −20.0 dB | 12.3 dB | 12.9 (16 at its fullest) | 4.34 | 0.78 |
| exhausted | −30.5 dB | 87 Hz | −24.6 dB | 14.1 dB | 4.6 (5.4 at its fullest) | 4.97 | 0.60 |
| nervous | −24.9 dB | 126 Hz, trembling 0.043 at 7.0 Hz (normal 0.017) | −24.0 dB | 13.4 dB | 6.2, trembling 0.08 | 4.13 | 0.59 |

Whisper's words (single runs; the four sentences as one clip, ten tuning
and ten held-out sentences), by the rules and, on the clip, with
tiny-voca's M2 prosody:

| State | Clip | 10 tuning | 10 held-out | Clip, `--neural M2` |
|---|---|---|---|---|
| normal | 24/31 | 81.5 % | 81.2 % | 28/31 |
| whisper | 21/31 | 43.2 % | 31.2 % | 21/31 |
| shout | 25/31 | 85.2 % | 82.5 % | 28/31 |
| exhausted | 24/31 | 84.0 % | 81.2 % | 26/31 |
| nervous | 26/31 | 76.5 % | 71.2 % | 27/31 |

The whisper is the weak one. It had been at 7 % of the tuning sentences'
words: with the voice asked for, the breath through the folds was taken
for that of vibrating folds and lost everything under 5 kHz, so nothing
excited the formants. In a whisper the folds are now not counted as
singing (25 %), and they are held a slit apart (tenseness 0.11) and not
a breath's width (43 %; at 0.06, 37 %). A recognizer trained on voices
still loses most single whispered sentences. The existing `whisper`
voice (tenseness 0) is as it was.

Thrown at a speaking voice in any order (states, morphs, coughs and
laughs, lung pressures, tenseness, block lengths; six seeds of a fuzz in
the gate) the sound stays finite and bounded, every utterance ends, and
the next sentence comes out within 6 dB of a fresh voice's.

### Interrupting and knocking a voice

A voice that is speaking can be broken off in the middle of a word, or
its speaker knocked, without anything in the tube being reset: the waves
that are in it go on through whatever the body does next.

```sh
say --engine physical --interrupt 1.25,cough --then "Excuse me. The juice of lemons makes fine punch." \
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."
say --engine physical --perturb 0.6,8 --perturb 1.6,-15 --perturb 2.9,6,1 "We were away a year ago. Mary may win more money."
```

```rust
use synthie_trombone::stream::{When, voice_stream};
use synthie_trombone::{Engine, Preset, Trombone, Vocalization};

let (mut stream, mut player) = voice_stream(1024);
let mut voice = Trombone::new(48_000.0);
let settings = Preset::Deep.settings().with_engine(Engine::Physical);
settings.apply(&mut voice);
stream.set_prosody(settings.prosody());
stream.push_text("The birch canoe slid on the smooth planks. ");
// … in the audio callback: player.render(&mut voice, &mut block) …
stream.interrupt_with(Vocalization::Cough, When::Now).unwrap(); // breaks in mid-word
stream.push_text("Excuse me. ");                                // said after the cough
stream.perturb(8.0, 0.0, When::Now).unwrap();                   // a bump to the chest
stream.interrupt_with_text("No. Run!");                         // other words instead
```

- **`interrupt_with(vocalization)`** (`VoiceStream`, `Talker`, `Speaker`;
  stream event tag 44; `F7` coughs in the terminal app): the speech's
  gestures let go where they are (a closure opens and the air behind it
  leaves as it would, the folds part), what was queued is dropped, the
  larynx is brought up to date at the next control block and not at the
  end of its own 10.7 ms block (`Trombone::react_now`), and the cough,
  sigh, laugh, chuckle or throat clearing starts there. What is said
  afterwards waits for it to be over. `interrupt_with_text` breaks off
  with other words instead.
- **`perturb(delta_lung_pressure, jaw_clamp)`** (`VoiceStream`, `Talker`,
  `Trombone`; event tag 45; `F8` in the app): a push on the lungs in
  cmH2O that reaches them in 6 ms and dies away in a tenth of a second
  (negative: a gasp; below nothing the lungs draw and the air turns
  inward), and a clamp of the jaw that carries the teeth and the lips up
  to three quarters of the way together and lets go in a fifth of a
  second. Knocks add up; the speech goes on.
- **A soft landing.** A command taken at once (stop pushing, part the
  folds) starts the lungs and the folds moving with a kink, and a kink in
  the flow through the glottis is a step in the sound: where a voice was
  cut off, 3 % of the speech's peak (30 dB under it). For a quarter of a
  second after an interruption or a knock, what the lungs and the folds
  are told reaches them through one more lag of 4 ms, so they start with
  no kink; and folds cut off while singing count as singing for 35 ms
  more, so their last pulses keep the voice's tilt (without it they came
  out 1.7 times as steep).

Measured against the same voice left alone, sample by sample
(`examples/interrupt.rs`, `tests/interrupt_gate.rs`), the deep and the
bright voice broken in on in the middle of a vowel, a stop's closure, an
/s/ and a nasal by each of the five vocalizations:

| Broken in on | Larynx or lungs answer after | The sound differs after | Largest step of the difference, first 3 ms | Steepest step in the next 60 ms against the speech's steepest |
|---|---|---|---|---|
| a vowel | 2.1 to 2.4 ms | 1.2 to 2.4 ms | 45 to 51 dB under the speech's peak | 0.80 to 0.84 |
| a stop's closure | 3.0 to 3.5 ms | 1.1 ms | 50 to 60 dB under (bright voice 36) | 0.12 to 0.26 |
| an /s/ | 1.3 ms | 1.4 ms | (the two are different noises) | 0.94 |
| a nasal | 3.3 to 3.5 ms | 1.5 ms | 32 to 40 dB under (bright voice 29) | 0.15 to 0.31 |

The answer comes at the next control block: 2.7 ms at most on the
physical engine as it is set up, 10.7 ms with a control block of 512.
Under 40 dB holds where the cut is all there is (the vowel); where a
closure opens with a voice or pressure behind it, what is heard is that
closure's own release, a fifth of the steepness of a stop's burst in
speech. Afterwards the next sentence is as long and as loud (within 3 dB)
as from a voice that was never broken off, and it does not start while
the vocalization lasts. Whisper on `28-s4-interrupt-cough.wav`: "The birch
can usually- Excuse me, the juice of lemons makes fine punch."

A knock, in the middle of a vowel: a push of 8 cmH2O raises the lungs by
3.7 to 4.5 cmH2O and the sound by 3.3 to 5.7 dB; a gasp of 15 takes the
lungs 7.3 to 8.0 under and turns the glottal flow inward (to −160 cm³/s
and more); the jaw knocked shut brings the lips 0.17 and more together
and muffles the vowel by about 1 dB; each starts with a step 43 dB and
more under the speech's peak and is gone 0.8 s later.

## The physical engine

Next to Pink Trombone's own model there is a second one, `Engine::Physical`,
that uses the same tract shape and controls but works in real units and
lets the sound come out of the air flow instead of a script:

| | Classic | Physical |
|---|---|---|
| Voice source | a fixed pulse shape | vocal folds as masses on springs, driven by lung pressure through the glottis; pitch from their tension |
| Losses | the same at every frequency | lip radiation, yielding walls and viscous losses, each with its own frequency dependence |
| Side cavities | one lossless nose | two piriform sinuses, two nasal passages with a sinus each |
| Hiss | noise wherever a finger narrows the tract | turbulence where a jet is fast enough, loudest where it strikes the teeth |
| Stop bursts | a scripted pop | pressure builds behind the closure and is released |
| Articulators | constant speed | critically damped, as tissue with inertia |

```sh
cargo run --release -p synthie-trombone-cli -- --engine physical   # `\` switches engines while playing
cargo run --release -p synthie-trombone --example say -- --engine physical "Hello there!"
cargo run --release -p synthie-trombone --example ab                # both engines side by side, in report/
cargo run --release -p synthie-trombone --example probe             # what each engine measures
```

```rust
use synthie_trombone::{Engine, Preset, Talker};

let voice = Preset::Deep.settings().with_engine(Engine::Physical);
let line = Talker::new(44_100.0, voice).bake("Stay a while, and listen.");
```

The physical voice also listens to itself. Each block it reads what its
own air is doing and corrects two things a person corrects by ear: a gap
that is meant to hiss but is shut, choked, or too fine for its stream to
turn turbulent is widened until it does (and the place remembers, so the
next fricative there starts right); and a vowel that starts to hiss, as a
close /u/ does in a loud voice, is opened a little around the place that
hisses and let back when it is quiet. Neither moves anything by more than
0.15 in diameter, and the default voice is left as it was. That is what
lets a persona make the body small or the breath strong without breaking
the consonants, and what lets a finger in the app find a fricative without
landing on its exact width. `examples/hiss.rs` prints the levels over body
size and lung pressure; `Trombone::set_regulation(false)` switches the
correction off to compare.

It is held to published measurements of real voices by tests
(`crates/synthie-trombone/tests/physical_gates.rs` and
`src/phys/tests.rs`); the classic engine is pinned bit for bit by
`tests/classic_golden.rs`, so it is the unchanged baseline to compare
against. A few of the figures:

- Formant bandwidths of 30 to 110 Hz, rising with frequency (classic: 70 to
  1900 Hz), and vowels fitted to Peterson & Barney's F1, F2 and F3.
- The walls resonate against the enclosed air near 190 Hz and give about
  2.7 cm³ per cmH2O, as measured in vivo (Hanna, Smith & Wolfe 2016).
- The piriform sinuses cut a notch near 4.4 kHz (Dang & Honda 1997).
- The narrow tube above the folds, the epilaryngeal tube, has numbers of
  its own (`AnatomyConfig::epilarynx_*`: its length in cells, its width,
  the width of the lower pharynx it opens into, a step or a ramp, and how
  far it narrows as the folds press together). The original's is seven
  cells of 0.45 cm² (2.55 cm) opening into 1.52 cm²; the default, the
  anatomical tract's, is four cells (1.46 cm) flaring into it over the next
  four (see "Two tracts" below; its tube's resonance stands near 3.8 kHz).
  The original's own resonance is F4, at 3.3 to 3.9 kHz in every vowel
  shape, with F3 near 2.6 kHz: no cluster at 3 kHz but in /i/. Eight cells (2.9 cm)
  of 0.25 cm² opening into 3.6 cm² ring at 2.9 to 3.2 kHz (as F3 in the
  front vowels, F4 in the others) and lift the transfer function's 2.8 to
  3.2 kHz band by 8 dB against 0 to 1 kHz; four cells (1.46 cm) put the
  tube's resonance near 4 kHz. Measured in speech, the cluster cost
  words; the four cells gained vowels but, opening in a step, left open
  vowels short of their F1 (`examples/epilarynx.rs`,
  `tests/epilarynx_gate.rs`).
- The folds sing within a cent of the requested note over every preset's
  range, stop below about 3.4 cmH2O of lung pressure, and keep voicing for
  some 40 ms into a /b/ before the mouth fills.
- /s/ and /sh/ come out about 10 dB louder than /f/ and 14 dB louder
  than /th/, held steady (the LibriSpeech reader's /f/ is 10.5 dB under
  his /s/). For /s z/ the jaw brings the incisors together: the jet
  strikes them through a narrow gap, and the small cavity behind them
  puts the peak of /s/ at 7 to 8 kHz (centroid 6 kHz, as a speaker's),
  with /sh/ 2.5 kHz lower. The jet of /f v/ strikes the edges of the
  upper teeth and the upper lip, and that of /θ ð/ the edges of the
  teeth: the force of its eddies there radiates straight out as a dipole
  (Curle 1955; Shadle 1985), flat from 2 to 7 kHz, without passing
  through the narrow gap behind it, which let the free jet's noise out
  some 20 dB weaker.
- Out of /s z ʃ ʒ/ into /f v/ the lower lip comes up to the teeth while
  the tongue still holds the sibilant and lets it go gradually: the hiss
  of /s/ fades over some 20 ms under the closing lip with no gap before
  the /f/ and no rush of air (lip flow up to about 480 cm³/s, where the
  tongue letting go at once let 950 out before the lip shut).

It costs more: about 6 times faster than real time on one core at the
native tract length, against 170 for the classic engine. With the default
`Fallback::Auto`, notes outside the folds' range are sung with a
prescribed glottal pulse instead: below it as vocal fry, above it as a
falsetto the folds have no register for. For the native-size voice at
modal tenseness that range is about 56 to 288 Hz, and it scales with the
inverse of the size (the critter voice's is 93 to 488 Hz, so the peaks
of its lively speech, 490 to 520 Hz, are sung by the pulse). Inside the
range the folds sing every note, stalls included; `Fallback::Never`
keeps the pulse out altogether, `GlottalSource::Pulse` uses nothing
else, `Trombone::fold_range` gives the range and `Gauges::sounding`
says which is singing. Body-cover folds under muscle control (Titze &
Story 2002) are implemented and tested but not yet what sets the pitch.

### How it speaks

Speech on the physical engine is made the way a speaker makes it, as far
as the model goes:

- **The voice.** In modal speech the folds close on every cycle, so the
  voice keeps a steady H1–H2 instead of drifting breathy, and the folds
  carry every sentence of the default voices; the pulse only takes notes
  out of their range (above). These folds shut abruptly: the derivative
  of their flow has its 2 to 4 kHz band some 20 dB under its 0 to 1 kHz
  band, where a Liljencrants–Fant source of a modal male voice (a return
  phase of 4 to 5 % of the period) has 33 to 37. So while the folds are
  asked to sing, the voice leaves through a first-order shelf that falls
  from its 500 Hz corner (where it is unity) and levels off from about 4
  kHz, 18 dB under the bottom (`voice_tilt`, `tilt_corner`;
  `src/phys/tilt.rs`), the same over every
  voiced sound; every noise source in the tract takes the shelf's inverse
  on the way in, so frication, bursts and breath are heard as without it.
  The breath noise of the glottal jet is pitched by the slit it comes
  through: while the folds vibrate it has little below 5 kHz
  (`voiced_breath_corner`) and is 6 dB stronger above it than an open
  glottis's at the same flow (`voiced_breath_gain`, round 10), where that
  of an open glottis in /h/ reaches down to 400 Hz. Measured in the
  LibriSpeech texts on round 10's engine, the accented vowels' 1 to 2 kHz
  band stands 21.0 dB under 0 to 1 kHz (the reader 20.9; 12.9 before
  round 9), 2 to 4 kHz 28.8 (his 28.9; 16.5), and 4 to 7.5 kHz, which is
  this breath noise and not the voice's harmonics, 26.9 (his 25.7; 30.0
  with round 9's breath). The tilt can take a second section
  (`tilt_top`, `tilt_top_corner`); it is off: a gentler shelf with a
  steeper top, or a shelf 4 dB less deep, brightened 2 to 4 kHz past the
  reader's and changed no score by more than a point. The pitch of the
  folds has a cycle-to-cycle jitter with a 1/f-like spectrum
  (`cycle_jitter`: RAP 0.23 %, the reader 0.25 %, before 0.12 %), its slow
  wander and vibrato now move in the physical engine too (they were frozen:
  its glottis clock never ran), and a 6 to 10 Hz flutter (`pitch_flutter`)
  moves it within a vowel (0.20 semitones about the line through it, the
  reader 0.21, before 0.11). The softening of nasals, liquids and
  unaccented vowels by a filter switched per phoneme (round 8,
  `Trombone::set_fold_softness`) is off: the owner heard the accented
  vowels jump out where it let go.
- **Voiced and voiceless.** For /p t k f s/ the arytenoids spread the
  glottis behind the folds, slower than the folds themselves move, and the
  voice stops at the moment the mouth actually shuts (the larynx is timed
  from `Trombone::mouth_shut`, not from a script). After a /p t k/ the
  glottis stays open for a voice onset time that grows from the lips to the
  velum, and the voice comes back a little high. In /b d g/ the throat and
  cheeks widen so the air the voice sends into the shut mouth has
  somewhere to go, and the voice carries on through the closure. A vowel
  is shorter before a voiceless consonant that ends its syllable and
  longer before a voiced one, as in English; not before the first sound of
  the next word (until round 10 the /s/ of "slid" clipped the /u/ of
  "canoe" to 128 ms, where the reference recordings hold it 180 to 215).
- **Stops** build pressure behind the closure and burst by themselves,
  without a click at the release. Released into an /l r/ ("blue",
  "play", "cream"), a stop is made on the liquid's tongue body, its lips
  or tongue back let go first, and the liquid's tip follows some 20 ms
  later (`liquid_transition`), so the burst and the stop's own transition
  are heard before the liquid; after /p k/ the start of the liquid is
  voiceless.
- **Taps.** A /t d/ between a vowel and an unstressed one, inside a word
  after its stress ("water", "city", "little") or at the end of a word
  before a vowel in the same phrase ("slid on", "get it", "a lot of"), is
  an alveolar tap: one voiced segment of 20 ms (`tap_length`) whose tip is
  thrown at the ridge 15 ms before it (`tap_lead`) and let go at once, so
  the mouth is shut for 3 to 7 ms, little pressure builds behind it, and
  there is no burst and no breath after it. The LibriSpeech reader's taps
  dip 5 to 13 dB under the vowels beside them and are voiced throughout;
  the engine's dip 6 to 15 dB, where its /t/ between vowels was a 98 ms
  voiceless stop 45 to 57 dB down. Not before a stressed vowel, after a
  consonant, at the end of a phrase, or for a /t/ before a syllabic /n/
  ("button"); the classic engine has no taps.
- **Nasals** close the mouth with the velum open, and the tongue drops
  behind the closure so that the closed mouth is a side branch of the
  nose with the antiresonance of its place (`nasal_cavity`); behind the
  tip of /n/ it drops quickly (`nasal_cavity_rate`), so the antiresonance
  is at 1.45 to 1.55 kHz from the start of the murmur, also before a
  consonant. Out of /n ŋ/ into a close vowel the velum shuts three times
  as fast as elsewhere (`velum_release`, round 11), so the nose is shut
  15 ms into the vowel: left open longer, it took over the /u/ of
  "canoe". Out of /n/ the tip lets go six times as fast as its own
  spring (`nasal_tip_release`, wave 1), closure and shoulder together:
  at its own speed the gap out of /n/ opened at 2 to 8 cm²/s (Stevens:
  50 to 100) and the second formant overshot onto a fronted vowel and
  fell back, a /j/. Keeping the tongue up behind an /n/ before the
  fronted /u/ (`nasal_cavity_held`) is off since wave 2: its
  antiresonance stood at 2.1 to 2.3 kHz there, toward /ŋ/ by the
  owner's ear; dropped as everywhere else it stands at 1.45 to 1.55.
- **Two speeds of articulator.** The tongue body and the jaw are heavy and
  slow, so the vowel shape glides from vowel to vowel underneath the
  consonants; the tongue tip, blade and lips are light and quick and make
  the consonants on top of it, and the back of the tongue is in between. A
  consonant is made on the body of the vowel it leads into (the stops,
  nasals and liquids that end a phrase on its last vowel), and, unless it
  shapes the lips itself, hisses or is made by the back of the tongue,
  with that vowel's lips: the lips of /u/ round through the /n/ of
  "canoe" (at the back of the tongue's speed, so the vowel before it stays
  unrounded) and the /n/ opens onto /u/. Each cell moves in place towards
  its target; the vowel shape is never slid along the tract, and all of a
  vowel's cells move with one speed, so between two vowels the tract
  passes through the cross-fade of their diameters (the jaw, a little
  quicker, aside). An /l/ holds a low tongue body of its own; the last
  10 ms of an /l/ before a vowel are made on the vowel's body
  (`lateral_release`), so its tip lets go onto the vowel rather than onto
  the open mouth over the /l/'s body.
- **Vowels** are the averages for the 45 men of Hillenbrand, Getty, Clark
  & Wheeler (1995), solved into shapes against the engine itself, with lip
  protrusion and a jaw; some of them were moved by a few per cent against
  the speech recognizer (`src/vowel_targets.rs` says which and by how
  much). They are held longer than the classic engine's, since the slow
  tongue body needs the time to get there. The lax vowels /ɪ ɛ ʌ ʊ/ are
  shorter than the tense ones: by 38 % when unstressed ("the", "of";
  `unstressed_lax_length` 0.62) and by 18 % when stressed ("sells",
  "punch"; `lax_length` 0.82, the owner's choice in round 6, with a 60 ms
  floor on the held part, `lax_floor`), and glide part of the way towards
  schwa over their last third, except /ɪ/, which closes a little there as
  the LibriSpeech reader's and the owner's do; /eɪ/ glides the whole way.
  /ɪ/ itself (round 9) is closer and further back than Hillenbrand's (F1
  400, F2 1900 Hz), away from the start of /eɪ/, where the engine's stood
  in sentences (448 / 1998 Hz against /eɪ/'s 510 / 1930; the reader's /ɪ/
  385 / 1747, his /eɪ/ 422 / 1892). An /u/ after /t d s z/ is made further
  front (F2 1600 Hz, the plain /u/ 1000) and glides part of the way back
  over its second half (`fronted_nucleus`), as the reader's does ("two"
  1690 to 1300 Hz), and after /n/ too since wave 1 (`fronted_after_n`;
  the owner's "canoe" holds 1860 to 1910 Hz); after the postalveolars
  and /j/, fronted the same way, it was heard as /ju/ ("the juice" as
  "the use"), so there it is the plain /u/. An unstressed /ə/ before a
  single consonant of the lips or the tip and then /i ɪ/ or a fronted
  /u/, or before /j/, is made higher and further front (F1 400, F2 1750
  Hz against 510 and 1530; `schwa_colour`, wave 1): the reader's /ə/
  before a front vowel stands at 372 and 1604 Hz, the owner's in "canoe"
  near 300 and 1800. A stressed /æ/
  can be made to glide a fifth of the way to schwa (`ae_nucleus` 0.55);
  it reaches its formants without, and with the glide the recognizer
  hears more /æ/ as /ɛ/, so that is off. A vowel shape may also set the tongue root, which widens or narrows
  the pharynx on its own and stays with the tongue body under the
  consonants around the vowel (`Shape::pharynx`); the built-in vowels
  leave it at rest. Round 6 solved it for /ɪ ʌ ʊ/ (pulled back by 0.20,
  0.12 and 0.29; `LAX_ROOTS` in `examples/vowel_size.rs`): the vowels of
  the LibriSpeech texts were heard better (error 27 to 24 %), but the
  tuning sentences gained nothing and the carrier phrases lost.
  `examples/nucleus.rs` measures, for every vowel of a list of sentences,
  its length and its formants through it.
- **Phrases.** The lungs build their pressure up over the start of a
  phrase, run down over it and let go over its last vowel, so a sentence
  swells in and dies away; within it, each vowel starts at full strength
  and its breath sinks by 2.5 dB of level to its end (`vowel_decline`),
  and after a voiceless consonant the arytenoids close the spread glottis
  in 12 ms (`spread_closing`; opening it takes 20), so the vowel is within
  3 dB of its peak 7.5 ms after its voice starts, as the LibriSpeech
  reader's (19 ms before, growing for 40 ms). Each stressed vowel gets a
  pitch accent that rises from the start of its syllable, peaks 55 % of
  the way through the vowel (`accent_peak`) and sinks back, so accented
  vowels no longer all rise (median across them −0.3 semitones, the
  reader's 0.0, before +1.2); the pitch declines over the phrase, and a
  statement falls at its end.
- **The lateral, jets and the last /s/ (round 10).** The physical /l/ is
  a lateral: its gap is never one meant to hiss (since round 7 it had been,
  by its number alone, so the jaw shut the incisors in front of it as for
  /s/ and its shoulder was the sibilant's), and it is carved as a narrow
  channel from its place to the front of the ridge with a short shoulder behind, the
  blade drawn down. In the middle of the /l/ the tract's F1 is now 390 to
  430 Hz (before 490 to 500, above the /ɪ/ after it; the LibriSpeech
  reader's 310 to 450 by LPC, the owner's 310 to 350), F2 1030 to 1130
  (his 950 to 1200), F3 3140 to 3300 (theirs 2450 to 2900; a channel run
  back from the place brought F3 to 2700 but F2 to 1400, and was heard as
  /r/), held 25 to 30 ms before the release (the reader's 20 to 35 in a
  cluster). In 36 short phrases the recognizer hears 28 of 28 /l/ after a
  consonant (before 22, 3 as /r/; after /s/ 4 of 8 before), 11 of 16 at the
  end of a syllable (7). While a gap is shutting fast, its jet is no
  faster than the pressure behind it (averaged over 4 ms) can drive it:
  bounded by the lungs' pressure alone, the averaged flow lagged the
  closing gap and the jet of a closing tip ran at its fastest, the loudest
  moment of an utterance (the tap of "a pot of" 8 dB over its vowel's
  loudest sample); fricatives keep their hiss. And a consonant after the
  last vowel of an utterance takes that vowel's tongue, not a /k g ŋ l r/'s
  own: the last /s/ of "planks" had taken the /k/'s velar place and was a
  hiss 56 dB under the vowel (now 18 dB). On the round's merged engine
  (single runs; without these three in brackets): phoneme error over the 30
  held-out sentences 25.9 % (27.4 %), Whisper's words 74.8 % (69.6 %), the
  ten `sentence_heldout` sentences 19.2 % and 81.2 % (21.2 and 76.2 %),
  vowels in the LibriSpeech texts heard wrong 25.5 % (23.8 %; the lateral
  alone costs about 2 points there, steady over three runs, mostly in
  slots away from any /l/), consonants there right 85.7 %; the
  four-sentence clip "The birch can also run the smooth planks. Glue the
  sheet to the dark blue background. She sells seashells by the seashore.
  The juice of lemons makes fine punch." (before: "... by the sea shawl").
- **Phrases against speakers (round 10).** Measured phone by phone against
  the LibriSpeech reader's 16 sentences (his own alignment, the rules saying
  the same text with a comma where he paused) and the two reference
  recordings, the rules' phrases were too even: his vowels lose some 3 dB
  of level over a phrase, the rules' gained 1; his accented vowels stand
  2.7 dB over the rest, the rules' 1.2; his pitch spans 10.4 semitones, the
  rules' 7.2. So the lungs now lose 30 % of their pressure over a phrase of
  two seconds or more (`phrase_decline`, 0.15 before; a shorter phrase in
  proportion to its time), push 1.5 dB less through an unaccented vowel and
  the nasals and liquids beside it, the obstruents keeping their breath
  (`unaccented_level`), and the accents rise by 4 semitones in the default
  voice (`accent_size` 8, 6 before). The reader draws out the consonants
  that close a phrase (1.2 to 1.6 times the rules' at his pace), so they
  are 1.25 times longer (`final_coda_stretch`); the references pause 0.54
  and 0.62 s between sentences, so the physical engine pauses 0.55 s there
  (`sentence_pause`, 0.38 before); function words' vowels are held 8 %
  more loosely (`light_laxness`; the reader's are 1.3 to 1.7 dB breathier
  in H1*-H2*); and a vowel is no longer clipped or drawn out by the first
  consonant of the next word. On the LibriSpeech texts the phrase fall
  (`natural.py`, -6 to -30 dB of the peak) went from 153 to 279 ms (the
  reader 710), the naturalness distance from 1.05 to 1.01 (its pitch terms
  0.76 to 0.59), the vowel levels' correlation with his from 0.25 to 0.30.
  Neither the place of the accent nor the length, level or pitch of the
  /u/ makes Whisper hear "canoe" in sentence 1 alone: every variant tried
  gave "can also", "can sleep" or "can use".
- **Transit and hold (round 11).** Played at 0.78 of its speed in an
  editor the engine is heard better, and it is heard better still when it
  speaks at that rate itself (the 30 held-out sentences' phoneme error
  25.9 to 19.2 %, the tuning sentences' 22.0 to 18.1 %); a larger tract at
  a lower pitch at the normal pace is heard worse (27.0 %), so the gain is
  the time, not the spectrum. Of that time, the consonants' share counts
  most (their lengths alone 1.28 times longer: 22.3 %; the vowels' alone:
  24.0 %), and the owner reads sentences 1 and 2 1.29 times slower than
  the rules, almost all of it in the consonants (the LibriSpeech reader is
  1.14 times faster than the rules). The engine does not lack steady
  shapes: in the LibriSpeech texts its stressed vowels hold F1 within 40
  Hz and F2 within 80 Hz for 58 % of their length, the reader's for 46 %,
  the reference recordings' and the owner's for 42 to 50 %, and its short
  vowels fall short of its long ones by less than his do; its formant
  transitions out of a stop are over in some 20 ms, his in 35.
  Articulators 28 % faster at the same lengths made the vowels heard
  better and the consonants worse (85.7 to 83.3 % right) and the sentences
  no better; the same motion slowed as a whole (rate and every speed and
  lead by 0.78) gave 22.1 %. `body_damping` lets the tongue body settle
  without its critically damped tail; it is off (see `Tuning`).
  `examples/plateau.rs` writes the tract's areas, targets and own formants
  every 2.5 ms through any lines of text.
- **The dark /l/ and phrases opening on a sonorant (round 11).** The two
  points of vowel error round 10's lateral cost in the LibriSpeech texts
  (23.8 to 25.5 %) were vowels *before* an /l/: of the 26 vowels before an
  /l/ that closes its syllable ("quilter", "welcome", "also", "middle",
  "principles"), 11 were heard wrong before the lateral and 15 after (the
  rest of the 286 slots 59 and 61); the recognizer's own choice among the
  vowels stood at a margin of −0.3 nats there (the reader's +3.0). The
  reader's vowels before such an /l/ are dark all through, F2 near 1000 Hz
  from a quarter of the way in; the engine's held their own shape up to
  the /l/ (1460 to 1660 Hz) and fell some 500 Hz into it, the more so with
  the new, darker lateral. So the last 60 % of such a vowel now glides
  towards /ʊ/, the back of the tongue rising for the /l/ before its tip
  (`dark_l_share`): there 10 of the 26 are heard wrong, at a margin of
  +0.2. And a phrase that opens on a nasal or a liquid starts that sound
  50 ms earlier, out of the pause (`sonorant_onset`): the lungs take some
  40 ms to bring the folds going, and that much of an opening /n m l/ was 33
  dB under its vowel, where the reader's /n m/ stand 8 to 20 dB under it
  from their first 10 ms; the recognizer heard no /n/ in four short
  phrases opening on one (now four of four), and 4 of 12 opening /l/ (now
  6, the rest still heard as /v ð/ before a front vowel). In the
  LibriSpeech texts the vowels are heard wrong 24.5 % of the time, the
  consonants right 88.2 % (85.7); the 30 held-out sentences 25.9 % and
  74.3 % (25.9, 74.8), the 30 tuning sentences 22.7 % and 72.4 % (22.0,
  73.2). Measured but not changed: a voiced /g/ builds almost no pressure
  behind its closure (some 1 cmH2O against the lungs' 5 to 7, with or
  without the pharynx widening), so its release is 21 to 37 dB under its
  voice at 1 to 2.5 kHz where the reader's is 4 to 12, and the recognizer
  hears /g/ as /d/ before stressed and unstressed vowels alike (8 of 16 /g/
  before an unstressed vowel right in short phrases, with or without the
  lighter unaccented syllables); and /ɪ/ in short phrases is heard as /eɪ/
  or /i/ after any consonant (1 of 26 right, after /l/ 8 of 19), not only
  after an /l/.
- **Rhythm (round 11).** Measured phone by phone against the LibriSpeech
  reader's 16 sentences, the two reference recordings and the owner's
  recording of sentences 1 and 2 (the rules saying the same text at the
  speakers' pauses; each speaker's durations over the rules', at the
  speaker's own pace; a fit of the log ratio on the phone's class and
  where it stands, the phrase's last syllable left out), the rules'
  syllables were too even, and not where the ear guessed: the vowels of
  stressed content words were not short (tense ones 0.89 to 0.99 of
  people's), but the consonants that open a stressed syllable were, for all
  four speakers (people's 1.30 to 1.37 times the rules'); each consonant
  of a cluster the reader holds for 0.81 of a single one (the rules made
  them alike); /ə/ is 0.69 to 0.76 of the rules' everywhere (the /ə/ of
  "canoe" was 100 ms, the reader's 65); the /ð v/ of "the", "of" 0.78, the
  reader's diphthongs 0.67. The normalised variability of successive
  vowel intervals (nPVI-V) was 30 to 43 for the rules and 57 to 58 for
  every speaker; the vowels' share of the time (%V) 48 to 51 %, people's
  40 to 46 %. So the physical engine's rules now draw out the consonants
  that open a stressed syllable 1.3 times (`stressed_onset_length`; not
  /f v θ ð h/, long already), shorten each consonant of a cluster to 0.85
  (`cluster_length`; not a sibilant closing one, which the reader holds
  longer), the consonants of function words but their stops to 0.9
  (`function_consonant_length`), every /ə/ to 0.75 (`schwa_length`) and
  the diphthongs to 0.75 (`diphthong_hold`), and hold everything else 2 %
  longer (`rhythm_pace`), the first and last syllable of a phrase excepted
  (where the breath is still building or the rules' own lengthening
  rules). The rate stays where it was: the four listening sentences last
  11.86 s (11.84), the reader's texts 10.96 phones a second of speech
  (10.78; the reader 12.37, the references 11.25, the owner 7.96: the
  owner's recording is 22 % slower than the rules, the reader 15 % and the
  references 10 % faster). Against the speakers the stressed onsets now
  stand at 1.02 to 1.07, the clusters at 0.92, /ə/ at 0.84 to 0.93; %V is
  46 to 50 %, the ratio of stressed to unstressed syllables 1.59 to 1.67
  (1.45 to 1.50 before; the reader 1.60, the references 2.5, the owner
  2.1); nPVI-V rose to 43 and 44 on the reference texts and stayed at 43
  on the reader's. Every consonant of a class 1.3 times as long instead
  (stops, fricatives, nasals or liquids, wherever they stood) gained 0.4
  to 1.4 points of phoneme error over the held-out and tuning sentences
  for 2 to 5 % more time; the stressed onsets and clusters alone gained
  2.9 for 1.4 %. On the engine of round 11 (main 18f28e1; single runs):
  the 30 held-out sentences' phoneme error 26.1 to 21.9 %, the tuning
  sentences' 22.3 to 20.8 %, vowels heard wrong in the LibriSpeech texts
  24.8 to 22.0 %, consonants right 88.2 % (88.2); Whisper's words fell,
  74.3 to 70.0 % and 72.8 to 71.5 %, and on the ten `sentence_heldout`
  sentences 81.2 to 71.2 % (every variant of the round that lengthened
  consonants lost 2 to 5 points of Whisper's words over the 60 sentences);
  the four-sentence clip went from 25 to 27 of 31 words ("The birch can
  use lid on the smooth planks. ... by the sea shore."). The naturalness
  distance to the reader is 1.013 (1.008), its timing part 0.26 (0.31).
- **The gesture score (Wave 1).** Speech on the physical engine is played
  as a task-dynamic gesture score (Saltzman & Munhall 1989; `gesture.rs`)
  on six tiers: `GlottalAdduction`, `LarynxAndPitch`, `Velum`,
  `TongueBody`, `TongueTip`, `LipsAndJaw`. Each segment's articulation
  becomes gestures on the tiers that make it (a /t/ is a tip gesture and
  a port gesture, an /m/ a lip and a port gesture); the larynx's
  settings and the pitch are driven gestures on the two laryngeal tiers;
  the velum's lead out of a nasal, a sibilant let go under the lower lip
  and the lip itself retarget or overlap gestures. Each tier blends the
  gestures active on it, weighted by activation and dominance, with a
  neutral attractor (a constriction let go opens, the velum shuts), and
  the tract's articulators move towards the blend as they always did:
  critically damped, each at the speed of its tier's stiffness
  (`ArticulationConfig`'s `TierDynamics`: `body_rate`, `articulation_rate`
  and now `lips_rate` are read through it). `gesture_score` 0 is the old
  segment path, kept for listening side by side (F6 in the app); with
  `gesture_overlap` 0 the gesture path renders bit for bit as it.
  The first family of rules is left to overlap by default
  (`gesture_overlap` 1): a consonant made by the tip or the lips, or /h/,
  brings only its own gestures, and the tongue body and the lips under it
  are those of the vowel it leans on, a gesture of the score spanning
  every consonant before that vowel; the tip's pull on the body is a weak
  tongue body gesture of the consonant's (dominance `w / (1 − w)` for the
  pull `w`), its own lips, or lips held unrounded under a hiss, a dominant
  lip gesture. The /u/'s rounding under the /n/ of "canoe" is that span:
  the /n/ has no lip gesture at all. Its gate (F2 into the /u/ after /n/)
  and every sentence set score as on the rule path (on the merged engine
  of Wave 1: the ten `sentence_heldout` sentences 22.0 % phoneme error and
  81.2 % of Whisper's words, the 30 held-out 26.3 % / 74.3 %, the 30
  tuning 22.2 % / 74.1 %, vowels in the LibriSpeech texts heard wrong
  23.8 %, consonants right 88.7 %, on both paths). Measured and left
  selectable: the liquid's tip on its way through the closure of the stop
  before it ("dark blue", `gesture_overlap` 2; the tip stood at 0.62 10 ms
  after the release where the rule's stands at 1.0 to 1.3, the stop's own
  transition mostly gone, and "glue" in sentence 2 alone lost its /g l/:
  the rule's phasing of the tip 20 ms after the release stays, as the
  liquid gesture's window); and the schwa as a weak body gesture
  (`schwa_dominance` 2) under the next vowel's span in place of the
  `schwa_colour` rule (`gesture_overlap` 4): its F2 before an /l/ falls
  from 1590 to 1398 Hz (the reader's 880), before a front vowel it stands
  as the rule's (1737 against 1746; the reader's 1604) but its F1 rises
  (489 against 420, the reader's 372; elsewhere 528 against 492, the
  reader's 460), and while the phoneme recognizer hears it better (the 30
  held-out 25.7 %, vowels wrong 22.7 %, consonants right 88.9 %), Whisper
  hears less (`sentence_heldout` 80.0 %, the four listening sentences
  71.0 % against 80.6 %). A series plant, the tiers' own task-space state
  sent to the articulators (`gesture_score` 2), only adds a lag: the
  stops of the LibriSpeech texts shut 21 ms into their phone instead of
  2 ms and 43 of 193 never do, and the speech falls apart (62 % phoneme
  error on `sentence_heldout`); the cells' contact, their aim past it and
  their quicker closing are the plant's own physics, not the tiers'.
- **A tract that changes length (Wave 2A, on by default since the switch below).** `phys/slide.rs` is a
  uniform tube of variable length: two delay lines read at a fractional
  delay by linear interpolation (passive at every fraction: at the half
  sample 0.04, 0.12 and 0.30 dB per pass at 3, 5 and 8 kHz; while it grows
  its reads are scaled by one less its growth, so it never gives back more
  than it took at any speed). Nothing can cross a slide in less than a
  tick, so a slide is at least one tick long, and one tick long it is a
  cell, bit for bit. With `lip_horn` above 0 (cm, the owner's 1.2) the
  lips' cell where rounding is centred (41) is a slide one tick plus the
  protrusion long, and the lumped lip tube is not used; the protrusion is
  a state of the tract on the lips' spring, aimed at `lip_horn` times the
  rounding the lips are told to make (a /b/ over an /u/ keeps them forward,
  lips passing through a rounded opening to a closure do not protrude),
  or at the `LipsAndJaw` tier's protrusion or `Trombone::set_lip_protrusion`.
  With `larynx_travel` above 0 cells 8 and 9 are one slide above the
  piriform sinuses, stretched up to 0.8 cm by a lowered larynx and
  shortened by up to one tick (0.36 cm) by a raised one, the larynx a
  spring of its own (`larynx_rate`, `larynx_damping`) aimed at the
  `LarynxAndPitch` tier's height or `Trombone::set_larynx_height`; the air
  the moving larynx sweeps is taken up at the slide's lower end (the
  piston; in a shut rigid tract lowering it 0.8 cm drops the pressure by
  69.1 cmH2O, as the 3.26 cm³ it adds to 63.8 says). Speech lowers it
  through voiced closures (`voiced_stop_larynx`) and in rounded vowels
  (`rounded_larynx`). Measured on the vowel shapes as solved for the old
  tract: the horn at 0.73 cm, the lumped tube's own length as a real tube,
  gives the same F1 to F3 within 0.1 % on /u ʊ oʊ w r ʃ/, so the lumped
  inertance was already the rounded lips' loading (without any protrusion
  /u/'s F2 stands 13 % higher, 1151 against 1018 Hz); 1.2 cm lowers /u/'s
  F1 and F2 by 2.3 and 5.0 %, /w/'s F1 by 5.6 %, and for the same F1 and F2
  the lips of /u/ can open from 0.24 to 0.30 cm² and those of the fronted
  /u/ from 0.40 to 0.52 (or /u/'s tongue come down from 2.05 to 2.17). The
  larynx lowered 0.8 cm lowers /u/'s F3 by 9 % and /ɑ/'s F1 to F3 by 3 to
  5 %; the slide at rest changes no formant by more than 0.05 %. The
  piston is small: 1.5 cm³ per cm lowered, 1.2 cm³ at 0.8 cm, where the
  pharynx's widening takes 8.7 cm³ in a 58 ms closure: lowered 0.8 cm
  through /b d g/ between vowels the voice bar rises 0.4 to 1.1 dB and the
  share of the closure voiced by up to 9 points, and the pressure behind a
  /g/ at its release falls a little further (the known limit: the widening
  drains the small cavity behind a velar closure), while the widening
  switched off costs 5 to 16 dB of voice bar. Every vowel moves off its
  target until the tables are solved again on this tract
  (`examples/vowel_size.rs`, then `examples/babble.rs`);
  `examples/tract_length.rs` measures all of this.
- **The tongue body from measured modes (Wave 2C, off).** With
  `tongue_model` 1 the physical tract's body, from the top of the tube
  above the folds (`epilarynx_cells`: cell 7 by default, 4 with four cells
  of tube) to where the lips begin (cell 38), is a measured
  neutral shape plus three principal modes, weighted by three numbers
  (`Shape::modes`, the `TongueBody` tier's `mode_1..3`), in place of the
  cosine hump. `examples/tract_modes.rs` computes them from Story's (2008)
  MRI area functions of eleven vowels of one man
  (`data/story2008_area_functions.csv`; a lossless tube of each row, with
  the lips' radiation inertance, gives the paper's calculated F3 within
  0.4 % and F2 within 2.4 %): over all 44 sections the mean and four modes
  carry 74.3, 14.7, 5.8 and 3.8 % of the variance, over the body's cells
  (4 to 38) 78.1, 13.7, 6.0 and 1.2 % (97.8 % with three; the fourth, over all
  sections the lips' and the front of the mouth's opening, is gone once the
  lips are left out, and the lips stay the `LipsAndJaw` tier's so that a
  vowel's rounding can span a consonant). Mode 1 is the pharynx wide and the
  front of the mouth narrow against the reverse (/i/ −1.6 to /ɑ ɔ/ +1.4:
  the owner's "jaw" mode, but without the jaw, which stays the lips' tier's);
  mode 2 a narrowing at the soft palate against an open middle (/u ʊ o/
  −1.3 to −1.9 against the rest: the front–back of the dorsum, as far as it
  is a separate thing in this speaker); mode 3 the whole lower tract wider,
  the lower pharynx most (/i/ +2.0, /ɑ u/ +1.0 to +1.2 against /æ ɛ ɪ/
  −0.8 to −1.2: the tongue root, which puts /i u/ against /ɪ ʊ/ and /ɑ/
  against /ʌ/, but /e o/ on the lax side). His shapes are scaled to the
  native tract as a similar tube (0.749 × the equivalent diameter; his own
  areas or a wider scale change the engine's formants of his vowels by
  about a point: a lossless tube depends on area ratios alone). His own
  epilarynx is four sections (1.5 cm) of diameters 0.51 to 0.61 (a uniform
  tube of the same air 0.56, the engine's 0.6; on the wave's tract, with
  the tables of 0.6, a tube of 0.56 heard the vowels wrong 24.1 % of the
  time against 25.5 %, the 30 held-out sentences 27.9 % and 70.9 % against
  28.8 % and 70.0 %); the body taken from the
  fifth over the original seven cells of tube put the tract's fourth
  formant at 3.8 to 4.3 kHz and the 4 to 8 kHz band 5 to 12 dB higher in
  every vowel, as four cells of tube do: the body begins where the tube
  above the folds ends (`Tract::modal_from`), and its measured lower
  pharynx takes the place of the epilarynx's flare. A tongue set by place and height (the app's
  handle, the consonants with a body of their own) is projected onto the
  modes (least squares over the cosine's cells); a consonant's tip draws
  the body by as much as it changes the vowel's cosine tongue (drawn
  towards the projection of the tip's own tongue, /ɔ/'s F2 rose from 1030
  to 1470 Hz under an alveolar, the cosine's to 1120, and it was heard as
  /ɑ/). The jaw follows the height of the cosine tongue the shape carries:
  following the modal body's own opening it opened /ʌ ɔ/ to 0.7 to 0.8 and
  /ʌ/ was heard as /ɑ/ 8 times in 18. Every vowel is solved again for the
  modal body (`src/vowel_modes_table.rs`), started from and pulled towards
  the speaker's shape for the eleven he said (a blend of his shapes for the
  others); on the default tract F1 and F2 land within 3 % at every size but
  /ɔ ɔɪ/ (F1 up to 3.7 % short in small bodies; on the wave's tract up to
  4.7 %, and /æ aɪ aʊ/ up to 3.2 %, where the cosine body misses /ɑ/ by up
  to 5 % and the modal body reaches it), the open vowels /ɑ ɔ æ aʊ ɔɪ/ and the end of
  /ʌ/'s glide with a narrowing of the lower pharynx as on the cosine body
  (the piriform sinuses, some 2 cm³ hung below, pull an open vowel's F1
  down by a quarter: the speaker's /ɑ/ on the engine is at 552 Hz with
  them, 745 without, his own 761 scaled). Solved, his vowels move by up to
  2.3 of their units from his own shapes (/ɔ/'s second number, /æ ɛ ʌ/'s
  third; /e o/ by under half a unit): one supine man holding vowels in a
  scanner, and targets that are 45 men's. The third number keeps /i/
  (+0.9) apart from /ɪ/ (−1.4) and /ɑ/ (+2.1) from /ʌ/ (−1.6), not /u/
  (−0.5) from /ʊ/ (0.0) or /e/ (−0.7) from /ɛ/ (+0.6). Measured on the merged engine (single runs): on the
  default tract of the time (the original one) the modal body hears vowels in the LibriSpeech texts wrong
  25.2 % of the time (the cosine 24.1 %), the ten `sentence_heldout`
  sentences 23.3 % phoneme error and 85.0 % of Whisper's words (22.0, 81.2),
  the 30 held-out 26.3 % and 77.4 % (26.3, 74.3), the 30 tuning 23.5 % and
  74.4 % (21.4, 74.1), consonants right 86.9 % (88.9); on the tract the
  wave is switched on with (`lip_horn` 1.2, `larynx_travel` 1,
  `rounded_larynx` 0.5, four cells of tube) the cosine body, solved again,
  gives 19.2 % vowels wrong, 16.3 % and 81.2 %, 20.4 % and 73.5 %, 17.1 %
  and 79.1 %, 90.5 % right, and the modal body 25.5 %, 24.5 % and 78.8 %,
  28.8 % and 70.0 %, 25.3 % and 74.1 %, 87.6 %. The modal body's front
  vowels are heard as well or better (/ɛ/ 0 % wrong on the default tract,
  the cosine's 17 %), its back ones worse (/ɔ/ 75 to 88 % against 62 %,
  mostly as /ɑ/; /ʌ/ 22 against 6 to 11 %), and the tense–lax confusions
  the third mode was to cure are few either way (9 of the cosine's 69
  wrong vowels; most are /ə/ heard as /ɪ æ/ and /ɔ ɑ/ heard as /ɑ æ/).
  The tables are solved per tract as well: a voice with the
  lengthened tract (`lip_horn` or `larynx_travel` above 0, or the tube
  above the folds off the original's) reads
  `src/vowel_size_wave_table.rs` or `src/vowel_modes_wave_table.rs`
  (`vowel_targets::lengthened`, `TractModel::of`), solved by
  `vowel_size -- --lengthened --both`; `babble -- --modal --lengthened`
  measures the modal body's babble map (`inverse::modal`), used by `mimic`
  (`Mimic::set_modal`), Formant Lock and a followed speaker's vowels on a
  modal voice.
- **Two tracts, each with its own tables and map (wave 2); the anatomical one is the default.**
  `TractModel::Original` is the tract above; `TractModel::Anatomical`
  (`Tuning::with_tract`) is the lip horn of 1.2 cm, the larynx's full
  travel lowered 0.5 cm in rounded vowels, and a tube of four cells above
  the folds that flares over the next four (`epilarynx_flare_cells` 4), as
  the measured speaker's narrow first four sections flare over sections 5
  to 8 (Story 2008; the mean of his eleven vowels there, scaled to the
  engine, is 0.63, 0.90, 1.08 and 1.05 cm², the ramp's 0.58, 0.85, 1.12
  and 1.39). Opening in a step into the 1.52 cm² of the original's lower
  pharynx, the four-cell tube put 1.2 cm³ under the open vowels' tongue
  and their F1 fell 3 to 6 % on the same shapes (/ɑ/ 780 Hz with seven
  cells of tube, 757 with four, 771 with the ramp); moving the piriform
  sinuses to the tube's top did not help (/ɑ/ 741 Hz: their 2.5 cm³ is
  as much compliance there). Solved again on that tract, with an open
  vowel's narrowing allowed down to cell 8 and the tongue no lower than
  3.4, every vowel meets its F1 and F2 within 3 % at every size (/ɑ/ was
  up to 5 % short, /ɔ ɔɪ aɪ aʊ/ 3 to 4 %), and the whisper's click into
  the /n/ of "British and", 6.6 times its loudest vowel sample with /æ/'s
  tongue at its lowest, is gone (1.6, as the original tract's 1.3). Each
  tract has a babble map of the cosine body (`inverse::Map::for_tuning`;
  `babble -- --lengthened` writes `src/inverse_wave_table.rs`, measured
  with the larynx at rest as `Trombone::set_shape` holds a shape), which
  `Mimic::set_tract`, a followed speaker's vowels, `say --vowels` and the
  app read; the free functions of `inverse` read the default tract's.
  The gates name the tract they hold. The anatomical tract with the cosine
  tongue is the default (`Tuning::DEFAULT`); `Tuning::with_tract` with
  `TractModel::Original`, or a profile setting `lip_horn`, `larynx_travel`
  and `rounded_larynx` to `0`, `epilarynx_cells` to 7 and
  `epilarynx_flare_cells` to 0, speaks with the original one.
- **A head and two ears (wave 3C; stereo is asked for, the mono mixdown off).**
  The tract's sound leaves by six ports, kept apart at the end of each
  tick: the lips (with the edge source of /f θ/), each nostril, the
  throat's walls and each cheek's. `phys/radiation.rs::Head` puts each on
  a rigid sphere of 8.75 cm (growing with the speaker) at its own place,
  pointing its own way, and sends it on its own path to each ear of a
  listener: Brown & Duda's one-pole sphere shadow from where the port
  points (the lips' highs beam forward), the straight line plus the arc
  round the talker's head and the listener's, 1/r; the mouth's sound comes
  back off the chest too (an image of the lips under a surface facing up
  and forward, 0.25 of it, low-passed at 1.5 kHz, 0.62 ms late straight
  ahead). All of it is relative to the mouth heard on its axis, which is
  what the engine was fitted to. `Trombone::set_stereo(Stereo::Binaural)`
  (or `Stereo::Pair`, an ORTF microphone pair) and
  `set_listener(Listener { .. })`, then `process_stereo` /
  `render_stereo`; `Talker` and the stream's `VoicePlayer` have the same,
  `Event::Listener` moves the listener from another thread, and the paths
  glide over 30 ms, so moving costs no click. `say --stereo [--listener
  AZ,EL,DIST] [--heard AZ] [--turn FROM,TO]` writes two channels, the app
  plays them with `--stereo`, the web engine has `renderStereo`. Measured
  (`tests/head_gate.rs`): straight ahead the ears are alike (a vowel to
  120 dB, a nasal to 29, the nostrils' own asymmetry); the talker heard at
  the listener's side reaches the far ear 0.78 ms later (Kuhn's 3a/c for a
  voice's lows; Woodworth's 0.66 for its highs) and 14 dB duller at 2 to
  6 kHz; from beside and behind the talker a vowel's 2 to 6 kHz fall 8 and
  12 to 14 dB, its lows below 300 Hz 2 to 3 dB, the voice bar of a /b/ (the
  walls alone) 2.2; from 60° below the face /m/'s 1.5 to 5 kHz are 5.8 dB
  stronger than from 60° above, a vowel's 0.5. The mono mixdown
  (`Tuning::head_mixdown` 1, the default since wave 3 was switched on) is the ports as heard straight ahead, each
  through its own beaming, with no path's delay: no two ports can cancel
  (the deepest notch two equal ports make below 8 kHz is 0.08 dB, where a
  listener's chest return makes 1.5), and it stays within 0.1 dB of the
  plain sum in vowels and 0 to −1.3 dB in nasals. Stereo costs about 5 %
  of the physical engine's time.
- **The tract and the tongue by name; affricates that hiss after their
  release (wave 3, `w3-juice`).** `say --tract anatomical|original` and
  `--tongue-model cosine|mri`, and the app's flags of the same names, lay
  `Tuning::with_tract` and `Tuning::with_tongue` (`TongueModel::Cosine`,
  `TongueModel::Measured`: `tongue_model` 0 or 1) over the voice after any
  profile; left out, the voice keeps its own. In the app they are
  top-level flags, as `--profile` is (the `sing` subcommand takes no
  profile, and its own `--tract` is the tract's length).
  `profiles/mri-tongue.json` is the measured tongue on the original tract,
  the same as `--tract original --tongue-model mri` field for field and
  sample for sample. The four (single runs; the ten `sentence_heldout`
  sentences and the 30 held-out ones, phoneme error and Whisper's words;
  vowels in the LibriSpeech texts heard wrong):

  | tract, tongue | ten held out | 30 held out | vowels wrong |
  |---|---|---|---|
  | anatomical, cosine (the default) | 15.9 %, 81.2 % | 21.1 %, 76.1 % | 20.6 % |
  | anatomical, mri (`--tongue-model mri`) | 26.5 %, 82.5 % | 28.3 %, 74.3 % | 24.8 % |
  | original, cosine (`--tract original`) | 22.0 %, 81.2 % | 26.8 %, 73.5 % | 23.8 % |
  | original, mri (`profiles/mri-tongue.json`) | 23.3 %, 87.5 % | 26.9 %, 77.8 % | 25.5 % |

  The 85.0 % and 77.4 % of Whisper's words that wave 2C measured with the
  measured tongue were on the original tract (the default then); on the
  anatomical tract it has always heard fewer (78.8 % and 70.0 % then).
  With `--tongue-model mri` alone the default voice loses phonemes
  everywhere and the listening sentences most (34.3 % phoneme error, 58.1 %
  of the words), and "juice" is heard as "dose" again.

  "The juice of lemons" was heard as "the dose of lemons" in the four
  sentences as one clip. The cause was not the channel's width: through
  the release of /dʒ/ the narrowest cell is as narrow as that of /ʒ/ (0.04
  to 0.05 cm²), on both tracts alike (the palate is the same in both), and
  narrowing it 0.05 or 0.1 diameter units lowers the Reynolds number
  further (at a given pressure it grows with the square root of the area)
  and silences a final /dʒ/. It was the air: the frication of /dʒ/ was
  made with the folds pressed as for its closure, so the channel carried
  some 20 cm³/s against the 90 of /ʒ/, the jet was turbulent for 45 to
  53 % of the frication between vowels (/ʒ/ 65 to 71 %) and for 16 to 24 %
  at the end of a word, where its hiss was 12 to 15 dB under a final /ʒ/'s;
  and the next sound's shape was sent 30 ms into its 60 ms frication, while
  the channel was still forming out of the closure (some 30 ms), so that it
  was narrow enough to hiss for about 10 ms. Now the frication of /dʒ/ has
  the larynx /ʒ/ would have in its place (inside a word the leak behind the
  folds and the widened pharynx, at its end the partly voiceless /ʒ/'s),
  and both affricates hand over to the next sound as late as a closure does
  (`release_lead`, not `lead`): /dʒ/ is turbulent for 60 to 68 % of its
  frication between vowels and 37 to 53 % at the end of a word (/ʒ/ 46 to
  75 %), its hiss within 7 dB of /ʒ/'s or louder, /tʃ/ as before from 1 dB
  under to 7 dB over /ʃ/ (`examples/affricate.rs`,
  `tests/affricate_gate.rs`). The clip is heard as "... The juice of lemons
  makes fine punch." (24 of 31 words), "Stand at the edge", "raging", "plunge
  in" and "He chose a cheap chair" are heard right where they were not.
  Before and after (the default voice): the ten held out 15.5 %, 83.8 % and
  15.9 %, 81.2 % (one sentence's first words, "The source", heard as "This
  horse"), the 30 held out 20.6 %, 75.7 % and 21.1 %, 76.1 %, vowels wrong
  21.0 % and 20.6 %, the four listening sentences 17.2 % and 15.2 %.
  Sentences without /tʃ dʒ/ are unchanged sample for sample.
- **Hiss.** /s/ is a jet on the teeth, brightest of the fricatives; /ʃ/ is
  broader and lower, with rounded but open lips.

Each of these has its numbers in `Tuning` (above) and its gate in
`tests/physical_gates.rs` or `tests/natural_gate.rs`.

### Speaking with a person's timing and pitch

`say --match-prosody schedule.json` speaks a text with the timing, pauses
and pitch contour of a recording of it, its level (within -10.5 to +5.2
dB), and since round 9 the speaker's voice: a baseline profile (tract
length, tenseness, breathiness, pitch) laid over the voice, each sonorant's
tenseness (`--prosody-voice`, `-tenseness`, `-speed off` leave a part
out) and, only with `--prosody-formants on`, each vowel's F1 and F2 (off
by default since round 10: with them the vowels were heard wrong more
often);
`tools/judge/prosody.py transfer speaker.wav` measures the schedule (the
text given with `--text`, from a `.txt` beside the recording, or heard by
Whisper) and renders both versions (its README has the details). The text
is compiled as always, so every articulatory rule stays; only the
durations, the phrase breaks and the pitch are the person's, and the pitch
kick after voiceless consonants is kept on top. `--prosody-pitch voice`
moves the person's contour to the voice's own median.

What it showed in round 8, on the 16 LibriSpeech anchors (the judge's
reference speaker) and the owner's recording of the first two test
sentences (the other two have no recording):

| | rule prosody | the person's | the person |
|---|---:|---:|---:|
| anchors: phoneme error | 17.5 % | 26.5 % | 6.4 % |
| anchors: Whisper words | 81.8 % | 72.7 % | 96.2 % |
| anchors: naturalness distance (`natural.py`) | 1.13 | 0.93 | 0 |
| owner's two sentences: phoneme error | 11.1 % | 9.3 % | 7.4 % |
| owner's two sentences: Whisper words | 13 of 16 | 13 of 16 | 16 of 16 |

- Of the naturalness distance, the pitch and timing measures move to the
  person's (their mean |z| 0.67 to 0.15); the warped spectral distance
  (15.8 to 15.4 dB), the level steps and walls and the hiss measures do
  not. Pitch and timing were about a fifth of the distance (0.22 of
  1.13); the rest is the tract, the source and how the level moves.
- With the owner's own timing and pitch Whisper still writes "The birch
  can you slide down the smooth planks": "canoe" and "slid on" are not a
  matter of prosody.
- The LibriSpeech reader speaks 14 % faster than the rules (median; from
  10 % slower to 38 % faster), and at his timing the engine is heard
  worse: the rules at his pace alone give 23.3 % phoneme error, his
  pattern of durations at the rules' pace 22.0 %. The rule durations are
  what this tract needs to reach its shapes; a person's shorter sounds
  leave the slow tongue body behind. The owner's recording is 24 % slower
  than the rules, and with that timing the engine is heard a little
  better.

In round 9 the transfer was grown into one of the speaker's voice
(`prosody.py transfer`; the same 16 anchors and the owner's sentences,
each part left out in turn from the final schedules; and the two synthetic
reference recordings the owner supplied, `report/listen/artificialsample.mp3`
and `artificalsample2.mp3`, whose text Whisper heard):

| anchors | rules | timing only | hybrid (everything) | without the voice | without tenseness | without vowels | without speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| phoneme error | 19.2 % | 22.8 % | 22.6 % | 21.0 % | 23.7 % | 23.6 % | 22.2 % |
| vowels heard wrong | 24.5 % | 27.3 % | 29.0 % | 24.1 % | 29.7 % | 26.6 % | 26.9 % |
| Whisper words | 79.4 % | 78.0 % | 78.9 % | 78.0 % | 77.0 % | 81.3 % | 78.0 % |
| naturalness distance | 1.05 | 0.86 | 0.88 | 0.88 | 0.89 | 0.89 | 0.89 |
| of it, warped spectral distance | 14.9 dB | 14.0 dB | 14.1 dB | 14.0 dB | 14.1 dB | 14.1 dB | 14.1 dB |

On the engine with round 9's source tilt, taps and fronted /u/. "Timing
only" has round 9's level range and 70 % floor on phone length; with
round 8's transfer the phoneme error was 26.9 % and Whisper 74.6 % (on
the engine before round 9's source and tongue changes, where the table
read: rules 18.3 / 24.1 / 81.3 % and 1.10; timing only 21.3 / 25.9 / 82.8
% and 0.89; hybrid 21.1 / 25.2 / 76.6 % and 0.91; without the speed-up
23.7 / 29.0 / 78.9 % and 0.92).

Single runs; Whisper over these 16 sentences moves by 4 to 6 points
between runs that differ only in the level loop's breath.

- Phones no shorter than 70 % of their rule length, with the tongue body
  sped up through those still shorter (up to 1.6 times), took the reader's
  own timing from 26.9 % phoneme error to 21.7 %; the speed-up was worth
  about 2.6 points of it before the source and tongue changes, and none
  after them (22.6 % with it, 22.2 % without). A faster tongue body in general did not help
  (the rules with it: 21.6 %); the /d/ he runs into the next sound ("and
  the") was the phone most often lost.
- The baseline voice puts the engine's formant scale within 0.5 % of his
  (5.5 % without) and H1*-H2* within 0.4 dB (0.9), but it did not bring
  the naturalness distance down (0.88 with it and without); the vowel
  formants cut the median F2 error against him from 20.0 to 17.0 % at 2.4
  points of vowels heard wrong; the tenseness took the correlation of the
  H1*-H2* contour with his from 0.27 to 0.59.
- On the two reference recordings (one utterance each, so a few phones
  either way): phoneme error against the engine's own phones 11 % for the
  Polly sample and 26 % for sample 2 with everything (by the rules 15 %
  and 13 %, with the timing only 7 % and 13 %); Whisper 14 of 16 words
  each.
- The floor by class (round 11). In the reader's 16 schedules the 70 %
  floor lengthens 258 of 736 phones by 4.7 s in all (7 % of the time;
  the references' 34 of 109, 0.5 s), most of it in vowels: the stressed
  ones of content words 34 %, those of function words 20 %, the
  unstressed ones of content words 16 %, the /ð v/ of function words 7 %;
  stops, sibilants and clusters a few per cent each. Left off one class at
  a time (16 anchors, his schedules spoken as fitted in round 9; phoneme
  error 22.5 %, vowels wrong 26.6 %, consonants wrong 17.0 %, Whisper
  81.8 % with it everywhere) it mattered most in the function words (off
  their vowels 24.7 %, off their consonants 25.2 % and Whisper 74.6 %),
  then the weak fricatives (24.7 %) and the stops (23.9 %), and not on
  the stressed vowels (22.2 %, Whisper 78.9 %) or the clusters (22.1 %).
  The floor lengthens 11 of the function words' 28 stops ("to", "but",
  "at"; 0.16 s), and off the stops the phoneme error rose by 1.4 points,
  so stops keep it whatever the word. So the floor is built by
  class (`speech::rhythm::held_in_transfer`): the stressed vowels of
  content words, the stops, the sibilants and affricates and the
  consonants of a content word's clusters keep `follow_shortest`, the rest
  take `follow_shortest_light`, and both are shares of a phone's own rule
  length, not of the length round 11's rhythm gives it. The light floor is
  0.7 as the rest: held only on those classes (`follow_shortest_light` 0,
  the floor asked for) the 16 anchors scored 28.2 % phoneme error, 32.2 %
  vowels wrong and 78.5 % of Whisper's words, against 21.0 %, 25.2 % and
  85.2 % with it (round 10's rules and floor 19.8 %, 23.4 %, 79.9 %); its
  phones lie 4.5 ms from the speaker's on average instead of 7.8.

The models and numbers come from: Steinecke & Herzel (1995) for the
two-mass folds; Story & Titze (1995) and Titze & Story (2002) for the
body-cover folds and muscle rules, with tissue constants as in Brad
Story's LeTalker 1.22; Titze (1984, 2002) for flow through the glottis;
Hanna, Smith & Wolfe (2016) for the walls; Dang & Honda (1997) and Dang,
Honda & Suzuki (1994) for the piriform sinuses and nose; Flanagan (1972)
for lip radiation; Stevens (1971) and Shadle (1985) for turbulence;
Peterson & Barney (1952) for the vowel targets.

### Neural prosody from text (tiny-voca, held)

`say --neural M2` (or `F2`, with `:SPEED` after it) speaks any text
with the timing, pauses and pitch of tiny-voca, a small neural
text-to-speech model (Apache-2.0, 4.5 M parameters), instead of the rules,
with no recording: the engine's phones for the text go to
`tools/judge/prosody.py neural`, which runs the model's ONNX graph on the
CPU (cut where its pitch, voicing and energy are computed: 3.5 ms a
sentence), lines its phones up with the engine's and writes the schedule
beside the output (`out.neural.json`); the text is then spoken following
it as with `--match-prosody`, the contour moved to the voice's own median
unless `--prosody-pitch speaker` says not. It needs the judge's `neural`
dependency group (`uv sync --group neural` in `tools/judge`, or
`SYNTHIE_NEURAL_PYTHON` naming a Python that has onnxruntime); the model
and its phonemizer are fetched into `tools/judge/.cache/tiny-voca` on first
use. The library, `speech.rs` and interactive play are untouched: without
`--neural` every output is the same to the byte. The model's own tempo is
23 % faster than the rules', so it is asked for 0.8 of it; its energy is not
laid on the breath by default (the rules' level already follows it, r =
0.68, and on top the phonemes were heard worse). A tag (`[sigh]`) keeps
the rules' pause for its vocalization, a filler the model's timing; the
sentence pauses are held at 0.5 s so that the breath between sentences is
drawn (in the model's 0.27 to 0.32 s it fell out), and the creak at a
statement's end stays. A floor on consonants (close-out of wave 4, on by
default; `prosody.py neural --consonant-floor 0 --spike 0` gives the
schedules of before to the byte) gives every stop, affricate, sibilant and
consonant of a cluster at least 0.85 of the rules' duration for it in that
place, taken from the vowels beside it in its own word, the model's pitch
and level carried along with each phone, and replaces single-frame pitch
spikes by the median of three (`tools/judge/README.md`, "Floor on
consonants, pitch spikes"). Measured on the deep voice (phoneme error /
Whisper words; single runs, and means of four renders with the pitch
jittered by 0.1 semitone):

| | rules | M2 | F2 |
|---|---:|---:|---:|
| ten held-out sentences | 15.1 / 81.2 % | 13.5 / 90.0 % | 20.8 / 68.8 % |
| four listening sentences | 16.2 / 87.1 % | 12.1 / 83.9 % | 15.2 / 83.9 % |
| the four as one clip | 12.1 / 77.4 % | 8.1 / 90.3 % | 7.1 / 87.1 % |
| 30 tuning sentences (four-run mean) | 16.5 / 74.5 % | 14.7 / 77.7 % | 19.5 / 74.6 % |
| 30 held-out sentences (four-run mean) | 21.0 / 77.1 % | 18.9 / 78.1 % | 20.8 / 71.2 % |

Before the floor the four-run means on the 30 held-out sentences were
22.9 / 73.8 % (M2) and 22.4 / 70.7 % (F2). With M2 Whisper writes the
clip "The birch can use slid on the smooth planks. Glue the sheet to the
dark blue background. She sells seashells by the sea shore. The juice of
lemons makes fine punch." (28 of 31 words in five renders of five, as
before the floor; "seashells" is Whisper's spelling, so "canoe" is the one
real loss); with F2 27 or 28. F2's contour marks the stressed syllables
less than M2's (79 against 93 % of its pitch peaks on a stressed vowel),
which keeps its words below M2's; scaling or smoothing it did not help.
A real-time path would put the `ort` crate in a std crate beside the
library (the CLI, not `synthie-trombone`), run the cut graph on a worker
thread once per sentence before it is spoken and hand the stream a
followed `Utterance`; the phonemizer (dizione, Python) would need a port.

### Breaths and creak (wave 3, stage 3A; on by default)

The physical engine breathes in between sentences and creaks at the end
of a statement, by default since wave 3 was switched on (below):
`ProsodyConfig::breath_intake` 5 with `breath_first` 0, `final_fry` 1
with `fry_time` 0.08. `say --breath-intake` (and the terminal app's
flag of the same name) takes the breath before the utterance as well,
`--no-breath` takes none, `--fry 0` ends statements in modal voice;
`VoiceSettings::with_breath(0.0, 0.0)` switches both off for a `Talker`
or the stream, `Tuning::with_living_voice(false)` all of wave 3 (the
head's mixdown and the hesitation fillers too). The stream, which compiles a sentence at a time,
opens a sentence with the breath the text's pause would have taken.

**Breathing in.** The lungs may draw below the air's pressure
(`Trombone::set_inhalation`, the glottal tier's `inhale` variable,
`Gesture::inhalation`, `Trombone::breathe_in`, `VoiceStream::breathe_in`):
the glottis is held open behind the folds as they draw
(`ArticulationConfig::breath_opening`, 0.6 cm² in all), the air goes in
through the tract, and what is heard is the noise of its own jets: the
glottal jet, and behind every narrowing of the tract a jet that forms on
the glottis's side of it (`Phys::tick` runs the jet rule in reverse while
the lungs draw; the folds feel the stream from above). Speech takes a
breath before an utterance (`breath_first`) and in every pause of at
least `breath_pause` (0.45 s: a sentence's, not a comma's), `breath_time`
(0.15 s) long, then 0.1 s of turn before the phrase; the mouth is readied
for the phrase's first vowel before it, the velum shut as for that vowel,
and the first sound's shape is sent half way through the turn. At 5 cmH2O
before a phrase of three seconds or more (seven tenths of it before one
of a second) the air goes in at 0.4 to 0.6 L/s, and the breath stands 31
to 37 dB under the vowels of the phrase after it (published speech
breaths: 20 to 40 dB under the speech); the turn's largest sample step
is under a fifth of the vowels' (`tests/breath_gate.rs`). In a pause the
breath takes its time from the pause; before the utterance it is added,
and every phone comes `breath_time` + 0.1 s later. Text with bracketed
phonemes and hand-made segments (singing) take no breath; a schedule
followed with `--match-prosody` keeps a breath where the speaker's pause
leaves room for it and drops it otherwise.

**Creak.** The standard folds are periodic wherever they vibrate (energy
at half the pitch 50 to 90 dB under the pitch's, `examples/breath.rs`).
Coupled more loosely between the two pieces of their cover, as a thick,
slack fold is (Titze & Story's rules give that coupling 0.065 of the lower
piece's stiffness, the standard folds 0.31), and pressed together, they
alternate strong and weak pulses by themselves: at 0.05 of the coupling,
pressed 0.005 cm and blown at 2 to 6 cmH2O, on /a i ə/ at 60 to 130 Hz,
the half-pitch stands 8 to 14 dB under the pitch
(`Trombone::set_fold_creak`, `AnatomyConfig::creak_coupling`,
`creak_press`; taken there within 10 ms they alternate from the third or
fourth pulse). Statements (`.` `!`) go into creak over the last
`fry_time` of their voice (0.08 s by default, stage 3A's 0.12 as an
option), never before the middle of the last vowel, up to the end of the
last sonorant: a voiced obstruent after it is spoken in modal voice (creaked
through, a final /ʒ/ lost 30 dB of its jet noise, a final /dʒ/ all of it,
the /z/ of "rose" 8.5 dB; `tests/living_voice.rs`). Questions and commas
do not creak. At 0.12 s, in the four listening sentences the last 100 ms
of voice alternate by 0.6 to 0.7 of their mean pulse height (0.03 to 0.2
without); at 0.08 s the 80 ms of creak alternate by 0.38 to 0.56 in the
breath gate's two statements. The share works as a switch: at 0.6 the
folds do not cross into alternation.

### Wave 3 switched on (`w3-switch`)

Since `w3-switch` the physical engine's default is the living voice
(`Tuning::with_living_voice(true)`): the mono output is the head's
mixdown (`head_mixdown` 1), the lungs draw a breath of 5 cmH2O in the
pauses between sentences (`breath_intake` 5, `breath_first` 0), and
statements creak over their last 80 ms of voice (`final_fry` 1,
`fry_time` 0.08). A single sentence is compiled exactly as without
breaths, and between sentences the breath takes its time from the pause:
the four listening sentences last as long as before. `Tuning::with_living_voice(false)`,
or a profile setting `head_mixdown`, `breath_intake` and `final_fry` to
`0`, gives back the engine before it sample for sample. The breath before
the utterance (`say --breath-intake`) and stage 3A's longer creak
(`fry_time` 0.12) stay optional: each costs about a point of phoneme error
on single sentences. Text with bracketed phonemes (the judge's probes) and
songs take neither breath nor creak. Measured (single runs; phoneme error
and Whisper's words; vowels in the LibriSpeech texts heard wrong; the
consonant probe `score physical --sets vcv`):

| | ten held out | four listening | the four as one clip | 30 held out | 30 tuning | vowels wrong | vcv |
|---|---|---|---|---|---|---|---|
| wave 3 off (`main` before) | 15.9 %, 81.2 % | 15.2 %, 83.9 % | 12.1 %, 77.4 % | 21.1 %, 76.1 % | 17.0 %, 74.1 % | 20.6 % | 60.2 % |
| the default | 15.1 %, 81.2 % | 16.2 %, 87.1 % | 12.1 %, 77.4 % | 21.0 %, 77.0 % | 16.5 %, 74.5 % | 20.3 % | 60.2 % |
| with `fry_time` 0.10 | 16.3 %, 81.2 % | 15.2 %, 83.9 % | 11.1 %, 77.4 % | 21.2 %, 76.5 % | | 20.3 % | |
| with `fry_time` 0.12 | 16.7 %, 81.2 % | 16.2 %, 83.9 % | 13.1 %, 83.9 % | 21.8 %, 75.7 % | | 20.3 % | |
| with the breath before (`breath_first` 1) | 16.3 %, 86.2 % | 19.2 %, 83.9 % | 11.1 %, 77.4 % | 21.7 %, 79.3 % | | 19.2 % | |

Whisper writes the four sentences as one text the same in every row ("The
birch can easily run the smooth planks. …", 24 of 31 words); of the last
words of the 30 held-out sentences one is gained ("boards", heard "bars"
with wave 3 off) and one lost ("reef", now "cliff").
With creak, probes creaked too at first and the consonant probe went to
64.0 %: they no longer do.

### Laughs, sighs, coughs and fillers (wave 3, stage 3B)

The physical engine can sigh, laugh, chuckle, clear its throat and cough
(`Vocalization`; `src/vocal.rs`). Each is a macro that lays gestures on
the tiers of the gesture score for one event, with the lungs' push as a
gesture of its own: the glottal tier has a `push` (cmH2O, in place of the
voice's own breath while above 0; `Trombone::set_lung_push`) and a `seal`
(the folds pressed shut, cm past touching; `Trombone::set_fold_seal`), the
larynx's tier a `squeeze` (the epilaryngeal tube drawn in;
`Trombone::set_epilarynx_squeeze`) and, since c3-valve, a `valve` and a
`flutter` (the ventricular valve, below), and a vocalization's breaths in are
Stage 3A's. Text takes them as tags, `"Well, [sigh] I suppose so.
[laugh]"` (exactly `[sigh]`, `[laugh]`, `[chuckle]`, `[clear-throat]`,
`[cough]`; anything else in brackets is phonemes as before), and a voice
as a call that can be made while it speaks: `Trombone::vocalize` (laid
over whatever plays), `Speaker::interject` and `Talker::vocalize` (the
sentence waits for it and carries on from where it was),
`VoiceStream::vocalize` / `Event::Vocalize` (wire tag 30). A tag ends the
phrase before it as a comma does (the voice held up), takes its own time,
and is followed by 0.15 s in which the mouth is readied for the next
phrase; the classic engine leaves a comma's pause. With the living voice's
breaths and creak (on by default since `w3-switch`) a tag is not
bracketed phonemes: the sentences around it breathe and creak as ever, but
no breath is taken in the pauses on either side of a tag (a sigh, a laugh
and a cough draw their own), and a statement before it creaks at its end
while a phrase cut by a tag mid-sentence ends as at a comma. A filler is a
phrase of its own ended as at a comma, so it creaks only where it ends a
statement ("Hmm."). `Vocal` sets how long
(`length`), how strong (`strength`: the lungs' pressure) and around what
pitch.

Measured on the deep voice at 100 Hz (`examples/vocal.rs`,
`tests/vocal_gate.rs`): a **laugh** is a breath in and five calls at 4.4 a
second (Bachorowski, Smoski & Owren 2001: 4.37), each made by a pulse of
the lungs of up to `laugh_pressure` (12 cmH2O, falling to seven tenths)
and a burst of the folds' adductors in time with it (Luschei et al. 2006),
on an open mouth held still: the voicing gate stays on through the bout,
each pulse blows the narrowed glottis into a call of some 0.13 to 0.18 s,
and between the pulses the folds stop by themselves and the breath goes
out through the glottis as it opens again (0.22 cm²). Pulses alone, the
glottis held at one opening, made the calls on a voice by itself (with
none, one long call), but in a sentence's pause the folds kept going
between some of them. The calls fall from 176 to 147 Hz and by 6 dB along
the bout. A **chuckle** is three such calls with the lips shut,
the voice out of the nose. A **cough** draws a breath, shuts the folds for
0.19 s while the pressure behind them builds to `cough_pressure` (40
cmH2O; people reach 50 to 100 and more in the chest, the model holds
finite and bounded to 100), and throws them open with the ventricular
valve above them (below): the flow peaks at 6.2 L/s 40 ms after the
opening (people: 6 to 12 L/s in 30 to 50 ms; 3.5 L/s before the valve),
the burst's loudest 10 ms some 11 dB over the speech of the same voice,
then a short breathy "uh". Folds pressed shut while the lungs already push are blown
open and vibrate (a pressed voice at 40 cmH2O); shut at rest first, their
pressed edges feel none of the pressure, so the closure holds. A **throat
clearing** is two weaker pushes (16 cmH2O), each held behind the shut
folds and valve and let out through the valve held narrow, its
aryepiglottic folds fluttering (below). A **sigh** draws
6.5 cmH2O (0.9 L/s inward), then lets out a long breathy voice (the
glottis held open: noise share 0.32 against 0.10 for a modal /a/) falling
from 120 to 90 Hz while the mouth relaxes from /ɑ/ towards schwa. None
clicks more than speech does (largest sample step against its 10 ms's
RMS: 4.6 to 7.5, speech 5.3). Laying one costs about 1 µs; playing it
costs what speech costs.

**Hesitation fillers** (`ProsodyConfig::fillers`, on by default; `0` speaks
them as words, as does `Tuning::with_living_voice(false)`): "uh", "um",
"hmm", "er" and "mm" (also "uhm", "erm", "hm") are marked by the front end
and spoken as fillers: a phrase of their own, held `filler_length`
(0.42 s; "um", "hmm" and "mm" a quarter more, "um" 45 % schwa and the rest
the hum), at `filler_pitch` (0.86) of the baseline falling a semitone and
a half (Shriberg & Lickley 1992: clause-internal filled pauses sit low,
fall a little and linearly), with `filler_pause` (0.3 s; six tenths after
"uh" and "er") of silence after; "er" is the schwa of "uh" (held as /ɝ/ it
was heard as "her"). Whisper writes them as "uh", "um" and "hmm" or leaves
them out. The judge's material has no fillers and no tags (none of its
words outside brackets is one): every benchmark is the same with them on.

### The ventricular valve (close-out, c3-valve)

A cough and a throat clearing are made with a valve of their own above the
folds (`src/phys/valve.rs`): the false folds, with the aryepiglottic folds
over them, at cell 1 (0.36 to 0.73 cm above the glottis, the lower end of
the four-cell epilaryngeal tube). It acts on the physical tract's areas
tick by tick, after the articulation model's shape, with its own state and
speed (the tube's own `squeeze` moves with the tongue body, at some 25
rad/s): `Trombone::set_ventricular_valve` (the larynx tier's `valve`)
from `0` (rest: nothing of it runs, so every sound without a cough or a
throat clearing is the same to the bit) to `1` (shut), or down to `−1`,
the tube and its flare (cells 0 to 7) thrown open to `valve_open` (2 cm²);
and `Trombone::set_aryepiglottic_flutter` (the tier's `flutter`), a
raised-cosine swing of the valve's area at `flutter_rate` (70 Hz). It
shuts over `valve_closing` (20 ms) and holds against any pressure below
it (at 100 cmH2O nothing leaves the mouth); let go with 5 cmH2O or more
behind it, it is thrown open within `valve_opening` (2 ms from 10 to 90 %;
1.6 ms from shut to 90 % of its rest area in a cough), with less as much
more slowly (let go at once with half a cmH2O behind it, it puffed). The
flow through it, its loss and its noise are those of any narrowing (the
joint's kinetic loss and its jet). The flutter is prescribed: in this
waveguide the pressure in a narrow cell carries no Bernoulli drop (the
whole kinetic loss sits at its exit joint), so a flap's driving force
would have to be drawn on top of the model anyway.

The **cough** now shuts the valve over the shut folds while the lungs
build `cough_pressure`, then throws the glottis and the valve open, the
tongue low and forward (/æ/) through the burst and back at /ʌ/ for the
"uh". Where its flow went before (40 cmH2O in the lungs, 3.5 L/s at the
peak): 13 cmH2O to the lungs and trachea, 3 to the glottis, 9 to the exit
of the tube above the folds (still drawn in to 0.36 cm²) and 11 to the
tongue of /ʌ/ (0.32 cm² at cells 14 to 19). Now: 6.2 L/s 40 ms after the
opening (10 to 90 % in 17 ms, against 39), 21 cmH2O to the lungs and
trachea, 12 to the glottis (held as wide as `set_glottal_spread` goes, 1
cm², the folds flapping between 0.8 and 1.5 cm² in the jet), 7 to the
tract. The owner's target, three times the old flow (10.5
L/s), is not reachable at 40 cmH2O: the lungs' and trachea's resistance
alone (about 3.5 cmH2O per L/s) would take 37 of it. Nor at the 14 to 18
cmH2O the owner named: there the cough peaks at 2.7 to 3.3 L/s (3.6 at
20), under the old cough's at 40. At 80 cmH2O (`Vocal::with_strength(2.0)`, people's range) it peaks
at 11.7 L/s. `cough_pressure` stays 40. The burst's loudest 10 ms are 11
dB over the speech of the same voice (6.5 before).

The **throat clearing** is two pushes (16 and 12 cmH2O) held behind the
shut folds and valve, each let out through the valve held at half its
area with the aryepiglottic folds fluttering in it (the narrowest 0.03
cm², the widest 0.21), the glottis open behind it and the voice off,
then shut again as the push dies away: two rough scrapes whose sound
repeats at 70 Hz, 6 % of their frames voiced (the cough: 49 %). Whisper
wrote the old one "uh, uh" in a sentence; now it leaves it out ("Clear
throat. [clear-throat] Cough. [cough]": "Clear throat. Puff.", before
"Clear throat. Uh, uh. Tough."; the passage of the living voice: "That was,
um, not what I expected. Shall we, uh, go run?", before "... expected. Uh,
uh, shall we, uh, go around?"). Alone, as before, it hallucinates letters
("P-P-P…", before "F-f-f…"). `tests/valve_gate.rs` measures all of this;
the valve costs some 1 to 3 % of the render while it acts, nothing at rest.

### A /dʒ/ before a pause (close-out, c2-badge; on by default)

"He wore a large orange badge." was heard as "... foreign batch". Measured
through the final /dʒ/ (`examples/affricate.rs --final TEXT`, which prints
the voice's swing and the hiss every 10 ms around the release): the vowel
before it is longer than before /tʃ/ and its closure is voiced, but at the
end of a phrase the glottis opened for the hiss as the frication began and
the folds all but stopped (the glottal flow repeated itself through half of
the frication of "badge", swinging by 80 to 100 cm³/s 10 to 40 ms after the
release against 135 just before it; "age?" 24 cm³/s). Changing one thing
at a time showed that the cue is in "badge" itself: the same change to the
/dʒ/ of "large" and "orange" alone left "batch", to that of "badge" alone
gave "badge"; cutting off the hiss after the frication changed nothing.

Now the folds stay together through the release of a /dʒ/ before a pause,
the cover slack and the pharynx widening as in its closure, behind the
voiced fricative's small leak, for `final_voicing_hold` (20 ms) after the
mouth actually opens, and then open for the hiss as before: the voice swings
by 118 cm³/s there ("age?" 56), the glottal gap opens some 10 ms later and
the hiss is as loud (0.3 to 0.5 dB under). Before a voiced sound the
frication of a word-final /dʒ/ was already voiced for 92 to 100 % of it, and
held there it only lost its hiss ("badge on" became "bags on"); before a
voiceless one English lets the voice go; so only a /dʒ/ before a pause
changes, and every other render is byte-identical (`tests/final_voicing.rs`;
`affricate_gate` now also holds a final /dʒ/'s voice through 65 to 82 % of
its closure and frication against 0 to 9 % for /tʃ/).

Whisper now writes "He are a large foreign badge." and "He are a large
badge." (both "batch" before); "Stand at the edge.", "What is your age?",
"Climb the ridge.", "Turn the page." and the four-sentence clip (byte for
byte) are unchanged. "orange" is still heard as "foreign" in that sentence,
as before (in "She has a large orange badge." it is right). The benchmarks
are unchanged but for one sentence: "A cup of sugar makes sweet fudge."
keeps its words and its last phonemes, but the phoneme recognizer drops the
"a" at its start (the audio is identical up to the /dʒ/), so the 30
held-out sentences' phoneme error is 21.1 % (21.0).

## A voice that listens

Four tools turn sound back into settings. All of them work on the physical
engine.

**The babble map** (`synthie_trombone::inverse`) goes from formants to a
tract shape: `solve(f1, f2, f3, near)` returns the tongue position, lip
rounding and narrowing that give those formants, found from a table of
54,000 shapes the engine was measured at (`examples/babble.rs` regenerates
it). Two hundred random reachable targets land within 1.7 % on F1 and
0.8 % on F2 when measured back through the engine. It covers F1 from about
250 to 710 Hz and F2 from 550 to 2550 Hz at the native tract length, so a
very open /a/ is pulled in a little. Formant Lock in the app is built on
it. The map is measured with the tongue root at rest: `solve` leaves it
there, and `predict` gives nothing for a shape whose root is not (the
vowel chart holds, for each vowel, the map's shape with that vowel's
formants).

**Mimic** copies a microphone: `--mimic` in the app, or in code a
`listen::Listener` (pitch, three formants, voicing, and how much and how
bright the hiss is, every 5 ms) feeding a `mimic::Mimic`, which picks the
shape through the babble map and a gap for /s/, /sh/ or /f/.

```sh
cargo run --release -p synthie-trombone-cli -- --mimic --voice bright     # wear headphones
cargo run --release -p synthie-trombone-cli -- --mimic --mic pipewire     # choose the input by name
```

In the tests a second voice copies a first from its sound alone: six sung
vowels within 4 % on F1 and F2, the copy's tract in place within 95 ms of
a change (its tongue body is as heavy and slow as the speaking voice's). The listener pre-emphasises in two stages (+6 dB per octave from
60 Hz and again from 1 kHz), as a voice as dark as a speaker's needs:
with one, F3 slipped into F2 in one frame in ten of the physical /æ/ once
its source took round 9's tilt, and a change of vowel took up to 220 ms
to settle (22 ms with two). On the LibriSpeech reader's sentences its
F2 jumps less from frame to frame and his vowels are told apart as well;
a real microphone in a real room is still untested. Formants are hard to read from high voices, where
the harmonics are far apart: above about 200 Hz the first formant reads
10 to 30 % off, and child-sized voices are worse. Pick a voice of about
your own size; vowels the voice cannot reach are pulled to the nearest one
it can.

**Matching a voice** from a recording: `tools/judge/match_voice.py`
measures pitch and its range, the scale of the formants (body size), how
firmly the folds close, breathiness and pace, and writes a profile that
makes the engine measure the same.

```sh
cd tools/judge && uv sync
uv run python match_voice.py someone.wav --out someone.json
cargo run --release -p synthie-trombone-cli -- --engine physical --profile tools/judge/someone.json
```

Recalibrated on round 10's engine, it re-measures eight hidden engine
voices within 4.5 % on pitch, 0.4 % on formant scale and 0.7 dB on H1-H2,
and finds every hidden tract length within 0.6 %. (Until round 10 three of
them came back with a tract length 16 to 38 % off, and a breathy one was
misread at every round since at least round 8: a breathy voice loses F2
from the formant tracker, and its formant spacing read the tract 21 to
48 % short; for a voice with H1*-H2* over 10.5 dB a spacing fitted to all
formant candidates at once now guides the estimate. The LibriSpeech
reader's H1*-H2* is 7 to 11 dB, so his analysis is as it was.) On six recordings of a human reader it meets those limits on four (on
last round's engine it met them on all six): one is matched 5.7 % low
in pitch, and in another the formant scale is misread as the longest tract
the engine allows. It copies those five things, not the person: accent,
intonation and everything else stay the engine's. Body size is unsteady on
real speech (five recordings of one reader gave tract lengths from 0.78 to
0.97), and very pressed voices are out of the engine's reach.

**A judge and a tuner** (`tools/judge`, see its README): a phoneme
recognizer (wav2vec2) and Whisper score how much of the engine's speech a
machine understands, and a naturalness measure compares its level, pitch,
timing and hiss with a person reading the same text. A CMA-ES search of the
`Tuning` numbers was tried in rounds 4 and 5 and retired: every tuned
profile fitted the tuning sentences and was no better, or worse, on the
held-out ones. Since round 6 the engine is changed one mechanism at a time
and scored directly (`judge.py score physical --fast` between changes). The
scoreboard, on material never tuned on (95 % intervals in brackets, from
resampling the sentences; one render each, since rendering is
deterministic):

| | Phoneme error, held-out sentences (30) | Words Whisper gets, held-out sentences | Vowels heard wrong (vowel lab, held out) | Voicing, forced choice, in sentences | Consonants right, in sentences |
|---|---|---|---|---|---|
| a person (LibriSpeech, his own 16 sentences) | 6 % [4, 8] | 96 % [92, 99] | 9 % [4, 13] | 99.5 % | 96 % [94, 98] |
| eSpeak NG | 16 % [13, 19] | 87 % [78, 93] | 17 % [14, 21] | 93 % [87, 98] | 89 % [84, 93] |
| classic engine | 77 % [72, 81] | 21 % [15, 28] | 88 % [85, 91] | 80 % [74, 86] | 46 % [40, 51] |
| physical engine, start of round 5 | 35 % [29, 40] | 56 % [47, 66] | 39 % [34, 44] | 97 % [94, 99] | 80 % [77, 84] |
| physical engine, end of round 5 | 26 % [22, 31] | 62 % [54, 70] | 30 % [27, 34] | 100 % | 91 % [88, 94] |
| physical engine, end of round 6 | 24 % | 66 % | 31 % | 98.5 % | 90.5 % |
| physical engine now (round 7) | 23 % | 70 % | – | 98.5 % | – |

Wave 1, all of it together (single scoring runs): speech is played as a
gesture score by default, with round 10's phone lengths and the four-part
"canoe" fix. The ten `sentence_heldout` sentences 22.0 % and 81.2 %, the
30 held-out sentences 26.3 % and 74.3 %, the 30 tuning sentences 22.2 %
and 74.1 %, vowels in the LibriSpeech texts heard wrong 23.8 %,
consonants right 88.7 %; the rule path (`gesture_score` 0) scores the same
on every one of these. The four-sentence clip: "The birch can easily run
the smooth planks. Glue the sheet to the dark blue background. She sells
seashells by the seashore. The juice of lemons makes fine punch." (24 of
31 words); sentence 1 alone "The birch can use it on the smooth planks.";
following the first reference recording "The birch can use slit on the
smooth planks.". No build yet is heard to say "canoe slid on".

After round 11 the owner's ears chose round 10's phone lengths: the six
rhythm numbers (`stressed_onset_length`, `cluster_length`,
`function_consonant_length`, `schwa_length`, `diphthong_hold`,
`rhythm_pace`) are 1 by default, and the figures of the paragraph below
"with them off" are the engine's: the ten `sentence_heldout` sentences
21.6 % and 81.2 %, the 30 held-out sentences 26.1 % and 74.3 %.

Round 11, all of it together (single scoring runs; against the end of
round 10): phoneme error over the 30 held-out sentences 21.9 % (25.9 %),
Whisper's words 70.0 % (74.8 %), the ten `sentence_heldout` sentences
15.5 % and 71.2 % (19.2 and 81.2 %), the 30 tuning sentences 20.8 % and
71.5 % (22.0 and 73.2 %), vowels in the LibriSpeech texts heard wrong
22.0 % (25.5 %), consonants there right 88.2 % (85.7 %). The four-sentence
clip: "The birch can use lid on the smooth planks. Glue the sheet to the
dark blue background. She sells seashells by the sea shore. The juice of
lemons makes fine punch." (27 of 31 words). Following the two reference
recordings: "The birch can go slid on the smooth planks." and "The birch
can move slid on the smooth plaques."; "canoe" is still not heard. The
new phone lengths account for the phoneme error (with them off: 26.1 %
held out, 22.3 % tuning) and for Whisper's loss on the held-out sentences
(74.3 % with them off, 72.8 % tuning); switched off one at a time no
single one of them does.

Round 10, all of it together (single scoring runs; against the end of
round 9): phoneme error over the 30 held-out sentences 25.9 % (25.1 %),
Whisper's words 74.8 % (68.3 %), the ten `sentence_heldout` sentences
19.2 % and 81.2 % (20.0 and 73.8 %), the 30 tuning sentences 22.0 % and
73.2 %, vowels in the LibriSpeech texts heard wrong 25.5 % (23.8 %),
consonants there right 85.7 %, the naturalness distance to the reader
about 1.00 (1.05), phrase fall 279 ms (153; his 710). The four-sentence
clip: "The birch can also run the smooth planks. Glue the sheet to the
dark blue background. She sells seashells by the seashore. The juice of
lemons makes fine punch." (24 of 31 words). Following the two reference
recordings (vowel formants off, schedules fitted again) the engine is
heard as "The birch can was slid on the smooth planks." and "The birch
can move slid on the smooth planks.", the second sentence right in both:
"canoe" is the one word still wrong. Each change of rounds 9 and 10
switched off in turn, on the held-out and on the tuning sentences: no one
change accounts for the phoneme error, the two sets disagreeing on most;
`final_coda_stretch` 1, `phrase_decline` 0.15, `vowel_decline` 0 and
`tap_length` 0 each take about a point off on both.

Round 9, all of it together (single scoring runs, the paragraphs below
measure each part by itself against the end of round 8): phoneme error
over the 30 held-out sentences 25.1 % (round 8: 22.9 %), Whisper's words
68.3 % (68.3 %), the ten `sentence_heldout` sentences 20.0 % and 73.8 %
(18.4 and 68.8 %), vowels in the LibriSpeech texts heard wrong 23.8 %
(23.1 %); the four-sentence clip 25 of 31 words (25), "The birch can also
count the smooth planks". The tap's gain on that clip ("canoes feed on",
27 words) does not survive the source's changes: switched off one at a
time, only the accent's place matters (`accent_peak` 1, as until round 9:
"The birch canoes sit on the smooth planks", 26 words; 0.7: 24; 0.85:
26, "can use three on"). It stays at 0.55, where the pitch across accented
vowels moves as the reader's does.

Round 9 (single scoring runs): the voice's source. The phoneme-switched
softening is off; the voice takes a source tilt and a voice's breath
noise is pitched higher (so its accented vowels are as dark as the
LibriSpeech reader's: 2 to 4 kHz 27.1 dB under 0 to 1 kHz, his 28.9,
before 16.5); the folds have cycle-to-cycle jitter, a flutter and a
working vibrato and wander; vowels start at once after voiceless
consonants and their breath sinks through them; the pitch accent peaks
inside its vowel. Phoneme error over the 30 held-out sentences 23.4 %
(23.2 % with the softening off and nothing else), Whisper's words 68.7 %
(69.1 %), the ten `sentence_heldout` sentences 16.3 % and 78.8 % (18.8
and 72.5 %), vowels in the LibriSpeech texts heard wrong 23.1 % (23.1 %),
consonants there right 89.1 % (90.0 %); the naturalness distance 1.08
(1.10), its warped spectral part 15.0 dB (15.6).

Round 9, the front of the tongue (single scoring runs; against the end of
round 8): /t d/ between a vowel and an unstressed one are taps, the /u/
after /t d s z/ is fronted and glides back, and /ɪ/ is closer and further
back and no longer opens towards schwa as it ends. Phoneme error over the
30 held-out sentences 24.0 % (22.9 %), Whisper's words 67.4 % (68.3 %),
vowels in the LibriSpeech texts heard wrong 23.1 % (23.1 %), consonants
there right 87.3 % (89.1 %), voicing forced choice 98.5 % (99.0 %); the
four-sentence clip 27 of 31 words (25), "The birch canoes feed on the
smooth planks".

Round 7 (single scoring runs again): the articulators and the tract's
areas are brought up to date every 2.7 ms instead of every 10.7 ms (the
folds' controls and the hiss regulator keep the long block), function words
take their weak forms before another word ("to" /tə/, "the" /ðɪ/ before a
vowel, "was", "has", "had", "than", "your"), a stressed /ɪ/ keeps the short
length, the physical /l/ is made closer (its first formant stood near 500
Hz and "slid" was heard as "slayed"), and the folds are held 12 % more
loosely through nasals, liquids, glides and unaccented vowels and slacken
over the last 150 ms of a phrase. Phoneme error over the 30 held-out
sentences 23.0 % (23.6 % after round 6), Whisper's words 70.0 % (65.7 %;
73.8 % on the ten `sentence_heldout` sentences), vowels in the LibriSpeech
texts heard wrong 22.7 % of the time (28.3 %).

Round 6 (no intervals: one scoring run of the final build; ten sentences
carry about ±5 points, thirty about ±4): a stop before /l r/ is released
with its burst and a short transition of its own ("blue", "glue",
"planks"; the lips used to part at half speed there), the mouth behind an
/n/ opens fast enough to keep its antiresonance through the murmur (/n/
heard as /ŋ/ 7 to 1 over the sentence sets), /f v θ ð/ have an edge source
where their jet strikes the teeth and lip, a sibilant fades into a
following /f v/ ("makes fine" was heard as "makes pine"), and stressed lax
vowels are 18 % shorter than tense ones instead of 38 %. Phoneme error over
the 30 held-out sentences went from 25.8 to 23.6 % and Whisper's words from
61.7 to 65.7 % (73.8 % on the ten `sentence_heldout` sentences); vowels in
the LibriSpeech texts are heard wrong 28.3 % of the time (26.9 % before;
26.2 % on this build with the stressed lax vowels at their old length).

Round 5 changed the sounds, not the tuning: nasals keep a mouth cavity of
their own, /ð/ has a dental blade, /p t k/ shut for at least 22 ms, the
voice dies away into a voiceless closure, the incisors close for /s z/,
and lax vowels are shorter and glide towards schwa. Against the start of
the round that is 9 points less phoneme error on the held-out sentences
[4.6, 13.6] and 4 points less over all 156 held-out items [2.4, 6.4]; the
consonants heard right in sentences went from 80 to 91 %. On its own
first ten held-out sentences it now scores 22 % [17, 27] with 67 % of the
words [53, 82].

The owner's targets for the round, on the held-out material:

| target | now | |
|---|---|---|
| phoneme error, held-out sentences, under 25 % | 22.0 % (first ten); 25.8 % over all 30 | met on the ten, at the line over 30 |
| Whisper word accuracy, held-out sentences, over 75 % | 67.1 % (first ten); 61.7 % over 30 | missed |
| vowel error in sentences, under 22 % | 26.9 % (LibriSpeech texts); 30.3 % over the vowel lab | missed |
| consonant error in sentences, under 22 % | 9.0 % | met |

The lax vowels' length was chosen by ear over the recognizer: at 0.62 of a
tense vowel's length (the default) against round 4's 0.905, the tuning
sentences score 7.1 points worse [2.9, 12.3], the held-out ones within
noise (pooled −1.0 [−2.8, 0.6]), and the vowel lab's held-out vowels 3.4
points better [0.4, 6.7].

**The round-5 tune.** The tuner was repaired first: its naturalness guard
charges every measure, unclipped, for moving further from the person than
the defaults, and bars any candidate whose level walls, level rises or
naturalness distance are worse than the defaults' beyond noise (round 4's
candidate made its rises steeper in all 16 sentences unseen); the sibilant
and lax-vowel gates are in the loop, nearly no gate is skipped, the best
candidates are run through every gate before one is chosen, and Whisper's
word errors are part of the objective (`tools/judge/README.md`). Pass a
(21 of the sound-making numbers, 209 evaluations, about 3 hours, the lax
lengths held at 0.62) improved the tuning split well beyond noise (phoneme
error −3.8 [−5.4, −2.2], Whisper +8.1 [3.0, 14.2]) and passed every gate,
but held out it is **worse** beyond noise (+2.9 [1.1, 4.8], worst on the
Harvard sentences, +5.2), with fewer consonants heard right in sentences:
it learned the tuning sentences, so it is kept as a rejected candidate.
About half of all candidates were barred by a gate: the voice bar in
closures, the onset pitch kick, the affricates, the pinch recovery and the
regulator sit close to their bounds at the defaults.

Pass b+c (timing, prosody and the lax nucleus; started again from the
defaults, not from pass a) was made harder to over-fit: sentences only (the
carrier syllables barely exercise timing, and in pass a they moved most on
tune and got worse held out), a ten times stronger pull to the defaults and
a smaller step. A first run had every candidate barred, so a gates-only
scan moved each of its 24 numbers alone by one and two steps of 5 % of its
range: `vowel_hold` fails a gate one step either side (the affricates, the
onset pitch kick, clicks); ten more fail on one side (`lead`: affricates,
voiced fricatives and the /eɪ/ glide when smaller; `closure_overshoot`:
affricates and clicks when smaller; `body_rate`: the steady H1–H2 and the
/eɪ/ glide when slower; `closure_floor`: the voiceless closures when
shorter; `prelenis_stretch`, `voiced_sibilant_length`, `accent_size`:
affricates when larger; `vot_alveolar`, `vot_velar`, `declination`: the
pitch kick or clicks when smaller), and several pass two steps out but fail
one, so the gates near their bounds are partly noise. The search was then
held to the side of each default that passed (at most two steps). Its
best, after 209 evaluations, improved the tuning sentences (phoneme error
−2.3 [−4.9, 0.0], Whisper +8.1 [2.1, 15.2]) and passed every gate, but held
out it is worse again (pooled phoneme error +1.6 [0.1, 3.0]; Whisper +3.0
[−2.6, 8.9], within noise; the LibriSpeech texts −1.1 [−4.4, 2.0]), so it
is kept as a rejected candidate too. Its moves are small (`release_lead`
0.011 to 0.014 s, `accent_decay` 0.20 to 0.18, `phrase_onset` 0.15 to
0.14, `velum_lead` 0.030 to 0.028, `dorsum_rate` 20 to 19.3). What the two passes show: with ten
tuning sentences and twenty Harvard ones, the recognizer's gains on the
tuning split are about the size of its noise, and the gates leave timing
almost no room.

Vowels are every vowel slot of the held-out sentences, carrier phrases and
LibriSpeech texts (673 slots; the person's figure is on his own
recordings); voicing and consonants are the slots of the 16 LibriSpeech
texts. Isolated items are another matter: the judge is harsh on them even
for a person (vowels in words cut out of the LibriSpeech recordings: 49 %
wrong), and the engine's isolated words score 58 % phoneme error [50, 66]
(eSpeak 51 %). How natural it sounds is in `tools/judge/report/natural.md`:
of all the systems, the physical engine is the closest to the person on
the guard's distance (1.11, eSpeak 1.63, classic 1.43; 0 is the person's
average sentence).

**Known limits.** Plainly:

- **Vowels in sentences** are heard wrong 23 % of the time in the
  LibriSpeech texts (31 % over the vowel lab's held-out material at the
  end of round 6)
  (eSpeak 15 and 17 %, the person 9 %; the target is 22 %). A third of
  the misses in the LibriSpeech texts are words the engine's front end
  says otherwise than eSpeak's reference (a full "and" for a reduced
  one): over the slots where they agree the error is 22 %. /ʊ/ comes
  through about a quarter of the time, /ɔ/ is
  mostly heard as /ɑ/, /ɑ/ often as /æ/ (its target is Hillenbrand's
  Michigan /ɑ/, F2 1333 Hz; at 1100 to 1180 Hz the recognizer hears it
  better, but that shape at the back of the tongue's reach moves the
  gates measured in "a pa"), /ɔɪ/ not at all, and schwa as /u/ or /ɚ/.
- **The tongue root is built and off.** Solved for /ɪ ʌ ʊ/ (set
  `LAX_ROOTS` in `examples/vowel_size.rs` and regenerate) it passes every
  gate, but on the round 6 build it moved vowel error in the LibriSpeech
  texts from 26.2 to 25.9 % and phoneme error on the ten held-out
  sentences from 18.8 to 21.2 %, all within noise (round 7, for /ɪ ʊ/
  alone: vowel error 22.7 to 24.8 %, the 30 held-out sentences 23.0 to
  24.0 %); with the longer
  stressed lax vowels as well, vowel error was 31.1 %. Drawing the tense
  vowels' roots forward broke the listener and mimic gates for no gain.
- **Lax vowels in the engine's sentences sit nearer the tense ones than a
  speaker's**: measured the same way in the LibriSpeech texts, the
  person's /ɪ/ has an F2 15 % over his schwa's and 16 % under his /i/;
  the engine's was 27 % over its schwa (round 9: 22 %, F1 448 to 429 Hz,
  F2 1998 to 1910); moved all the way there, its /ɪ/ is heard as a high
  central vowel. At the stressed length of 0.82 the recognizer hears
  more stressed /ɪ/ as /eɪ/ (vowel error 28.3 against 26.2 % at 0.62)
  while Whisper gets a few more words; the length is the owner's choice.
- **Whisper gets 70 % of the words of the 30 held-out sentences** (74 % of
  the ten; target 75 %). "Makes fine" is heard as "makes fine" since round
  6 (it was "makes pine": the /s/ cut off in a few milliseconds, the air
  rushed out before the lip shut, and the /f/'s only source, the free jet
  in the lip's gap, radiated some 20 dB too weakly); in 24 short /f/ and
  /θ/ sentences Whisper now gets 19 target words where it got 7. Still
  short of a speaker: /f/ between vowels is some 22 dB under the vowel
  where the reader's is 16 (the mouth pressure behind a short /f/ builds
  slowly); after /s/ the /f/ is only 8 dB under it where the reader's is
  16; "thin" is still heard as "so nice" or "been". In the four listening
  sentences "canoe slid on" is still not heard (below).
- **What depends on the control block.** The articulators and the tract
  go by a 2.7 ms block, and the same gesture gives the same trajectory
  within 1 % at 64 to 512 samples (`tests/block_invariance.rs`); the
  folds' pitch, tenseness and voicing are still read once every 10.7 ms, so
  a voice onset can fall up to that much late.
- **Weak forms the recognizer's reference keeps full** ("and", "of", "for",
  "can") are said in full: reduced to schwa, as a speaker says them, the
  vowel error in the LibriSpeech texts rose from 24.1 to 28.0 % with
  Whisper unchanged, because the reference transcription has the full
  vowel there.
- **In the four listening sentences** "canoe slid on" is heard as
  "canoes feed on" (round 8: "can use with"): with the tap joining "slid"
  to "on" Whisper hears the /u/ of "canoe" as /u/, and takes the /s/ of
  "slid" for a plural. Sentence 1 alone is still heard as "The birch can
  also eat on this new plane". Measured one change at a time against
  round 8 (four-sentence clip, words of 31): the tap alone 27 ("canoes
  rid on"), the /ɪ/ alone and the fronted /u/ after /t d s z/ alone 25,
  all three 27. What is left: the /l/ of "slid" is heard as /r/, /w/ or
  not at all, and of 15 /ɪ/ after /l/ in short phrases the recognizer
  hears 5 right (round 8: 2), 4 as /eɪ/ and 5 as /i/. The /u/ of "canoe"
  is the plain back /u/: fronted after /n/ (F2 1600 to 1800 Hz, as the
  reader's and the owner's are), it was heard as /ju/, "can use". The
  engine's /u/ and /i/ have a first formant 50 to 80 Hz above the reader's
  in sentences (LPC: /u/ 410 against 274, /i/ 397 against 317), and the
  tongue body, arriving late from the vowel before, still makes the start
  of a short /ɪ/ open (F1 falls some 100 Hz over its first 40 %; the
  reader's rises 10). "Background" is on the edge: heard right or as
  "back drawing" as the stop release speed moves by a tenth.
- **Taps cost a little elsewhere.** Over the 30 held-out sentences the
  phoneme error with the tap is 24.0 % against 22.9 % (within the ±4 of
  the set) and Whisper's words 69.6 % against 68.3 %; a tap after /ɑ/ is
  sometimes heard as /r/ ("a rod is", "a pot of"). With all of round 9
  the consonants of the LibriSpeech texts are heard right 87.3 % of the
  time against 89.1 %: four slots are a /t d/ the recognizer writes as
  the flap where the reference has the stop ("middle", "fled in",
  "lucidity", "not exactly"; the reader taps them), three the /w/ of
  "quilter" before the new /ɪ/, and two a /d/ after /n/ before an /ɪ/
  ("send him", "and the [ðɪ]"), heard voiceless since the /ɪ/ moved (not
  yet measured why; the closure is made on the /ɪ/'s tongue body, which is
  now closer).
- **The softer voice is a filter, and it is off.** Held more loosely
  (round 7), the folds open: past 12 % the child voice fell out of its
  folds' range with a click and a /g/ lost its voice bar, and at 12 % the
  nasals and liquids came out brighter, not darker. Closing them more
  softly does not work in this model either (round 8), so the softening
  is a low-pass on the voice as it leaves, switched by phoneme; since
  round 9 it is off by default. With the source tilt the engine's nasals
  are 1.9 dB and its liquids 3.8 dB less dark against the accented vowels
  than the reader's (his: 5.6 and 5.1 dB darker in 2 to 4 kHz); turning
  the switch back on for that rest was not tried against the owner's ear.
- **H1–H2 of the nasals and glides** stays under the reader's: his
  nasals have 4.1 dB more H1–H2 than his accented vowels, the engine's 1.5
  dB less. The engine's nasal murmur resonance stands near 290 Hz with a
  58 Hz bandwidth, which lifts H2 of the deep voice by 5 dB against /a/; a
  source that keeps the folds' adduction does not change H1–H2 (the
  low-pass leaves it within 0.1 dB). The glides are already darker than
  his (11.6 dB under the accented vowels in 2 to 4 kHz, his 4.4) and are
  not softened.
- **The voice's brightness is a filter on the source, not the folds.**
  The fold model's abrupt closure and the ripple of its flow under the
  tract's pressure make the bright source; the shelf stands for the
  return phase they lack. The formants of the tract are narrower than a
  speaker's at the top (closed glottis: F3 30 to 230 Hz, F4 40 to 90, where
  100 to 150 and 150 to 250 are typical): the boundary-layer loss is the
  textbook value at 1.5 kHz, and three times it would give F3 85 to 270
  and F4 100 to 150 Hz but widen F1 to 70 to 110 as well. Above 4 kHz the
  reader's vowels have more energy than the engine's (4 to 7.5 kHz 25.7
  dB under 0 to 1 kHz, the engine's 28.6): his breath noise lies higher
  still. With the tilt, the listener (`listen.rs`) reads close vowels'
  F1 10 to 16 % high and settles slowly on /æ/ after /u/ (221 ms), and
  mimic follows /i/ and /æ/ late; their gates hold those cases apart.
- **Isolated words** score about 58 % phoneme error, against a 13-of-26
  ceiling the judge gives a real person's isolated words; the isolated
  figures are a relative measure only.
- **Remaining confusions**: stops dropped (/t/ and /d/ in clusters, /d/
  heard as /n/ in carrier phrases), /z/ as /ð/; an /l/ that starts an
  utterance is often heard as /t/ or /f/, and /ɪ/ after /l/ as /i/ or /eɪ/.
- **"canoe" (round 10).** The /u/ after /n/ is a back one, F2 near 970 Hz
  from 20 % of it on, where the owner's falls from 1920 to 1390 and the
  reader's after /n t d/ starts at 1600 to 2200; its first formant reads
  400 to 460 Hz (the owner's 340 to 360, the reader's 250 to 330): the
  tract's own is 345 to 375 (the target 378), the open glottis adds some
  40, and the velum, still half open as the /n/ lets go, closes only 30 to
  35 ms into the vowel, which raises the first 20 % and makes a 3.5 dB
  step in the level there. Under tight lips the back /u/ is dull: its 2 to
  4 kHz band stands 53 dB under its 0 to 1 kHz band, the owner's 30, and
  the /n/ before it is 3 dB louder than it. The recognizer hears it as
  /ʌ/ or /l/ ("can also"). Fronted after /n/ (F2 1700 at its middle, 2 to
  4 kHz 15 dB stronger) Whisper hears "canoe" in 2 of 10 carrier phrases
  as before, sentence 1 as "can use lead on"; raising the velum 20 to 40
  ms before the release takes the step away and changes nothing heard.
  Neither is on. Pitch, length and level were ruled out by r10-prosody.
- **"canoe" (round 11): the nose is shut sooner, the word is still not
  heard.** The velum that stayed open into the /u/ was a real fault: even
  a quarter open (0.2 cm²) it takes over a close vowel (the coupled /u/'s
  loudest peak at 704 Hz, above its own F1 at 322; the nose's peaks at 2.5
  to 3.4 kHz 17 dB up), and for 35 ms after the release the 550 to 950 Hz
  band of "canoe" stood 7 to 9 dB under its 200 to 450 Hz band, where the
  owner's "canoe" and "smooth" and the reader's "the new" stand 16 to 26
  dB under from 5 ms on. Out of /n ŋ/ into a close vowel the velum now
  shuts three times as fast (`velum_release`; not after /m/, whose slow
  lips then made "smooth" dip 6 to 8 dB, and not into open vowels, where
  "no" and "gnawed" were heard with /m/ and /l/): the band is 18 dB under
  from 20 ms on, the level step is gone, and the recognizer hears N UW in
  sentence 1 and in about 7 of 10 carrier phrases (2 to 3 before). Whisper
  does not follow: sentence 1 is "can use it on" (before "can also turn"),
  carriers 1 of 10 as before, the references re-fitted "conch was slid"
  and "camo slid" ("can was", "can move" in round 10). Measured, not the
  cause: the level (the /n/ against the /u/ is 2.3 dB under in the
  reader's "the new", +0.5 in the owner's "canoe", +4.1 and +3.3 in the
  engine's; the engine's /u/ is 5 dB quieter against its other vowels
  than the reader's, its nasals 1.6 dB louder; 3 dB more /u/ changed
  nothing heard), the first vowel (100 ms where the reader's schwa is 65
  and the references' 50, and loud; at 55 ms, or as [ɪ], or left out,
  still "can also"), the 2 to 4 kHz band (a back /u/ is dull by nature:
  its transfer function 14.5 dB down there against a fronted one's 0.8;
  20 dB more on the spliced vowel changed nothing). Spliced into the
  owner's own sentence 1, every engine /u/ tried (back, fronted, longer,
  lower F1, the one of "smooth") turns "canoe" into "can now", "can go"
  or "can slide", while the engine's /k/, schwa and /n/ there are heard as
  "canoe"; a fronted /u/ (F2 1800) with its first 20 ms cut off is heard
  as "canoe" there. Those 20 ms are the tip of the /n/ letting go slowly
  onto a fronted tongue (F2 overshooting to 2230 Hz with F3 at 3700, a
  /j/): so people's fronted /u/ is right, and the engine needs a quicker
  tip release before it can use it. A fronted /u/ with tight lips is out
  of this tract's reach (the tongue body stops at cell 29; with lips
  rounded F2 falls to 1270 and F3 stays at 2700).
- **"canoe" (wave 1): the owner's three-part fix is on; the recognizers
  do not hear the word.** By ear (round 11's experiment clip: "much
  smoother, no fake 'you' glide") the owner chose: the tip quick out of
  /n/ (`nasal_tip_release` 6, onto a vowel's shape only), the fronted /u/
  after /n/ (`fronted_after_n`), the tongue kept up behind an /n/ before
  it (`nasal_cavity_held`; switched off again in wave 2, the /n/ leaned
  toward "ng"), and a higher, fronter schwa before it
  (`schwa_colour`). In the tract (its own formants): the schwa of
  "canoe" stands at F1 320 to 400, F2 1750 to 1790 Hz (before 380 to
  510 and 1645 to 1570; the owner's near 300 and 1800); out of the /n/
  F2 rises from 1580 to 1590 and glides back to 1480 (before: the back
  /u/, 1350 falling to 1000; with the fronted /u/ and the tip at its own
  speed, 1896 falling to 1644 over 25 ms, the /j/). Each part can be
  switched off (`0`, the tip release `1`); all four off is the earlier
  engine sample for sample. Whisper writes sentence 1 "The birch can use
  it on the smooth planks." (as before), the four-sentence clip "...can
  easily run the smooth planks..." (24 of 31 words, 25 before), "canoe"
  in 0 of 10 carrier phrases (1 before). The phoneme recognizer hears
  the fronted /u/ after /n/ less often than the back one ("the new king",
  "I knew him" lose their /u/; "a new car", "noon" gain it), and one /n/
  before it as /ŋ/: kept up, the tongue puts the /n/'s antiresonance at
  2.1 to 2.3 kHz (1.45 to 1.55 dropped). The same fronted /u/ after /d/
  is heard as /u/. Over the 30 held-out sentences 26.3 % phoneme error,
  74.3 % of Whisper's words (26.1, 74.3 before), the 30 tuning ones 22.2
  and 74.1 % (22.3, 72.8), vowels heard wrong in the LibriSpeech texts
  23.8 % (24.8), consonants right 88.7 % (88.2). Hastened into /ð s l/
  as well (the tip moving on to its next place, not letting go), the
  quick tip cost "in the", "on the", "at once": 74.3 to 72.2 % of the
  held-out words; it no longer does.
- **Taps after /ɑ/** are heard as /r/ in 7 of 12 short phrases (other
  vowels 1 of 8). Not their third formant (2700 to 3100 Hz at the tap, the
  reader's 2500 to 2800) nor their length (12, 20 or 30 ms, approach 8 to
  25 ms: 7 to 9 of 12 each): from /ɑ/ the first formant falls smoothly
  from 780 to 380 Hz and the second dips to 1250 with it, an /ɹ/'s
  pattern; a faster tongue body (the alveolar locus reached) did not help
  either.
- **/s/ is darker** than a person's in sentences.
- **Closures are steeper than a person's**: about 3.4 level walls (20 dB
  in 10 ms) a second against his 1.1, and level rises of 15 dB per 10 ms
  (95th percentile) against his 12. About 70 % of that gap is the room in
  his recordings (the engine's output is dry), so the tuner holds walls
  and rises to the defaults rather than to his figures.
- **The listener's F1 bias on close vowels**: the recognizer hears the
  engine's /ɪ/ as /eɪ/ and its schwa as /ɪ/ or /u/, which is partly its own
  F1 bias on close vowels and partly the engine.
- **The whisper voice** has looser click bounds than the others (clicks up
  to 6 vowel peaks against 2.5, steps between samples not bounded, bursts
  up to 6 dB over its weak vowels).
- **The pulse source sings notes above the folds' range** (the peaks of
  the small voices' lively speech, the critter's 490 to 520 Hz), so those
  stretches are a prescribed pulse, not the folds.
- **Mimic's tract arrives slower** than before: the copy's tract is in
  place within 95 ms of a change (the bound was 60 ms before the tongue
  body was made heavy and slow).
- **Voice matching** is less sure on real speech (four of six
  recordings within its limits, above).
- **The gates leave the tuner little room**: the defaults sit close to
  many gate bounds, so moving one number by a few per cent often fails a
  gate (the voice bar, the onset pitch kick, the affricates, the pinch
  recovery, the regulator, clicks at stop releases). In timing, a step of
  5 % of its range fails a gate for `vowel_hold` either way and for ten
  more numbers one way (above); about half of round 5's candidates were
  barred, and the tuning sets are small enough to learn (both passes did).

## Making your game talk

Add the library (`synthie-trombone`, in `crates/synthie-trombone`) to your
project, tune a `Talker` once, and turn each line into mono `f32` samples
for whatever plays your audio:

```rust
use synthie_trombone::{Preset, Talker, VoiceSettings, normalize};

let mut shopkeeper = Talker::new(44_100.0, Preset::Elder);
let mut line = shopkeeper.bake("Welcome, traveller!");
normalize(&mut line, 0.9);
// with game-toolkit: audio.play_samples(&line, 44_100)?;

let goblin = VoiceSettings { nasality: 0.4, rate: 1.5, ..Preset::Critter.settings() };
let squeak = Talker::new(44_100.0, goblin).bake("Shiny! Mine!");
```

For live dialogue, `Talker::say` queues text and `Talker::render` fills your
audio buffers as they come. The library never opens an audio device, so it
doesn't compete with kira (or whatever else you use) for the output, and it
builds for `wasm32-unknown-unknown` and without `std` (enable `libm`, plus
`alloc` for speech).

If you'd rather play it as an instrument from your own audio callback,
`Trombone` gives you the voice directly: note on and off, MIDI pitch, glide,
tongue, up to four constrictions, and the velum, with no allocation in the
audio path.

## When the pronunciation is off

Text is looked up in a pronouncing dictionary first: a core of about 2,000
common and irregular words is always built in, and the `dictionary`
feature (on by default, about 22 KB) adds 1,200 less frequent ones. A word
it does not know is taken apart into a known stem and a regular ending
(`-s`, `-ed`, `-ing`, `-ly` ...), and otherwise spelled out by English
letter-to-sound rules (digraphs, the silent *e*, soft *c* and *g*,
unstressed vowels); numbers are spelled out. On the judge's material the
front end now agrees with eSpeak on nearly every word
(`tools/judge/frontend_check.py`). Consonants take the tongue position of
the next vowel, stops get a closure, a burst and a breath, and the pitch
drifts down over a phrase, falls at `.` and rises at `?`. When a word
comes out odd, you can write its phonemes in ARPAbet in square brackets (a
`1` marks the stressed vowel):

```text
[HH AX L OW1], pink trombone!
```

Or give the front end your own words: `speech::phonemize_with(text,
&lexicon)` consults `lexicon` (a lower-case word to ARPAbet such as
`"K AX N UW1"`) in place of the built-in dictionary and spells out
everything else as `phonemize` does; `&|_| None` hears the spelling rules
alone.

## A trombone walks into a synthesizer

I had been building [synthie](https://github.com/sunsided/synth), a
chiptune synthesizer in Rust, and at some point the question came up whether
the trombone could live there too. It turns out somebody already did the
port: the [`pink-trombone`](https://crates.io/crates/pink-trombone) crate
translates [chdh's TypeScript version](https://github.com/chdh/pink-trombone-mod)
to Rust, more or less line by line. I did my own anyway, partly because I
wanted it inside an audio callback (the port is a faithful translation, but
it allocates in the audio path, and a real-time callback is the one place
that can't have that), and partly because once the tract was running I
wanted it to sing notes, speak typed text, and sound like more than one
person.

The vowels were the part I enjoyed most. My first formant probe was a plain
DFT peak search, which happily reported 210 Hz for every vowel I threw at it
(the glottal source tilts the spectrum so hard that the low harmonics win
every time). Switching to LPC gave sensible numbers, and from there a grid
search over tongue position, tongue height and lip rounding fitted each vowel
to Peterson & Barney's 1952 averages for adult male speakers; all nine land
within about 2 % on F1 and F2. The search runs for about four minutes
(`task tune:vowels`), and `task formants` prints what each vowel measures
today.

The screenshots above come from the real UI driven by a real voice;
`task screenshots` renders them again.

## Development

| Task | What it does |
|---|---|
| `task check` | rustfmt, Clippy with warnings as errors, `cargo check`, the `no_std` builds |
| `task test` | all tests |
| `task lint:actions` | actionlint and zizmor over the workflows, same versions as CI |

CI runs the same gates (plus wasm32, docs and the MSRV) on every push and
pull request. Pushing a `v*` tag that matches the workspace version builds
the terminal app for Linux, macOS and Windows and attaches it to a GitHub
release.

## License

EUPL-1.2 (`LICENSE`), like synthie and game-toolkit. Derived from Pink
Trombone, © 2017 Neil Thapen, used under the MIT licence; its notice is in
`NOTICE-pink-trombone`.
