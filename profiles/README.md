# Profiles

A profile is a JSON file of overrides for the physical engine's `Tuning` numbers, `{"tuning": {...}}`
(`say --profile file.json`, the app's `--profile`, `SYNTHIE_TROMBONE_PROFILE` for the gate tests), by
the fields' names: flat, `{"tuning": {"voice_tilt": 14}}`, or within their part of `Tuning` (`anatomy`,
`articulation`, `prosody`), `{"tuning": {"anatomy": {"voice_tilt": 14}}}`, as `say --dump-profile`
writes them, or both mixed; a field given both ways is refused. The defaults live in the code (`Tuning::DEFAULT`, `crates/synthie-trombone/src/tuning.rs`), so no profile is
needed to hear the engine as it is.

One is kept here: `mri-tongue.json`, the measured (MRI) tongue body on the original tract
(`tongue_model` 1 and the original tract's numbers; the same as `say --tract original --tongue-model
mri`), the configuration whose Whisper words were the project's best (wave 2C: 85.0 % on the ten
held-out sentences and 77.4 % on the 30; with the affricates of wave 3, 87.5 % and 77.8 %). The tuned
profiles of rounds 3 to 5 were all rejected (they fitted the tuning sentences
and were no better on held-out ones) and are in the git history, last present in commit b531112. To try a
variant, write one, for example a stressed /æ/ gliding a fifth of the way to schwa:

```json
{"tuning": {"ae_nucleus": 0.55}}
```
