# Profiles

A profile is a JSON file of overrides for the physical engine's `Tuning` numbers, `{"tuning": {...}}`
(`say --profile file.json`, the app's `--profile`, `SYNTHIE_TROMBONE_PROFILE` for the gate tests). The
defaults live in the code (`Tuning::DEFAULT`, `crates/synthie-trombone/src/tuning.rs`), so no profile is
needed to hear the engine as it is.

None is kept here. The tuned profiles of rounds 3 to 5 were all rejected (they fitted the tuning sentences
and were no better on held-out ones) and are in the git history, last present in commit b531112. To try a
variant, write one, for example a stressed /æ/ gliding a fifth of the way to schwa:

```json
{"tuning": {"ae_nucleus": 0.55}}
```
