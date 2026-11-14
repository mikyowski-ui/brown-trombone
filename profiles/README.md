# Profiles

Tuning profiles for the physical engine (`say --profile`, `SYNTHIE_TROMBONE_PROFILE` for the gates), each
`{"tuning": {...}}` over the defaults of `Tuning`.

| profile | what |
|---|---|
| `stressed-lax.json` | stressed lax vowels 18 % shorter than tense ones (`lax_length` 0.82) instead of 38 %, as the owner asked in round 6; the recognizer hears the stressed /ɪ/ as /eɪ/ far more often at this length, so it is not the default. |
| `long-lax.json` | lax vowels at the round 4 length (`lax_length`, `unstressed_lax_length` 0.905); the default is 0.62, chosen by ear. |
| `tuned-r5a-candidate.json` | round 5, pass a (21 sound parameters, CMA-ES on the tuning split). **Rejected**: it fits the tuning split (pooled phoneme error -3.8 points, Whisper words +8.1) but held out it is worse beyond noise (pooled phoneme error +2.9 [1.1, 4.8], harvard_heldout +5.2, carrier_heldout +2.3), with consonants in sentences 91.0 to 87.3. |
| `tuned-r5-candidate.json` | round 5, pass b+c (23 timing, prosody and lax-nucleus parameters, from the defaults, sentences only, each held to the side of its default a gate scan found free). **Rejected**: better on the tuning sentences (phoneme error -2.3, Whisper +8.1) and passes every gate, but held out pooled phoneme error +1.6 [0.1, 3.0], Whisper +3.0 [-2.6, 8.9]. |
| `tuned-r4-candidate.json` | round 4, pass a; kept out of the defaults for steeper level rises and more level walls. Stale: pins the old `teeth_gain`. |
| `old/` | round 3 profiles, which now score worse than the defaults. |
