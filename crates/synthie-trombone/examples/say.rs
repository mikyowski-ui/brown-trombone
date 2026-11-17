//! Speak text into `say.wav`.
//!
//! ```sh
//! cargo run -p synthie-trombone --example say --release -- "Hello, pink trombone!"
//! cargo run -p synthie-trombone --example say --release -- --voice critter "Are you there?"
//! cargo run -p synthie-trombone --example say --release -- --pitch 180 --rate 0.8 "Hello?"
//! cargo run -p synthie-trombone --example say --release -- --engine physical "Hello there!"
//! ```
//!
//! `--engine physical` uses the physical sound model; add `--source pulse`
//! to drive it with a prescribed glottal pulse instead of the
//! self-oscillating folds, and `--out file.wav` to choose the file.
//!
//! A voice can be described instead of dialled: `--size 0.8 --effort 0.7
//! --breathiness 0.2 --clarity 0.8` (see `Persona`), or loaded from a JSON
//! profile with `--profile voice.json` (see `VoiceSettings::with_profile`).
//! `--dump-profile` prints the voice that results; `--dump-parameters`
//! prints every tunable number with its range.
//!
//! A profile may also move the physical engine's vowels, for trying out
//! other targets without regenerating the vowel table: a `"vowels"` object
//! of F1, F2 and F3 for an adult man, Hz (see `vowel_targets`), keyed by
//! the vowel (`"IH"`), a diphthong's start (`"AY<"` or `"AY"`) or end
//! (`"AY>"`):
//!
//! ```json
//! { "engine": "Physical", "vowels": { "IH": [410, 2050, 2700], "AY>": [400, 2100, 2800] } }
//! ```
//!
//! Their shapes are solved through the babble map (`inverse::solve_for`)
//! at start-up, scaled to the voice's size, starting from the built-in
//! shape so that they stay close to it; they land within about 2 % of
//! the formants asked for.
//!
//! `--match-prosody schedule.json` speaks the text with a person's timing
//! and pitch: the schedule `tools/judge/prosody.py` measures on a
//! recording of that text (one span per phone, a pitch track, a breath
//! share per phone; see `speech::Schedule`). `--prosody-pitch speaker`
//! (the default) keeps the speaker's pitch as it is, `--prosody-pitch
//! voice` moves the contour to the voice's own median; `--prosody-kick
//! off` leaves out the engine's pitch kick after voiceless consonants
//! (the speaker's contour has his own). A schedule's voice profile,
//! tenseness per phone and quicker articulators are taken unless
//! `--prosody-voice`, `-tenseness` or `-speed off` says not; its vowel
//! formants only with `--prosody-formants on` (moved towards the
//! speaker's, the vowels were heard wrong more often: 29.0 against 26.6 %
//! on the LibriSpeech texts). `--phone-times file.json` writes where each
//! phone starts and ends in the output.
//!
//! `--neural VOICE` (or `VOICE:SPEED`) speaks any text with the timing,
//! pitch and level of tiny-voca, a small neural text-to-speech model, in
//! its voice `F2` or `M2`: the engine's phones for the text go to
//! `tools/judge/prosody.py neural`, which writes the schedule beside the
//! output (`out.neural.json`), and the text is spoken following it as with
//! `--match-prosody` (the `--prosody-*` flags apply; the pitch is moved
//! to the voice's own median unless `--prosody-pitch speaker` says
//! not). SPEED is the model's speaking rate (default 0.8, about the
//! engine's rule pace; larger is faster). The bridge runs in
//! the Python of `SYNTHIE_NEURAL_PYTHON` (default: `tools/judge/.venv`,
//! with the judge's `neural` dependency group installed) from the judge
//! folder `SYNTHIE_JUDGE_DIR` (default: this checkout's).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/wav.rs"]
mod wav;

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;
use synthie_trombone::speech::{self, FollowOptions, PhoneSpan, PitchMode, Schedule, Utterance};
use synthie_trombone::vowel_targets::{self, VowelKey};
use synthie_trombone::{Engine, GlottalSource, Manner, Persona, Phoneme, Preset, Talker, inverse};
use synthie_trombone::{Trombone, normalize};

/// A prosody schedule (`tools/judge/prosody.py`) from its JSON.
fn read_schedule(json: &str) -> Schedule {
    let v: serde_json::Value =
        serde_json::from_str(json).unwrap_or_else(|e| panic!("not a schedule: {e}"));
    let num = |v: &serde_json::Value, what: &str| {
        v.as_f64()
            .unwrap_or_else(|| panic!("schedule: {what} must be a number")) as f32
    };
    let phones = v["phones"]
        .as_array()
        .expect("schedule: no \"phones\"")
        .iter()
        .map(|p| {
            let mut span = PhoneSpan::new(
                p["phone"].as_str().and_then(Phoneme::from_symbol),
                num(&p["start"], "start"),
                num(&p["end"], "end"),
            );
            span.breath = p["breath"].as_f64().map_or(1.0, |b| b as f32);
            span.tenseness = p["tenseness"].as_f64().map_or(1.0, |t| t as f32);
            span.formants = p["formants"]
                .as_array()
                .filter(|f| f.len() == 2)
                .and_then(|f| Some((f[0].as_f64()? as f32, f[1].as_f64()? as f32)));
            span
        })
        .collect();
    let pitch = &v["pitch"];
    Schedule {
        phones,
        pitch_start: num(&pitch["start"], "pitch.start"),
        pitch_hop: num(&pitch["hop"], "pitch.hop"),
        pitch: pitch["hz"]
            .as_array()
            .expect("schedule: no pitch.hz")
            .iter()
            .map(|x| num(x, "pitch.hz"))
            .collect(),
        median_hz: num(&v["median_hz"], "median_hz"),
    }
}

/// The phones of `utterance` as `--phone-times` names them: each by the
/// segment that starts it.
fn phone_names(utterance: &Utterance) -> Vec<String> {
    let mut phones = vec![String::new(); utterance.phone_spans().len()];
    for (i, s) in utterance.segments().iter().enumerate() {
        if let Some(p) = utterance.phone_of(i)
            && phones[p].is_empty()
        {
            phones[p] = s.phoneme.to_string();
        }
    }
    phones
}

/// `--neural VOICE[:SPEED]`: the engine's phones for `text` (by the
/// rules, with `settings`) handed to `prosody.py neural`, which writes a
/// schedule with tiny-voca's prosody beside `out`. Returns its path.
fn neural_schedule(
    spec: &str,
    text: &str,
    out: &str,
    settings: &synthie_trombone::VoiceSettings,
) -> std::io::Result<String> {
    let (voice, speed) = spec
        .split_once(':')
        .map_or((spec, None), |(v, s)| (v, Some(s)));
    let utterance = Utterance::from_tokens(&speech::phonemize(text), &settings.prosody());
    let spans: Vec<[f32; 2]> = utterance
        .phone_spans()
        .iter()
        .map(|&(a, b)| [a, b])
        .collect();
    let phones_file = Path::new(out).with_extension("phones.json");
    let schedule_file = Path::new(out).with_extension("neural.json");
    std::fs::write(
        &phones_file,
        serde_json::json!({ "phones": phone_names(&utterance), "spans": spans }).to_string(),
    )?;
    let judge = std::env::var("SYNTHIE_JUDGE_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/judge").to_string());
    let python = std::env::var("SYNTHIE_NEURAL_PYTHON")
        .unwrap_or_else(|_| format!("{judge}/.venv/bin/python"));
    // The bridge's report goes to stderr: the first line of stdout is the
    // phones, which the judge reads.
    let status = Command::new(&python)
        .arg(format!("{judge}/prosody.py"))
        .arg("neural")
        .args(["--voice", voice])
        .args(speed.map(|s| ["--speed", s]).into_iter().flatten())
        .arg("--phones")
        .arg(&phones_file)
        .arg("--out")
        .arg(&schedule_file)
        .arg(text)
        .stdin(Stdio::null())
        .stdout(Stdio::from(std::io::stderr()))
        .status()
        .unwrap_or_else(|e| panic!("--neural: cannot run {python}: {e}"));
    assert!(
        status.success(),
        "--neural: prosody.py neural failed ({status})"
    );
    std::fs::remove_file(&phones_file)?;
    Ok(schedule_file.to_string_lossy().into_owned())
}

/// A schedule's `"voice"` object (a profile: the speaker's tract length,
/// tenseness, breathiness and pitch), as profile JSON.
fn schedule_voice(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    v.get("voice")
        .filter(|v| v.is_object())
        .map(ToString::to_string)
}

/// A key of a profile's `"vowels"` object: `IH`, `AY<` / `AY`, `AY>`, and
/// `UW+` for a fronted vowel.
fn vowel_key(name: &str) -> Option<VowelKey> {
    if let Some(symbol) = name.strip_suffix('+') {
        let key = VowelKey::Fronted(Phoneme::from_symbol(symbol)?);
        return vowel_targets::target(key).map(|_| key);
    }
    let (symbol, end) = match name.strip_suffix('>') {
        Some(s) => (s, true),
        None => (name.strip_suffix('<').unwrap_or(name), false),
    };
    let p = Phoneme::from_symbol(symbol)?;
    let key = match (p.manner(), end) {
        (_, true) => VowelKey::End(p),
        (Manner::Diphthong, false) => VowelKey::Start(p),
        _ => VowelKey::Vowel(p),
    };
    vowel_targets::target(key).map(|_| key)
}

/// Take the `"vowels"` object out of a profile: `(key, [F1, F2, F3])`.
fn take_vowels(json: &str) -> (String, Vec<(VowelKey, [f32; 3])>) {
    let mut value: serde_json::Value =
        serde_json::from_str(json).unwrap_or_else(|e| panic!("not a profile: {e}"));
    let Some(vowels) = value.as_object_mut().and_then(|o| o.remove("vowels")) else {
        return (json.to_string(), Vec::new());
    };
    let serde_json::Value::Object(vowels) = vowels else {
        panic!("\"vowels\" must be an object of [F1, F2, F3]");
    };
    let targets = vowels
        .iter()
        .map(|(name, f)| {
            let key = vowel_key(name).unwrap_or_else(|| panic!("no vowel shape {name:?}"));
            let f: Vec<f32> = f
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|v| v as f32))
                        .collect()
                })
                .unwrap_or_default();
            let f: [f32; 3] = f
                .try_into()
                .unwrap_or_else(|_| panic!("vowels.{name} must be [F1, F2, F3]"));
            (key, f)
        })
        .collect();
    (value.to_string(), targets)
}

/// Shape `voice`'s vowels for `targets` (adult man, Hz), at its size.
fn hold_vowels(voice: &mut Trombone, targets: &[(VowelKey, [f32; 3])]) {
    let length = voice.tract_length();
    let scale = vowel_targets::ADULT_MALE_LENGTH / length;
    let tuning = *voice.tuning();
    let map = inverse::Map::for_tuning(&tuning);
    for &(key, f) in targets {
        let near = key
            .articulation()
            .and_then(|a| a.physical_shape_with(length, &tuning));
        let shape = map.solve_for(
            f[0] * scale,
            f[1] * scale,
            Some(f[2] * scale),
            near.as_ref(),
            length,
        );
        voice.set_vowel_shape(key, Some(&shape));
    }
}

#[allow(clippy::too_many_lines)] // one flat argument loop
fn main() -> std::io::Result<()> {
    const SR: u32 = 48_000;
    let mut settings = Preset::Deep.settings();
    let mut engine = Engine::Classic;
    let mut source = GlottalSource::Folds;
    let mut path = String::from("say.wav");
    let mut persona = Persona::NEUTRAL;
    let mut profile: Option<String> = None;
    let mut dump = false;
    let mut schedule: Option<String> = None;
    let mut pitch_mode = None;
    let mut phone_times: Option<String> = None;
    let mut kick = true;
    let mut follow_voice = true;
    let mut follow_tenseness = true;
    let mut follow_formants = false;
    let mut follow_speed = true;
    let mut neural: Option<String> = None;
    let mut text = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value"))
        };
        match a.as_str() {
            "--voice" => {
                let v = value("--voice");
                let preset = Preset::from_name(&v).unwrap_or_else(|| {
                    let names: Vec<_> = Preset::ALL.iter().map(|p| p.name()).collect();
                    panic!("unknown voice {v:?}; try one of {}", names.join(", "))
                });
                settings = preset.settings();
            }
            "--engine" => {
                engine = match value("--engine").as_str() {
                    "classic" => Engine::Classic,
                    "physical" => Engine::Physical,
                    other => panic!("unknown engine {other:?}; try classic or physical"),
                };
            }
            "--source" => {
                source = match value("--source").as_str() {
                    "folds" => GlottalSource::Folds,
                    "pulse" => GlottalSource::Pulse,
                    other => panic!("unknown source {other:?}; try folds or pulse"),
                };
            }
            "--out" => path = value("--out"),
            "--size" => persona.size = value("--size").parse().expect("--size needs a number"),
            "--effort" => {
                persona.effort = value("--effort").parse().expect("--effort needs a number");
            }
            "--breathiness" => {
                persona.breathiness = value("--breathiness")
                    .parse()
                    .expect("--breathiness needs a number");
            }
            "--clarity" => {
                persona.clarity = value("--clarity")
                    .parse()
                    .expect("--clarity needs a number");
            }
            "--profile" => profile = Some(value("--profile")),
            "--match-prosody" => schedule = Some(value("--match-prosody")),
            "--prosody-pitch" => {
                pitch_mode = Some(match value("--prosody-pitch").as_str() {
                    "speaker" => PitchMode::Speaker,
                    "voice" => PitchMode::Voice,
                    other => panic!("unknown pitch mode {other:?}; try speaker or voice"),
                });
            }
            "--phone-times" => phone_times = Some(value("--phone-times")),
            "--prosody-kick" => kick = value("--prosody-kick") != "off",
            "--prosody-voice" => follow_voice = value("--prosody-voice") != "off",
            "--prosody-tenseness" => follow_tenseness = value("--prosody-tenseness") != "off",
            "--prosody-formants" => follow_formants = value("--prosody-formants") == "on",
            "--prosody-speed" => follow_speed = value("--prosody-speed") != "off",
            "--neural" => neural = Some(value("--neural")),
            "--dump-profile" => dump = true,
            "--dump-parameters" => {
                println!("{}", synthie_trombone::tuning_parameters());
                return Ok(());
            }
            "--pitch" => settings.pitch = value("--pitch").parse().expect("--pitch needs Hz"),
            "--rate" => settings.rate = value("--rate").parse().expect("--rate needs a number"),
            "--expr" => {
                settings.expressiveness = value("--expr").parse().expect("--expr needs a number");
            }
            _ => text.push(a),
        }
    }
    let text = if text.is_empty() {
        "Hello! I am the pink trombone. Can you understand me?".to_string()
    } else {
        text.join(" ")
    };

    let mut settings = settings.with_engine(engine).with_persona(&persona);
    if let Some(spec) = &neural {
        schedule = Some(neural_schedule(spec, &text, &path, &settings)?);
    }
    // The model's voice is not this one: by default its contour is moved to
    // this voice's own median.
    let pitch_mode = pitch_mode.unwrap_or(if neural.is_some() {
        PitchMode::Voice
    } else {
        PitchMode::Speaker
    });
    // A schedule's own voice: the speaker's tract length, voice quality and
    // pitch, as a profile laid over the preset (a --profile goes on top).
    let schedule_json = match &schedule {
        Some(file) => Some(std::fs::read_to_string(file)?),
        None => None,
    };
    if let Some(json) = &schedule_json
        && follow_voice
        && let Some(voice) = schedule_voice(json)
    {
        settings = settings
            .with_profile(&voice)
            .unwrap_or_else(|e| panic!("the schedule's voice: {e}"));
    }
    let mut vowels = Vec::new();
    if let Some(file) = profile {
        let json = std::fs::read_to_string(&file)?;
        let (json, own) = take_vowels(&json);
        vowels = own;
        settings = settings
            .with_profile(&json)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
    }
    if dump {
        println!("{}", settings.to_profile());
        return Ok(());
    }

    let mut talker = Talker::new(SR as f32, settings);
    talker.voice_mut().set_glottal_source(source);
    hold_vowels(talker.voice_mut(), &vowels);
    let mut followed = None;
    let utterance = if let (Some(file), Some(json)) = (&schedule, &schedule_json) {
        let schedule = read_schedule(json);
        let tokens = schedule.phrase(&speech::phonemize(&text));
        let mut u = Utterance::from_tokens(&tokens, &talker.settings().prosody());
        let mut options = FollowOptions::new(
            pitch_mode,
            &talker.settings().tuning,
            talker.settings().tract_length,
        );
        if !follow_tenseness {
            options.tenseness_range = (1.0, 1.0);
        }
        if !follow_formants {
            options.formant_range = 0.0;
        }
        if !follow_speed {
            options.max_speed = 1.0;
        }
        let report = u
            .follow_with(&schedule, &options)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        followed = Some(report);
        if !kick {
            // The speaker's contour has his own rise after voiceless
            // consonants; this leaves out the engine's on top of it.
            for s in u.segments_mut() {
                s.gesture.kick = 0.0;
            }
        }
        u
    } else {
        talker.compile(&text)
    };
    let spans = utterance.phone_spans();
    let phones = phone_names(&utterance);
    // One symbol per segment, as the judge reads it; a vowel's glide on
    // another shape than its own (a lax vowel's towards schwa, a vowel's
    // towards the dark /l/ after it) is part of the vowel, not a segment of
    // its own.
    let mut last = None;
    for s in utterance.segments() {
        let offglide = s.phoneme.manner() == Manner::Vowel
            && s.articulation.vowel.is_some()
            && s.articulation.vowel != Some(VowelKey::Vowel(s.phoneme))
            && last == Some(s.phoneme);
        if !offglide {
            print!("{} ", s.phoneme);
        }
        last = Some(s.phoneme);
    }
    println!();
    let started = Instant::now();
    let mut out = talker.bake_utterance(utterance);
    let took = started.elapsed().as_secs_f32();
    normalize(&mut out, 0.89);

    wav::write(&path, SR, &out)?;
    let seconds = out.len() as f32 / SR as f32;
    println!(
        "wrote {path} ({seconds:.1} s, rendered in {took:.2} s: {:.0}x real time)",
        seconds / took
    );
    if let Some(r) = followed {
        println!(
            "followed: {} phones, {} below the engine's floor (+{:.0} ms), contour moved {:+.1} st, {} breath shares limited, {} tenseness shares limited, {} vowels shaped ({} limited), {} phones sped up (up to {:.2}x)",
            r.spans.len(),
            r.too_short,
            1000.0 * r.added,
            r.shift,
            r.breath_limited,
            r.tenseness_limited,
            r.shaped,
            r.formants_limited,
            r.sped,
            r.fastest
        );
    }
    if let Some(file) = phone_times {
        let spans: Vec<[f32; 2]> = spans.iter().map(|&(a, b)| [a, b]).collect();
        std::fs::write(
            &file,
            serde_json::json!({ "spans": spans, "phones": phones }).to_string(),
        )?;
    }
    Ok(())
}
