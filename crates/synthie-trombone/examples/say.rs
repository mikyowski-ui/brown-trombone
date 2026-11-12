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

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/wav.rs"]
mod wav;

use std::time::Instant;
use synthie_trombone::{Engine, GlottalSource, Persona, Preset, Talker, normalize};

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
    if let Some(file) = profile {
        let json = std::fs::read_to_string(&file)?;
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
    let utterance = talker.compile(&text);
    for s in utterance.segments() {
        print!("{} ", s.phoneme);
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
    Ok(())
}
