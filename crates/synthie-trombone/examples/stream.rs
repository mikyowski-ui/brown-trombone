//! Feed a voice through a [`VoiceStream`] and write what it says to a WAV
//! file: first a scripted list of timed events (a hummed phrase with
//! gestures on the velum and the lips), then the lines of stdin, sent a
//! word at a time from another thread as a language model would send them.
//!
//! ```sh
//! echo "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background." |
//!     cargo run --release -p synthie-trombone --example stream -- --engine physical --out stream.wav
//! ```
//!
//! `--engine classic|physical`, `--voice NAME` (a preset), `--out FILE`
//! (default `stream.wav`), `--words-per-second N` (default 4: how fast the
//! text arrives, on the voice's clock), `--phrases` (cut the text at
//! commas too, not only at full stops).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/wav.rs"]
mod wav;

use std::io::BufRead;
use std::time::{Duration, Instant};
use synthie_trombone::stream::standin::{Gesture, Tier};
use synthie_trombone::stream::{Event, Granularity, VoiceStream, voice_stream};
use synthie_trombone::{Engine, Phoneme, Preset, Trombone};

const RATE: f32 = 48_000.0;

/// The hummed opening: seconds on the voice's clock and what happens then.
fn script() -> Vec<(f32, Event)> {
    vec![
        (0.00, Event::Articulate(Phoneme::M)),
        (0.00, Event::NoteOn(45.0)),
        (0.35, Event::Articulate(Phoneme::Aa)),
        (0.50, Event::Note(47.0)),
        (0.70, Event::Note(49.0)),
        // The velum opens (a nasal /a/), then the lips close for a hum.
        (0.90, Event::Gesture(Gesture::new(Tier::Velum, 0.8))),
        (1.20, Event::Gesture(Gesture::new(Tier::LipsAndJaw, 1.0))),
        (1.50, Event::Gesture(Gesture::new(Tier::LipsAndJaw, 0.0))),
        (1.50, Event::Gesture(Gesture::new(Tier::Velum, 0.0))),
        (1.50, Event::Note(45.0)),
        (1.90, Event::NoteOff),
    ]
}

struct Args {
    engine: Engine,
    voice: Preset,
    out: String,
    words_per_second: f32,
    granularity: Granularity,
}

fn args() -> Result<Args, String> {
    let mut a = Args {
        engine: Engine::Classic,
        voice: Preset::Deep,
        out: "stream.wav".into(),
        words_per_second: 4.0,
        granularity: Granularity::Sentence,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--engine" => {
                a.engine = match value()?.as_str() {
                    "classic" => Engine::Classic,
                    "physical" => Engine::Physical,
                    e => return Err(format!("unknown engine {e:?}")),
                }
            }
            "--voice" => {
                let name = value()?;
                a.voice = Preset::from_name(&name).ok_or(format!("unknown voice {name:?}"))?;
            }
            "--out" => a.out = value()?,
            "--words-per-second" => {
                a.words_per_second = value()?.parse().map_err(|e| format!("{e}"))?;
            }
            "--phrases" => a.granularity = Granularity::Phrase,
            _ => return Err(format!("unexpected {arg:?}")),
        }
    }
    Ok(a)
}

fn main() -> Result<(), String> {
    let args = args()?;
    let settings = args.voice.settings().with_engine(args.engine);
    let (mut stream, mut player) = voice_stream(4096);
    let mut voice = Trombone::new(RATE);
    settings.apply(&mut voice);
    stream.set_prosody(settings.prosody());
    stream.set_granularity(args.granularity);
    let script = script();
    let opening = script.iter().map(|(t, _)| *t).fold(0.0, f32::max) + 0.3;
    for (t, e) in &script {
        stream
            .push_at((t * RATE) as u64, *e)
            .map_err(|e| e.to_string())?;
    }

    // The controller: stdin, a word at a time, paced on the voice's clock.
    let lines: Vec<String> = std::io::stdin()
        .lock()
        .lines()
        .map_while(Result::ok)
        .collect();
    let words_per_second = args.words_per_second.max(0.1);
    let controller = std::thread::spawn(move || {
        let mut due = opening;
        let wait_for = |stream: &VoiceStream<'_>, t: f32| {
            while (stream.now() as f32) < t * RATE {
                std::thread::sleep(Duration::from_micros(200));
            }
        };
        let mut sent = 0;
        for line in &lines {
            for word in line.split_inclusive(' ') {
                wait_for(&stream, due);
                sent += stream.push_text(word).utterances;
                due += 1.0 / words_per_second;
            }
            sent += stream.push_text("\n").utterances;
        }
        sent += stream.flush_text().utterances;
        // Stay until the player has taken everything, so that what it
        // is done with is freed here.
        while !stream.held_text().is_empty() {
            sent += stream.flush_text().utterances;
            std::thread::sleep(Duration::from_millis(1));
        }
        (stream, sent)
    });

    // The audio side: render until the controller is done and the voice
    // has said everything.
    let started = Instant::now();
    let mut out = Vec::new();
    let mut block = [0.0f32; 256];
    loop {
        player.render(&mut voice, &mut block);
        out.extend_from_slice(&block);
        if controller.is_finished() && player.is_idle() {
            break;
        }
        if out.len() as f32 > 600.0 * RATE {
            return Err("ten minutes of sound and still going".into());
        }
    }
    let (mut stream, sent) = controller.join().map_err(|_| "the controller panicked")?;
    stream.collect();
    // A little silence at the end.
    out.extend(std::iter::repeat_n(0.0, (0.2 * RATE) as usize));
    wav::write(&args.out, RATE as u32, &out).map_err(|e| e.to_string())?;
    println!(
        "{}: {:.2} s, {} timed events and {sent} pieces of text, rendered in {:.2} s",
        args.out,
        out.len() as f32 / RATE,
        script.len(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}
