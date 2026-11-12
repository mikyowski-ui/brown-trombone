//! How a sentence moves on the physical engine: a timeline of the level,
//! the breath, the voice and the hiss, segment by segment.
//!
//! ```sh
//! cargo run -p synthie-trombone --example phrase --release -- "The birch canoe slid on the smooth planks."
//! cargo run -p synthie-trombone --example phrase --release -- --every 5 "Glue the sheet."   # a row per 5 ms
//! cargo run -p synthie-trombone --example phrase --release -- --profile voice.json --out phrase.wav "..."
//! ```
//!
//! For every segment it prints its phoneme, start, length and pitch, and
//! inside it the output level (dB under the sentence's loudest 10 ms), the
//! pressure below the folds (cmH2O), the fold pitch the engine measures and
//! the noise of the strongest jet in the mouth. Below, every level step
//! steeper than 20 dB in 10 ms and every stretch more than 30 dB under the
//! peak inside the sentence, with the segments they fall in: the hard
//! edges and dead gaps a listener hears as a machine.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/wav.rs"]
mod wav;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Preset};

const SR: f32 = 48_000.0;
/// Samples per analysis frame (2.5 ms) and per level window (10 ms).
const HOP: usize = 120;
const WINDOW: usize = 480;

#[allow(clippy::too_many_lines)] // one flat report
fn main() -> std::io::Result<()> {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut every_ms: Option<f32> = None;
    let mut out: Option<String> = None;
    let mut text = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--every" => every_ms = args.next().and_then(|v| v.parse().ok()),
            "--out" => out = args.next(),
            "--profile" => {
                let file = args.next().expect("--profile needs a file");
                let json = std::fs::read_to_string(&file)?;
                settings = settings
                    .with_profile(&json)
                    .unwrap_or_else(|e| panic!("{file}: {e}"));
            }
            _ => text.push(a),
        }
    }
    let text = if text.is_empty() {
        "The birch canoe slid on the smooth planks.".to_string()
    } else {
        text.join(" ")
    };

    let mut voice = synthie_trombone::Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(&text, &settings.prosody());
    let segments = utterance.segments().to_vec();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut audio = Vec::new();
    let mut sub = Vec::new();
    let mut pitch = Vec::new();
    let mut jet = Vec::new();
    let mut flows = Vec::new();
    let mut index = Vec::new();
    let tail = (0.3 * SR) as usize;
    let mut after = 0;
    while after < tail {
        if !speaker.is_speaking() {
            after += 1;
        }
        index.push(speaker.current().map_or(usize::MAX, |(k, _)| k));
        audio.push(speaker.process(&mut voice));
        let g = voice.gauges();
        sub.push(g.subglottal_pressure);
        pitch.push(g.fold_pitch);
        jet.push(g.jet_noise);
        flows.push((g.lip_flow, g.skin_flow, g.glottal_flow));
    }

    // Level per 2.5 ms over a 10 ms window, relative to the loudest.
    let frames = audio.len() / HOP;
    let level: Vec<f32> = (0..frames)
        .map(|f| {
            let c = f * HOP + HOP / 2;
            let a = c.saturating_sub(WINDOW / 2);
            let b = (c + WINDOW / 2).min(audio.len());
            let ms = audio[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a).max(1) as f32;
            10.0 * ms.max(1e-15).log10()
        })
        .collect();
    let peak = level.iter().fold(f32::MIN, |m, &v| m.max(v));
    let rel: Vec<f32> = level.iter().map(|v| v - peak).collect();
    let name = |k: usize| {
        segments
            .get(k)
            .map_or("-".to_string(), |s| s.phoneme.to_string())
    };

    let every = every_ms.map(|ms| ((ms / 1000.0 * SR) as usize / HOP).max(1));
    let mut start = 0.0f32;
    let mut f = 0;
    for (k, s) in segments.iter().enumerate() {
        let end = start + s.duration;
        let (a, b) = ((start * SR) as usize / HOP, (end * SR) as usize / HOP);
        let span = &rel[a.min(frames)..b.min(frames)];
        let (lo, hi) = span
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        let mid = (f32::midpoint(start, end) * SR) as usize;
        println!(
            "{k:3} {:>4} {:6.3} s {:4.0} ms {:5.1}->{:5.1} Hz {} level {:6.1}..{:6.1} dB  Psub {:4.1}  f0 {:5.1}  jet {:6.3}",
            s.phoneme.to_string(),
            start,
            1000.0 * s.duration,
            s.pitch.0,
            s.pitch.1,
            if s.voiced { "v" } else { "-" },
            lo,
            hi,
            sub.get(mid).copied().unwrap_or(0.0),
            pitch.get(mid).copied().unwrap_or(0.0),
            jet.get(mid).copied().unwrap_or(0.0),
        );
        if let Some(n) = every {
            while f < b.min(frames) {
                if f >= a {
                    let i = f * HOP + HOP / 2;
                    let (lips, skin, glottis) = flows[i];
                    println!(
                        "      {:6.3} {:6.1} dB  Psub {:5.2}  f0 {:5.1}  jet {:6.3}  U lips {lips:6.1} skin {skin:6.1} glottis {glottis:6.1}",
                        i as f32 / SR,
                        rel[f],
                        sub[i],
                        pitch[i],
                        jet[i]
                    );
                }
                f += n;
            }
        }
        start = end;
    }

    let (at, top) =
        audio.iter().enumerate().fold(
            (0, 0.0f32),
            |(k, m), (i, v)| if v.abs() > m { (i, v.abs()) } else { (k, m) },
        );
    println!(
        "\nloudest sample {top:.3} at {:.3} s in {}",
        at as f32 / SR,
        name(index[at])
    );

    // Steps steeper than 20 dB in 10 ms, and dips below -30 dB inside the
    // sentence.
    let lag = 4;
    let first = rel.iter().position(|&v| v > -30.0).unwrap_or(0);
    let last = rel.iter().rposition(|&v| v > -30.0).unwrap_or(0);
    println!("\nwalls (> 20 dB in 10 ms):");
    let mut f = first;
    while f + lag <= last {
        let d = rel[f + lag] - rel[f];
        if d.abs() > 20.0 && rel[f].max(rel[f + lag]) > -40.0 {
            let i = (f + lag / 2) * HOP;
            println!(
                "  {:6.3} s {:+5.1} dB  in {}",
                i as f32 / SR,
                d,
                name(index[i])
            );
            f += lag;
        } else {
            f += 1;
        }
    }
    println!("dips (< -30 dB inside the sentence):");
    let mut f = first;
    while f < last {
        if rel[f] < -30.0 {
            let a = f;
            while f < last && rel[f] < -30.0 {
                f += 1;
            }
            let deepest = rel[a..f].iter().fold(0.0f32, |m, &v| m.min(v));
            println!(
                "  {:6.3} s {:4.0} ms down to {:6.1} dB  {} .. {}",
                (a * HOP) as f32 / SR,
                ((f - a) * HOP) as f32 / SR * 1000.0,
                deepest,
                name(index[a * HOP]),
                name(index[(f * HOP).min(index.len() - 1)])
            );
        }
        f += 1;
    }
    if let Some(path) = out {
        let mut audio = audio;
        synthie_trombone::normalize(&mut audio, 0.89);
        wav::write(&path, SR as u32, &audio)?;
    }
    Ok(())
}
