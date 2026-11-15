//! How the physical engine's articulators move through running speech: for
//! every line of text, the tract's area function, its targets, the velum and
//! the tract's own F1, F2 and F3 (from its transfer function, no LPC) every
//! 2.5 ms, with the phones' spans, so that a script can measure how much of
//! each phone the shape spends on its way and how much holding its target.
//!
//! ```sh
//! cargo run -p synthie-trombone --example plateau --release -- lines.txt out_dir
//! cargo run -p synthie-trombone --example plateau --release -- --profile p.json lines.txt out_dir
//! ```
//!
//! One line of text per line of the file; for line `k` it writes
//! `out_dir/k.wav` (48 kHz) and `out_dir/k.json`: `phones` (symbol, stressed,
//! start and end in seconds), `hop`, and per frame `t`, `d` (the 44 cell
//! diameters), `g` (their targets), `v` (velum opening) and `f` (F1 F2 F3, Hz,
//! every other frame; zero where there is no resonance below 4 kHz).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/wav.rs"]
mod wav;

use std::fmt::Write as _;
use synthie_trombone::speech::{Speaker, Token, Utterance, phonemize};
use synthie_trombone::{Engine, Preset, Trombone, VoiceSettings};

const SR: f32 = 48_000.0;
/// Frame hop, output samples (2.5 ms).
const HOP: usize = 120;

/// F1, F2 and F3 of the voice's tract as it is now.
fn formants(voice: &Trombone) -> [f32; 3] {
    let step = 6.0;
    let rate = voice.tick_rate();
    let mut x = vec![0.0f32; (0.06 * rate) as usize];
    voice.impulse_response(&mut x);
    let mag = spectrum::spectrum(&x, rate, 4000.0, step);
    let r = spectrum::resonances(&mag, step);
    let mut f = r.iter().map(|r| r.hz).filter(|&hz| hz > 150.0);
    [
        f.next().unwrap_or(0.0),
        f.next().unwrap_or(0.0),
        f.next().unwrap_or(0.0),
    ]
}

fn list(out: &mut String, xs: &[f32], digits: usize) {
    out.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{x:.digits$}");
    }
    out.push(']');
}

fn line(settings: &VoiceSettings, text: &str) -> (String, Vec<f32>) {
    let stresses: Vec<(&str, bool)> = phonemize(text)
        .iter()
        .filter_map(|t| match *t {
            Token::Phone { phoneme, stressed } => Some((phoneme.symbol(), stressed)),
            _ => None,
        })
        .collect();
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let spans = utterance.phone_spans();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut audio = Vec::new();
    let mut frames = String::from("[");
    let mut n = 0usize;
    while speaker.is_speaking() {
        audio.push(speaker.process(&mut voice));
        if n % HOP == 0 {
            if n > 0 {
                frames.push(',');
            }
            let _ = write!(frames, "{{\"t\":{:.4},\"d\":", n as f32 / SR);
            list(&mut frames, voice.diameters(), 3);
            frames.push_str(",\"g\":");
            list(&mut frames, voice.target_diameters(), 3);
            let _ = write!(frames, ",\"v\":{:.3}", voice.velum());
            if (n / HOP) % 2 == 0 {
                frames.push_str(",\"f\":");
                list(&mut frames, &formants(&voice), 0);
            }
            frames.push('}');
        }
        n += 1;
    }
    frames.push(']');
    let mut out = String::from("{\"phones\":[");
    for (k, &(a, b)) in spans.iter().enumerate() {
        let (p, s) = stresses.get(k).copied().unwrap_or(("?", false));
        if k > 0 {
            out.push(',');
        }
        let _ = write!(out, "[\"{p}\",{},{a:.4},{b:.4}]", u8::from(s));
    }
    let _ = write!(out, "],\"hop\":{},\"frames\":{frames}}}", HOP as f32 / SR);
    (out, audio)
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut rest = Vec::new();
    while let Some(a) = args.next() {
        if a == "--profile" {
            let path = args.next().expect("--profile FILE");
            let text = std::fs::read_to_string(&path)?;
            settings = settings.with_profile(&text).expect("profile");
        } else {
            rest.push(a);
        }
    }
    let [file, dir] = <[String; 2]>::try_from(rest).expect("lines.txt out_dir");
    let text = std::fs::read_to_string(file)?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    std::fs::create_dir_all(&dir)?;
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (lines, settings, dir) = (&lines, &settings, &dir);
            s.spawn(move || {
                for (k, l) in lines.iter().enumerate().filter(|(k, _)| k % threads == t) {
                    let (json, audio) = line(settings, l);
                    std::fs::write(format!("{dir}/{k}.json"), json).expect("json");
                    wav::write(&format!("{dir}/{k}.wav"), SR as u32, &audio).expect("wav");
                }
            });
        }
    });
    Ok(())
}
