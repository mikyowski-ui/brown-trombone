//! How the tract gets from one shape to the next: F1 to F3 of the tract's
//! transfer function every 5 ms while a phrase is spoken on the physical
//! engine (deep voice), with the segment, the narrowest opening, the velum
//! and the diameters of a few cells. While the velum is open the peaks are
//! those of the tract and the nose together.
//!
//! ```sh
//! cargo run -p synthie-trombone --example glide --release -- "a canoe" "slid"
//! cargo run -p synthie-trombone --example glide --release -- --cells 14,20,24,28,34,41 "the new"
//! cargo run -p synthie-trombone --example glide --release -- --set lateral_release=0 "slid"
//! cargo run -p synthie-trombone --example glide --release -- --targets "lid"   # target diameters
//! cargo run -p synthie-trombone --example glide --release -- --shapes          # every vowel's shape
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::format_push_string,
    clippy::too_many_lines
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, N, Phoneme, Preset, Trombone};

const SR: f32 = 48_000.0;
/// Samples per frame (5 ms).
const FRAME: usize = 240;

fn formants(voice: &Trombone) -> [f32; 3] {
    let step = 8.0;
    let mag = spectrum::transfer(voice, 0.05, 4200.0, step);
    let peaks: Vec<f32> = spectrum::resonances(&mag, step)
        .iter()
        .map(|r| r.hz)
        .filter(|&hz| hz >= 180.0)
        .collect();
    [0, 1, 2].map(|k| peaks.get(k).copied().unwrap_or(0.0))
}

fn main() {
    let mut cells: Vec<usize> = vec![14, 18, 21, 23, 25, 27, 30, 33, 36, 40, 42];
    let mut texts = Vec::new();
    let mut targets = false;
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cells" => {
                cells = args
                    .next()
                    .expect("--cells needs a list")
                    .split(',')
                    .map(|c| c.parse().expect("cell index"))
                    .filter(|&c: &usize| c < N)
                    .collect();
            }
            "--set" => {
                let kv = args.next().expect("--set needs name=value");
                let (k, v) = kv.split_once('=').expect("--set needs name=value");
                assert!(
                    settings.tuning.set(k, v.parse().expect("a number")),
                    "no tuning field {k}"
                );
            }
            "--targets" => targets = true,
            "--shapes" => {
                for p in Phoneme::ALL.iter().filter(|p| p.manner() == Manner::Vowel) {
                    println!("{p}\t{:?}", p.articulation().physical_shape(1.0));
                }
                return;
            }
            _ => texts.push(a),
        }
    }
    if texts.is_empty() {
        texts = [
            "a canoe",
            "canoe slid on",
            "the new",
            "to do",
            "slid",
            "lid",
            "blue",
        ]
        .map(String::from)
        .to_vec();
    }
    for text in &texts {
        let mut voice = Trombone::new(SR);
        settings.apply(&mut voice);
        Speaker::prepare(&mut voice);
        let utterance = Utterance::from_text(text, &settings.prosody());
        let mut speaker = Speaker::new();
        speaker.speak(utterance);
        println!("# {text}");
        let mut head = String::from("ms\tseg\tF1\tF2\tF3\tnarrow\tat\tvelum\tnudge");
        for c in &cells {
            head.push_str(&format!("\td{c}"));
        }
        println!("{head}");
        let mut n = 0usize;
        let mut last = usize::MAX;
        while speaker.is_speaking() {
            if n.is_multiple_of(FRAME) {
                let (index, phoneme) = speaker
                    .current()
                    .map_or((usize::MAX, Phoneme::Sil), |(i, s)| (i, s.phoneme));
                let d = if targets {
                    voice.target_diameters()
                } else {
                    voice.diameters()
                };
                let (at, narrow) = (10..N)
                    .map(|i| (i, d[i]))
                    .fold((10, f32::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
                let f = if voice.diameters()[10..].iter().all(|&x| x > 0.05) {
                    formants(&voice)
                } else {
                    [0.0; 3]
                };
                // A new segment is marked with `*`.
                let mark = if index == last { " " } else { "*" };
                last = index;
                let mut row = format!(
                    "{:.0}\t{mark}{phoneme}\t{:.0}\t{:.0}\t{:.0}\t{narrow:.2}\t{at}\t{:.2}\t{:.3}",
                    n as f32 / SR * 1000.0,
                    f[0],
                    f[1],
                    f[2],
                    voice.velum(),
                    voice.gauges().gap_nudge
                );
                for &c in &cells {
                    row.push_str(&format!("\t{:.2}", d[c]));
                }
                println!("{row}");
            }
            speaker.process(&mut voice);
            n += 1;
        }
        println!();
    }
}
