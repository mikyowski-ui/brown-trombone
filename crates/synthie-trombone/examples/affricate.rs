//! The affricates /tʃ dʒ/ on the physical engine, measured where their
//! hiss is made: through the release of each one, the narrowest place in
//! the mouth (its area), the flow through it and its Reynolds number, how
//! much of the time a jet there is turbulent, and the level of the hiss
//! (2 to 8 kHz) against the vowel next to it, beside the same figures for
//! the fricatives /ʃ ʒ/ in the same vowels, on both tracts.
//!
//! ```sh
//! cargo run -p synthie-trombone --example affricate --release
//! cargo run -p synthie-trombone --example affricate --release -- --trace "the juice of lemons"
//! ```
//!
//! Each line is one consonant: its frication (for an affricate the
//! segment after its closure), the median and smallest area of the
//! narrowest cell from the tongue body to the teeth (cells 24 to 39), the
//! median and largest Reynolds number of the lip flow through it, the share
//! of the frication in which the engine's jet noise is on, the 2 to 8 kHz
//! level of the loudest 10 ms frame and of the frication's mean, both
//! against the loudest 10 ms of the vowel after it (all bands), and the
//! centroid of the hiss over 1 to 8 kHz. `--trace TEXT` prints the same
//! readings every 2.5 ms through every affricate and fricative of TEXT.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, TractModel, Trombone, Tuning};

const SR: f32 = 48_000.0;
/// Kinematic viscosity of air, cm²/s (the engine's μ/ρ).
const NU: f32 = 1.86e-4 / 1.14e-3;
/// Cells searched for the narrowest place: tongue body to the teeth.
const FROM: usize = 24;
const TO: usize = 40;

/// One output sample's readings.
#[derive(Clone, Copy, Default)]
struct Reading {
    audio: f32,
    area: f32,
    cell: usize,
    flow: f32,
    jet: f32,
    shut: bool,
    /// Pressure behind the narrowest place, cmH2O.
    behind: f32,
    /// Flow through the glottis, cm³/s, and its area, cm².
    glottal: f32,
    glottal_area: f32,
}

fn reynolds(flow: f32, area: f32) -> f32 {
    if area <= 0.0 {
        return 0.0;
    }
    flow.max(0.0) * (4.0 / (core::f32::consts::PI * area)).sqrt() / NU
}

fn say(text: &str, tract: TractModel, tongue: f32) -> (Vec<Reading>, Vec<(Phoneme, f32, f32)>) {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    settings.tuning = Tuning::DEFAULT.with_tract(tract);
    settings.tuning.anatomy.tongue_model = tongue;
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut t = 0.0;
    let segments = utterance
        .segments()
        .iter()
        .map(|s| {
            t += s.duration;
            (s.phoneme, t - s.duration, t)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let len = voice.tract_length();
    let scale = voice.area_scale();
    let mut out = Vec::new();
    while speaker.is_speaking() {
        let audio = speaker.process(&mut voice);
        let d = voice.diameters();
        let mut cell = FROM;
        for k in FROM..TO {
            if d[k] < d[cell] {
                cell = k;
            }
        }
        let dd = d[cell].max(0.0) * len;
        let g = voice.gauges();
        out.push(Reading {
            audio,
            area: scale * 0.25 * core::f32::consts::PI * dd * dd,
            cell,
            flow: g.lip_flow,
            jet: g.jet_noise,
            shut: g.mouth_shut,
            behind: voice.pressure(cell.saturating_sub(3)),
            glottal: g.glottal_flow,
            glottal_area: g.glottal_area,
        });
    }
    (out, segments)
}

/// Power of x near hz (a Hann-windowed DFT bin).
fn bin(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = 2.0 * core::f32::consts::PI * hz / SR;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (k, &v) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * k as f32 / n as f32).cos();
        re += v * h * (w * k as f32).cos();
        im -= v * h * (w * k as f32).sin();
    }
    (re * re + im * im) / n as f32
}

/// Power between lo and hi, in 100 Hz bins.
fn band(x: &[f32], lo: f32, hi: f32) -> f32 {
    let mut sum = 0.0;
    let mut hz = lo + 50.0;
    while hz < hi {
        sum += bin(x, hz);
        hz += 100.0;
    }
    sum
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-20).log10()
}

/// The loudest 10 ms frame (hop 2.5 ms) of x between lo and hi Hz.
fn loudest(x: &[f32], lo: f32, hi: f32) -> f32 {
    let (win, hop) = ((0.01 * SR) as usize, (0.0025 * SR) as usize);
    let mut best = 0.0f32;
    let mut a = 0;
    while a + win <= x.len() {
        best = best.max(band(&x[a..a + win], lo, hi));
        a += hop;
    }
    best
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

struct Row {
    phoneme: Phoneme,
    area: f32,
    least: f32,
    re: f32,
    re_max: f32,
    on: f32,
    peak: f32,
    mean: f32,
    centroid: f32,
    ms: f32,
    /// Mean square of the jet noise gauge over the frication, dB re 1 cmH2O.
    source: f32,
}

/// The frication of every affricate and /ʃ ʒ/ of text.
fn rows(text: &str, tract: TractModel, tongue: f32) -> Vec<Row> {
    let (r, segments) = say(text, tract, tongue);
    let audio: Vec<f32> = r.iter().map(|x| x.audio).collect();
    let mut out = Vec::new();
    for (k, s) in segments.iter().enumerate() {
        let p = s.0;
        let wanted = match p.manner() {
            // The second of an affricate's two segments is its frication.
            Manner::Affricate => {
                segments[..k].iter().rev().take_while(|x| x.0 == p).count() % 2 == 1
            }
            Manner::Fricative => matches!(p, Phoneme::Sh | Phoneme::Zh),
            _ => false,
        };
        if !wanted {
            continue;
        }
        let a = (s.1 * SR) as usize;
        let b = ((s.2 * SR) as usize).min(r.len());
        if b <= a + 480 {
            continue;
        }
        let part = &r[a..b];
        let open: Vec<&Reading> = part.iter().filter(|x| !x.shut).collect();
        let area = median(open.iter().map(|x| x.area).collect());
        let least = open.iter().map(|x| x.area).fold(f32::MAX, f32::min);
        let res: Vec<f32> = open.iter().map(|x| reynolds(x.flow, x.area)).collect();
        let re_max = res.iter().copied().fold(0.0, f32::max);
        let on = part.iter().filter(|x| x.jet > 0.0).count() as f32 / part.len() as f32;
        // The vowel after it (or before, at the end of a word).
        let vowel = segments
            .get(k + 1)
            .filter(|v| matches!(v.0.manner(), Manner::Vowel | Manner::Diphthong))
            .or_else(|| {
                segments[..k]
                    .iter()
                    .rev()
                    .find(|v| matches!(v.0.manner(), Manner::Vowel | Manner::Diphthong))
            });
        let reference = vowel.map_or(1.0, |v| {
            let (va, vb) = ((v.1 * SR) as usize, ((v.2 * SR) as usize).min(audio.len()));
            loudest(&audio[va..vb], 0.0, 8000.0)
        });
        let x = &audio[a..b];
        let peak = db(loudest(x, 2000.0, 8000.0) / reference);
        let mean = db(band(x, 2000.0, 8000.0) * (0.01 * SR) / x.len() as f32 / reference);
        let (mut num, mut den) = (0.0, 0.0);
        let mut hz = 1050.0;
        while hz < 8000.0 {
            let v = bin(x, hz);
            num += v * hz;
            den += v;
            hz += 100.0;
        }
        out.push(Row {
            phoneme: p,
            area,
            least: if least == f32::MAX { 0.0 } else { least },
            re: median(res),
            re_max,
            on,
            peak,
            mean,
            centroid: num / den.max(1e-20),
            ms: 1000.0 * (s.2 - s.1),
            source: db(part.iter().map(|x| x.jet * x.jet).sum::<f32>() / part.len() as f32),
        });
    }
    out
}

fn trace(text: &str, tract: TractModel, tongue: f32) {
    let (r, segments) = say(text, tract, tongue);
    println!("{tract:?}: {text}");
    for s in &segments {
        if !matches!(s.0.manner(), Manner::Affricate | Manner::Fricative) {
            continue;
        }
        println!("  {} {:.3} to {:.3} s", s.0, s.1, s.2);
        let (a, b) = (
            ((s.1 - 0.01) * SR) as usize,
            ((s.2 * SR) as usize).min(r.len()),
        );
        let hop = (0.0025 * SR) as usize;
        let mut k = a;
        while k < b {
            let x = r[k];
            println!(
                "    {:6.1} ms cell {:2} area {:.4} cm² flow {:6.0} Re {:6.0} jet {:7.3} behind {:5.2} glottis {:5.0} cm³/s {:.3} cm²{}",
                1000.0 * (k as f32 / SR - s.1),
                x.cell,
                x.area,
                x.flow,
                reynolds(x.flow, x.area),
                x.jet,
                x.behind,
                x.glottal,
                x.glottal_area,
                if x.shut { " shut" } else { "" }
            );
            k += hop;
        }
    }
}

/// Words and nonsense words with /tʃ dʒ/ and /ʃ ʒ/ in front, back and
/// rounded vowels.
const TEXTS: [&str; 10] = [
    "The juice of lemons.",
    "[AX0 JH IY1 S]",
    "[AX0 JH AA1 S]",
    "[AX0 JH UW1 S]",
    "[AX0 ZH IY1 S]",
    "[AX0 ZH AA1 S]",
    "[AX0 ZH UW1 S]",
    "[AX0 CH IY1 S] [AX0 CH AA1 S] [AX0 CH UW1 S]",
    "[AX0 SH IY1 S] [AX0 SH AA1 S] [AX0 SH UW1 S]",
    "The judge ate much cheese at the edge.",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tongue = if args.iter().any(|a| a == "--modal") {
        1.0
    } else {
        0.0
    };
    if let Some(k) = args.iter().position(|a| a == "--trace") {
        let text = args.get(k + 1).expect("--trace needs a text");
        for tract in [TractModel::Anatomical, TractModel::Original] {
            trace(text, tract, tongue);
        }
        return;
    }
    // `--text TEXT` (any number of them) measures those instead.
    let own: Vec<&str> = args
        .iter()
        .enumerate()
        .filter(|(k, _)| *k > 0 && args[k - 1] == "--text")
        .map(|(_, t)| t.as_str())
        .collect();
    let texts: Vec<&str> = if own.is_empty() { TEXTS.to_vec() } else { own };
    for tract in [TractModel::Anatomical, TractModel::Original] {
        println!("{tract:?}");
        println!(
            "  text                                     ph   ms  area  least     Re  Re max  jet on  peak  mean  centroid source  (cm², dB re vowel, Hz, dB re 1 cmH2O)"
        );
        for &text in &texts {
            for row in rows(text, tract, tongue) {
                println!(
                    "  {text:<40} {:>3} {:4.0} {:.3} {:.3} {:6.0} {:7.0} {:6.0}% {:5.1} {:5.1} {:8.0} {:6.1}",
                    row.phoneme.to_string(),
                    row.ms,
                    row.area,
                    row.least,
                    row.re,
                    row.re_max,
                    100.0 * row.on,
                    row.peak,
                    row.mean,
                    row.centroid,
                    row.source
                );
            }
        }
    }
}
