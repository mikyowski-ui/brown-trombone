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
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use std::fmt::Write as _;
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, TractModel, Trombone, Tuning};

const SR: f32 = 48_000.0;
/// Kinematic viscosity of air, cm²/s (the engine's μ/ρ).
const NU: f32 = 1.86e-4 / 1.14e-3;
/// Cells searched for the narrowest place: tongue body to the teeth.
const FROM: usize = 24;
const TO: usize = 40;

/// Tuning fields set with `--set NAME=VALUE` (any number of them).
static SETS: std::sync::OnceLock<Vec<(String, f32)>> = std::sync::OnceLock::new();
/// Where `--raw FILE` writes the audio of `--final`.
static RAW: std::sync::OnceLock<String> = std::sync::OnceLock::new();

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
    /// The part of the glottal area between the vibrating edges, cm².
    edge_area: f32,
    /// Pressure below and above the folds, cmH2O.
    below: f32,
    above: f32,
    /// Pitch of the self-oscillating folds, Hz (0 while they are still).
    fold_pitch: f32,
}

fn reynolds(flow: f32, area: f32) -> f32 {
    if area <= 0.0 {
        return 0.0;
    }
    flow.max(0.0) * (4.0 / (core::f32::consts::PI * area)).sqrt() / NU
}

fn say(text: &str, tract: TractModel, tongue: f32) -> (Vec<Reading>, Vec<(Phoneme, f32, f32)>) {
    let (r, s, _) = say_planned(text, tract, tongue);
    (r, s)
}

/// The readings, the segments (phoneme, start, end) and each segment's
/// larynx as planned (its gesture controls).
type Planned = (
    Vec<Reading>,
    Vec<(Phoneme, f32, f32)>,
    Vec<synthie_trombone::speech::Controls>,
);

/// As `say`, with each segment's larynx as planned.
fn say_planned(text: &str, tract: TractModel, tongue: f32) -> Planned {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    settings.tuning = Tuning::DEFAULT.with_tract(tract);
    settings.tuning.anatomy.tongue_model = tongue;
    for (name, value) in SETS.get().map_or(&[][..], Vec::as_slice) {
        assert!(settings.tuning.set(name, *value), "no tuning field {name}");
    }
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
    let planned = utterance.segments().iter().map(|s| s.gesture).collect();
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
            edge_area: g.edge_area,
            below: g.subglottal_pressure,
            above: g.supraglottal_pressure,
            fold_pitch: g.fold_pitch,
        });
    }
    (out, segments, planned)
}

/// Whether the glottal flow of the 10 ms from `a` is a voice: it swings by
/// more than 20 cm³/s and repeats itself (normalised autocorrelation over
/// 0.5) at some lag between 2.5 and 12.5 ms.
fn periodic(r: &[Reading], a: usize) -> bool {
    let win = (0.01 * SR) as usize;
    let (lo, hi) = ((0.0025 * SR) as usize, (0.0125 * SR) as usize);
    if a + win + hi > r.len() {
        return false;
    }
    // Centred on the frame: from half the longest lag before it.
    let a = a.saturating_sub(usize::midpoint(win, hi) - win / 2);
    if a + win + hi > r.len() {
        return false;
    }
    let x: Vec<f32> = r[a..a + win + hi].iter().map(|q| q.glottal).collect();
    let (min, max) = x[..win]
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    if max - min < 20.0 {
        return false;
    }
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let m0 = mean(&x[..win]);
    (lo..=hi).step_by(12).any(|lag| {
        let m1 = mean(&x[lag..lag + win]);
        let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
        for k in 0..win {
            let (p, q) = (x[k] - m0, x[k + lag] - m1);
            xy += p * q;
            xx += p * p;
            yy += q * q;
        }
        xy / (xx * yy).sqrt().max(1e-12) > 0.5
    })
}

/// Share of the span's 10 ms frames (hop 2.5 ms) that are voiced.
fn voiced_share(r: &[Reading], a: usize, b: usize) -> f32 {
    let hop = (0.0025 * SR) as usize;
    let (mut n, mut v) = (0, 0);
    let mut k = a;
    while k + hop <= b {
        n += 1;
        v += usize::from(periodic(r, k));
        k += hop;
    }
    if n == 0 { 0.0 } else { v as f32 / n as f32 }
}

/// `--final TEXT...`: through every affricate of each text and the vowel
/// before it (and the sound after it), segment by segment: its length, the
/// share of it the glottal flow is periodic, the folds' mean gap and the
/// part of it between the vibrating edges, the mean pressure across the
/// folds, the share the jet is turbulent, the loudest 10 ms of the 2 to 8
/// kHz hiss against the vowel before, and the larynx planned for it
/// (spread, slack, widening, creak).
fn finals(text: &str, tract: TractModel, tongue: f32) {
    let (r, segments, planned) = say_planned(text, tract, tongue);
    let audio: Vec<f32> = r.iter().map(|x| x.audio).collect();
    if let Some(path) = RAW.get()
        && tract == TractModel::Anatomical
    {
        // `--raw FILE`: the audio, 32-bit floats at 48 kHz, and the
        // segments' ends beside it (FILE.txt).
        let bytes: Vec<u8> = audio.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes).expect("--raw: cannot write");
        let mut ends = String::new();
        for s in &segments {
            writeln!(ends, "{} {:.4} {:.4}", s.0, s.1, s.2).expect("a string");
        }
        std::fs::write(format!("{path}.txt"), ends).expect("--raw: cannot write");
    }
    println!("{tract:?}: {text}");
    println!(
        "  seg    ms  voiced  vib%  gap cm²  edge cm²  ΔP cmH2O  jet on  hiss dB  spread  slack  expand  creak"
    );
    let is_v = |p: Phoneme| matches!(p.manner(), Manner::Vowel | Manner::Diphthong);
    for (k, s) in segments.iter().enumerate() {
        if s.0.manner() != Manner::Affricate || segments.get(k + 1).is_none_or(|n| n.0 != s.0) {
            continue;
        }
        let Some(v) = (0..k).rev().find(|&j| is_v(segments[j].0)) else {
            continue;
        };
        let (va, vb) = (
            (segments[v].1 * SR) as usize,
            ((segments[v].2 * SR) as usize).min(r.len()),
        );
        let reference = loudest(&audio[va..vb], 0.0, 8000.0);
        // As the mouth did it: the closure from the moment the mouth shuts
        // (within 30 ms of the segment) to the release, then the frication
        // while the jet is turbulent (gaps under 5 ms bridged).
        let shut_from = ((segments[k].1 - 0.03).max(0.0) * SR) as usize;
        let until = ((segments[k + 1].2 + 0.06) * SR) as usize;
        let until = until.min(r.len());
        if let Some(c0) = (shut_from..until).find(|&i| r[i].shut)
            && let Some(c1) = (c0..until).find(|&i| !r[i].shut)
        {
            let gap = (0.005 * SR) as usize;
            let mut f1 = c1;
            let mut quiet = 0;
            for (i, x) in r.iter().enumerate().take(until).skip(c1) {
                if x.jet > 0.0 {
                    f1 = i;
                    quiet = 0;
                } else {
                    quiet += 1;
                    if quiet > gap && f1 > c1 {
                        break;
                    }
                }
            }
            let hiss = if f1 > c1 + 480 {
                db(loudest(&audio[c1..f1], 2000.0, 8000.0) / reference)
            } else {
                f32::NAN
            };
            // Every 10 ms from 50 ms before the release to 150 ms after it:
            // shut (|) or open, voiced (v) or not, and the hiss of that frame
            // (2 to 8 kHz) in dB against the vowel's loudest 10 ms (all bands).
            let mut line = String::new();
            let step = (0.01 * SR) as usize;
            for j in 0..20 {
                let Some(at) = (c1 + j * step).checked_sub(5 * step) else {
                    continue;
                };
                if at + step > r.len() {
                    break;
                }
                let h = db(band(&audio[at..at + step], 2000.0, 8000.0) / reference);
                // The voice's swing: the glottal flow's range over the frame
                // while it repeats, cm³/s.
                let (lo, hi) = r[at..at + step]
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(l, h), q| {
                        (l.min(q.glottal), h.max(q.glottal))
                    });
                write!(
                    line,
                    " {}{}{:.0}/{:.0}",
                    if r[at + step / 2].shut { "|" } else { "" },
                    if periodic(&r, at) { "v" } else { "-" },
                    if periodic(&r, at) { hi - lo } else { 0.0 },
                    h
                )
                .expect("a string");
            }
            println!("  10 ms frames from -50 ms (voice swing cm³/s / hiss dB):{line}");
            println!(
                "  as made: closure {:.0} ms, {:.0}% voiced; frication {:.0} ms, {:.0}% voiced, hiss {:.1} dB; voice to {:.0} ms after the release",
                1000.0 * (c1 - c0) as f32 / SR,
                100.0 * voiced_share(&r, c0, c1),
                1000.0 * (f1 - c1) as f32 / SR,
                100.0 * voiced_share(&r, c1, f1),
                hiss,
                {
                    let hop = (0.0025 * SR) as usize;
                    let mut k = c1;
                    while k < r.len() && periodic(&r, k) {
                        k += hop;
                    }
                    1000.0 * (k - c1) as f32 / SR
                },
            );
        }
        let last = (k + 2).min(segments.len() - 1);
        for j in v..=last {
            let q = segments[j];
            let (a, b) = ((q.1 * SR) as usize, ((q.2 * SR) as usize).min(r.len()));
            if b <= a {
                continue;
            }
            let part = &r[a..b];
            let n = part.len() as f32;
            let mean = |f: fn(&Reading) -> f32| part.iter().map(f).sum::<f32>() / n;
            let label = if j == k {
                "clos"
            } else if j == k + 1 {
                "fric"
            } else {
                ""
            };
            let c = planned[j];
            println!(
                "  {:>3} {:4} {:4.0} {:6.0}% {:4.0}% {:8.4} {:9.4} {:9.2} {:6.0}% {:8.1} {:7.3} {:6.3} {:7.0} {:6.2}",
                q.0.to_string(),
                label,
                1000.0 * (q.2 - q.1),
                100.0 * voiced_share(&r, a, b),
                100.0 * part.iter().filter(|x| x.fold_pitch > 0.0).count() as f32 / n,
                mean(|x| x.glottal_area),
                mean(|x| x.edge_area),
                mean(|x| x.below - x.above),
                100.0 * part.iter().filter(|x| x.jet > 0.0).count() as f32 / n,
                if b > a + 480 {
                    db(loudest(&audio[a..b], 2000.0, 8000.0) / reference)
                } else {
                    f32::NAN
                },
                c.spread,
                c.slack,
                c.expansion,
                c.creak,
            );
        }
    }
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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut sets = Vec::new();
    while let Some(k) = args.iter().position(|a| a == "--set") {
        let pair = args.remove(k + 1);
        args.remove(k);
        let (name, value) = pair.split_once('=').expect("--set NAME=VALUE");
        sets.push((name.to_string(), value.parse::<f32>().expect("a number")));
    }
    SETS.set(sets).expect("once");
    if let Some(k) = args.iter().position(|a| a == "--raw") {
        let path = args.remove(k + 1);
        args.remove(k);
        RAW.set(path).expect("once");
    }
    let tongue = if args.iter().any(|a| a == "--modal") {
        1.0
    } else {
        0.0
    };
    if let Some(k) = args.iter().position(|a| a == "--final") {
        for text in args[k + 1..].iter().filter(|a| !a.starts_with("--")) {
            for tract in [TractModel::Anatomical, TractModel::Original] {
                if tract == TractModel::Original && args.iter().any(|a| a == "--anatomical") {
                    continue;
                }
                finals(text, tract, tongue);
            }
        }
        return;
    }
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
