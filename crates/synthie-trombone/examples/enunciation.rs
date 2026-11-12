//! How clearly does each engine speak? Scores three sentences phoneme by
//! phoneme: do the articulators get where they are going, do vowels reach
//! their formants, is the voice there for the whole vowel, do fricatives
//! hiss, does the voice come back after a voiced stop.
//!
//! ```sh
//! cargo run -p synthie-trombone --example enunciation --release
//! cargo run -p synthie-trombone --example enunciation --release -- --segments   # every phoneme
//! ```

#![allow(
    clippy::format_push_string,
    clippy::manual_midpoint,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use synthie_trombone::speech::{Segment, Speaker, Utterance};
use synthie_trombone::{Articulation, Engine, Manner, N, Phoneme, Preset, Trombone};

const SR: f32 = 48_000.0;
/// Samples per analysis window (5 ms).
const WINDOW: usize = 240;

const SENTENCES: &[&str] = &[
    "Hello there, how are you today?",
    "She sells sea shells by the sea shore.",
    "The big dog barked at the old man.",
];

/// The shape and formants an articulation settles to if given all the time
/// in the world.
struct Target {
    shape: [f32; N],
    f1: f32,
    f2: f32,
}

fn target(engine: Engine, a: &Articulation) -> Target {
    let mut t = Trombone::new(SR);
    t.set_engine(engine);
    t.articulate(a);
    t.settle();
    let f = spectrum::formants(&t, 3500.0);
    Target {
        shape: *t.diameters(),
        f1: f.first().map_or(0.0, |r| r.hz),
        f2: f.get(1).map_or(0.0, |r| r.hz),
    }
}

fn distance(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// Everything above 3 kHz (second-order high-pass).
fn high_band(x: &[f32]) -> Vec<f32> {
    let w = 2.0 * std::f64::consts::PI * 3000.0 / f64::from(SR);
    let alpha = w.sin() / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1) = ((1.0 + w.cos()) / 2.0 / a0, -(1.0 + w.cos()) / a0);
    let (a1, a2) = (-2.0 * w.cos() / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            let y = b0 * v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = v;
            y2 = y1;
            y1 = y;
            y as f32
        })
        .collect()
}

/// What happened during one segment.
struct Spoken {
    segment: Segment,
    /// First and one-past-last output sample.
    range: (usize, usize),
    /// Tract shape at the middle of the segment.
    mid_shape: [f32; N],
    /// F1 and F2 of the tract as it was at the middle.
    mid_formants: (f32, f32),
    /// Share of 5 ms windows in which the folds were vibrating.
    voiced_share: f32,
    /// Whether the folds were vibrating in each 5 ms window.
    voicing: Vec<bool>,
    /// Share of 5 ms windows in which the tract was shut somewhere.
    sealed_share: f32,
    sealed: usize,
}

fn speak(engine: Engine, text: &str) -> (Vec<Spoken>, Vec<f32>) {
    let settings = Preset::Deep.settings().with_engine(engine);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let segments: Vec<Segment> = utterance.segments().to_vec();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);

    let mut out = Vec::new();
    let mut spoken: Vec<Spoken> = Vec::new();
    let mut current = usize::MAX;
    let (mut lo, mut hi, mut in_window) = (f32::INFINITY, 0.0f32, 0usize);
    let mut mid_done = false;
    while speaker.is_speaking() {
        let index = speaker.current().map_or(usize::MAX, |(i, _)| i);
        if index != current {
            if let Some(last) = spoken.last_mut() {
                last.range.1 = out.len();
            }
            current = index;
            mid_done = false;
            (lo, hi, in_window) = (f32::INFINITY, 0.0, 0);
            spoken.push(Spoken {
                segment: segments[index],
                range: (out.len(), out.len()),
                mid_shape: [0.0; N],
                mid_formants: (0.0, 0.0),
                voiced_share: 0.0,
                voicing: Vec::new(),
                sealed_share: 0.0,
                sealed: 0,
            });
        }
        out.push(speaker.process(&mut voice));
        let this = spoken.last_mut().expect("a segment is open");
        let done = out.len() - this.range.0;
        let length = (this.segment.duration * SR) as usize;

        // Are the folds vibrating? The physical engine shows it in the gap
        // between them; the classic one only has its voicing envelope.
        let vibrating = match engine {
            Engine::Physical => {
                let area = voice.gauges().glottal_area;
                lo = lo.min(area);
                hi = hi.max(area);
                hi - lo > 0.02
            }
            Engine::Classic => {
                lo = lo.min(voice.intensity());
                lo > 0.5
            }
        };
        in_window += 1;
        if in_window == WINDOW {
            if voice.diameters()[8..].iter().any(|&d| d <= 0.0) {
                this.sealed += 1;
            }
            this.voicing.push(vibrating);
            (lo, hi, in_window) = (f32::INFINITY, 0.0, 0);
        }

        if !mid_done && done >= length / 2 {
            mid_done = true;
            this.mid_shape = *voice.diameters();
            if matches!(
                this.segment.phoneme.manner(),
                Manner::Vowel | Manner::Diphthong
            ) {
                let f = spectrum::formants(&voice, 3500.0);
                this.mid_formants = (
                    f.first().map_or(0.0, |r| r.hz),
                    f.get(1).map_or(0.0, |r| r.hz),
                );
            }
        }
    }
    if let Some(last) = spoken.last_mut() {
        last.range.1 = out.len();
    }
    for s in &mut spoken {
        let n = s.voicing.len().max(1) as f32;
        s.voiced_share = s.voicing.iter().filter(|&&v| v).count() as f32 / n;
        s.sealed_share = s.sealed as f32 / n;
    }
    // Let the tail ring out so later analysis sees the release.
    for _ in 0..(0.2 * SR) as usize {
        out.push(voice.process());
    }
    (spoken, out)
}

#[derive(Default)]
struct Tally {
    reach_mid: Vec<f32>,
    f1_error: Vec<f32>,
    f2_error: Vec<f32>,
    vowel_voiced: Vec<f32>,
    sibilant_share: Vec<f32>,
    sibilant_level: Vec<f32>,
    faint_level: Vec<f32>,
    stop_voiced: Vec<f32>,
    voice_return: Vec<f32>,
    stressed_level: Vec<f32>,
    unstressed_level: Vec<f32>,
    closure_share: Vec<f32>,
    weak_vowels: usize,
}

fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        f32::NAN
    } else {
        x.iter().sum::<f32>() / x.len() as f32
    }
}

/// The value a tenth of the entries exceed.
fn tenth(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(f32::total_cmp);
    v[(v.len() * 9 / 10).min(v.len() - 1)]
}

fn median(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let every_segment = std::env::args().any(|a| a == "--segments");
    for engine in [Engine::Classic, Engine::Physical] {
        let mut tally = Tally::default();
        for text in SENTENCES {
            let (spoken, out) = speak(engine, text);
            let hiss = high_band(&out);
            let vowel_level = {
                let levels: Vec<f32> = spoken
                    .iter()
                    .filter(|s| s.segment.phoneme.manner() == Manner::Vowel)
                    .map(|s| rms(&out[s.range.0..s.range.1]))
                    .collect();
                median(&levels)
            };
            if every_segment {
                println!("{engine:?}: {text}");
            }
            let mut previous: Option<Target> = None;
            for (k, s) in spoken.iter().enumerate() {
                let p = s.segment.phoneme;
                let goal = target(engine, &s.segment.articulation);
                let level = rms(&out[s.range.0..s.range.1]);
                let mut note = String::new();

                // Reach: how much of the way from the last target to this one.
                if let Some(from) = &previous {
                    let whole = distance(&from.shape, &goal.shape);
                    if whole > 0.3 {
                        let mid = 1.0 - distance(&s.mid_shape, &goal.shape) / whole;
                        tally.reach_mid.push(mid.clamp(0.0, 1.0));
                        note += &format!(" reach {:>3.0}%", 100.0 * mid);
                    }
                }

                match p.manner() {
                    Manner::Vowel | Manner::Diphthong => {
                        if goal.f1 > 0.0 && s.mid_formants.0 > 0.0 && p.manner() == Manner::Vowel {
                            let e1 = (s.mid_formants.0 / goal.f1 - 1.0).abs();
                            let e2 = (s.mid_formants.1 / goal.f2 - 1.0).abs();
                            tally.f1_error.push(e1);
                            tally.f2_error.push(e2);
                            note += &format!(
                                " F1 {:>4.0} (goal {:>4.0}) F2 {:>4.0} (goal {:>4.0})",
                                s.mid_formants.0, goal.f1, s.mid_formants.1, goal.f2
                            );
                        }
                        tally.vowel_voiced.push(s.voiced_share);
                        if s.voiced_share < 0.6 {
                            tally.weak_vowels += 1;
                        }
                        note += &format!(" voiced {:>3.0}%", 100.0 * s.voiced_share);
                        if p.manner() == Manner::Vowel {
                            if s.segment.duration > 0.1 {
                                tally.stressed_level.push(db(level / vowel_level));
                            } else {
                                tally.unstressed_level.push(db(level / vowel_level));
                            }
                        }
                    }
                    Manner::Fricative if !s.segment.voiced => {
                        let windows: Vec<f32> =
                            hiss[s.range.0..s.range.1].chunks(WINDOW).map(rms).collect();
                        let audible = 0.025 * vowel_level;
                        let share = windows.iter().filter(|&&w| w > audible).count() as f32
                            / windows.len().max(1) as f32;
                        let hiss_level = db(rms(&hiss[s.range.0..s.range.1]) / vowel_level);
                        if matches!(p, Phoneme::S | Phoneme::Sh) {
                            tally.sibilant_share.push(share);
                            tally.sibilant_level.push(hiss_level);
                        } else {
                            tally.faint_level.push(hiss_level);
                        }
                        note += &format!(
                            " hiss {:>3.0}% of it, {hiss_level:>5.1} dB re vowel",
                            100.0 * share
                        );
                    }
                    Manner::Stop | Manner::Nasal
                        if s.segment
                            .articulation
                            .constriction
                            .is_some_and(|c| c.diameter <= 0.3)
                            && !(p.manner() == Manner::Stop && s.segment.voiced) =>
                    {
                        tally.closure_share.push(s.sealed_share);
                        note += &format!(" shut {:>3.0}% of it", 100.0 * s.sealed_share);
                    }
                    Manner::Stop if s.segment.voiced => {
                        tally.closure_share.push(s.sealed_share);
                        note += &format!(" shut {:>3.0}% of it,", 100.0 * s.sealed_share);
                        tally.stop_voiced.push(s.voiced_share);
                        // How long after the release until the folds swing.
                        let later: Vec<bool> = spoken[k + 1..]
                            .iter()
                            .flat_map(|n| n.voicing.iter().copied())
                            .collect();
                        let back = later
                            .iter()
                            .position(|&v| v)
                            .map_or(f32::NAN, |w| w as f32 * 5.0);
                        if back.is_finite() {
                            tally.voice_return.push(back);
                        }
                        note += &format!(
                            " voiced {:>3.0}% of the closure, voice back after {back:.0} ms",
                            100.0 * s.voiced_share
                        );
                    }
                    _ => {}
                }
                if every_segment {
                    println!(
                        "  {:<3} {:>4.0} ms {:>6.1} dB{note}",
                        p.symbol(),
                        1000.0 * s.segment.duration,
                        db(level / vowel_level)
                    );
                }
                previous = Some(goal);
            }
        }
        println!(
            "{engine:?}: shape {:.0}% reached by mid-segment | closures shut {:.0}% of their segment | vowels: F1 off {:.1}%, F2 off {:.1}% mid-vowel (worst tenth {:.0}%, {:.0}%), voiced {:.0}% of the time, {} of {} mostly unvoiced | /s sh/: hiss for {:.0}% of the segment at {:.1} dB re vowel; /f th/ {:.1} dB | voiced stops: voiced {:.0}% of the closure, voice back after {:.0} ms | stressed vowels {:+.1} dB over unstressed",
            100.0 * mean(&tally.reach_mid),
            100.0 * mean(&tally.closure_share),
            100.0 * median(&tally.f1_error),
            100.0 * median(&tally.f2_error),
            100.0 * tenth(&tally.f1_error),
            100.0 * tenth(&tally.f2_error),
            100.0 * mean(&tally.vowel_voiced),
            tally.weak_vowels,
            tally.vowel_voiced.len(),
            100.0 * mean(&tally.sibilant_share),
            mean(&tally.sibilant_level),
            mean(&tally.faint_level),
            100.0 * mean(&tally.stop_voiced),
            mean(&tally.voice_return),
            mean(&tally.stressed_level) - mean(&tally.unstressed_level),
        );
    }
}
