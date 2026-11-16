//! Do the self-oscillating folds close on every cycle? Holds a vowel,
//! cuts the area between the vibrating edges into cycles, and counts the
//! cycles in which the edges never meet: when they happen, how far apart,
//! and what the voice's H1–H2 (first minus second harmonic, the usual
//! measure of how abruptly the folds close) does over 50 ms frames.
//!
//! ```sh
//! cargo run -p synthie-trombone --example closure --release -- --vowel UW
//! cargo run -p synthie-trombone --example closure --release -- --vowel AA --preset bright --size 0.8 --lung 12
//! cargo run -p synthie-trombone --example closure --release -- --sweep
//! cargo run -p synthie-trombone --example closure --release -- --sweep --no-trim
//! cargo run -p synthie-trombone --example closure --release -- --sentences
//! ```
//!
//! `--sweep` holds all twelve vowels on every voiced preset, at body sizes
//! 0.6 to 1.4, three pitches around the voice's own and lung pressures of
//! 5 to 14 cmH2O, and reports which of them lose closure. `--gate` holds
//! only what the closure gate in `tests/physical_gates.rs` holds: every
//! preset as it is, and the default voice at every size.
//!
//! Parts of the voice can be switched off to find out where a behaviour
//! comes from: `--no-vibrato`, `--no-jitter`, `--no-trim` (the pitch
//! trim), `--no-regulation` (the hiss regulator), `--no-monitor` (the
//! stall watch that presses stalled folds together again), `--no-supra`
//! and `--no-sub` (the folds stop feeling the pressure sent back by the
//! tract above them or the trachea below).
//!
//! `--sentences` speaks the judge's sentences (or the files given with
//! `--material`) and reports how much of the voiced time the pulse source
//! sang, and around which sounds. `--fallback never` keeps the pulse out
//! entirely.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Fallback, GlottalSource, Persona, Phoneme, Preset, Trombone};

const SR: f32 = 48_000.0;
/// The twelve monophthongs held.
const VOWELS: [&str; 12] = [
    "IY", "IH", "EH", "AE", "AH", "AA", "AO", "UH", "UW", "OW", "ER", "AX",
];
const SIZES: [f32; 5] = [0.6, 0.8, 1.0, 1.2, 1.4];
const LUNGS: [f32; 4] = [5.0, 8.0, 11.0, 14.0];
/// Pitches tried, relative to the voice's own (about ±6 semitones).
const PITCHES: [f32; 3] = [0.7, 1.0, 1.4];
/// Seconds of onset left out of the count.
const ONSET: f32 = 0.2;
/// H1–H2 frame hop, seconds.
const FRAME: f32 = 0.05;
/// A trough deeper than this share of the typical peak counts as closed.
const CLOSED: f32 = 0.01;
/// Smallest swing, as a share of the typical peak, that counts as a cycle.
const SWING: f32 = 0.15;
const DEFAULT_MATERIAL: [&str; 2] = [
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/judge/material/sentence_tune.tsv"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/judge/material/sentence_heldout.tsv"
    ),
];

#[derive(Clone, Copy)]
struct Switches {
    vibrato: bool,
    jitter: bool,
    trim: bool,
    regulation: bool,
    monitor: bool,
    supra: bool,
    sub: bool,
    fallback: Fallback,
}

#[derive(Clone, Copy)]
struct Condition {
    vowel: Phoneme,
    preset: Preset,
    size: f32,
    /// Pitch relative to the voice's own, or an absolute pitch in Hz.
    pitch: Pitch,
    lung: f32,
}

#[derive(Clone, Copy)]
enum Pitch {
    Relative(f32),
    Hz(f32),
}

/// What one held vowel did.
struct Held {
    hz: f32,
    /// What the folds can sing, Hz, at the end of the render.
    range: (f32, f32),
    cycles: usize,
    /// Times (s) and periods (ms) of the cycles that did not close.
    open: Vec<(f32, f32)>,
    /// Seconds without any cycle at all.
    stalled: f32,
    /// Share of the voiced time the pulse was sounding.
    pulse: f32,
    /// H1–H2 of the radiated sound and of the glottal flow, dB, per frame.
    h1h2: Vec<f32>,
    h1h2_flow: Vec<f32>,
    /// Shallowest trough relative to the typical peak.
    worst_trough: f32,
}

fn voice(c: &Condition, sw: Switches) -> (Trombone, f32) {
    let persona = Persona {
        size: c.size,
        ..Persona::NEUTRAL
    };
    let settings = c
        .preset
        .settings()
        .with_engine(Engine::Physical)
        .with_persona(&persona);
    let hz = match c.pitch {
        Pitch::Relative(r) => settings.pitch * r,
        Pitch::Hz(hz) => hz,
    };
    let mut t = Trombone::new(SR);
    settings.apply(&mut t);
    t.set_fallback(sw.fallback);
    t.set_lung_pressure(c.lung);
    t.set_frequency(hz);
    if !sw.vibrato {
        t.set_vibrato(0.0, 0.0);
    }
    if !sw.jitter {
        t.set_jitter(0.0);
    }
    t.set_regulation(sw.regulation);
    t.probe_folds(sw.trim, sw.monitor, sw.supra, sw.sub);
    t.articulate_phoneme(c.vowel);
    t.settle();
    t.note_on();
    (t, hz)
}

/// Troughs of `a` (index, value), keeping only swings of at least `swing`.
fn troughs(a: &[f32], swing: f32) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let (mut lo, mut hi) = ((0usize, a[0]), (0usize, a[0]));
    let mut rising = false;
    for (i, &v) in a.iter().enumerate() {
        if rising {
            if v > hi.1 {
                hi = (i, v);
            }
            if hi.1 - v > swing {
                rising = false;
                lo = (i, v);
            }
        } else {
            if v < lo.1 {
                lo = (i, v);
            }
            if v - lo.1 > swing {
                out.push(lo);
                rising = true;
                hi = (i, v);
            }
        }
    }
    out
}

/// Magnitude of the strongest component of `x` within ±8 % of `hz`, under
/// a Hann window.
fn harmonic(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let mut best = 0.0f64;
    for step in -8..=8 {
        let f = f64::from(hz) * (1.0 + 0.01 * f64::from(step));
        let w = 2.0 * std::f64::consts::PI * f / f64::from(SR);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, &v) in x.iter().enumerate() {
            let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
            let v = f64::from(v) * hann;
            re += v * (w * k as f64).cos();
            im += v * (w * k as f64).sin();
        }
        best = best.max((re * re + im * im).sqrt());
    }
    best as f32
}

/// H1–H2 in dB over frames `FRAME` apart from `start`, each at least four
/// periods long, at the pitch the cycles show there.
fn h1h2(x: &[f32], start: usize, periods: &[(usize, f32)], fallback_hz: f32) -> Vec<f32> {
    let hop = (FRAME * SR) as usize;
    let mut out = Vec::new();
    let mut centre = start + hop / 2;
    while centre + hop / 2 <= x.len() {
        let near: Vec<f32> = periods
            .iter()
            .filter(|(at, _)| at.abs_diff(centre) < hop)
            .map(|p| p.1)
            .collect();
        let hz = if near.is_empty() {
            fallback_hz
        } else {
            let mut sorted = near.clone();
            sorted.sort_by(f32::total_cmp);
            SR / sorted[sorted.len() / 2]
        };
        let len = (hop as f32).max(4.0 * SR / hz) as usize;
        let from = centre.saturating_sub(len / 2);
        let to = (from + len).min(x.len());
        let frame = &x[from..to];
        let (h1, h2) = (harmonic(frame, hz), harmonic(frame, 2.0 * hz));
        out.push(20.0 * (h1.max(1e-12) / h2.max(1e-12)).log10());
        centre += hop;
    }
    out
}

fn hold(c: &Condition, sw: Switches, seconds: f32) -> Held {
    let (mut t, hz) = voice(c, sw);
    let n = (seconds * SR) as usize;
    let mut edge = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    let mut flow = Vec::with_capacity(n);
    let (mut voiced, mut pulse) = (0usize, 0usize);
    for _ in 0..n {
        out.push(t.process());
        let g = t.gauges();
        edge.push(g.edge_area);
        flow.push(g.glottal_flow);
        match g.sounding {
            Some(GlottalSource::Pulse) => {
                voiced += 1;
                pulse += 1;
            }
            Some(GlottalSource::Folds) => voiced += 1,
            None => {}
        }
    }
    let start = (ONSET * SR) as usize;
    let span = &edge[start..];
    let mut sorted = span.to_vec();
    sorted.sort_by(f32::total_cmp);
    let peak = sorted[(sorted.len() as f32 * 0.95) as usize].max(1e-6);
    // One trough per cycle: a notch in the open phase (closer than half a
    // period to the trough before it) is merged with it, the deeper kept.
    let mut found: Vec<(usize, f32)> = Vec::new();
    for t in troughs(span, SWING * peak) {
        match found.last_mut() {
            Some(last) if ((t.0 - last.0) as f32) < 0.5 * SR / hz => {
                if t.1 < last.1 {
                    *last = t;
                }
            }
            _ => found.push(t),
        }
    }
    let mut open = Vec::new();
    let mut periods = Vec::new();
    let mut stalled = 0.0;
    let mut worst = 0.0f32;
    for (k, &(at, depth)) in found.iter().enumerate() {
        let period = found.get(k + 1).map(|next| (next.0 - at) as f32);
        if let Some(p) = period {
            periods.push((start + at, p));
            if p > 3.0 * SR / hz {
                stalled += p / SR;
            }
        }
        // The last trough has no following cycle to judge it by.
        if k > 0 {
            worst = worst.max(depth / peak);
            if depth > CLOSED * peak {
                let time = (start + at) as f32 / SR;
                open.push((time, period.unwrap_or(0.0) / SR * 1000.0));
            }
        }
    }
    if found.is_empty() {
        stalled = span.len() as f32 / SR;
    }
    Held {
        hz,
        range: t.fold_range(),
        cycles: found.len().saturating_sub(1),
        open,
        stalled,
        pulse: pulse as f32 / voiced.max(1) as f32,
        h1h2: h1h2(&out, start, &periods, hz),
        h1h2_flow: h1h2(&flow, start, &periods, hz),
        worst_trough: worst,
    }
}

/// Largest change between neighbouring frames, and the range, dB.
fn roughness(contour: &[f32]) -> (f32, f32) {
    let jump = contour
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    let lo = contour.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = contour.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (jump, if contour.is_empty() { 0.0 } else { hi - lo })
}

fn describe(c: &Condition) -> String {
    let pitch = match c.pitch {
        Pitch::Relative(r) => format!("pitch x{r:.1}"),
        Pitch::Hz(hz) => format!("{hz:.0} Hz"),
    };
    format!(
        "{:<2} {:<8} size {:.1} {pitch} lung {:>4.1}",
        c.vowel.symbol(),
        c.preset.name(),
        c.size,
        c.lung
    )
}

fn detail(c: &Condition, sw: Switches, seconds: f32) {
    let h = hold(c, sw, seconds);
    println!(
        "{}: {:.1} Hz (the folds sing {:.0} to {:.0} Hz)",
        describe(c),
        h.hz,
        h.range.0,
        h.range.1
    );
    println!(
        "  {} cycles, {} did not close, {:.3} s stalled, pulse {:.0}% of the voiced time, shallowest trough {:.1}% of the peak",
        h.cycles,
        h.open.len(),
        h.stalled,
        100.0 * h.pulse,
        100.0 * h.worst_trough
    );
    if !h.open.is_empty() {
        let mut episodes: Vec<(f32, usize)> = Vec::new();
        for &(time, period) in &h.open {
            println!("    open cycle at {time:.4} s, period {period:.2} ms");
            match episodes.last_mut() {
                Some(e) if time - e.0 < 0.012 + e.1 as f32 * 1.5 / h.hz => e.1 += 1,
                _ => episodes.push((time, 1)),
            }
        }
        let gaps: Vec<String> = episodes
            .windows(2)
            .map(|w| format!("{:.0}", 1000.0 * (w[1].0 - w[0].0)))
            .collect();
        println!(
            "  {} episodes; ms between them: {}",
            episodes.len(),
            gaps.join(" ")
        );
    }
    let (jump, range) = roughness(&h.h1h2);
    let (fjump, frange) = roughness(&h.h1h2_flow);
    println!(
        "  H1-H2 (sound): largest step {jump:.2} dB, range {range:.2} dB; (flow): step {fjump:.2}, range {frange:.2}"
    );
    let line: Vec<String> = h.h1h2.iter().map(|v| format!("{v:.1}")).collect();
    println!(
        "  H1-H2 sound per {:.0} ms: {}",
        FRAME * 1000.0,
        line.join(" ")
    );
    let line: Vec<String> = h.h1h2_flow.iter().map(|v| format!("{v:.1}")).collect();
    println!(
        "  H1-H2 flow  per {:.0} ms: {}",
        FRAME * 1000.0,
        line.join(" ")
    );
}

/// Tally of open cycles by one dimension of the sweep.
#[derive(Default)]
struct Tally {
    conditions: usize,
    failing: usize,
    cycles: usize,
    open: usize,
    pulse: usize,
    worst_jump: f32,
    worst_range: f32,
}

/// The whole grid, or (`gate`) only what `tests/physical_gates.rs` holds:
/// each preset as it is, and the default voice at every size.
fn sweep(sw: Switches, seconds: f32, presets: &[Preset], gate: bool) {
    let mut conditions = Vec::new();
    for &preset in presets {
        for v in VOWELS {
            for size in SIZES {
                for pitch in PITCHES {
                    for lung in LUNGS {
                        let own = size == 1.0 || preset == Preset::Deep;
                        if gate && (pitch != 1.0 || lung != 8.0 || !own) {
                            continue;
                        }
                        conditions.push(Condition {
                            vowel: Phoneme::from_symbol(v).expect("a vowel"),
                            preset,
                            size,
                            pitch: Pitch::Relative(pitch),
                            lung,
                        });
                    }
                }
            }
        }
    }
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Held)>> = Mutex::new(Vec::new());
    let threads =
        std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(2).max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(c) = conditions.get(k) else { break };
                    let h = hold(c, sw, seconds);
                    results.lock().expect("results").push((k, h));
                }
            });
        }
    });
    let mut results = results.into_inner().expect("results");
    results.sort_by_key(|r| r.0);

    let mut by: BTreeMap<(usize, String), Tally> = BTreeMap::new();
    let (mut all_jump, mut all_range) = (Vec::new(), Vec::new());
    println!("Conditions that lost closure (open cycles / cycles) or used the pulse:");
    for (k, h) in &results {
        let c = &conditions[*k];
        let (jump, range) = roughness(&h.h1h2);
        let pulsed = h.pulse > 0.0;
        if !pulsed {
            all_jump.push(jump);
            all_range.push(range);
        }
        if !h.open.is_empty() || h.stalled > 0.0 || pulsed {
            println!(
                "  {}: {:>5.1} Hz {:>4}/{:<4} stalled {:.2} s pulse {:>3.0}% H1-H2 step {jump:.1} range {range:.1}",
                describe(c),
                h.hz,
                h.open.len(),
                h.cycles,
                h.stalled,
                100.0 * h.pulse
            );
        }
        let keys = [
            (0, c.vowel.symbol().to_string()),
            (1, c.preset.name().to_string()),
            (2, format!("size {:.1}", c.size)),
            (
                3,
                match c.pitch {
                    Pitch::Relative(r) => format!("pitch x{r:.1}"),
                    Pitch::Hz(hz) => format!("{hz:.0} Hz"),
                },
            ),
            (4, format!("lung {:>4.1}", c.lung)),
        ];
        for key in keys {
            let t = by.entry(key).or_default();
            t.conditions += 1;
            t.cycles += h.cycles;
            t.open += h.open.len();
            if !h.open.is_empty() || h.stalled > 0.0 {
                t.failing += 1;
            }
            if pulsed {
                t.pulse += 1;
            } else {
                t.worst_jump = t.worst_jump.max(jump);
                t.worst_range = t.worst_range.max(range);
            }
        }
    }
    println!();
    println!(
        "{:<14} {:>10} {:>12} {:>9} {:>11} {:>14} {:>14}",
        "", "conditions", "losing it", "open %", "pulse sang", "H1-H2 step", "H1-H2 range"
    );
    let mut last = usize::MAX;
    for ((dim, name), t) in &by {
        if *dim != last && last != usize::MAX {
            println!();
        }
        last = *dim;
        println!(
            "{name:<14} {:>10} {:>12} {:>8.2}% {:>11} {:>11.2} dB {:>11.2} dB",
            t.conditions,
            t.failing,
            100.0 * t.open as f32 / t.cycles.max(1) as f32,
            t.pulse,
            t.worst_jump,
            t.worst_range
        );
    }
    let total_open: usize = results.iter().map(|r| r.1.open.len()).sum();
    let total: usize = results.iter().map(|r| r.1.cycles).sum();
    let failing = results
        .iter()
        .filter(|r| !r.1.open.is_empty() || r.1.stalled > 0.0)
        .count();
    let pulsed = results.iter().filter(|r| r.1.pulse > 0.0).count();
    all_jump.sort_by(f32::total_cmp);
    all_range.sort_by(f32::total_cmp);
    let pct = |v: &[f32], q: f32| {
        v.get(((v.len() as f32 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    println!();
    println!(
        "All: {failing} of {} conditions lost closure or stalled; {total_open} of {total} cycles open ({:.3}%); the pulse sang in {pulsed}",
        results.len(),
        100.0 * total_open as f32 / total.max(1) as f32
    );
    println!(
        "H1-H2 largest frame-to-frame step on the folds: median {:.2}, 95th {:.2}, worst {:.2} dB; range median {:.2}, 95th {:.2}, worst {:.2} dB",
        pct(&all_jump, 0.5),
        pct(&all_jump, 0.95),
        pct(&all_jump, 1.0),
        pct(&all_range, 0.5),
        pct(&all_range, 0.95),
        pct(&all_range, 1.0)
    );
}

/// Speak each sentence of the given files and report the pulse's share.
fn sentences(sw: Switches, files: &[String], presets: &[Preset]) {
    let mut lines = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        for row in text.lines().skip(1) {
            let cols: Vec<&str> = row.split('\t').collect();
            if cols.len() >= 3 {
                lines.push((cols[0].to_string(), cols[2].to_string()));
            }
        }
    }
    for &preset in presets {
        let (mut voiced, mut pulse, mut inside) = (0usize, 0usize, 0usize);
        let mut contexts: BTreeMap<String, (usize, f32, f32)> = BTreeMap::new();
        for (id, text) in &lines {
            let settings = preset.settings().with_engine(Engine::Physical);
            let mut t = Trombone::new(SR);
            settings.apply(&mut t);
            t.set_fallback(sw.fallback);
            Speaker::prepare(&mut t);
            let utterance = Utterance::from_text(text, &settings.prosody());
            let segments = utterance.segments().to_vec();
            let mut speaker = Speaker::new();
            speaker.speak(utterance);
            let (mut v, mut p) = (0usize, 0usize);
            while speaker.is_speaking() {
                let index = speaker.current().map_or(0, |(i, _)| i);
                speaker.process(&mut t);
                match t.gauges().sounding {
                    Some(GlottalSource::Pulse) => {
                        v += 1;
                        p += 1;
                        let before = index
                            .checked_sub(1)
                            .map_or("#", |i| segments[i].phoneme.symbol());
                        let after = segments.get(index + 1).map_or("#", |s| s.phoneme.symbol());
                        let key =
                            format!("{before} [{}] {after}", segments[index].phoneme.symbol());
                        let hz = t.frequency();
                        let (lo, hi) = t.fold_range();
                        if (lo..=hi).contains(&hz) {
                            inside += 1;
                        }
                        let entry = contexts.entry(key).or_insert((0, f32::INFINITY, 0.0));
                        entry.0 += 1;
                        entry.1 = entry.1.min(hz);
                        entry.2 = entry.2.max(hz);
                    }
                    Some(GlottalSource::Folds) => v += 1,
                    None => {}
                }
            }
            if p > 0 {
                println!(
                    "  {preset} {id}: pulse {:.1}% of {:.2} s voiced",
                    100.0 * p as f32 / v.max(1) as f32,
                    v as f32 / SR
                );
            }
            voiced += v;
            pulse += p;
        }
        println!(
            "{preset}: pulse {:.2}% of {:.1} s voiced ({:.3} s), {:.3} s of it on notes inside the folds' range",
            100.0 * pulse as f32 / voiced.max(1) as f32,
            voiced as f32 / SR,
            pulse as f32 / SR,
            inside as f32 / SR
        );
        let mut ranked: Vec<_> = contexts.into_iter().collect();
        ranked.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        for (context, (n, lo, hi)) in ranked.iter().take(12) {
            println!(
                "    {context}: {:.1} ms at {lo:.0} to {hi:.0} Hz",
                1000.0 * *n as f32 / SR
            );
        }
    }
}

fn main() {
    let mut sw = Switches {
        vibrato: true,
        jitter: true,
        trim: true,
        regulation: true,
        monitor: true,
        supra: true,
        sub: true,
        fallback: Fallback::Auto,
    };
    let mut c = Condition {
        vowel: Phoneme::Uw,
        preset: Preset::Deep,
        size: 1.0,
        pitch: Pitch::Relative(1.0),
        lung: 8.0,
    };
    let mut seconds = 2.0 + ONSET;
    let (mut do_sweep, mut do_sentences, mut gate) = (false, false, false);
    let mut presets: Vec<Preset> = Preset::ALL
        .into_iter()
        .filter(|&p| p != Preset::Whisper)
        .collect();
    let mut chosen_preset = false;
    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--vowel" => {
                let v = value();
                c.vowel = Phoneme::from_symbol(&v).unwrap_or_else(|| panic!("unknown vowel {v}"));
            }
            "--preset" => {
                let v = value();
                c.preset = Preset::from_name(&v).unwrap_or_else(|| panic!("unknown preset {v}"));
                if !chosen_preset {
                    presets.clear();
                    chosen_preset = true;
                }
                presets.push(c.preset);
            }
            "--size" => c.size = value().parse().expect("a size"),
            "--pitch" => c.pitch = Pitch::Hz(value().parse().expect("Hz")),
            "--lung" => c.lung = value().parse().expect("cmH2O"),
            "--seconds" => seconds = value().parse::<f32>().expect("seconds") + ONSET,
            "--material" => files.push(value()),
            "--fallback" => {
                sw.fallback = match value().as_str() {
                    "auto" => Fallback::Auto,
                    "never" => Fallback::Never,
                    other => panic!("unknown fallback {other}; try auto or never"),
                };
            }
            "--sweep" => do_sweep = true,
            "--gate" => {
                do_sweep = true;
                gate = true;
            }
            "--sentences" => do_sentences = true,
            "--no-vibrato" => sw.vibrato = false,
            "--no-jitter" => sw.jitter = false,
            "--no-trim" => sw.trim = false,
            "--no-regulation" => sw.regulation = false,
            "--no-monitor" => sw.monitor = false,
            "--no-supra" => sw.supra = false,
            "--no-sub" => sw.sub = false,
            other => panic!("unknown argument {other}"),
        }
    }
    if do_sentences {
        if files.is_empty() {
            files = DEFAULT_MATERIAL.iter().map(|s| (*s).to_string()).collect();
        }
        sentences(sw, &files, &presets);
    } else if do_sweep {
        sweep(sw, seconds, &presets, gate);
    } else {
        detail(&c, sw, seconds);
    }
}
