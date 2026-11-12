//! How the hiss of fricatives, and the unwanted hiss of vowels, depend on
//! the size of the speaker and on lung pressure (physical engine).
//!
//! ```text
//! cargo run -p synthie-trombone --example hiss --release
//! ```
//!
//! For each body size and lung pressure it prints the level of sustained
//! /s sh f th/ relative to the vowel /a/ at the same setting, and for every
//! vowel the noise of the strongest jet in the mouth relative to the jet of
//! that setting's /s/. A fricative should keep its level against the vowel
//! whatever the voice; a vowel should not hiss at all.
//!
//! `--manual` instead places a constriction by hand around each fricative's
//! position (±2 cells, ±0.15 diameter) and reports how much hiss is left.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::format_push_string,
    clippy::needless_range_loop
)]

use synthie_trombone::{Constriction, Engine, Phoneme, SLOT_MAIN, Trombone, Tuning};

const SR: f32 = 48_000.0;
const SIZES: [f32; 5] = [0.6, 0.8, 1.0, 1.2, 1.4];
const LUNGS: [f32; 4] = [5.0, 8.0, 11.0, 14.0];
const FRICATIVES: [Phoneme; 4] = [Phoneme::S, Phoneme::Sh, Phoneme::F, Phoneme::Th];
const VOWELS: [Phoneme; 12] = [
    Phoneme::Iy,
    Phoneme::Ih,
    Phoneme::Eh,
    Phoneme::Ae,
    Phoneme::Ah,
    Phoneme::Aa,
    Phoneme::Ao,
    Phoneme::Uh,
    Phoneme::Uw,
    Phoneme::Ow,
    Phoneme::Er,
    Phoneme::Ax,
];

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// A voice of the given size and lung pressure, at the pitch a body of that
/// size would speak at.
fn voice(size: f32, lung: f32, tenseness: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_tract_length(size);
    let tuning = Tuning {
        lung_pressure: lung,
        ..Tuning::DEFAULT
    };
    t.set_tuning(&tuning);
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(120.0 * size.powf(-2.5));
    t.set_tenseness(tenseness);
    t
}

/// Level, mean jet noise (cmH2O), mean jet speed (cm/s) and the jet's cell
/// over `seconds`, after `lead` seconds to settle.
fn hold(t: &mut Trombone, lead: f32, seconds: f32) -> (f32, f32, f32, usize) {
    for _ in 0..(lead * SR) as usize {
        t.process();
    }
    let n = (seconds * SR) as usize;
    let mut x = Vec::with_capacity(n);
    let (mut jet, mut speed) = (0.0f64, 0.0f64);
    let mut cells = [0u32; 64];
    for _ in 0..n {
        x.push(t.process());
        let g = t.gauges();
        jet += f64::from(g.jet_noise);
        speed += f64::from(g.jet_speed);
        if g.jet_noise > 0.0 {
            cells[g.jet_cell.min(63)] += 1;
        }
    }
    let cell = (0..64).max_by_key(|&c| cells[c]).unwrap_or(0);
    (
        rms(&x),
        (jet / n as f64) as f32,
        (speed / n as f64) as f32,
        cell,
    )
}

struct Row {
    size: f32,
    lung: f32,
    vowel: f32,
    fric: [(f32, f32, f32); 4],
    /// Worst vowel: its jet noise and which vowel.
    worst: (f32, Phoneme, f32),
    hissing: Vec<(Phoneme, f32)>,
}

fn measure(size: f32, lung: f32, tenseness: f32) -> Row {
    let vowel = {
        let mut t = voice(size, lung, tenseness);
        t.articulate_phoneme(Phoneme::Aa);
        t.settle();
        t.note_on();
        hold(&mut t, 0.6, 0.4).0
    };
    let mut fric = [(0.0, 0.0, 0.0); 4];
    for (k, &p) in FRICATIVES.iter().enumerate() {
        let mut t = voice(size, lung, tenseness);
        t.articulate_phoneme(Phoneme::Aa);
        t.settle();
        t.articulate_phoneme(p);
        let (level, jet, speed, _) = hold(&mut t, 0.5, 0.3);
        fric[k] = (level, jet, speed);
    }
    let s_jet = fric[0].1.max(1e-9);
    let mut worst = (0.0f32, Phoneme::Aa, 0.0f32);
    let mut hissing = Vec::new();
    for &p in &VOWELS {
        let mut t = voice(size, lung, tenseness);
        t.articulate_phoneme(p);
        t.settle();
        t.note_on();
        let (_, jet, speed, _) = hold(&mut t, 0.5, 0.3);
        if jet > worst.0 {
            worst = (jet, p, speed);
        }
        if db(jet / s_jet) > -26.0 {
            hissing.push((p, db(jet / s_jet)));
        }
    }
    Row {
        size,
        lung,
        vowel,
        fric,
        worst,
        hissing,
    }
}

fn sweep(tenseness: f32) {
    let mut conditions = Vec::new();
    for &size in &SIZES {
        for &lung in &LUNGS {
            conditions.push((size, lung));
        }
    }
    let rows: Vec<Row> = std::thread::scope(|s| {
        let handles: Vec<_> = conditions
            .iter()
            .map(|&(size, lung)| s.spawn(move || measure(size, lung, tenseness)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let reference = rows
        .iter()
        .find(|r| (r.size - 1.0).abs() < 1e-6 && (r.lung - 8.0).abs() < 1e-6)
        .map(|r| r.fric.map(|f| db(f.0 / r.vowel)))
        .unwrap();
    println!("tenseness {tenseness}");
    println!(
        "size lung | vowel dB |  s re V (Δ)   sh re V (Δ)    f re V (Δ)   th re V (Δ) | s jet cm/s | worst vowel jet re s"
    );
    let mut worst_delta = 0.0f32;
    let mut hissers = 0;
    for r in &rows {
        let mut line = format!("{:4.1} {:4.0} | {:8.1} |", r.size, r.lung, db(r.vowel));
        for k in 0..4 {
            let level = db(r.fric[k].0 / r.vowel);
            let delta = level - reference[k];
            if k < 2 {
                worst_delta = worst_delta.max(delta.abs());
            }
            line += &format!(" {level:6.1} ({delta:+5.1})");
        }
        let s_jet = r.fric[0].1.max(1e-9);
        line += &format!(
            " | {:10.0} | {} {:6.1} dB at {:.0} cm/s",
            r.fric[0].2,
            r.worst.1,
            db(r.worst.0 / s_jet),
            r.worst.2
        );
        if !r.hissing.is_empty() {
            hissers += r.hissing.len();
            line += "  HISS:";
            for (p, level) in &r.hissing {
                line += &format!(" {p} {level:.0}");
            }
        }
        println!("{line}");
    }
    println!(
        "worst /s sh/ change re default: {worst_delta:.1} dB; vowels hissing (> -26 dB re /s/ jet): {hissers}"
    );
}

/// A constriction placed by hand near each fricative.
fn manual() {
    println!("manual constrictions, unvoiced, size 1, 8 cmH2O: level re the fricative itself, dB");
    for &p in &FRICATIVES {
        let c = p.articulation().constriction.unwrap();
        let reference = {
            let mut t = voice(1.0, 8.0, 0.6);
            t.articulate_phoneme(Phoneme::Aa);
            t.settle();
            t.articulate_phoneme(p);
            hold(&mut t, 0.5, 0.3).0
        };
        let mut line = format!("{p:>3} at {:4.1}, {:.2}:", c.index, c.diameter);
        for dd in [-0.15, -0.075, 0.0, 0.075, 0.15] {
            for di in [-2.0, 0.0, 2.0] {
                let mut t = voice(1.0, 8.0, 0.6);
                t.articulate_phoneme(Phoneme::Aa);
                t.settle();
                t.set_constriction(
                    SLOT_MAIN,
                    Some(Constriction::new(c.index + di, c.diameter + dd)),
                );
                let level = hold(&mut t, 0.5, 0.3).0;
                line += &format!(" {:6.1}", db(level / reference));
            }
            line += " |";
        }
        println!("{line}");
    }
    println!("(columns: diameter −0.15, −0.075, 0, +0.075, +0.15; within each, cell −2, 0, +2)");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--manual") {
        manual();
        return;
    }
    let tenseness = args
        .iter()
        .position(|a| a == "--tenseness")
        .and_then(|k| args.get(k + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6);
    sweep(tenseness);
}
