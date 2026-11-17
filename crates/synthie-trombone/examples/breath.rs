//! The living voice's breath and creak (wave 3, stage 3A): where the
//! self-oscillating folds alternate strong and weak pulses, and what an
//! inhalation sounds like.
//!
//! ```sh
//! cargo run -p synthie-trombone --example breath --release -- --fry [iy|ax]   # where the folds creak
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::{Engine, GlottalSource, Phoneme, Trombone};

const SR: f32 = 48_000.0;

fn voice(vowel: Phoneme, hz: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_glottal_source(GlottalSource::Folds);
    t.articulate_phoneme(vowel);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(hz);
    t.set_tenseness(0.65);
    t.note_on();
    t
}

fn run(t: &mut Trombone, seconds: f32) -> (Vec<f32>, Vec<f32>) {
    let n = (seconds * SR) as usize;
    let mut out = Vec::with_capacity(n);
    let mut flows = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(t.process());
        flows.push(t.gauges().glottal_flow);
    }
    (out, flows)
}

/// The pulses of a stretch of glottal flow: (sample of each peak, its
/// height above the floor). A pulse is a run above a quarter of the swing.
fn pulses(flows: &[f32]) -> Vec<(usize, f32)> {
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let swing = peak - floor;
    let mut out = Vec::new();
    if swing <= 1.0 {
        return out;
    }
    let lo = floor + 0.15 * swing;
    let mut k = 0;
    while k < flows.len() {
        if flows[k] > lo {
            let mut best = (k, flows[k]);
            while k < flows.len() && flows[k] > lo {
                if flows[k] > best.1 {
                    best = (k, flows[k]);
                }
                k += 1;
            }
            out.push((best.0, best.1 - floor));
        }
        k += 1;
    }
    out
}

/// Magnitude of x (Hann window) at hz.
fn tone(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(SR);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (k, &v) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
        re += f64::from(v) * h * (w * k as f64).cos();
        im += f64::from(v) * h * (w * k as f64).sin();
    }
    (re * re + im * im).sqrt() as f32
}

fn peak_near(x: &[f32], hz: f32) -> f32 {
    (-6..=6)
        .map(|c| tone(x, hz * (1.0 + c as f32 * 0.01)))
        .fold(0.0f32, f32::max)
}

/// What a stretch of voice is: (pulse rate Hz, alternation of pulse
/// heights, alternation of pulse spacing, energy at half the pitch against
/// the pitch dB).
pub fn creak(out: &[f32], flows: &[f32]) -> (f32, f32, f32, f32) {
    let p = pulses(flows);
    if p.len() < 6 {
        return (0.0, 0.0, 0.0, -99.0);
    }
    let spacing: Vec<f32> = p.windows(2).map(|w| (w[1].0 - w[0].0) as f32).collect();
    let mean_space = spacing.iter().sum::<f32>() / spacing.len() as f32;
    let rate = SR / mean_space;
    let mean_h = p.iter().map(|q| q.1).sum::<f32>() / p.len() as f32;
    let alt_h = p.windows(2).map(|w| (w[1].1 - w[0].1).abs()).sum::<f32>()
        / (p.len() - 1) as f32
        / mean_h;
    let alt_t = spacing.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>()
        / (spacing.len() - 1) as f32
        / mean_space;
    let sub = peak_near(out, 0.5 * rate);
    let f0 = peak_near(out, rate);
    (rate, alt_h, alt_t, 20.0 * (sub.max(1e-12) / f0.max(1e-12)).log10())
}

fn fine_survey(args: &[String]) {
    let vowel = if args.iter().any(|a| a == "iy") { Phoneme::Iy } else if args.iter().any(|a| a == "ax") { Phoneme::Ax } else { Phoneme::Aa };
    println!("Weakly coupled cover pieces, pressed folds, low pressure, on {vowel:?}");
    println!("  coupl tension gap    lung   rate   alt-h  alt-t  sub dB   heights");
    for coupling in [0.2f32, 0.12, 0.08, 0.05, 0.03] {
        for tension in [0.45f32, 0.6, 0.8, 1.0] {
            for gap in [-0.01f32, -0.005, -0.002] {
                for lung in [3.0f32, 4.0, 6.0] {
                    let mut t = voice(vowel, 60.0);
                    t.probe_folds(false, false, true, true);
                    t.override_folds(Some(tension), Some(gap));
                    t.override_fold_coupling(Some(coupling));
                    t.set_lung_pressure(lung);
                    run(&mut t, 0.5);
                    let (out, flows) = run(&mut t, 0.4);
                    let (rate, ah, at, sub) = creak(&out, &flows);
                    let h: Vec<String> = pulses(&flows).iter().take(8).map(|p| format!("{:.0}", p.1)).collect();
                    println!(
                        "  {coupling:.2}  {tension:>5.2} {gap:>6.3} {lung:>5.1}  {rate:>6.1}  {ah:>5.2}  {at:>5.2}  {sub:>6.1}   {}", h.join(" ")
                    );
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--fry") {
        fine_survey(&args);
    }
}
