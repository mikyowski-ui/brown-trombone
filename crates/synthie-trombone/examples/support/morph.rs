//! Measurements of a voice morphing from one body into another (hybrid
//! roadmap, stage 1), shared by the `morph` example and
//! `tests/morph_gate.rs`: a held vowel recorded sample by sample while
//! the body changes, and what a click, a stall of the folds and a formant
//! are in it.

#![allow(
    clippy::many_single_char_names,
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::speech::Speaker;
use synthie_trombone::{Engine, MorphTarget, Phoneme, Trombone, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;

/// A held vowel, one entry per output sample.
#[derive(Default)]
pub struct Rec {
    pub out: Vec<f32>,
    /// Glottal flow, cm³/s.
    pub flow: Vec<f32>,
    /// The folds' measured pitch, Hz (0 while they do not vibrate).
    pub pitch: Vec<f32>,
    /// The tract's relative length.
    pub length: Vec<f32>,
}

/// A voice of `settings` on the physical engine holding the vowel `p` at
/// `hz`, its voice started and steady.
pub fn held(settings: &VoiceSettings, p: Phoneme, hz: f32) -> Trombone {
    let mut v = Trombone::new(SR);
    settings.with_engine(Engine::Physical).apply(&mut v);
    Speaker::prepare(&mut v);
    v.articulate_phoneme(p);
    v.settle();
    v.set_frequency(hz);
    v.note_on();
    run(&mut v, 0.5, &mut Rec::default());
    v
}

/// Render `seconds` of `v` into `rec`.
pub fn run(v: &mut Trombone, seconds: f32, rec: &mut Rec) {
    for _ in 0..(seconds * SR) as usize {
        let x = v.process();
        let g = v.gauges();
        rec.out.push(x);
        rec.flow.push(g.glottal_flow);
        rec.pitch.push(g.fold_pitch);
        rec.length.push(v.tract_length());
    }
}

/// Glide `v` to `target` and record until it is there and `after` seconds
/// more.
pub fn glide(v: &mut Trombone, target: MorphTarget, after: f32, rec: &mut Rec) {
    v.set_morph(target);
    let mut left = (20.0 * SR) as usize;
    while v.morph() != v.morph_target() && left > 0 {
        run(v, 0.01, rec);
        left -= (0.01 * SR) as usize;
    }
    run(v, after, rec);
}

pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// The roughest 10 ms of `x`: the largest sample-to-sample step of any
/// 10 ms against that 10 ms's own level (a click is a step far above what
/// the sound around it makes), among those within 30 dB of the loudest.
pub fn click_ratio(x: &[f32]) -> f32 {
    let n = (0.01 * SR) as usize;
    let loudest = x.chunks(n).map(rms).fold(0.0, f32::max);
    x.chunks(n)
        .filter(|c| c.len() == n && db(rms(c)) > db(loudest) - 30.0)
        .map(|c| {
            c.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0, f32::max)
                / rms(c).max(1e-9)
        })
        .fold(0.0, f32::max)
}

/// The weakest stretch of the folds' vibration in `flow` at about `hz`:
/// the glottal flow's swing over each two periods against the median swing
/// of the fifth of a second around it. A stall is a stretch near 0; folds
/// that go on vibrating stay near 1.
pub fn weakest_swing(flow: &[f32], hz: f32) -> f32 {
    let n = (2.0 * SR / hz) as usize;
    let swing: Vec<f32> = flow
        .chunks(n)
        .filter(|c| c.len() == n)
        .map(|c| {
            c.iter().fold(f32::MIN, |m, &v| m.max(v)) - c.iter().fold(f32::MAX, |m, &v| m.min(v))
        })
        .collect();
    let reach = ((0.1 * SR) as usize / n).max(2);
    (0..swing.len())
        .map(|k| {
            let mut near: Vec<f32> =
                swing[k.saturating_sub(reach)..(k + reach + 1).min(swing.len())].to_vec();
            near.sort_by(f32::total_cmp);
            swing[k] / near[near.len() / 2].max(1e-9)
        })
        .fold(f32::MAX, f32::min)
}

/// Magnitude of `x` (Hann window) at `hz`.
pub fn tone(x: &[f32], hz: f32) -> f32 {
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

/// The largest magnitude of `x` within 4 % of `hz`.
pub fn peak_near(x: &[f32], hz: f32) -> f32 {
    (-8..=8)
        .map(|k| tone(x, hz * (1.0 + 0.005 * k as f32)))
        .fold(0.0, f32::max)
}

/// First harmonic over the second, dB, of a waveform at pitch `hz`.
pub fn h1_h2(x: &[f32], hz: f32) -> f32 {
    db(peak_near(x, hz)) - db(peak_near(x, 2.0 * hz))
}

/// Level of the harmonics of `hz` in `x` between `lo` and `hi` Hz, dB.
pub fn band(x: &[f32], hz: f32, lo: f32, hi: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut k = (lo / hz).ceil().max(1.0);
    while k * hz <= hi {
        let m = peak_near(x, k * hz);
        sum += m * m;
        k += 1.0;
    }
    db(sum.sqrt())
}

/// Share of a period the glottis is shut: flow under a tenth of its swing.
pub fn closed_share(flow: &[f32]) -> f32 {
    let hi = flow.iter().fold(f32::MIN, |m, &v| m.max(v));
    let lo = flow.iter().fold(f32::MAX, |m, &v| m.min(v));
    flow.iter().filter(|&&v| v < lo + 0.1 * (hi - lo)).count() as f32 / flow.len().max(1) as f32
}
