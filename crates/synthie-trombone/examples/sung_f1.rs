//! Where is a physical vowel's F1 while it is sung? Three readings: the
//! impulse response with the glottis shut (as the babble map and the vowel
//! targets use), the response with the glottis held open at its mean
//! phonating area around its steady flow, and the peak of the harmonics
//! of the voice singing for a second at 70, 110 and 220 Hz (a parabola
//! through the strongest harmonic near F1 and its neighbours; only the
//! lower pitches have harmonics dense enough to say).
//!
//! ```sh
//! cargo run -p synthie-trombone --example sung_f1 --release -- IY
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::format_push_string
)]
use synthie_trombone::{Engine, Phoneme, Trombone};
const SR: f32 = 48_000.0;
fn dft(x: &[f32], hz: f32, rate: f32) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(rate);
    let (rr, ri) = (w.cos(), -w.sin());
    let (mut pr, mut pi) = (1.0f64, 0.0f64);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for &v in x {
        let v = f64::from(v);
        re += v * pr;
        im += v * pi;
        let t = pr * rr - pi * ri;
        pi = pr * ri + pi * rr;
        pr = t;
    }
    re * re + im * im
}
fn first_peak(ir: &[f32], rate: f32, lo: f32, hi: f32) -> f32 {
    let mut prev = 0.0;
    let mut rising = false;
    let mut f = lo;
    while f < hi {
        let m = dft(ir, f, rate);
        if m < prev && rising {
            return f - 1.0;
        }
        rising = m >= prev;
        prev = m;
        f += 1.0;
    }
    0.0
}
fn biggest_peak(ir: &[f32], rate: f32, lo: f32, hi: f32) -> f32 {
    let mags: Vec<(f32, f64)> = {
        let mut v = Vec::new();
        let mut f = lo;
        while f < hi {
            v.push((f, dft(ir, f, rate)));
            f += 1.0;
        }
        v
    };
    let mut best = (0.0f32, 0.0f64);
    for k in 1..mags.len() - 1 {
        if mags[k].1 > mags[k - 1].1 && mags[k].1 >= mags[k + 1].1 && mags[k].1 > best.1 {
            best = mags[k];
        }
    }
    best.0
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = Phoneme::from_symbol(&a[1]).unwrap();
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.articulate_phoneme(p);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    let rate = t.tick_rate();
    let mut ir = vec![0.0f32; (0.25 * rate) as usize];
    t.impulse_response(&mut ir);
    let fa = first_peak(&ir, rate, 150.0, 1200.0);
    let mut line = format!("{:<2} (a) closed {fa:4.0}", a[1]);
    for hz in [70.0f32, 110.0, 220.0] {
        let mut v = t.clone();
        v.set_frequency(hz);
        v.note_on();
        for _ in 0..24000 {
            v.process();
        }
        let n = 48000;
        let mut x = Vec::with_capacity(n);
        let mut area = 0.0f64;
        let mut lung = 0.0f64;
        for _ in 0..n {
            x.push(v.process());
            let g = v.gauges();
            area += f64::from(g.glottal_area);
            lung += f64::from(g.subglottal_pressure);
        }
        let f0 = {
            let mut f = v.gauges().fold_pitch;
            if f <= 0.0 {
                f = hz;
            }
            f
        };
        let mean = (area / n as f64) as f32;
        let ps = (lung / n as f64) as f32;
        let mut irb = vec![0.0f32; (0.25 * rate) as usize];
        v.impulse_response_open(mean, ps, &mut irb);
        let fb = biggest_peak(&irb, rate, 0.6 * fa, 1.6 * fa);
        // harmonic amplitudes, dB; strongest harmonic in [0.6, 1.6] x fa, parabola
        let hs: Vec<(f32, f64)> = (1..60)
            .map(|k| k as f32 * f0)
            .filter(|&f| f > 0.6 * fa && f < 1.6 * fa)
            .map(|f| (f, 10.0 * dft(&x, f, SR).log10()))
            .collect();
        let mut best = 0;
        for i in 0..hs.len() {
            if hs[i].1 > hs[best].1 {
                best = i;
            }
        }
        let fc = if best > 0 && best + 1 < hs.len() {
            let (l, m, r) = (hs[best - 1].1, hs[best].1, hs[best + 1].1);
            let c = l - 2.0 * m + r;
            hs[best].0
                + if c < 0.0 {
                    (0.5 * (l - r) / c) as f32 * f0
                } else {
                    0.0
                }
        } else {
            hs[best].0
        };
        line += &format!(" | {hz:3.0} Hz: (b) open {mean:.3} cm2 {fb:4.0} (c) sung {fc:4.0}");
    }
    println!("{line}");
}
