//! Measure what the engines actually do, against published human data.
//!
//! ```sh
//! cargo run -p synthie-trombone --example probe --release
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use synthie_trombone::{Engine, Phoneme, Trombone};

/// Peterson & Barney (1952), adult men: F1, F2, F3.
const TARGETS: &[(Phoneme, [f32; 3])] = &[
    (Phoneme::Iy, [270.0, 2290.0, 3010.0]),
    (Phoneme::Ih, [390.0, 1990.0, 2550.0]),
    (Phoneme::Eh, [530.0, 1840.0, 2480.0]),
    (Phoneme::Ae, [660.0, 1720.0, 2410.0]),
    (Phoneme::Ah, [640.0, 1190.0, 2390.0]),
    (Phoneme::Aa, [730.0, 1090.0, 2440.0]),
    (Phoneme::Ao, [570.0, 840.0, 2410.0]),
    (Phoneme::Uh, [440.0, 1020.0, 2240.0]),
    (Phoneme::Uw, [300.0, 870.0, 2240.0]),
];

fn voice(engine: Engine, p: Phoneme) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(engine);
    t.articulate_phoneme(p);
    t.settle();
    t
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn main() {
    println!("Vowel resonances: frequency/bandwidth in Hz (closed glottis)");
    println!("            target F1/F2/F3      engine     F1          F2          F3");
    for &(p, target) in TARGETS {
        for engine in [Engine::Classic, Engine::Physical] {
            let f = spectrum::formants(&voice(engine, p), 4000.0);
            print!(
                "{:<4} {:>5.0} {:>5.0} {:>5.0}   {:<9}",
                p.symbol(),
                target[0],
                target[1],
                target[2],
                format!("{engine:?}")
            );
            for (k, r) in f.iter().take(3).enumerate() {
                let err = 100.0 * (r.hz / target[k] - 1.0);
                print!("  {:>5.0}/{:<3.0}{:>+4.0}%", r.hz, r.bandwidth, err);
            }
            println!();
        }
    }

    println!();
    println!("Sustained /a/ at 110 Hz: level and peak");
    for engine in [Engine::Classic, Engine::Physical] {
        let mut t = Trombone::new(48_000.0);
        t.set_engine(engine);
        t.articulate_phoneme(Phoneme::Aa);
        t.set_frequency(110.0);
        t.note_on();
        let mut warm = vec![0.0f32; 24_000];
        t.render(&mut warm);
        let mut x = vec![0.0f32; 48_000];
        t.render(&mut x);
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("{engine:?}: rms {:.4}  peak {:.3}", rms(&x), peak);
    }
}
