//! The hiss of each fricative on the physical engine, as a listener would
//! measure it: its spectrum, how broad it is, and how it starts and stops.
//!
//! ```sh
//! cargo run -p synthie-trombone --example frication --release
//! cargo run -p synthie-trombone --example frication --release -- --profile voice.json
//! ```
//!
//! Each fricative is said between two vowels (`AA C AA`, as a word). For the
//! middle of its hiss it prints the level in 500 Hz bands up to 8 kHz (dB
//! under the loudest band), the spectral centroid and flatness over 1 to 8
//! kHz and the width of the band within 6 dB of the peak; for the hiss
//! above 2 kHz (3 kHz for /s z/), its rise and fall (15 dB under its peak to 3 dB under)
//! and how long it stays within 10 dB of its peak, and its level at the
//! peak against the loudest 10 ms of the vowels. People's /s/ has its
//! energy from about 4 kHz up and peaks at 5 to 8 kHz, /ʃ/ from about 2 kHz
//! with a broad peak at 2.5 to 4 kHz; /f θ/ are flat and weak; the hiss
//! rises and falls over tens of milliseconds.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/hiss.rs"]
mod hiss;

use synthie_trombone::{Engine, Preset};

fn main() {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--profile" {
            let file = args.next().expect("--profile needs a file");
            let json = std::fs::read_to_string(&file).expect("profile");
            settings = settings
                .with_profile(&json)
                .unwrap_or_else(|e| panic!("{file}: {e}"));
        }
    }
    print!("       ");
    for b in 0..hiss::BANDS {
        print!("{:5.1}", (b as f32 + 0.5) * hiss::BAND_HZ / 1000.0);
    }
    println!("  centroid flat width6  rise  fall  dur level  (kHz, ms, dB)");
    for p in ["S", "Z", "SH", "ZH", "F", "V", "TH", "DH"] {
        let h = hiss::measure(&settings, p);
        print!("{p:>4}   ");
        for v in &h.bands {
            print!("{v:5.0}");
        }
        println!(
            "  {:6.0} {:5.2} {:5.0} {:5.1} {:5.1} {:5.0} {:5.1}",
            h.centroid, h.flatness, h.width6, h.rise_ms, h.fall_ms, h.dur_ms, h.level
        );
    }
}
