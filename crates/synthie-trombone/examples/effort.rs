//! Measure the vocal effort states (hybrid roadmap, stage 3): one sentence
//! through each, with its level, pitch, spectral balance, H1-H2, lungs
//! and tremor.
//!
//! ```sh
//! cargo run -p synthie-trombone --example effort --release
//! cargo run -p synthie-trombone --example effort --release -- "Any sentence."
//! ```

#![allow(clippy::cast_precision_loss)]

#[path = "support/effort.rs"]
mod effort;

use effort::{settings, signature, speak};
use synthie_trombone::Effort;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = if args.is_empty() {
        "We were away a year ago. Mary may win more money.".to_string()
    } else {
        args.join(" ")
    };
    println!(
        "state      level dB | pitch Hz voiced | 1-2.5k 2.5-5k dB | H1-H2 dB | lungs cmH2O | tremor Hz, share | seconds | jaw"
    );
    for e in Effort::ALL {
        let s = signature(&speak(&settings(e), &text, |_, _| {}));
        println!(
            "{:10} {:7.1} | {:6.0} {:5.0} % | {:6.1} {:6.1} | {:6.1} | {:6.1} | {:5.2} {:6.3} | {:5.2} | {:4.2}",
            e.name(),
            s.level,
            s.pitch,
            100.0 * s.voiced,
            s.mid,
            s.tilt,
            s.h1_h2,
            s.lung,
            s.tremor.0,
            s.tremor.1,
            s.seconds,
            s.jaw
        );
    }
}
