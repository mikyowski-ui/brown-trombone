//! Measure a voice interrupted or knocked in the middle of a word (hybrid
//! roadmap, stage 4): the same sentence with and without something
//! happening at one sample, and what the difference shows.
//!
//! ```sh
//! cargo run -p synthie-trombone --example interrupt --release
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/interrupt.rs"]
mod interrupt;

use interrupt::{Run, SR, cut, db, deep, peak, points, run};
use synthie_trombone::Vocalization;

fn main() {
    let text = "The birch canoe slid on the smooth planks.";
    let settings = deep();
    let plain = run(&settings, text, 4.0, None, |_| {});
    println!(
        "broken in at            with          | body ms  sound ms | step at the cut dB | steps after / speech's | stopped vocalizing"
    );
    for (name, at) in points(&settings, text) {
        for v in Vocalization::ALL {
            let with = run(&settings, text, 4.0, Some(at), |t| {
                t.interrupt_with(v);
            });
            let c = cut(&plain, &with, at);
            println!(
                "{name:14} {:5.2} s  {:12} | {:6.2}  {:7.2} | {:8.1} | {:6.2} | {} {}",
                at as f32 / SR,
                v.name(),
                c.body_ms,
                c.sound_ms,
                c.step_db,
                c.after_step,
                c.stopped,
                c.vocalizing
            );
        }
        // Knocks: a bump to the chest, a gasp, the jaw knocked shut.
        for (what, lung, jaw) in [
            ("bump +8", 8.0, 0.0),
            ("gasp -15", -15.0, 0.0),
            ("jaw 1.0", 0.0, 1.0),
        ] {
            let with = run(&settings, text, 4.0, Some(at), |t| t.perturb(lung, jaw));
            let c = cut(&plain, &with, at);
            let window = (0.3 * SR) as usize;
            let level = |r: &Run| db(peak(&r.out[at..at + window]));
            let lungs = with.lung[at..at + window]
                .iter()
                .zip(&plain.lung[at..at + window])
                .map(|(a, b)| a - b)
                .fold(0.0f32, |m, d| if d.abs() > m.abs() { d } else { m });
            let least = with.flow[at..at + window]
                .iter()
                .fold(f32::MAX, |m, &f| m.min(f));
            println!(
                "{name:14} {:5.2} s  {what:12} | {:6.2}  {:7.2} | {:8.1} | {:6.2} | level {:+.1} dB, lungs {lungs:+.1} cmH2O, least flow {least:.0}",
                at as f32 / SR,
                c.body_ms,
                c.sound_ms,
                c.step_db,
                c.after_step,
                level(&with) - level(&plain),
            );
        }
    }
}
