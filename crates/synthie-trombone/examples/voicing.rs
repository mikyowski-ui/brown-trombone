//! How the physical engine voices its consonants: for every stop in
//! initial, medial and final position the closure length, the voice bar in
//! the closure, the burst, the voice onset time, the mouth pressure at the
//! release and the pitch at the start of the vowel; for the fricative pairs
//! the voice bar and the hiss; and how much shorter vowels are before a
//! voiceless coda than before a voiced one.
//!
//! ```sh
//! cargo run -p synthie-trombone --example voicing --release
//! cargo run -p synthie-trombone --example voicing --release -- --profile voice.json
//! ```
//!
//! The voice is the one the judge listens to (`deep`). Levels are in dB
//! re the neighbouring stressed vowel; the voice bar is the energy below
//! 300 Hz; the pressure is the mouth's over the pressure below the glottis.

#![allow(clippy::cast_precision_loss)]

#[path = "support/stops.rs"]
mod stops;

use stops::{FRICATIVES, STOPS, context, fricative, judge_voice, stop, vowel_length};

fn main() -> std::io::Result<()> {
    let mut settings = judge_voice();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--profile" {
            let file = args.next().expect("--profile needs a file");
            let json = std::fs::read_to_string(&file)?;
            settings = settings
                .with_profile(&json)
                .unwrap_or_else(|e| panic!("{file}: {e}"));
        }
    }

    println!(
        "stop  position  closure ms  voiced %  voice bar dB  burst dB  VOT ms  p_mouth/p_sub  F0 0-25/25-50 ms re steady (steady Hz)"
    );
    for position in ["initial", "medial", "final"] {
        for &(voiceless, voiced) in &STOPS {
            for p in [voiceless, voiced] {
                let m = stop(&settings, &context(p, position), p);
                println!(
                    "{:4}  {position:8}  {:10.0}  {:8.0}  {:12.1}  {:8.1}  {:6.0}  {:13.2}  {:+5.1} / {:+5.1} ({:.0})",
                    p.symbol(),
                    m.closure,
                    100.0 * m.voiced_share,
                    m.voice_bar,
                    m.burst,
                    m.vot,
                    m.pressure,
                    m.f0_onset,
                    m.f0_next,
                    m.f0_steady,
                );
            }
        }
    }

    println!();
    println!("fricative  length ms  voiced %  voice bar dB  hiss dB   (between AA1 and AA)");
    for &(voiceless, voiced) in &FRICATIVES {
        for p in [voiceless, voiced] {
            let m = fricative(&settings, &context(p, "vcv"), p);
            println!(
                "{:9}  {:9.0}  {:8.0}  {:12.1}  {:7.1}",
                p.symbol(),
                m.length,
                100.0 * m.voiced_share,
                m.voice_bar,
                m.hiss
            );
        }
    }

    println!();
    println!("vowel before voiceless / voiced coda (compiled lengths)");
    let mut ratios = Vec::new();
    for (a, b) in [
        ("[B AE1 T]", "[B AE1 D]"),
        ("[R OW1 P]", "[R OW1 B]"),
        ("[B AE1 K]", "[B AE1 G]"),
        ("[B UW1 S]", "[B UW1 Z]"),
        ("[L IY1 F]", "[L IY1 V]"),
    ] {
        let (x, y) = (vowel_length(&settings, a), vowel_length(&settings, b));
        ratios.push(x / y);
        println!(
            "  {a:10} {:4.0} ms   {b:10} {:4.0} ms   ratio {:.2}",
            1000.0 * x,
            1000.0 * y,
            x / y
        );
    }
    println!(
        "  mean ratio {:.2}",
        ratios.iter().sum::<f32>() / ratios.len() as f32
    );
    Ok(())
}
