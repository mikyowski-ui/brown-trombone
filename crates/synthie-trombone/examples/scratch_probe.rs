#![allow(clippy::pedantic, dead_code)]
#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/survey.rs"]
mod survey;

use std::time::Instant;
use survey::{Lcg, Meter};
use synthie_trombone::inverse::Shape;

fn random_shape(rng: &mut Lcg) -> Shape {
    let lips = if rng.unit() < 0.3 { None } else { Some(rng.range(0.65, 1.45)) };
    let narrowing = if rng.unit() < 0.3 {
        None
    } else {
        Some((rng.range(14.0, 34.0), rng.range(0.7, 1.4)))
    };
    Shape {
        tongue_index: rng.range(12.0, 29.0),
        tongue_diameter: rng.range(2.05, 3.5),
        lips,
        narrowing,
    }
}

fn main() {
    use synthie_trombone::{Engine, Phoneme, Trombone};
    let vowels = [
        Phoneme::Iy, Phoneme::Ih, Phoneme::Eh, Phoneme::Ae, Phoneme::Ah, Phoneme::Aa,
        Phoneme::Ao, Phoneme::Uh, Phoneme::Uw, Phoneme::Ow, Phoneme::Er, Phoneme::Ax,
    ];
    let lengths = [0.6f32, 0.8, 1.0, 1.2, 1.4];
    for p in vowels {
        let mut native = [0.0f32; 3];
        let mut row = format!("{:3}", p.symbol());
        let mut all = Vec::new();
        for &len in &lengths {
            let mut t = Trombone::new(48_000.0);
            t.set_engine(Engine::Physical);
            t.set_tract_length(len);
            t.articulate_phoneme(p);
            t.settle();
            let f = survey::formants(&t).unwrap();
            if len == 1.0 {
                native = f;
            }
            all.push((len, f));
        }
        row += &format!("  native {:6.1} {:6.1} {:6.1} |", native[0], native[1], native[2]);
        for (len, f) in all {
            row += &format!(
                " {len}: {:+5.1}% {:+5.1}% {:+5.1}% |",
                100.0 * (f[0] * len / native[0] - 1.0),
                100.0 * (f[1] * len / native[1] - 1.0),
                100.0 * (f[2] * len / native[2] - 1.0)
            );
        }
        println!("{row}");
    }
}
