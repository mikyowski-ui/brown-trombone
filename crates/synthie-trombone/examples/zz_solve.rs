#![allow(clippy::all, clippy::pedantic)]
//! TEMPORARY (r11-canoe): solve F1 F2 F3 targets (adult man) and measure the shape.
#[path = "support/spectrum.rs"]
mod spectrum;
use synthie_trombone::vowel_targets::{self, VowelKey};
use synthie_trombone::{Engine, Phoneme, Trombone, inverse};
fn main() {
    let a: Vec<f32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let len = 1.0f32;
    let scale = vowel_targets::ADULT_MALE_LENGTH / len;
    let near = VowelKey::Fronted(Phoneme::Uw).articulation().and_then(|a| a.physical_shape(len));
    for f in a.chunks(3) {
        let shape = inverse::solve_for(f[0] * scale, f[1] * scale, Some(f[2] * scale), near.as_ref(), len);
        let mut t = Trombone::new(48_000.0);
        t.set_engine(Engine::Physical);
        t.set_tract_length(len);
        t.set_shape(&shape);
        t.settle();
        let mag = spectrum::transfer(&t, 0.2, 5000.0, 2.0);
        let r = spectrum::resonances(&mag, 2.0);
        print!("target {:?} -> {:?}\n   ", f, shape);
        for x in r.iter().take(4) {
            print!("{:5.0}/{:3.0}/{:5.1}  ", x.hz / scale, x.bandwidth, x.level);
        }
        println!();
        let key = VowelKey::Fronted(Phoneme::Uw);
        let mut t = Trombone::new(48_000.0);
        t.set_engine(Engine::Physical);
        t.set_tract_length(len);
        t.set_vowel_shape(key, Some(&shape));
        t.articulate(&key.articulation().unwrap());
        t.settle();
        let mag = spectrum::transfer(&t, 0.2, 5000.0, 2.0);
        let r = spectrum::resonances(&mag, 2.0);
        print!("   articulated: ");
        for x in r.iter().take(4) {
            print!("{:5.0}/{:3.0}/{:5.1}  ", x.hz / scale, x.bandwidth, x.level);
        }
        print!("\n   cells:");
        for i in (8..44).step_by(2) {
            print!(" {:.2}", t.diameters()[i]);
        }
        println!();
    }
}
