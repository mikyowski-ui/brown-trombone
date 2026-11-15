#![allow(clippy::all, clippy::pedantic)]
//! TEMPORARY (r11-canoe): measure shapes given directly: zz_shape ti td lips ncell ndia [... ...]
//! (lips 0 = spread). Prints the first resonances (adult-man Hz) and the 2-4 kHz band.
#[path = "support/spectrum.rs"]
mod spectrum;
use synthie_trombone::inverse::Shape;
use synthie_trombone::vowel_targets;
use synthie_trombone::{Engine, Trombone};
fn main() {
    let a: Vec<f32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let len = 1.0f32;
    let scale = vowel_targets::ADULT_MALE_LENGTH / len;
    for f in a.chunks(5) {
        let shape = Shape {
            tongue_index: f[0],
            tongue_diameter: f[1],
            lips: if f[2] > 0.0 { Some(f[2]) } else { None },
            narrowing: if f[4] > 0.0 { Some((f[3], f[4])) } else { None },
            pharynx: 0.0,
        };
        let mut t = Trombone::new(48_000.0);
        t.set_engine(Engine::Physical);
        t.set_tract_length(len);
        t.set_shape(&shape);
        t.settle();
        let mag = spectrum::transfer(&t, 0.2, 5000.0, 2.0);
        let r = spectrum::resonances(&mag, 2.0);
        let band = |lo: f32, hi: f32| {
            let s: f32 = mag[(lo / 2.0) as usize..(hi / 2.0) as usize].iter().map(|v| v * v).sum();
            10.0 * s.log10()
        };
        print!("{:?} ->", f);
        for x in r.iter().take(4) {
            print!(" {:5.0}/{:3.0}/{:5.1}", x.hz / scale, x.bandwidth, x.level);
        }
        println!("  | 2-4k re 0-1k {:5.1}", band(2000.0, 4000.0) - band(100.0, 1000.0));
        if std::env::var("ZZ_D").is_ok() {
            print!("   cells:");
            for i in (8..44).step_by(2) {
                print!(" {:.2}", t.diameters()[i]);
            }
            println!();
        }
    }
}
