#![allow(clippy::all, clippy::pedantic)]
#[path = "support/spectrum.rs"]
mod spectrum;
use synthie_trombone::vowel_targets::VowelKey;
use synthie_trombone::{Engine, Phoneme, Trombone};
fn show(t: &Trombone, label: &str) {
    let mag = spectrum::transfer(t, 0.2, 5000.0, 2.0);
    let r = spectrum::resonances(&mag, 2.0);
    let band = |lo: f32, hi: f32| {
        let s: f32 = mag[(lo / 2.0) as usize..(hi / 2.0) as usize].iter().map(|v| v * v).sum();
        10.0 * s.log10()
    };
    print!("{:10} ", label);
    for x in r.iter().take(6) {
        print!("{:5.0}/{:3.0}/{:5.1}  ", x.hz, x.bandwidth, x.level);
    }
    println!(
        " | 0-1k {:5.1}  2-4k re 0-1k {:5.1}",
        band(100.0, 1000.0),
        band(2000.0, 4000.0) - band(100.0, 1000.0)
    );
}
fn main() {
    let len: f32 = std::env::args().nth(1).map_or(1.0, |s| s.parse().unwrap());
    let keys = [
        VowelKey::Vowel(Phoneme::Uw),
        VowelKey::Fronted(Phoneme::Uw),
        VowelKey::End(Phoneme::Uw),
        VowelKey::Vowel(Phoneme::W),
        VowelKey::Vowel(Phoneme::Uh),
        VowelKey::Vowel(Phoneme::Ax),
        VowelKey::Vowel(Phoneme::Ih),
        VowelKey::Vowel(Phoneme::Iy),
        VowelKey::Vowel(Phoneme::Y),
        VowelKey::Vowel(Phoneme::Aa),
    ];
    for k in keys {
        let a = k.articulation().unwrap();
        let mut t = Trombone::new(48_000.0);
        t.set_engine(Engine::Physical);
        t.set_tract_length(len);
        t.articulate(&a);
        t.settle();
        show(&t, &format!("{k}"));
        if std::env::var("ZZ_VEL").is_ok() && matches!(k, VowelKey::Vowel(Phoneme::Uw) | VowelKey::Fronted(Phoneme::Uw) | VowelKey::Vowel(Phoneme::Aa) | VowelKey::Vowel(Phoneme::Iy)) {
            for v in [0.25f32, 0.5, 1.0] {
                t.set_velum(v);
                t.settle();
                show(&t, &format!("  velum{v}"));
            }
        }
        if std::env::var("ZZ_D").is_ok() {
            let d = t.diameters();
            print!("      d:");
            for (i, v) in d.iter().enumerate() {
                if i % 2 == 0 || i >= 38 {
                    print!(" {:.2}", v);
                }
            }
            println!();
        }
    }
    // the /n/ murmur on the /u/ body
    for p in [Phoneme::N, Phoneme::M] {
        let mut t = Trombone::new(48_000.0);
        t.set_engine(Engine::Physical);
        t.set_tract_length(len);
        t.articulate(&VowelKey::Vowel(Phoneme::Uw).articulation().unwrap());
        t.settle();
        t.articulate(&p.articulation());
        t.settle();
        show(&t, &format!("{p} (u)"));
    }
}
