//! Measure a voice morphing between bodies (hybrid roadmap, stage 1).
//!
//! ```sh
//! cargo run -p synthie-trombone --example morph --release              # the sweep
//! cargo run -p synthie-trombone --example morph --release -- folds     # fold size at a pitch
//! cargo run -p synthie-trombone --example morph --release -- formants  # a voice and its morphs
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/morph.rs"]
mod morph;
#[path = "support/spectrum.rs"]
mod spectrum;

use morph::{
    Rec, SR, band, click_ratio, closed_share, glide, h1_h2, held, rms, run, weakest_swing,
};
use synthie_trombone::{MorphTarget, Phoneme, Preset, VoiceSettings};

fn sweep() {
    println!(
        "voice vowel | click steady / sweeping | weakest swing | pitch lo..hi | F1 F2 at 0.75, 1, 1.25 of the tract"
    );
    for (name, preset, hz) in [
        ("deep", Preset::Deep, 105.0),
        ("bright", Preset::Bright, 210.0),
    ] {
        for p in [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw, Phoneme::Ae] {
            let settings = preset.settings();
            let mut v = held(&settings, p, hz);
            let mut steady = Rec::default();
            run(&mut v, 1.0, &mut steady);
            let mut rec = Rec::default();
            glide(&mut v, MorphTarget::sized(0.75), 0.3, &mut Rec::default());
            let f = |v: &synthie_trombone::Trombone| {
                let r = spectrum::formants(v, 4000.0);
                (
                    r.first().map_or(0.0, |r| r.hz),
                    r.get(1).map_or(0.0, |r| r.hz),
                )
            };
            let small = f(&v);
            glide(&mut v, MorphTarget::sized(1.25), 0.0, &mut rec);
            let large = f(&v);
            glide(&mut v, MorphTarget::sized(0.75), 0.0, &mut rec);
            glide(&mut v, MorphTarget::NEUTRAL, 0.2, &mut rec);
            let mid = f(&v);
            let (lo, hi) = rec
                .pitch
                .iter()
                .fold((f32::MAX, 0.0f32), |(l, h), &p| (l.min(p), h.max(p)));
            println!(
                "{name:6} {p:3} | {:5.2} / {:5.2} | {:4.2} | {lo:5.1}..{hi:5.1} Hz | {:4.0} {:4.0}  {:4.0} {:4.0}  {:4.0} {:4.0}  ({:.1} s)",
                click_ratio(&steady.out),
                click_ratio(&rec.out),
                weakest_swing(&rec.flow, hz),
                small.0,
                small.1,
                mid.0,
                mid.1,
                large.0,
                large.1,
                rec.out.len() as f32 / SR,
            );
        }
    }
    // Every number at once, there and back.
    let to = MorphTarget {
        tract_scale: 1.2,
        larynx_depth: 0.6,
        fold_thickness: 0.7,
        epilarynx_flare: 1.4,
    };
    let back = MorphTarget {
        tract_scale: 0.8,
        larynx_depth: -0.3,
        fold_thickness: 1.4,
        epilarynx_flare: 0.7,
    };
    for (name, preset, hz) in [
        ("deep", Preset::Deep, 105.0),
        ("bright", Preset::Bright, 210.0),
    ] {
        let mut v = held(&preset.settings(), Phoneme::Aa, hz);
        let mut rec = Rec::default();
        glide(&mut v, to, 0.1, &mut rec);
        glide(&mut v, back, 0.1, &mut rec);
        glide(&mut v, MorphTarget::NEUTRAL, 0.1, &mut rec);
        println!(
            "{name:6} all four, there and back: click {:5.2}, weakest swing {:4.2}",
            click_ratio(&rec.out),
            weakest_swing(&rec.flow, hz)
        );
    }
}

fn folds() {
    println!(
        "voice thickness | fold range Hz | pitch | H1-H2 flow dB | closed | flow peak | 0-1k, 1-2.5k, 2.5-5k dB of the sound (/a/)"
    );
    for (name, preset, hz) in [
        ("bright", Preset::Bright, 210.0),
        ("deep", Preset::Deep, 105.0),
    ] {
        for th in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] {
            let m = MorphTarget {
                fold_thickness: th,
                ..MorphTarget::NEUTRAL
            };
            let settings: VoiceSettings = preset.settings().with_morph(&m);
            let mut v = held(&settings, Phoneme::Aa, hz);
            let mut rec = Rec::default();
            run(&mut v, 0.6, &mut rec);
            let (lo, hi) = v.fold_range();
            let pitch = rec.pitch.iter().sum::<f32>() / rec.pitch.len() as f32;
            let peak = rec.flow.iter().fold(0.0f32, |m, &x| m.max(x));
            let low = band(&rec.out, hz, 0.0, 1000.0);
            println!(
                "{name:6} {th:4.1} | {lo:4.0}..{hi:4.0} | {pitch:5.1} | {:5.1} | {:4.2} | {peak:5.0} | {low:5.1} {:5.1} {:5.1}  rms {:.4}",
                h1_h2(&rec.flow, hz),
                closed_share(&rec.flow),
                band(&rec.out, hz, 1000.0, 2500.0) - low,
                band(&rec.out, hz, 2500.0, 5000.0) - low,
                rms(&rec.out),
            );
        }
    }
}

fn formants(args: &[String]) {
    let num = |k: usize, d: f32| args.get(k).and_then(|a| a.parse().ok()).unwrap_or(d);
    let m = MorphTarget {
        tract_scale: num(0, 1.07),
        larynx_depth: num(1, 0.3),
        fold_thickness: num(2, 1.0),
        epilarynx_flare: num(3, 1.0),
    };
    println!("bright against bright with {m:?}: F1 F2 F3 F4 (Hz) and the morph's share of each");
    for p in [
        Phoneme::Iy,
        Phoneme::Ih,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Aa,
        Phoneme::Ao,
        Phoneme::Uh,
        Phoneme::Uw,
        Phoneme::Ah,
        Phoneme::Er,
    ] {
        let f = |s: &VoiceSettings| {
            let v = held(s, p, 210.0);
            let r = spectrum::formants(&v, 5500.0);
            [0, 1, 2, 3].map(|k| r.get(k).map_or(0.0, |r| r.hz))
        };
        let a = f(&Preset::Bright.settings());
        let b = f(&Preset::Bright.settings().with_morph(&m));
        println!(
            "{p:3} | {:4.0} {:4.0} {:4.0} {:4.0} | {:4.0} {:4.0} {:4.0} {:4.0} | {:.3} {:.3} {:.3} {:.3}",
            a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[1],
            b[2],
            b[3],
            b[0] / a[0],
            b[1] / a[1],
            b[2] / a[2],
            b[3] / a[3]
        );
    }
}

/// The roughest 10 ms stretches of `x`: (seconds, step over level).
fn roughest(x: &[f32]) -> Vec<(f32, f32)> {
    let n = (0.01 * SR) as usize;
    let loudest = x.chunks(n).map(rms).fold(0.0, f32::max);
    let mut all: Vec<(f32, f32)> = x
        .chunks(n)
        .enumerate()
        .filter(|(_, c)| c.len() == n && morph::db(rms(c)) > morph::db(loudest) - 30.0)
        .map(|(k, c)| {
            let step = c
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0, f32::max);
            (k as f32 * 0.01, step / rms(c).max(1e-9))
        })
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    all.truncate(4);
    all
}

fn speech() {
    use synthie_trombone::{Engine, Talker};
    let text =
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
    let speak = |settings: &VoiceSettings, from: f32, to: f32| {
        let mut talker = Talker::new(SR, *settings);
        let u = talker.compile(text);
        let frames = ((u.duration() + 0.25) * SR) as usize;
        talker.say_utterance(u);
        let mut out = vec![0.0; frames];
        for (k, block) in out.chunks_mut(64).enumerate() {
            let at = (k * 64) as f32 / frames as f32;
            talker.set_morph(MorphTarget::sized(from + (to - from) * at));
            talker.render(block);
        }
        out
    };
    for preset in [Preset::Deep, Preset::Bright] {
        let s = preset.settings().with_engine(Engine::Physical);
        println!("{preset}: plain {:?}", roughest(&speak(&s, 1.0, 1.0)));
        for scale in [0.8, 0.9, 1.1, 1.2] {
            let fixed = s.with_morph(&MorphTarget::sized(scale));
            println!("  made {scale}: {:?}", roughest(&speak(&fixed, 1.0, 1.0)));
        }
        for scale in [0.84, 0.845, 0.85, 0.855, 0.86] {
            let fixed = s.with_morph(&MorphTarget::sized(scale));
            println!(
                "  made {scale}: {:?}",
                &roughest(&speak(&fixed, 1.0, 1.0))[..2]
            );
        }
        for (a, b) in [
            (0.81, 1.21),
            (0.79, 1.19),
            (0.8, 1.25),
            (0.82, 1.18),
            (0.78, 1.22),
            (0.8, 1.15),
            (0.75, 1.25),
            (1.25, 0.75),
        ] {
            println!("  glide {a}..{b}: {:?}", &roughest(&speak(&s, a, b))[..2]);
        }
        println!("  glide 0.84..0.86: {:?}", roughest(&speak(&s, 0.84, 0.86)));
        println!("  glide 0.8..1.2: {:?}", roughest(&speak(&s, 0.8, 1.2)));
        println!("  glide 1.2..0.8: {:?}", roughest(&speak(&s, 1.2, 0.8)));
        println!("  glide 0.95..1.05: {:?}", roughest(&speak(&s, 0.95, 1.05)));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("speech") => speech(),
        Some("folds") => folds(),
        Some("formants") => formants(&args[1..]),
        _ => sweep(),
    }
}
