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
    source_report();
    println!();
    fricative_report();
    println!();
    stop_report();
}

const SR: f32 = 48_000.0;

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

fn render(t: &mut Trombone, seconds: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; (seconds * SR) as usize];
    t.render(&mut x);
    x
}

/// Spectral centre of gravity, Hz, between 500 Hz and 12 kHz.
fn centroid(x: &[f32]) -> f32 {
    let step = 50.0;
    let mag = spectrum::spectrum(x, SR, 12_000.0, step);
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (k, m) in mag.iter().enumerate().skip(10) {
        let p = m * m;
        num += p * k as f32 * step;
        den += p;
    }
    num / den.max(1e-20)
}

fn sustained(engine: Engine, tenseness: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(engine);
    t.articulate_phoneme(Phoneme::Aa);
    t.set_frequency(110.0);
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_tenseness(tenseness);
    t.note_on();
    render(&mut t, 0.5);
    t
}

fn source_report() {
    println!("Sustained /a/ at 110 Hz");
    for engine in [Engine::Classic, Engine::Physical] {
        let mut t = sustained(engine, 0.65);
        let x = render(&mut t, 1.0);
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "  {engine:?}: rms {:.4} ({:.1} dB)  peak {:.3}",
            rms(&x),
            db(rms(&x)),
            peak
        );
    }
    println!("  Glottal flow of the physical engine by tenseness");
    println!("  (people: mean 100-300 cm3/s, open quotient 0.4-0.8 in modal voice)");
    for tense in [0.2, 0.4, 0.65, 0.9] {
        let mut t = sustained(Engine::Physical, tense);
        let n = (SR * 0.5) as usize;
        let (mut sum, mut peak, mut open, mut psub) = (0.0f32, 0.0f32, 0usize, 0.0f32);
        let mut flows = Vec::with_capacity(n);
        for _ in 0..n {
            t.process();
            let g = t.gauges();
            sum += g.glottal_flow;
            peak = peak.max(g.glottal_flow);
            psub += g.subglottal_pressure;
            flows.push(g.glottal_flow);
        }
        let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        for &f in &flows {
            if f > floor + 0.1 * (peak - floor) {
                open += 1;
            }
        }
        println!(
            "    tenseness {tense:.2}: mean {:>4.0}  peak {:>4.0}  leak {:>4.0} cm3/s  open quotient {:.2}  subglottal {:.1} cmH2O",
            sum / n as f32,
            peak,
            floor.max(0.0),
            open as f32 / n as f32,
            psub / n as f32
        );
    }
}

fn fricative_report() {
    println!("Fricatives against the vowel /a/ (physical engine)");
    println!(
        "  (people: /s/ and /sh/ 10-20 dB under the vowel, /f/ and /th/ 10-20 dB under those;"
    );
    println!("   /s/ centred near 5-8 kHz, /sh/ near 3-4.5 kHz)");
    let vowel = {
        let mut t = sustained(Engine::Physical, 0.65);
        rms(&render(&mut t, 0.5))
    };
    for p in [
        Phoneme::S,
        Phoneme::Sh,
        Phoneme::F,
        Phoneme::Th,
        Phoneme::Hh,
    ] {
        let mut t = Trombone::new(SR);
        t.set_engine(Engine::Physical);
        t.articulate_phoneme(Phoneme::Aa);
        t.articulate_phoneme(p);
        if p == Phoneme::Hh {
            t.set_tenseness(0.0);
            t.note_on();
        }
        render(&mut t, 0.3);
        let x = render(&mut t, 0.3);
        let g = t.gauges();
        println!(
            "  {:<3} {:>6.1} dB re vowel   centre {:>5.0} Hz   flow {:>4.0} cm3/s   pressure behind it {:.1} cmH2O",
            p.symbol(),
            db(rms(&x) / vowel),
            centroid(&x),
            g.glottal_flow,
            t.pressure(26)
        );
    }
    // Classic, for comparison: its hiss does not depend on the place.
    let classic_vowel = {
        let mut t = sustained(Engine::Classic, 0.65);
        rms(&render(&mut t, 0.5))
    };
    print!("  Classic engine:");
    for p in [Phoneme::S, Phoneme::Sh, Phoneme::F, Phoneme::Th] {
        let mut t = Trombone::new(SR);
        t.articulate_phoneme(Phoneme::Aa);
        t.articulate_phoneme(p);
        render(&mut t, 0.3);
        let x = render(&mut t, 0.3);
        print!(
            "  {} {:.1} dB / {:.0} Hz",
            p.symbol(),
            db(rms(&x) / classic_vowel),
            centroid(&x)
        );
    }
    println!();
}

fn stop_report() {
    println!("Stops between two /a/ (physical engine)");
    for (p, voiced) in [
        (Phoneme::P, false),
        (Phoneme::B, true),
        (Phoneme::T, false),
        (Phoneme::D, true),
    ] {
        let mut t = sustained(Engine::Physical, 0.65);
        // Articulators as fast as the speech front end sets them.
        t.set_articulation_speed(2.2);
        let before = render(&mut t, 0.1);
        let vowel_rms = rms(&before);

        // Close. Every 5 ms: are the folds still vibrating (does the gap
        // between them still open and shut), and are the lips shut (does
        // next to no air leave them)?
        t.articulate_phoneme(p);
        t.set_voiced(voiced);
        let mut shut_at = None;
        let mut voicing_until = 0.0f32;
        let mut max_pressure = 0.0f32;
        let mut bar = Vec::new();
        for k in 0..24 {
            let (mut lo, mut hi, mut lips) = (f32::INFINITY, 0.0f32, 0.0f32);
            let mut out = Vec::with_capacity(240);
            for _ in 0..240 {
                out.push(t.process());
                let g = t.gauges();
                lo = lo.min(g.glottal_area);
                hi = hi.max(g.glottal_area);
                lips = lips.max(g.lip_flow.abs());
            }
            let ms = (k + 1) as f32 * 5.0;
            if shut_at.is_none() && lips < 20.0 {
                shut_at = Some(ms);
            }
            if let Some(at) = shut_at {
                if hi - lo > 0.02 {
                    voicing_until = ms - at;
                }
                bar.extend(out);
            }
            max_pressure = max_pressure.max(t.pressure(30));
        }
        let bar_level = if bar.len() > 1440 {
            db(rms(&bar[..1440]) / vowel_rms)
        } else {
            f32::NAN
        };

        // Release.
        t.articulate_phoneme(Phoneme::Aa);
        t.set_voiced(true);
        let after = render(&mut t, 0.15);
        let burst = after[..(0.03 * SR) as usize]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        // Voice onset: first 10 ms window whose level reaches half the vowel's.
        let onset = after
            .chunks(480)
            .position(|c| rms(c) > 0.5 * vowel_rms)
            .map_or(f32::NAN, |k| k as f32 * 10.0);
        println!(
            "  {:<2} mouth pressure {:>4.1} cmH2O   voicing lasts {:>3.0} ms into the closure (sound {:>5.1} dB re vowel)   burst peak {:.3}   voice back after {:>3.0} ms",
            p.symbol(),
            max_pressure,
            voicing_until,
            bar_level,
            burst,
            onset
        );
    }
}

#[allow(dead_code)]
fn unused() {
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
