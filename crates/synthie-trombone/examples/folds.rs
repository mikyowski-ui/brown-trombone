//! How the self-oscillating folds behave: pitch against the requested
//! note, flow, and the lung pressure below which they stop.
//!
//! ```sh
//! cargo run -p synthie-trombone --example folds --release
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::{Engine, GlottalSource, Phoneme, Trombone};

const SR: f32 = 48_000.0;

fn voice(hz: f32, tenseness: f32, length: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_glottal_source(GlottalSource::Folds);
    t.set_tract_length(length);
    t.articulate_phoneme(Phoneme::Aa);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(hz);
    t.set_tenseness(tenseness);
    t.note_on();
    t
}

/// Instants (in samples) at which the glottal flow rises through the
/// middle of its swing; it must fall to a quarter of the swing in between.
fn rises(flows: &[f32]) -> Vec<f32> {
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let (hi, lo) = (floor + 0.5 * (peak - floor), floor + 0.25 * (peak - floor));
    let mut crossings = Vec::new();
    let mut armed = false;
    for k in 1..flows.len() {
        if flows[k] < lo {
            armed = true;
        }
        if armed && flows[k - 1] <= hi && flows[k] > hi {
            let frac = (hi - flows[k - 1]) / (flows[k] - flows[k - 1]);
            crossings.push(k as f32 - 1.0 + frac);
            armed = false;
        }
    }
    crossings
}

/// Pitch over a whole stretch of glottal flow.
fn pitch(flows: &[f32]) -> f32 {
    let c = rises(flows);
    if c.len() < 3 {
        return 0.0;
    }
    (c.len() - 1) as f32 * SR / (c[c.len() - 1] - c[0])
}

/// Pitch over the first five cycles.
fn first_cycles(flows: &[f32]) -> f32 {
    let c = rises(flows);
    if c.len() < 6 {
        return 0.0;
    }
    5.0 * SR / (c[5] - c[0])
}

fn cents(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

fn run(t: &mut Trombone, seconds: f32) -> (Vec<f32>, Vec<f32>) {
    let n = (seconds * SR) as usize;
    let mut out = Vec::with_capacity(n);
    let mut flows = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(t.process());
        flows.push(t.gauges().glottal_flow);
    }
    (out, flows)
}

/// Steady-state reading of folds held at a fixed tension and gap, on one
/// vowel: (pitch, mean flow, flow swing, level dB).
fn held_on(vowel: Phoneme, tension: f32, gap: f32) -> (f32, f32, f32, f32) {
    let mut t = voice(110.0, 0.65, 1.0);
    t.articulate_phoneme(vowel);
    t.settle();
    t.override_folds(Some(tension), Some(gap));
    run(&mut t, 0.5);
    let (out, flows) = run(&mut t, 0.4);
    let mean = flows.iter().sum::<f32>() / flows.len() as f32;
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let swing = peak - floor;
    let f = if swing > 0.15 * peak {
        pitch(&flows)
    } else {
        0.0
    };
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    (f, mean, swing, 20.0 * rms.max(1e-9).log10())
}

/// The same averaged over four different vowels, since the tract above
/// the folds pulls their pitch a few percent either way.
fn held(tension: f32, gap: f32) -> (f32, f32, f32, f32) {
    let vowels = [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw, Phoneme::Ax];
    let mut log_pitch = 0.0f32;
    let (mut mean, mut swing, mut level) = (0.0f32, 0.0f32, 0.0f32);
    for v in vowels {
        let (f, m, s, l) = held_on(v, tension, gap);
        if f == 0.0 {
            return (0.0, m, s, l);
        }
        log_pitch += f.ln() / 4.0;
        mean += m / 4.0;
        swing += s / 4.0;
        level += l / 4.0;
    }
    (log_pitch.exp(), mean, swing, level)
}

fn survey() {
    println!("Pitch (Hz) by tension (rows) and rest gap in cm (columns); 0 = not vibrating");
    let gaps = [0.0, 0.01, 0.018, 0.025, 0.035, 0.045, 0.06];
    print!("        ");
    for g in gaps {
        print!("{g:>8.3}");
    }
    println!();
    for k in 0..14 {
        let tension = 0.4 * 1.22f32.powi(k);
        print!("  {tension:>5.2} ");
        for g in gaps {
            let (f, _, _, _) = held(tension, g);
            print!("{f:>8.1}");
        }
        println!();
    }
    println!("Mean flow cm3/s / level dB at tension 1 by gap");
    for g in gaps {
        let (f, mean, swing, level) = held(1.0, g);
        println!(
            "  gap {g:.3}: pitch {f:>6.1}  mean {mean:>4.0}  swing {swing:>4.0}  level {level:>6.1}"
        );
    }
}

/// Body-cover folds under muscle control: pitch, flow and level against
/// cricothyroid activation, for a few thyroarytenoid settings and gaps.
fn muscles() {
    println!("Body-cover folds (Titze & Story 2002 rules), lateral cricoarytenoid 0.5, vowel /a/");
    println!("  ct    ta    gap    pitch   mean flow  swing   level");
    for ta in [0.1f32, 0.25, 0.5] {
        for gap in [0.0f32, 0.01] {
            for k in 0..=8 {
                let ct = k as f32 / 8.0;
                let mut t = voice(110.0, 0.65, 1.0);
                t.override_muscles(Some((ct, ta, 0.5)));
                t.override_folds(Some(1.0), Some(gap));
                run(&mut t, 0.6);
                let (out, flows) = run(&mut t, 0.4);
                let mean = flows.iter().sum::<f32>() / flows.len() as f32;
                let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
                let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let swing = peak - floor;
                let f = if swing > 0.15 * peak {
                    pitch(&flows)
                } else {
                    0.0
                };
                let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
                println!(
                    "  {ct:.3} {ta:.2}  {gap:.3}  {f:>6.1}   {mean:>6.0}    {swing:>5.0}   {:>6.1} dB",
                    20.0 * rms.max(1e-9).log10()
                );
            }
        }
    }
}

/// Print the pitch table embedded in `src/phys/source.rs`.
fn table() {
    let gaps = [0.0f32, 0.007, 0.014];
    println!(
        "pub(crate) const TABLE_GAPS: [f32; {}] = {gaps:?};",
        gaps.len()
    );
    println!("pub(crate) const TABLE_TENSION_START: f32 = 0.35;");
    println!("pub(crate) const TABLE_TENSION_RATIO: f32 = 1.18;");
    println!(
        "pub(crate) const PITCH_TABLE: [[f32; 15]; {}] = [",
        gaps.len()
    );
    let mut notes = Vec::new();
    for g in gaps {
        print!("    [");
        for k in 0..15 {
            let (f, mean, swing, level) = held(0.35 * 1.18f32.powi(k), g);
            print!("{f:.1}, ");
            if (g - 0.014).abs() < 1e-6 {
                notes.push(format!("// tension {:.2}: {f:.0} Hz, mean flow {mean:.0}, swing {swing:.0}, level {level:.1} dB", 0.35 * 1.18f32.powi(k)));
            }
        }
        println!("],");
    }
    println!("];");
    for n in notes {
        println!("{n}");
    }
}

fn main() {
    if std::env::args().any(|a| a == "--survey") {
        survey();
        return;
    }
    if std::env::args().any(|a| a == "--muscles") {
        muscles();
        return;
    }
    if std::env::args().any(|a| a == "--table") {
        table();
        return;
    }

    println!("Requested note against what the folds sing (tenseness 0.65)");
    println!("   Hz  length   first 5 cycles   settled      mean flow  peak  open q.  level");
    for (hz, length) in [
        (62.0, 1.4),
        (80.0, 1.0),
        (105.0, 1.0),
        (125.0, 1.0),
        (160.0, 1.0),
        (210.0, 0.86),
        (220.0, 1.0),
        (290.0, 0.72),
        (330.0, 1.0),
        (360.0, 0.6),
        (440.0, 1.0),
    ] {
        let mut t = voice(hz, 0.65, length);
        let (_, onset) = run(&mut t, 0.25);
        let early = first_cycles(&onset);
        run(&mut t, 0.5);
        let (out, flows) = run(&mut t, 0.5);
        let f = pitch(&flows);
        let mean = flows.iter().sum::<f32>() / flows.len() as f32;
        let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
        let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let open = flows
            .iter()
            .filter(|&&v| v > floor + 0.1 * (peak - floor))
            .count() as f32
            / flows.len() as f32;
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        println!(
            "  {hz:>4.0}  {length:>4.2}   {early:>6.1} {:>+5.0}c   {f:>6.1} {:>+5.0}c   {mean:>5.0} {peak:>6.0}   {open:>4.2}   {:>6.1} dB",
            cents(early.max(1.0), hz),
            cents(f.max(1.0), hz),
            20.0 * rms.max(1e-9).log10()
        );
    }

    println!();
    println!("By tenseness at 110 Hz");
    for tense in [0.0, 0.2, 0.4, 0.65, 0.9, 1.0] {
        let mut t = voice(110.0, tense, 1.0);
        run(&mut t, 0.6);
        let (out, flows) = run(&mut t, 0.5);
        let f = pitch(&flows);
        let mean = flows.iter().sum::<f32>() / flows.len() as f32;
        let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
        let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        println!(
            "  tenseness {tense:.2}: pitch {f:>6.1}  mean flow {mean:>4.0}  peak {peak:>4.0}  leak {:>4.0}  level {:>6.1} dB",
            floor.max(0.0),
            20.0 * rms.max(1e-9).log10()
        );
    }
}
