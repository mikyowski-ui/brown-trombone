//! The living voice's breath and creak (wave 3, stage 3A): where the
//! self-oscillating folds alternate strong and weak pulses, and what an
//! inhalation sounds like.
//!
//! ```sh
//! cargo run -p synthie-trombone --example breath --release -- --fry [iy|ax]   # where the folds creak
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/breath.rs"]
mod breath;

use breath::{SR, breaths, creak, db, phrase_vowels, pulses, rms, span, take, vowel_after};
use synthie_trombone::{Engine, GlottalSource, Phoneme, Preset, Trombone, VoiceSettings};

fn voice(vowel: Phoneme, hz: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_glottal_source(GlottalSource::Folds);
    t.articulate_phoneme(vowel);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(hz);
    t.set_tenseness(0.65);
    t.note_on();
    t
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

fn fine_survey(args: &[String]) {
    let vowel = if args.iter().any(|a| a == "iy") {
        Phoneme::Iy
    } else if args.iter().any(|a| a == "ax") {
        Phoneme::Ax
    } else {
        Phoneme::Aa
    };
    println!("Weakly coupled cover pieces, pressed folds, low pressure, on {vowel:?}");
    println!("  coupl tension gap    lung   rate   alt-h  alt-t  sub dB   heights");
    for coupling in [0.2f32, 0.12, 0.08, 0.05, 0.03] {
        for tension in [0.45f32, 0.6, 0.8, 1.0] {
            for gap in [-0.01f32, -0.005, -0.002] {
                for lung in [3.0f32, 4.0, 6.0] {
                    let mut t = voice(vowel, 60.0);
                    t.probe_folds(false, false, true, true);
                    t.override_folds(Some(tension), Some(gap));
                    t.override_fold_coupling(Some(coupling));
                    t.set_lung_pressure(lung);
                    run(&mut t, 0.5);
                    let (out, flows) = run(&mut t, 0.4);
                    let (rate, ah, at, sub) = creak(&out, &flows);
                    let h: Vec<String> = pulses(&flows)
                        .iter()
                        .take(8)
                        .map(|p| format!("{:.0}", p.1))
                        .collect();
                    println!(
                        "  {coupling:.2}  {tension:>5.2} {gap:>6.3} {lung:>5.1}  {rate:>6.1}  {ah:>5.2}  {at:>5.2}  {sub:>6.1}   {}",
                        h.join(" ")
                    );
                }
            }
        }
    }
}

fn speech_report(settings: &VoiceSettings, text: &str) {
    let t = take(settings, text);
    println!("{text}");
    for (a, b) in breaths(&t) {
        let low = t.lung[a..b].iter().copied().fold(0.0f32, f32::min);
        let inflow = t.flow[a..b].iter().copied().fold(0.0f32, f32::min);
        let deep: Vec<f32> = (a..b)
            .filter(|&k| t.lung[k] < 0.5 * low)
            .map(|k| t.out[k])
            .collect();
        let (va, vb) = vowel_after(&t, b).unwrap_or((b, b + 1));
        // The vowels of the phrase after it, together.
        let phrase = phrase_vowels(&t, b);
        let vowel = rms(&t.out[va..vb]);
        let vpeak = t.out[va..vb].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // The turn: 40 ms either side of the lungs crossing zero.
        let turn = (b..t.lung.len()).find(|&k| t.lung[k] >= 0.0).unwrap_or(b);
        let (ta, tb) = (turn.saturating_sub(1920), (turn + 1920).min(t.out.len()));
        let step = t.out[ta..tb]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let vstep = t.out[va..vb]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let mid = usize::midpoint(a, b);
        println!(
            "    middle: area {:.3} cm2, below the folds {:.2} cmH2O, lungs {:.2}",
            t.area[mid], t.sub[mid], t.lung[mid]
        );
        println!(
            "  breath {:.3}-{:.3} s: lungs {low:.2} cmH2O, inflow {inflow:.0} cm3/s, noise {:+.1} dB re the phrase's vowels, {:+.1} dB re the vowel after ({:.0} ms; peak {:+.1} dB re its peak); turn at {:.3} s, largest step {:.2} x the vowel's",
            a as f32 / SR,
            b as f32 / SR,
            db(rms(&deep)) - db(rms(&phrase)),
            db(rms(&deep)) - db(vowel),
            1000.0 * deep.len() as f32 / SR,
            db(deep.iter().fold(0.0f32, |m, v| m.max(v.abs()))) - db(vpeak),
            turn as f32 / SR,
            step / vstep.max(1e-9),
        );
    }
    // The last 100 ms of voice before each pause.
    for k in 1..t.phonemes.len() {
        if t.phonemes[k] != Phoneme::Sil || t.phonemes[k - 1] == Phoneme::Sil {
            continue;
        }
        let Some(last) = (0..k).rev().find(|&i| t.voiced[i]) else {
            continue;
        };
        let (_, end) = span(&t, last);
        let a = end.saturating_sub((0.1 * SR) as usize);
        let (rate, ah, at, sub) = creak(&t.out[a..end], &t.flow[a..end]);
        let h: Vec<String> = pulses(&t.flow[a..end])
            .iter()
            .take(14)
            .map(|p| format!("{:.0}", p.1))
            .collect();
        let lungs = t.lung[a..end].iter().sum::<f32>() / (end - a).max(1) as f32;
        println!(
            "    lungs {lungs:.2} cmH2O on average over it; at its start {:.2}, end {:.2}",
            t.lung[a],
            t.lung[end.saturating_sub(1)]
        );
        println!(
            "  last 100 ms of voice before {:.3} s ({:?}): rate {rate:.1} Hz, alt-h {ah:.2}, alt-t {at:.2}, half-pitch {sub:+.1} dB; pulses {}",
            end as f32 / SR,
            t.phonemes[last],
            h.join(" ")
        );
    }
}

fn low_survey() {
    println!("Creak at low lung pressure, on /a/");
    println!("  coupl tension gap    lung   rate   alt-h  alt-t  sub dB   heights");
    for coupling in [1.0f32, 0.1, 0.05, 0.03] {
        for tension in [0.45f32, 0.6, 0.8] {
            for gap in [-0.004f32, -0.002, 0.0, 0.003] {
                for lung in [2.0f32, 2.5, 3.0, 3.5] {
                    let mut t = voice(Phoneme::Aa, 60.0);
                    t.probe_folds(false, false, true, true);
                    t.override_folds(Some(tension), Some(gap));
                    t.override_fold_coupling(Some(coupling));
                    // Started at 6 cmH2O, then let down: folds keep going
                    // below the pressure they need to start.
                    t.set_lung_pressure(6.0);
                    run(&mut t, 0.3);
                    t.set_lung_pressure(lung);
                    run(&mut t, 0.3);
                    let (out, flows) = run(&mut t, 0.3);
                    let (rate, ah, at, sub) = creak(&out, &flows);
                    let h: Vec<String> = pulses(&flows)
                        .iter()
                        .take(8)
                        .map(|p| format!("{:.0}", p.1))
                        .collect();
                    println!(
                        "  {coupling:.2}  {tension:>5.2} {gap:>6.3} {lung:>5.1}  {rate:>6.1}  {ah:>5.2}  {at:>5.2}  {sub:>6.1}   {}",
                        h.join(" ")
                    );
                }
            }
        }
    }
}

fn onset_survey() {
    println!("Folds under the voice's own control, then taken into creak");
    for hz in [70.0f32, 85.0, 100.0] {
        for lung in [3.0f32, 4.0, 6.0] {
            let mut t = voice(Phoneme::Aa, hz);
            t.set_lung_pressure(lung);
            run(&mut t, 0.4);
            let (out0, f0) = run(&mut t, 0.2);
            t.set_fold_creak(1.0);
            let (out1, f1) = run(&mut t, 0.1);
            let (out2, f2) = run(&mut t, 0.3);
            let c0 = creak(&out0, &f0);
            let c1 = creak(&out1, &f1);
            let c2 = creak(&out2, &f2);
            let h: Vec<String> = pulses(&f1)
                .iter()
                .chain(pulses(&f2).iter())
                .take(24)
                .map(|p| format!("{:.0}", p.1))
                .collect();
            println!(
                "  {hz:>5.0} Hz {lung:.0} cmH2O: modal {:.1} Hz sub {:+.1} dB; first 100 ms of creak {:.1} Hz alt {:.2} sub {:+.1}; after {:.1} Hz alt {:.2} sub {:+.1}\n     {}",
                c0.0,
                c0.3,
                c1.0,
                c1.1,
                c1.3,
                c2.0,
                c2.1,
                c2.3,
                h.join(" ")
            );
        }
    }
}

fn main() {
    if std::env::args().any(|a| a == "--onset") {
        onset_survey();
    }
    if std::env::args().any(|a| a == "--low") {
        low_survey();
    }
    if let Some(i) = std::env::args().position(|a| a == "--speech") {
        let args: Vec<String> = std::env::args().collect();
        let text = args
            .get(i + 1)
            .cloned()
            .unwrap_or_else(|| "The birch canoe slid on the smooth planks. Is it there?".into());
        let mut settings = Preset::Deep.settings();
        for a in &args {
            if let Some((name, value)) = a.split_once('=')
                && let Ok(v) = value.parse::<f32>()
            {
                assert!(settings.tuning.set(name, v), "no tuning field {name}");
            }
        }
        speech_report(&settings, &text);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--fry") {
        fine_survey(&args);
    }
}
