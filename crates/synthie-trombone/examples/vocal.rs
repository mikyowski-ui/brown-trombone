//! The living voice's non-verbal sounds (wave 3, stage 3B): each
//! vocalization alone on a voice, what its calls, its burst and its breath
//! measure, and where it was written.
//!
//! ```sh
//! cargo run -p synthie-trombone --example vocal --release -- [OUT_DIR]
//! ```

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/vocal.rs"]
mod vocal;
#[path = "support/wav.rs"]
mod wav;

use synthie_trombone::{Vocal, Vocalization};
use vocal::{SR, calls, db, deep, envelope, largest_step, median, noise_share, record, rms, voice};

#[allow(clippy::too_many_lines)] // one report per vocalization
fn main() {
    let dir = std::env::args().nth(1);
    {
        // The same voice speaking, for comparison.
        use synthie_trombone::speech::{Speaker, Utterance};
        let settings = deep().with_engine(synthie_trombone::Engine::Physical);
        let mut t = voice(&settings);
        let mut s = Speaker::new();
        s.speak(Utterance::from_text(
            "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.",
            &settings.prosody(),
        ));
        let mut out = Vec::new();
        while s.is_speaking() {
            out.push(s.process(&mut t));
        }
        println!(
            "speech        loudest 10 ms {:.1} dB, click ratio {:.1}",
            db(envelope(&out, 10.0, 5.0).into_iter().fold(0.0, f32::max)),
            vocal::click_ratio(&out, 50.0)
        );
    }
    let strength: f32 = std::env::var("STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    {
        // What a vocalization costs: the call that lays its gestures, and
        // rendering it against rendering speech.
        use std::time::Instant;
        let mut voices: Vec<_> = (0..100).map(|_| voice(&deep())).collect();
        let (mut total, mut longest) = (0.0f64, 0.0f64);
        for (v, kind) in voices.iter_mut().zip(Vocalization::ALL.iter().cycle()) {
            let t0 = Instant::now();
            assert!(v.vocalize(*kind).is_some());
            let took = t0.elapsed().as_secs_f64();
            total += took;
            longest = longest.max(took);
        }
        println!(
            "vocalize()    {:.1} us a call on average, {:.1} us at most",
            1e6 * total / 100.0,
            1e6 * longest
        );
        let mut v = voice(&deep());
        let seconds = v.vocalize(Vocalization::Laugh).unwrap() + 2.0;
        let t0 = Instant::now();
        let r = record(&mut v, seconds);
        let laugh = t0.elapsed().as_secs_f64() / f64::from(seconds);
        let _ = r;
        let mut v = voice(&deep());
        v.articulate_phoneme(synthie_trombone::Phoneme::Aa);
        v.note_on();
        let t0 = Instant::now();
        record(&mut v, seconds);
        let vowel = t0.elapsed().as_secs_f64() / f64::from(seconds);
        println!(
            "rendering     {laugh:.3} s a second of audio with a laugh playing (then silence), {vowel:.3} s for a held vowel"
        );
        // The ventricular valve's cost: a held vowel with it shut a third
        // and fluttering the whole time, against the same vowel with it at
        // rest (the best of five of each, taken in turn).
        let held = |engaged: bool| {
            let mut v = voice(&deep());
            v.articulate_phoneme(synthie_trombone::Phoneme::Aa);
            v.note_on();
            if engaged {
                v.set_ventricular_valve(0.3);
                v.set_aryepiglottic_flutter(0.5);
            }
            let mut buf = [0.0f32; 256];
            let t0 = Instant::now();
            for _ in 0..(2.0 * SR / 256.0) as usize {
                v.render(&mut buf);
            }
            t0.elapsed().as_secs_f64() / 2.0
        };
        let (mut rest, mut engaged) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            rest = rest.min(held(false));
            engaged = engaged.min(held(true));
        }
        println!(
            "valve         {engaged:.3} s a second of audio with it shut a third and fluttering, {rest:.3} s at rest ({:+.1} %)",
            100.0 * (engaged / rest - 1.0)
        );
    }
    for kind in Vocalization::ALL {
        let mut v = voice(&deep());
        v.set_frequency(100.0);
        let seconds = v
            .vocalize(Vocal::new(kind).with_strength(strength))
            .expect("room");
        let r = record(&mut v, seconds + 0.3);
        let peak_lung = r.lung.iter().copied().fold(f32::MIN, f32::max);
        let low_lung = r.lung.iter().copied().fold(f32::MAX, f32::min);
        let peak_sub = r.sub.iter().copied().fold(f32::MIN, f32::max);
        let out_peak = r.out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let env = envelope(&r.out, 10.0, 5.0);
        let env_peak = env.iter().copied().fold(0.0, f32::max);
        println!(
            "{:<13} {seconds:.2} s  lungs {low_lung:+.1}..{peak_lung:+.1} cmH2O  below folds max {peak_sub:.1}  peak {out_peak:.3} (10 ms RMS {:.1} dB)  step {:.3}  finite {}",
            kind.name(),
            db(env_peak),
            largest_step(&r.out),
            r.finite
        );
        if std::env::var("TRACE").is_ok() {
            let n = (0.02 * SR) as usize;
            let vib = vocal::vibrating(&r);
            for k in (0..r.len()).step_by(n) {
                println!(
                    "   {:.2}  lung {:+6.1} sub {:+6.1} flow {:7.0} area {:.3} edge {:.3} pitch {:5.0} vib {} out {:6.1} dB",
                    k as f32 / SR,
                    r.lung[k],
                    r.sub[k],
                    r.flow[k],
                    r.area[k],
                    r.edge[k],
                    r.pitch[k],
                    u8::from(vib[k / (SR as usize / 100)]),
                    db(rms(&r.out[k..(k + n).min(r.len())]))
                );
            }
        }
        if let Ok(spec) = std::env::var("FINE") {
            let (name, a, b) = {
                let p: Vec<&str> = spec.split(',').collect();
                (
                    p[0].to_string(),
                    p[1].parse::<f32>().unwrap(),
                    p[2].parse::<f32>().unwrap(),
                )
            };
            if name == kind.name() {
                let n = (0.002 * SR) as usize;
                for k in ((a * SR) as usize..(b * SR) as usize).step_by(n) {
                    println!(
                        "   {:.3}  lung {:+6.1} sub {:+6.1} flow {:7.0} area {:.3} lips {:8.1} nose {:7.1} skin {:8.2} out {:+.4}",
                        k as f32 / SR,
                        r.lung[k],
                        r.sub[k],
                        r.flow[k],
                        r.area[k],
                        r.lips[k],
                        r.nose[k],
                        r.skin[k],
                        r.out[k]
                    );
                }
            }
        }
        println!(
            "      click ratio {:.1} (within 50 dB of the loudest 10 ms)",
            vocal::click_ratio(&r.out, 50.0)
        );
        let c = calls(&r);
        for (t, l, hz, level) in &c {
            println!(
                "      call at {t:.3} s for {:.0} ms, {hz:.0} Hz, {level:.1} dB",
                1000.0 * l
            );
        }
        if c.len() > 1 {
            let rate = (c.len() - 1) as f32 / (c[c.len() - 1].0 - c[0].0);
            println!("      {} calls, {rate:.2} per second", c.len());
        }
        if kind == Vocalization::Cough {
            let shut: Vec<usize> = (0..r.len())
                .filter(|&k| r.area[k] < 0.002 && r.sub[k] > 5.0)
                .collect();
            if let Some(&last) = shut.last() {
                let env = envelope(&r.out[last..], 10.0, 1.0);
                let at = env
                    .iter()
                    .enumerate()
                    .take(150)
                    .fold((0, 0.0f32), |m, (k, &x)| if x > m.1 { (k, x) } else { m });
                let flow = &r.flow[last..(last + (0.2 * SR) as usize).min(r.len())];
                let peak = flow.iter().copied().fold(0.0f32, f32::max);
                let rise = flow.iter().position(|&f| f >= 0.9 * peak).unwrap_or(0);
                let from = flow.iter().position(|&f| f >= 0.1 * peak).unwrap_or(0);
                println!(
                    "      shut {:.3} s with {:.1} cmH2O behind; burst's loudest 10 ms {} ms after the opening, {:.1} dB; flow peak {peak:.0} cm3/s, 10-90 % in {:.1} ms",
                    shut.len() as f32 / SR,
                    shut.iter().map(|&k| r.sub[k]).fold(0.0f32, f32::max),
                    at.0,
                    db(at.1),
                    1000.0 * (rise - from) as f32 / SR
                );
            }
        }
        if kind == Vocalization::Cough || kind == Vocalization::ClearThroat {
            // The ventricular valve: how long it was shut, how fast it
            // opened (to 90 % of its rest area), how wide it went, and the
            // rate of its flutter where it was narrowed.
            let rest = r.valve[0];
            let shut = r.valve.iter().filter(|&&a| a == 0.0).count();
            let last = r.valve.iter().rposition(|&a| a == 0.0).unwrap_or(0);
            let opened = r.valve[last..].iter().position(|&a| a >= 0.9 * rest);
            let widest = r.valve.iter().copied().fold(0.0f32, f32::max);
            let narrowed: Vec<usize> = (0..r.len())
                .filter(|&k| r.valve[k] > 0.0 && r.valve[k] < 0.7 * rest)
                .collect();
            let troughs: Vec<usize> = (1..narrowed.len().saturating_sub(1))
                .map(|i| narrowed[i])
                .filter(|&k| r.valve[k] < r.valve[k - 1] && r.valve[k] <= r.valve[k + 1])
                .collect();
            // The mean period between troughs within a scrape.
            let periods: Vec<usize> = troughs
                .windows(2)
                .map(|w| w[1] - w[0])
                .filter(|&d| (d as f32) < 0.03 * SR)
                .collect();
            let hz = if periods.is_empty() {
                0.0
            } else {
                SR * periods.len() as f32 / periods.iter().sum::<usize>() as f32
            };
            let peak = r.flow.iter().copied().fold(0.0f32, f32::max);
            println!(
                "      valve shut {:.3} s, then {} ms to 90 % of its rest area {rest:.2} cm2, widest {widest:.2} cm2; narrowed {:.3} s, fluttering at {:.0} Hz; peak flow {peak:.0} cm3/s",
                shut as f32 / SR,
                opened.map_or_else(
                    || "-".to_string(),
                    |k| format!("{:.1}", 1000.0 * k as f32 / SR)
                ),
                narrowed.len() as f32 / SR,
                hz
            );
        }
        if kind == Vocalization::Sigh {
            let inward = r.flow.iter().copied().fold(0.0f32, f32::min);
            let outward = r.flow.iter().copied().fold(0.0f32, f32::max);
            let runs = vocal::voiced_runs(&r, 100.0);
            if let Some(&(a, b)) = runs.first() {
                let q = (b - a) / 4;
                let hz = median(&r.pitch[a..b]);
                let mut modal = voice(&deep());
                modal.articulate_phoneme(synthie_trombone::Phoneme::Aa);
                modal.set_frequency(100.0);
                modal.note_on();
                let m = record(&mut modal, 1.0);
                println!(
                    "      a modal /a/ at 100 Hz: noise share {:.2}",
                    noise_share(&m.out[(0.3 * SR) as usize..], 100.0)
                );
                println!(
                    "      flow {inward:.0}..{outward:.0} cm3/s; voice {:.2}-{:.2} s, pitch {:.0} -> {:.0} Hz, noise share {:.2}",
                    a as f32 / SR,
                    b as f32 / SR,
                    median(&r.pitch[a..a + q]),
                    median(&r.pitch[b - q..b]),
                    noise_share(&r.out[a..b], hz)
                );
            }
        }
        if let Some(d) = &dir {
            let path = format!("{d}/vocal-{}.wav", kind.name());
            let mut out = r.out.clone();
            synthie_trombone::normalize(&mut out, 0.89);
            wav::write(&path, SR as u32, &out).expect("write");
        }
        let _ = rms;
    }
}
