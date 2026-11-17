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
    let strength: f32 = std::env::var("STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
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
        if kind == Vocalization::Sigh {
            let inward = r.flow.iter().copied().fold(0.0f32, f32::min);
            let outward = r.flow.iter().copied().fold(0.0f32, f32::max);
            let runs = vocal::voiced_runs(&r, 100.0);
            if let Some(&(a, b)) = runs.first() {
                let q = (b - a) / 4;
                let hz = median(&r.pitch[a..b]);
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
