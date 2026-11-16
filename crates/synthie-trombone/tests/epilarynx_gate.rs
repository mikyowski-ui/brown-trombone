//! The epilaryngeal tube against the 2.8 to 3.2 kHz cluster (physical
//! engine; `AnatomyConfig::epilarynx_*`). For every vowel shape of the
//! table, closed glottis: the resonances of the tract's transfer function,
//! the share of each one's acoustic energy that the tube holds
//! (`Trombone::energy_share`), and the level of the 2.8 to 3.2 kHz band
//! against 0 to 1 kHz. `examples/epilarynx.rs` prints the same per shape.
//!
//! The gate is for the setting it is meant for: with the tube that gives
//! the cluster ([`CLUSTER`]) every shape must have a resonance between 2.85 and 3.3
//! kHz that holds at least half its energy in the tube (its F3 in the front
//! vowels, F4 in the others), and the band must be lifted; the original's
//! tube ([`ORIGINAL`], today's default) is held to what it does, ringing
//! above the cluster. The vowel shapes are the table's, solved for the
//! default tube.
//!
//! ```sh
//! cargo test -p synthie-trombone --release --test epilarynx_gate -- --nocapture
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use synthie_trombone::vowel_targets::TARGETS;
use synthie_trombone::{Engine, Trombone, Tuning};

/// The tube that gives the cluster: eight cells (2.92 cm), 0.25 cm²
/// (diameter 0.45), opening in a step into 3.6 cm² (diameter 1.7), not
/// tied to the folds.
const CLUSTER: [(&str, f32); 5] = [
    ("epilarynx_cells", 8.0),
    ("epilarynx_diameter", 0.45),
    ("epilarynx_flare", 1.7),
    ("epilarynx_flare_cells", 0.0),
    ("epilarynx_effort", 0.0),
];

/// The original's tube: seven cells of 0.45 cm² (diameter 0.6), a step
/// to 1.52 cm² (1.1); today's defaults, set by name.
const ORIGINAL: [(&str, f32); 5] = [
    ("epilarynx_cells", 7.0),
    ("epilarynx_diameter", 0.6),
    ("epilarynx_flare", 1.1),
    ("epilarynx_flare_cells", 0.0),
    ("epilarynx_effort", 0.0),
];

fn tuning(fields: &[(&str, f32)]) -> Tuning {
    let mut t = Tuning::DEFAULT;
    for &(name, value) in fields {
        assert!(t.set(name, value), "{name}");
    }
    t
}

/// Spectrum step, Hz.
const STEP: f32 = 4.0;

/// One vowel shape: its resonances (Hz, bandwidth Hz, share of their
/// energy in the tube) up to 5.5 kHz, and the band's level, dB.
struct Shape {
    name: String,
    resonances: Vec<(f32, f32, f32)>,
    band: f32,
}

/// Peaks of `mag` above 150 Hz: the largest within ±60 Hz, down to 60 dB
/// under the tallest, with their half-power widths.
fn peaks(mag: &[f32]) -> Vec<(f32, f32)> {
    let reach = (60.0 / STEP) as usize;
    let first = (150.0 / STEP) as usize;
    let n = mag.len();
    let tallest = mag[first..].iter().fold(0.0f32, |m, &v| m.max(v));
    let mut out: Vec<(f32, f32)> = Vec::new();
    for k in first.max(reach)..n.saturating_sub(reach) {
        let window = &mag[k - reach..=k + reach];
        if mag[k] < 1e-3 * tallest || window.iter().any(|&v| v > mag[k]) {
            continue;
        }
        let hz = k as f32 * STEP;
        if out.last().is_some_and(|r| hz - r.0 < 2.0 * STEP) {
            continue;
        }
        let half = mag[k] * std::f32::consts::FRAC_1_SQRT_2;
        let (mut lo, mut hi) = (k, k);
        while lo > 0 && mag[lo] > half {
            lo -= 1;
        }
        while hi < n - 1 && mag[hi] > half {
            hi += 1;
        }
        out.push((hz, (hi - lo) as f32 * STEP));
    }
    out
}

fn band(mag: &[f32], lo: f32, hi: f32) -> f32 {
    let (a, b) = ((lo / STEP) as usize, (hi / STEP) as usize);
    let p: f32 = mag[a..b].iter().map(|m| m * m).sum::<f32>() / (b - a) as f32;
    10.0 * p.max(1e-30).log10()
}

fn measure(tuning: &Tuning, key: usize) -> Shape {
    let target = &TARGETS[key];
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    t.set_tuning(tuning);
    t.set_tenseness(0.65);
    t.articulate(&target.key.articulation().expect("a vowel shape"));
    t.settle();
    let rate = t.tick_rate();
    let mut x = vec![0.0f32; (0.2 * rate) as usize];
    t.impulse_response(&mut x);
    let mag = spectrum::spectrum(&x, rate, 5500.0, STEP);
    let tube = tuning.anatomy.epilarynx_cells.ceil() as usize;
    let resonances = peaks(&mag)
        .into_iter()
        .map(|(hz, bw)| (hz, bw, t.energy_share(hz, 0..tube)))
        .collect();
    Shape {
        name: format!("{}", target.key),
        resonances,
        band: band(&mag, 2800.0, 3200.0) - band(&mag, 0.0, 1000.0),
    }
}

fn survey(tuning: &Tuning) -> Vec<Shape> {
    let keys: Vec<usize> = (0..TARGETS.len()).collect();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = keys.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = keys
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || part.iter().map(|&k| measure(tuning, k)).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    })
}

/// The resonance of the tube in `lo..=hi` Hz: one held sharply (bandwidth
/// under 400 Hz) with at least half its energy in the tube, the most so.
fn rings(shape: &Shape, lo: f32, hi: f32) -> Option<f32> {
    shape
        .resonances
        .iter()
        .filter(|&&(hz, bw, share)| (lo..=hi).contains(&hz) && bw < 400.0 && share >= 0.5)
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|r| r.0)
}

/// The tube's own resonance from 2.4 kHz on.
fn own(shape: &Shape) -> Option<f32> {
    rings(shape, 2400.0, 5500.0)
}

fn report(name: &str, shapes: &[Shape]) -> f32 {
    println!("{name}: the tube's own resonance and the 2.8-3.2 kHz band against 0-1 kHz");
    for s in shapes {
        println!(
            "  {:<6} {:>6.0} Hz  {:+6.1} dB",
            s.name,
            own(s).unwrap_or(f32::NAN),
            s.band
        );
    }
    let mean = shapes.iter().map(|s| s.band).sum::<f32>() / shapes.len() as f32;
    println!("  mean band {mean:+.1} dB");
    mean
}

#[test]
fn the_cluster_tube_rings_in_the_cluster() {
    let shapes = survey(&tuning(&CLUSTER));
    let mean = report("cluster", &shapes);
    for s in &shapes {
        assert!(
            rings(s, 2850.0, 3300.0).is_some(),
            "{}: no resonance of the tube at 2.85 to 3.3 kHz: {:?}",
            s.name,
            s.resonances
        );
    }
    // Measured -0.2 dB; the original's tube -8.2.
    assert!(mean > -2.0, "band {mean} dB");
}

#[test]
fn the_original_tube_rings_above_the_cluster() {
    let shapes = survey(&tuning(&ORIGINAL));
    let mean = report("original", &shapes);
    // Its own resonance (F4 in most shapes) at 3.3 to 3.9 kHz.
    let at: Vec<f32> = shapes.iter().filter_map(own).collect();
    let above = at.iter().filter(|&&hz| hz > 3300.0).count();
    assert!(
        above * 4 >= shapes.len() * 3,
        "{above} of {} above 3.3 kHz: {at:?}",
        shapes.len()
    );
    assert!(mean < -6.0, "band {mean} dB");
}
