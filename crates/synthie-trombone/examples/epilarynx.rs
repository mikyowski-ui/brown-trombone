//! The epilaryngeal tube against the 2.8 to 3.2 kHz cluster: for every
//! vowel shape of the table (`vowel_targets::TARGETS`) on the physical
//! engine, F1 to F5 of the tract's transfer function (closed glottis), the
//! bandwidths of F3 to F5, the share of each one's acoustic energy that the
//! tube holds (which resonance is the tube's own), and the level of the
//! transfer function's 2.8 to 3.2 kHz band against its 0 to 1 kHz band.
//!
//! ```sh
//! cargo run -p synthie-trombone --example epilarynx --release
//! cargo run -p synthie-trombone --example epilarynx --release -- --set epilarynx_cells=8
//! cargo run -p synthie-trombone --example epilarynx --release -- --sweep
//! ```
//!
//! `--set name=value` sets any `Tuning` field (repeatable); `--length L`
//! the tract length (default 1, the `deep` voice); `--sweep` prints one
//! summary line per tube (lengths 4 to 8 cells, areas 0.2 to 0.8 cm²,
//! flare-to-tube area ratios 2 to 8) instead of the table.
//!
//! A cluster is two neighbouring formants among F3, F4 and F5 less than
//! 500 Hz apart, both between 2.5 and 3.5 kHz, their mean within 2.8 to
//! 3.2 kHz (scaled by `1 / length`).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use synthie_trombone::vowel_targets::TARGETS;
use synthie_trombone::{Engine, GlottalSource, Phoneme, Trombone, Tuning};

/// Spectrum step, Hz.
const STEP: f32 = 4.0;
/// Highest frequency analysed, Hz (native size).
const MAX_HZ: f32 = 7000.0;
/// Area of a cell of articulation diameter 1 at the native size, cm².
const UNIT_AREA: f32 = 1.6 * std::f32::consts::FRAC_PI_4;

/// What one vowel shape measured.
#[derive(Clone, Copy, Debug, Default)]
struct Row {
    /// F1..F5 and their half-power bandwidths, Hz.
    f: [f32; 5],
    bw: [f32; 5],
    /// Share of the energy of F3, F4, F5 held in the tube.
    share: [f32; 3],
    /// 2.8-3.2 kHz band against 0-1 kHz, dB (mean power per Hz).
    band: f32,
    /// Gap of the cluster near 3 kHz, Hz, if there is one.
    cluster: Option<f32>,
}

/// Peaks of `mag` above 150 Hz: the largest value within ±60 Hz, down to
/// 60 dB under the tallest, each refined by a parabola, with its
/// half-power width.
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
        let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
        let hz = hz + STEP * 0.5 * (a - c) / (a - 2.0 * b + c);
        let half = mag[k] * std::f32::consts::FRAC_1_SQRT_2;
        let mut lo = k;
        while lo > 0 && mag[lo] > half {
            lo -= 1;
        }
        let mut hi = k;
        while hi < n - 1 && mag[hi] > half {
            hi += 1;
        }
        out.push((hz, (hi - lo) as f32 * STEP));
    }
    out
}

/// Mean power per Hz of `mag` from `lo` to `hi` Hz, dB.
fn band(mag: &[f32], lo: f32, hi: f32) -> f32 {
    let (a, b) = ((lo / STEP) as usize, (hi / STEP) as usize);
    let p: f32 = mag[a..b].iter().map(|m| m * m).sum::<f32>() / (b - a) as f32;
    10.0 * p.max(1e-30).log10()
}

fn measure(tuning: &Tuning, length: f32, tenseness: f32, key: usize) -> Row {
    let target = &TARGETS[key];
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    t.set_tuning(tuning);
    t.set_tract_length(length);
    t.set_tenseness(tenseness);
    t.articulate(&target.key.articulation().expect("a vowel shape"));
    t.settle();
    let rate = t.tick_rate();
    let mut x = vec![0.0f32; (0.2 * rate) as usize];
    t.impulse_response(&mut x);
    let scale = 1.0 / length;
    let mag = spectrum::spectrum(&x, rate, MAX_HZ * scale, STEP);
    let found = peaks(&mag);
    let mut row = Row::default();
    for (k, &(hz, bw)) in found.iter().take(5).enumerate() {
        row.f[k] = hz;
        row.bw[k] = bw;
    }
    row.band = band(&mag, 2800.0 * scale, 3200.0 * scale) - band(&mag, 0.0, 1000.0 * scale);
    let top = tuning.anatomy.epilarynx_cells.ceil() as usize;
    for k in 0..3 {
        if row.f[k + 2] > 0.0 {
            row.share[k] = t.energy_share(row.f[k + 2], 0..top);
        }
    }
    for k in 2..4 {
        let (a, b) = (row.f[k], row.f[k + 1]);
        let mean = f32::midpoint(a, b);
        let inside = |f: f32| (2500.0 * scale..=3500.0 * scale).contains(&f);
        if b > 0.0
            && b - a < 500.0 * scale
            && inside(a)
            && inside(b)
            && (2800.0 * scale..=3200.0 * scale).contains(&mean)
        {
            row.cluster = Some(row.cluster.map_or(b - a, |g: f32| g.min(b - a)));
        }
    }
    row
}

/// Every vowel shape, on all threads.
fn measure_all(tuning: &Tuning, length: f32, tenseness: f32) -> Vec<Row> {
    let keys: Vec<usize> = (0..TARGETS.len()).collect();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = keys.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = keys
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|&k| measure(tuning, length, tenseness, k))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    })
}

/// One line: means over the vowels.
fn summary(rows: &[Row]) -> String {
    let n = rows.len() as f32;
    let mean = |f: &dyn Fn(&Row) -> f32| rows.iter().map(f).sum::<f32>() / n;
    let clusters = rows.iter().filter(|r| r.cluster.is_some()).count();
    // Which of F3..F5 holds most of the tube's energy, most often.
    let mut tube = [0usize; 3];
    for r in rows {
        let k = (0..3)
            .max_by(|&a, &b| r.share[a].total_cmp(&r.share[b]))
            .unwrap_or(0);
        tube[k] += 1;
    }
    format!(
        "F3 {:5.0}/{:3.0}  F4 {:5.0}/{:3.0}  F5 {:5.0}/{:3.0}  F4-F3 {:4.0}  F5-F4 {:4.0}  band {:+5.1} dB  cluster {:2}/{}  tube's own F3/F4/F5 {}/{}/{}  shares {:.2}/{:.2}/{:.2}",
        mean(&|r| r.f[2]),
        mean(&|r| r.bw[2]),
        mean(&|r| r.f[3]),
        mean(&|r| r.bw[3]),
        mean(&|r| r.f[4]),
        mean(&|r| r.bw[4]),
        mean(&|r| r.f[3] - r.f[2]),
        mean(&|r| r.f[4] - r.f[3]),
        mean(&|r| r.band),
        clusters,
        rows.len(),
        tube[0],
        tube[1],
        tube[2],
        mean(&|r| r.share[0]),
        mean(&|r| r.share[1]),
        mean(&|r| r.share[2]),
    )
}

/// Output rate of the coupling measurements, Hz.
const SR: f32 = 48_000.0;

/// A voice on the folds holding `vowel` at `hz`, settled, about to start.
fn held(tuning: &Tuning, vowel: Phoneme, hz: f32, tenseness: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_tuning(tuning);
    t.set_engine(Engine::Physical);
    t.set_glottal_source(GlottalSource::Folds);
    t.set_tenseness(tenseness);
    t.articulate_phoneme(vowel);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(hz);
    t
}

/// `seconds` of output and glottal flow.
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

/// Instants (samples) at which the glottal flow rises through the middle of
/// its swing, having fallen to a quarter of it in between.
fn rises(flows: &[f32]) -> Vec<f32> {
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let (hi, lo) = (floor + 0.5 * (peak - floor), floor + 0.25 * (peak - floor));
    let mut out = Vec::new();
    let mut armed = false;
    for k in 1..flows.len() {
        if flows[k] < lo {
            armed = true;
        }
        if armed && flows[k - 1] <= hi && flows[k] > hi {
            out.push(k as f32 - 1.0 + (hi - flows[k - 1]) / (flows[k] - flows[k - 1]));
            armed = false;
        }
    }
    out
}

/// Magnitude of `x` at `hz` (Hann window).
fn tone(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(SR);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (k, &v) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
        let v = f64::from(v) * win;
        re += v * (w * k as f64).cos();
        im += v * (w * k as f64).sin();
    }
    (re * re + im * im).sqrt() as f32
}

/// Energy of `x` from `lo` to `hi` Hz, in 20 Hz steps (only ratios of it
/// are printed).
fn energy(x: &[f32], lo: f32, hi: f32) -> f32 {
    let step = 20.0;
    let mag = spectrum::spectrum(x, SR, hi, step);
    mag[(lo / step) as usize..].iter().map(|m| m * m).sum()
}

/// What the folds do on one held vowel: onset ms (the flow's swing within
/// 10 % of its steady swing), pitch, period jitter %, cycle-to-cycle
/// alternation of the flow's peaks %, subharmonics dB (energy half-way
/// between harmonics against at them, harmonics 1 to 10), open quotient,
/// H1-H2 of the sound, and its 1-2, 2-4, 4-7.5 kHz bands against 0-1 kHz.
fn held_report(tuning: &Tuning, vowel: Phoneme, hz: f32, tenseness: f32) -> [f32; 10] {
    let mut t = held(tuning, vowel, hz, tenseness);
    t.note_on();
    let (_, start) = run(&mut t, 0.4);
    let (out, flows) = run(&mut t, 0.5);
    let c = rises(&flows);
    if c.len() < 6 {
        return [f32::NAN; 10];
    }
    let periods: Vec<f32> = c.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = periods.iter().sum::<f32>() / periods.len() as f32;
    let f0 = SR / mean;
    let jitter = 100.0
        * (periods.iter().map(|p| (p - mean) * (p - mean)).sum::<f32>() / periods.len() as f32)
            .sqrt()
        / mean;
    let peaks: Vec<f32> = c
        .windows(2)
        .map(|w| {
            flows[w[0] as usize..w[1] as usize]
                .iter()
                .fold(f32::MIN, |m, &v| m.max(v))
        })
        .collect();
    let alternation = 100.0 * peaks.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>()
        / (peaks.len() - 1) as f32
        / (peaks.iter().sum::<f32>() / peaks.len() as f32);
    let (mut sub, mut harm) = (0.0f32, 0.0f32);
    for k in 1..=10 {
        harm += tone(&out, k as f32 * f0).powi(2);
        sub += tone(&out, (k as f32 - 0.5) * f0).powi(2);
    }
    let subharmonics = 10.0 * (sub / harm).log10();
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let swing = peak - floor;
    let open =
        flows.iter().filter(|&&v| v > floor + 0.1 * swing).count() as f32 / flows.len() as f32;
    let h12 = 20.0 * (tone(&out, f0) / tone(&out, 2.0 * f0)).log10();
    // Onset: the first period-long window whose swing is within 10 % of the
    // steady one.
    let window = mean as usize;
    let mut onset = f32::NAN;
    for k in (0..start.len().saturating_sub(window)).step_by(window / 4 + 1) {
        let w = &start[k..k + window];
        let s =
            w.iter().fold(f32::MIN, |m, &v| m.max(v)) - w.iter().fold(f32::MAX, |m, &v| m.min(v));
        if s >= 0.9 * swing {
            onset = 1000.0 * (k + window) as f32 / SR;
            break;
        }
    }
    let base = energy(&out, 50.0, 1000.0);
    let band = |lo: f32, hi: f32| 10.0 * (energy(&out, lo, hi) / base).log10();
    [
        onset,
        f0,
        jitter,
        alternation,
        subharmonics,
        open,
        h12,
        band(1000.0, 2000.0),
        band(2000.0, 4000.0),
        band(4000.0, 7500.0),
    ]
}

/// Lowest lung pressure, cmH2O, at which the folds keep vibrating on a held
/// /ɑ/ at 110 Hz (bisection, as `examples/folds.rs`).
fn threshold(tuning: &Tuning, tenseness: f32) -> f32 {
    let vibrating = |pressure: f32| {
        let mut t = held(tuning, Phoneme::Aa, 110.0, tenseness);
        t.note_on();
        t.set_lung_pressure(pressure);
        run(&mut t, 0.8);
        let (_, f) = run(&mut t, 0.3);
        let peak = f.iter().fold(0.0f32, |m, &v| m.max(v));
        let floor = f.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        peak - floor > 0.5 * peak
    };
    let (mut lo, mut hi) = (1.0f32, 8.0f32);
    for _ in 0..9 {
        let mid = f32::midpoint(lo, hi);
        if vibrating(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn coupling(tuning: &Tuning, tenseness: f32) {
    println!(
        "vowel   Hz  onset ms  pitch  jitter %  altern. %  subharm dB   OQ   H1-H2  1-2 kHz  2-4 kHz  4-7.5 kHz (dB re 0-1 kHz)"
    );
    let jobs: Vec<(Phoneme, f32)> = [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw]
        .into_iter()
        .flat_map(|v| [90.0f32, 120.0, 200.0].map(|hz| (v, hz)))
        .collect();
    let rows: Vec<[f32; 10]> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(v, hz)| s.spawn(move || held_report(tuning, v, hz, tenseness)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    for (&(v, hz), r) in jobs.iter().zip(&rows) {
        println!(
            "{:<5} {hz:4.0}  {:8.1}  {:5.1}  {:8.2}  {:9.2}  {:10.1}  {:4.2}  {:+6.1}  {:+7.1}  {:+7.1}  {:+9.1}",
            v.symbol(),
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9]
        );
    }
    println!(
        "phonation threshold (/ɑ/, 110 Hz): {:.2} cmH2O",
        threshold(tuning, tenseness)
    );
}

fn main() {
    let mut tuning = Tuning::DEFAULT;
    let mut length = 1.0f32;
    let mut sweep = false;
    let mut couple = false;
    let mut tenseness = 0.65f32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--set" => {
                let kv = args.next().expect("--set name=value");
                let (k, v) = kv.split_once('=').expect("--set name=value");
                assert!(tuning.set(k, v.parse().expect("a number")), "no field {k}");
            }
            "--length" => length = args.next().expect("--length L").parse().expect("a number"),
            "--tenseness" => {
                tenseness = args
                    .next()
                    .expect("--tenseness T")
                    .parse()
                    .expect("a number");
            }
            "--sweep" => sweep = true,
            "--coupling" => couple = true,
            _ => panic!("unknown argument {a}"),
        }
    }
    let tuning = tuning.clamped();
    if couple {
        coupling(&tuning, tenseness);
        return;
    }
    if sweep {
        println!("tube: cells, area cm², flare area cm² (ratio)");
        for cells in [4.0f32, 5.0, 6.0, 7.0, 8.0] {
            for area in [0.2f32, 0.35, 0.5, 0.8] {
                for ratio in [2.0f32, 3.4, 5.0, 8.0] {
                    let mut t = tuning;
                    t.anatomy.epilarynx_cells = cells;
                    t.anatomy.epilarynx_diameter = (area / UNIT_AREA).sqrt();
                    t.anatomy.epilarynx_flare = (ratio * area / UNIT_AREA).sqrt();
                    let t = t.clamped();
                    let a = UNIT_AREA * t.anatomy.epilarynx_diameter.powi(2);
                    let af = UNIT_AREA * t.anatomy.epilarynx_flare.powi(2);
                    let rows = measure_all(&t, length, tenseness);
                    println!(
                        "{cells:.0} {a:.2} {af:4.2} ({:3.1})  {}",
                        af / a,
                        summary(&rows)
                    );
                }
            }
        }
        return;
    }
    let a = &tuning.anatomy;
    println!(
        "tube {:.2} cells ({:.2} cm), {:.3} cm², flare {:.2} cm² over {} cells, effort {}; length {length}",
        a.epilarynx_cells,
        a.epilarynx_cells * 0.3645 * length,
        UNIT_AREA * a.epilarynx_diameter.powi(2),
        UNIT_AREA * a.epilarynx_flare.powi(2),
        a.epilarynx_flare_cells,
        a.epilarynx_effort,
    );
    println!("shape     F1    F2    F3/bw  share   F4/bw  share   F5/bw  share   3k band  cluster");
    let rows = measure_all(&tuning, length, tenseness);
    for (target, r) in TARGETS.iter().zip(&rows) {
        println!(
            "{:<8} {:5.0} {:5.0} {:5.0}/{:<4.0} {:.2}  {:5.0}/{:<4.0} {:.2}  {:5.0}/{:<4.0} {:.2}  {:+5.1}   {}",
            format!("{}", target.key),
            r.f[0],
            r.f[1],
            r.f[2],
            r.bw[2],
            r.share[0],
            r.f[3],
            r.bw[3],
            r.share[1],
            r.f[4],
            r.bw[4],
            r.share[2],
            r.band,
            r.cluster
                .map_or(String::from("-"), |g| format!("{g:.0} Hz")),
        );
    }
    println!("mean: {}", summary(&rows));
}
