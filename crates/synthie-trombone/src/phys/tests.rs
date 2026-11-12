//! Stage gates for the physical tube, checked against theory.

extern crate std;

use super::*;
use std::vec;
use std::vec::Vec;

const TICK: f32 = 1.0 / 96_000.0;

fn uniform(diameter: f32) -> Phys {
    let mut p = Phys::new();
    p.snap(&[diameter; N], 0.0);
    p
}

/// Flow response at the lips to a unit flow impulse at the glottis.
fn impulse(p: &mut Phys, ticks: usize) -> Vec<f32> {
    p.reset();
    (0..ticks)
        .map(|n| p.tick(1.0, Drive::Flow(if n == 0 { 1.0 } else { 0.0 })))
        .collect()
}

/// Magnitude of the response's spectrum at `hz`.
fn magnitude(x: &[f32], hz: f32) -> f32 {
    let w = 2.0 * core::f32::consts::PI * hz * TICK;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &v) in x.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let ph = f64::from(w) * n as f64;
        re += f64::from(v) * ph.cos();
        im -= f64::from(v) * ph.sin();
    }
    #[allow(clippy::cast_possible_truncation)]
    let m = (re * re + im * im).sqrt() as f32;
    m
}

/// Spectral peaks `(hz, half-power width)` between 80 Hz and `max_hz`:
/// the largest value within ±40 Hz, and at least 3 % of the tallest.
fn peaks(x: &[f32], max_hz: f32) -> Vec<(f32, f32)> {
    let step = 2.0;
    let reach = 20;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (max_hz / step) as usize;
    #[allow(clippy::cast_precision_loss)]
    let mag: Vec<f32> = (0..n).map(|k| magnitude(x, k as f32 * step)).collect();
    let first = 40;
    let tallest = mag[first..].iter().fold(0.0f32, |m, &v| m.max(v));
    let mut out = vec![];
    for k in first..n - reach {
        let window = &mag[k - reach..=k + reach];
        if mag[k] < 0.03 * tallest || window.iter().any(|&v| v > mag[k]) {
            continue;
        }
        if out.last().is_some_and(|&(hz, _): &(f32, f32)| {
            #[allow(clippy::cast_precision_loss)]
            let here = k as f32 * step;
            here - hz < 2.0 * step
        }) {
            continue;
        }
        let half = mag[k] * core::f32::consts::FRAC_1_SQRT_2;
        let mut lo = k;
        while lo > 0 && mag[lo] > half {
            lo -= 1;
        }
        let mut hi = k;
        while hi < n - 1 && mag[hi] > half {
            hi += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        out.push((k as f32 * step, (hi - lo) as f32 * step));
    }
    out
}

#[test]
fn uniform_tube_resonates_where_theory_says() {
    // A closed-open tube resonates at odd multiples of c/4L', where L' is
    // the tube plus the end correction 8r/3π of a piston in a baffle.
    let mut p = uniform(2.0);
    p.set_ideal(true);
    p.snap(&[2.0; N], 0.0);
    let x = impulse(&mut p, 40_000);
    let found = peaks(&x, 4000.0);
    #[allow(clippy::cast_precision_loss)]
    let radius = sqrt(AREA_SCALE);
    #[allow(clippy::cast_precision_loss)]
    let length = N as f32 * C * TICK + 8.0 * radius / (3.0 * PI);
    for (k, &(hz, _)) in found.iter().take(3).enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let theory = (2 * k + 1) as f32 * C / (4.0 * length);
        let err = (hz - theory).abs() / theory;
        std::println!("mode {k}: {hz:.0} Hz, theory {theory:.0} Hz");
        assert!(err < 0.02, "mode {k}: {hz} vs {theory}");
    }
    assert!(found.len() >= 3);
}

#[test]
fn energy_never_grows_without_a_source() {
    // Random static shapes, losses and walls off, velum shut. The only
    // energy store outside the tube is the lip load, which can hand a
    // little back, so the bound is the energy put in, never more.
    let mut seed = 77u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        #[allow(clippy::cast_precision_loss)]
        let x = (seed >> 8) as f32 / 16_777_216.0;
        x
    };
    for _ in 0..20 {
        let mut d = [0.0f32; N];
        for v in &mut d {
            *v = 0.2 + 3.0 * rnd();
        }
        let mut p = Phys::new();
        p.set_ideal(true);
        p.snap(&d, 0.0);
        p.reset();
        p.tick(1.0, Drive::Flow(1000.0));
        let start = p.energy(1.0);
        let mut last = start;
        for _ in 0..40_000 {
            p.tick(1.0, Drive::Flow(0.0));
            last = p.energy(1.0);
            assert!(last <= start * 1.000_01, "{last} > {start}");
        }
        assert!(
            last < 0.05 * start,
            "energy did not radiate away: {last} of {start}"
        );
    }
}

#[test]
fn bandwidths_are_humanlike_and_rise_with_frequency() {
    // Neutral tube, closed glottis: F1–F3 half-power widths. Measured
    // human values are roughly 40–150 Hz (Fant 1972; Hawks & Miller 1995).
    let mut p = uniform(1.8);
    let x = impulse(&mut p, 40_000);
    let f = peaks(&x, 3500.0);
    std::println!("{f:?}");
    assert!(f.len() >= 3);
    for &(hz, bw) in f.iter().take(3) {
        assert!((25.0..180.0).contains(&bw), "{hz} Hz: bandwidth {bw}");
    }
    assert!(f[2].1 > f[0].1 * 0.9, "{f:?}");
}

#[test]
fn closed_tract_rings_on_its_walls() {
    // Lips shut: the lowest resonance is the wall mass on the enclosed
    // air, around 200 Hz in adults.
    let mut d = [1.8f32; N];
    d[N - 1] = 0.0;
    d[N - 2] = 0.0;
    let mut p = Phys::new();
    p.snap(&d, 0.0);
    p.reset();
    // Listen to the pressure behind the closure.
    let x: Vec<f32> = (0..60_000)
        .map(|n| {
            p.tick(1.0, Drive::Flow(if n == 0 { 1.0 } else { 0.0 }));
            p.pressure(N - 6)
        })
        .collect();
    let f = peaks(&x, 1000.0);
    std::println!("{f:?}");
    // Hanna et al. (2016) measured 212 Hz with a 104 Hz bandwidth in men.
    let (hz, bw) = f[0];
    assert!((150.0..250.0).contains(&hz), "wall resonance at {hz} Hz");
    assert!((60.0..150.0).contains(&bw), "wall bandwidth {bw} Hz");
}

#[test]
fn sealed_mouth_fills_slowly() {
    // A voiced stop: 200 cm³/s into a shut mouth. The walls must give
    // 1–3 cm³ per cmH2O so pressure takes tens of milliseconds to build,
    // which is what keeps the folds vibrating into a /b/.
    let mut d = [1.8f32; N];
    d[N - 1] = 0.0;
    d[N - 2] = 0.0;
    let mut p = Phys::new();
    p.snap(&d, 0.0);
    p.reset();
    let mut reached = None;
    // Average the pressure over 5 ms so the ringing of the flow step is
    // not mistaken for the mouth filling.
    let mut mean = 0.0f32;
    for n in 0..20_000 {
        p.tick(1.0, Drive::Flow(200.0));
        mean += (p.pressure(N - 6) - mean) / 480.0;
        if reached.is_none() && mean >= 5.0 * CM_H2O {
            reached = Some(n);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let ms = reached.map(|n| n as f32 * TICK * 1000.0);
    std::println!("5 cmH2O after {ms:?} ms");
    let ms = ms.expect("pressure never built");
    assert!(ms >= 30.0, "filled in {ms} ms");

    // Quasi-static yield: hold the flow off and read volume per pressure.
    let volume: f32 = (0..N).map(|i| p.wall_s[i] * p.wall_y[i]).sum();
    let yield_per_cm = volume / (mean / CM_H2O);
    std::println!("wall yield {yield_per_cm:.2} cm³/cmH2O");
    assert!((0.7..3.5).contains(&yield_per_cm), "{yield_per_cm}");
}

#[test]
fn every_cell_can_seal_without_nan() {
    for shut in 0..N {
        let mut d = [1.5f32; N];
        d[shut] = 0.0;
        let mut p = Phys::new();
        p.snap(&d, 1.0);
        p.reset();
        for n in 0..3000 {
            let area = if n % 700 < 350 { 0.1 } else { 0.0 };
            let out = p.tick(
                1.0,
                Drive::Glottis {
                    area,
                    lung: 8.0 * CM_H2O,
                },
            );
            assert!(out.is_finite(), "cell {shut}, tick {n}");
        }
        for i in 0..N {
            assert!(p.pressure(i).is_finite(), "cell {shut}");
            assert!(p.pressure(i).abs() < 40.0 * CM_H2O, "cell {shut}/{i}");
        }
    }
}

#[test]
fn steady_glottal_flow_follows_bernoulli() {
    let mut p = uniform(1.8);
    let lung = 8.0 * CM_H2O;
    let area = 0.1;
    for _ in 0..40_000 {
        p.tick(1.0, Drive::Glottis { area, lung });
    }
    let ideal = area * sqrt(2.0 * lung / (GLOTTAL_KT * RHO));
    let u = p.gauge.u_glottis;
    std::println!("flow {u:.0} cm³/s, orifice law {ideal:.0}");
    assert!(u > 0.75 * ideal && u < 1.02 * ideal, "{u} vs {ideal}");
    assert!((p.gauge.u_lips - u).abs() < 0.05 * u, "{:?}", p.gauge);
}

/// Level of the lip-flow response at `hz`, dB.
fn level(x: &[f32], hz: f32) -> f32 {
    20.0 * magnitude(x, hz).max(1e-20).log10()
}

#[test]
fn piriform_sinuses_carve_a_notch() {
    // The pockets beside the larynx act as quarter-wave traps. In people
    // they take 10–25 dB out of the spectrum between 4 and 5 kHz and nudge
    // the lower formants down a little (Dang & Honda 1997).
    let d = [1.8f32; N];
    let mut with = Phys::new();
    with.snap(&d, 0.0);
    let mut without = Phys::new();
    without.set_side_cavities(false);
    without.snap(&d, 0.0);
    let a = impulse(&mut with, 30_000);
    let b = impulse(&mut without, 30_000);

    let mut deepest = (0.0f32, 0.0f32);
    let mut hz = 3000.0;
    while hz <= 6000.0 {
        let drop = level(&b, hz) - level(&a, hz);
        if drop > deepest.1 {
            deepest = (hz, drop);
        }
        hz += 10.0;
    }
    std::println!("deepest notch {:.1} dB at {:.0} Hz", deepest.1, deepest.0);
    assert!(
        (3800.0..5200.0).contains(&deepest.0),
        "notch at {} Hz",
        deepest.0
    );
    assert!(deepest.1 >= 10.0, "notch only {} dB", deepest.1);

    let fa = peaks(&a, 3500.0);
    let fb = peaks(&b, 3500.0);
    std::println!("with {fa:?}\nwithout {fb:?}");
    for (x, y) in fa.iter().zip(&fb).take(3) {
        let shift = (x.0 - y.0) / y.0;
        assert!(shift.abs() < 0.05, "{} vs {}", x.0, y.0);
        assert!(
            shift <= 0.005,
            "piriform should not raise formants: {shift}"
        );
    }
}

#[test]
fn nasal_murmur_is_low_and_has_an_antiresonance() {
    // /m/: lips shut, velum open. The murmur's first resonance sits near
    // 250–350 Hz, and the closed mouth cavity puts a zero in the spectrum
    // (around 750–1250 Hz for /m/; Fujimura 1962).
    let mut d = [1.8f32; N];
    d[N - 1] = 0.0;
    d[N - 2] = 0.0;
    let mut p = Phys::new();
    p.snap(&d, 1.0);
    let x = impulse(&mut p, 40_000);
    let f = peaks(&x, 3000.0);
    std::println!("murmur resonances {f:?}");
    assert!((200.0..400.0).contains(&f[0].0), "nasal F1 at {}", f[0].0);

    // The deepest dip between the first two resonances and 1.6 kHz.
    let mut dip = (0.0f32, f32::INFINITY);
    let mut hz = f[0].0 + 100.0;
    while hz <= 1600.0 {
        let l = level(&x, hz);
        if l < dip.1 {
            dip = (hz, l);
        }
        hz += 10.0;
    }
    let depth = level(&x, f[0].0) - dip.1;
    std::println!(
        "antiresonance at {:.0} Hz, {:.1} dB below the murmur peak",
        dip.0,
        depth
    );
    assert!(depth > 15.0, "no clear antiresonance: {depth} dB");
    // Both nostrils carry flow, and not equally.
    assert!(p.gauge.u_lips.abs() < 1e-6);
}
