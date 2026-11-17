//! Stage gates for the physical tube, checked against theory.

#![allow(clippy::many_single_char_names)]

extern crate std;

use super::*;
use std::vec;
use std::vec::Vec;

const TICK: f32 = 1.0 / 96_000.0;

fn uniform(diameter: f32) -> Phys {
    let mut p = original();
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
        let mut p = original();
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
    let mut p = original();
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
    let mut p = original();
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
        let mut p = original();
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
    let mut with = original();
    with.snap(&d, 0.0);
    let mut without = original();
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
    let mut p = original();
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

// ----- the tract's length (wave 2A) ----------------------------------------

/// A tract with the lip horn and the larynx slide as given
/// (`AnatomyConfig::lip_horn`, `AnatomyConfig::larynx_travel`).
fn with_slides(horn: f32, travel: f32) -> Phys {
    let mut t = crate::Tuning::DEFAULT.with_tract(crate::TractModel::Original);
    t.anatomy.lip_horn = horn;
    t.anatomy.larynx_travel = travel;
    let mut p = Phys::new();
    p.set_tuning(&t);
    p
}

/// A tube of the original tract: no slides, whatever the default tuning.
fn original() -> Phys {
    with_slides(0.0, 0.0)
}

/// Ticks in a control block of the physical engine (128 internal samples).
const BLOCK_TICKS: usize = 256;

/// Play `blocks` control blocks of shape `d` with the lengths `at(t)` (cm:
/// protrusion, lengthening) and their speeds `speed(t)` (cm/s) at each
/// block's end, `t` in seconds; `each` sees the tract after every tick.
fn moving(
    p: &mut Phys,
    d: &[f32; N],
    blocks: usize,
    at: impl Fn(f32) -> [f32; 2],
    speed: impl Fn(f32) -> [f32; 2],
    mut each: impl FnMut(&Phys, usize),
) {
    #[allow(clippy::cast_precision_loss)]
    let block_dt = BLOCK_TICKS as f32 * TICK;
    let mut n = 0;
    for b in 0..blocks {
        #[allow(clippy::cast_precision_loss)]
        let t = (b + 1) as f32 * block_dt;
        p.set_lengths((at(t), speed(t)));
        p.begin_block(d, 0.0, block_dt, false);
        for k in 0..BLOCK_TICKS {
            #[allow(clippy::cast_precision_loss)]
            let lambda = k as f32 / BLOCK_TICKS as f32;
            p.tick(lambda, Drive::Flow(0.0));
            each(p, n);
            n += 1;
        }
    }
}

#[test]
fn slides_at_rest_change_nothing() {
    // The lip horn with the lips not pushed forward is the lips' cell, bit
    // for bit, in the whole tract (losses, walls, nose, sinuses, noise),
    // for shapes whose lips are open wide enough not to protrude today.
    let mut seed = 5u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        #[allow(clippy::cast_precision_loss)]
        let x = (seed >> 8) as f32 / 16_777_216.0;
        x
    };
    for case in 0..6 {
        let mut d = [1.8f32; N];
        if case > 0 {
            for v in &mut d {
                *v = 0.3 + 2.5 * rnd();
            }
            for v in &mut d[39..] {
                *v = v.max(1.3);
            }
        }
        let velum = if case % 2 == 0 { 0.0 } else { 0.6 };
        let mut plain = original();
        let mut horn = with_slides(1.2, 0.0);
        plain.snap(&d, velum);
        horn.set_lengths(([0.0; 2], [0.0; 2]));
        horn.snap(&d, velum);
        for n in 0..6000 {
            let drive = if n < 3000 {
                Drive::Flow(if n == 0 { 1.0 } else { 0.0 })
            } else {
                Drive::Glottis {
                    area: if n % 400 < 200 { 0.12 } else { 0.02 },
                    lung: 8.0 * CM_H2O,
                }
            };
            let (a, b) = (plain.tick(1.0, drive), horn.tick(1.0, drive));
            assert_eq!(a.to_bits(), b.to_bits(), "case {case}, tick {n}");
        }
    }
}

#[test]
fn the_larynx_slide_at_rest_is_the_cells_it_replaces() {
    // Uniform and lossless, two cells and a slide two ticks long are
    // the same tube (but for the rounding at the joint inside).
    let d = [1.8f32; N];
    let mut cells = original();
    cells.set_ideal(true);
    cells.snap(&d, 0.0);
    let mut slide = with_slides(0.0, 1.0);
    slide.set_ideal(true);
    slide.set_lengths(([0.0; 2], [0.0; 2]));
    slide.snap(&d, 0.0);
    let a = impulse(&mut cells, 20_000);
    let b = impulse(&mut slide, 20_000);
    let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(&b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    std::println!("largest difference {:.2e} of the peak", worst / peak);
    assert!(worst <= 1e-4 * peak, "{worst} of {peak}");
}

#[test]
fn lengthened_tubes_resonate_where_theory_says() {
    // A closed-open uniform tube lengthened by ΔL at the lips (the horn) or
    // at the larynx (the slide) resonates at (2n − 1)·c/4(L + ΔL), L with
    // the end correction of a piston in a baffle: its resonances move from
    // the tube's own by L/(L + ΔL) within 1 %, and are within 2 % of the
    // formula as the tube's own are (`uniform_tube_resonates_where_theory_says`;
    // the end correction shrinks a little at higher frequencies).
    #[allow(clippy::cast_precision_loss)]
    let length = N as f32 * C * TICK + 8.0 * sqrt(AREA_SCALE) / (3.0 * PI);
    let mut own = with_slides(1.2, 1.0);
    own.set_ideal(true);
    own.set_lengths(([0.0; 2], [0.0; 2]));
    own.snap(&[2.0; N], 0.0);
    let own = peaks(&impulse(&mut own, 40_000), 4000.0);
    for (name, horn, lengthening) in [
        ("lips +1.2 cm", 1.2f32, 0.0f32),
        ("lips +0.73 cm", 0.73, 0.0),
        ("larynx +0.8 cm", 0.0, 0.8),
        ("larynx -0.35 cm", 0.0, -0.35),
        ("both", 1.2, 0.8),
    ] {
        let mut p = with_slides(1.2, 1.0);
        p.set_ideal(true);
        p.set_lengths(([horn, lengthening], [0.0; 2]));
        p.snap(&[2.0; N], 0.0);
        let x = impulse(&mut p, 40_000);
        let found = peaks(&x, 4000.0);
        let total = length + horn + lengthening;
        for (k, &(hz, _)) in found.iter().take(3).enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let theory = (2 * k + 1) as f32 * C / (4.0 * total);
            let err = (hz - theory) / theory;
            let moved = hz / own[k].0;
            let shift = moved / (length / total) - 1.0;
            std::println!(
                "{name}: mode {k} {hz:.0} Hz, theory {theory:.0} ({:+.2} %); moved by {moved:.4}, theory {:.4} ({:+.2} %)",
                100.0 * err,
                length / total,
                100.0 * shift
            );
            assert!(shift.abs() < 0.01, "{name}, mode {k}: moved by {moved}");
            assert!(err.abs() < 0.02, "{name}, mode {k}: {hz} vs {theory}");
        }
        assert!(found.len() >= 3);
    }
}

/// The slides' lengths swept through their ranges and back over `period`
/// seconds, cm, and their speeds, cm/s: the horn 0 to 1.2, the larynx −0.35
/// to +0.8.
fn sweep(period: f32) -> (impl Fn(f32) -> [f32; 2], impl Fn(f32) -> [f32; 2]) {
    let w = core::f32::consts::TAU / period;
    let at = move |t: f32| {
        let c = (w * t).cos();
        [0.6 - 0.6 * c, 0.225 - 0.575 * c]
    };
    let speed = move |t: f32| {
        let s = (w * t).sin();
        [0.6 * w * s, 0.575 * w * s]
    };
    (at, speed)
}

#[test]
fn energy_never_grows_while_the_slides_move() {
    // Lossless, the glottis and the lips shut, a sound ringing inside (a
    // pulse of air in and out again, so no air is added): the slides swept
    // through their whole ranges and back in 80, 40 and 10 ms (one way in
    // 40, 20 and 5 ms; the quickest gesture takes 40). The delay lines
    // alone never add energy. With the pistons the moving ends do work on
    // the air (the pressure there pushes on the volume they sweep; in this
    // rigid tract the larynx's sweep alone changes the pressure by tens of
    // cmH2O, far more energy than the ringing sound), and the energy never
    // exceeds what was put in plus the work done on it.
    let mut d = [1.8f32; N];
    d[N - 1] = 0.0;
    for pistons in [false, true] {
        for period in [0.08f32, 0.04, 0.01] {
            let mut p = with_slides(1.2, 1.0);
            p.set_ideal(true);
            p.set_pistons(pistons);
            p.set_lengths(([0.0, 0.225 - 0.575], [0.0; 2]));
            p.snap(&d, 0.0);
            p.reset();
            p.tick(1.0, Drive::Flow(1000.0));
            p.tick(1.0, Drive::Flow(-1000.0));
            let start = f64::from(p.energy(1.0));
            let (at, speed) = sweep(period);
            // The work done on the air, in the units of `energy`.
            let mut work = 0.0f64;
            let mut most = 0.0f64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let blocks = (2.0 * period / (BLOCK_TICKS as f32 * TICK)) as usize;
            moving(&mut p, &d, blocks, at, speed, |p, _| {
                work -= f64::from(p.gauge.piston * RHO_C);
                let e = f64::from(p.energy(1.0));
                let bound = start + work;
                most = most.max(e / bound);
                assert!(e <= bound * 1.000_1, "{e} > {start} + {work}");
            });
            std::println!(
                "pistons {pistons}, period {:.0} ms: at most {most:.5} of what was put in and done (the work {:.1} times the sound's energy)",
                1000.0 * period,
                work / start
            );
        }
    }
}

#[test]
fn the_piston_drops_the_pressure_by_the_volume_it_sweeps() {
    // A shut, rigid, lossless tract: lowering the larynx by 0.8 cm grows
    // the air's volume by the slide's area times that, and its pressure
    // falls by ρc² times the share of the volume added.
    let mut d = [1.8f32; N];
    d[N - 1] = 0.0;
    let mut p = with_slides(0.0, 1.0);
    p.set_ideal(true);
    p.set_lengths(([0.0; 2], [0.0; 2]));
    p.snap(&d, 0.0);
    p.reset();
    let lowering = 0.8f32;
    let time = 0.1f32;
    let at = move |t: f32| {
        let x = (t / time).clamp(0.0, 1.0);
        [0.0, lowering * x * x * (3.0 - 2.0 * x)]
    };
    let speed = move |t: f32| {
        let x = (t / time).clamp(0.0, 1.0);
        [0.0, lowering * 6.0 * x * (1.0 - x) / time]
    };
    let mut mean = 0.0f64;
    let mut count = 0.0f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let blocks = (0.2 / (BLOCK_TICKS as f32 * TICK)) as usize;
    moving(&mut p, &d, blocks, at, speed, |p, n| {
        #[allow(clippy::cast_precision_loss)]
        if n as f32 * TICK > 0.15 {
            // The pressure averaged over the air in the tract.
            let (mut pv, mut v) = (0.0f32, 0.0f32);
            for i in 0..N - 1 {
                let a = p.a1[i];
                let len = if LARYNX_CELLS.contains(&i) {
                    p.slide_ticks().1 / LARYNX_REST
                } else {
                    1.0
                };
                pv += p.pressure(i) * a * len;
                v += a * len;
            }
            mean += f64::from(pv / v);
            count += 1.0;
        }
    });
    #[allow(clippy::cast_possible_truncation)]
    let drop = (mean / count) as f32;
    let area = p.a1[8];
    #[allow(clippy::cast_precision_loss)]
    let volume = (N - 1) as f32 * area * C * TICK;
    let added = area * lowering;
    let theory = -RHO_C * C * added / (volume + added);
    std::println!(
        "{added:.2} cm³ added to {volume:.1}: {:.1} cmH2O, theory {:.1}",
        drop / CM_H2O,
        theory / CM_H2O
    );
    assert!((drop / theory - 1.0).abs() < 0.02, "{drop} vs {theory}");
}

// ----- the epilaryngeal tube (`AnatomyConfig::epilarynx_*`) -------------

/// A physical-engine tract (no voice) with `tuning`.
fn tract_with(tuning: crate::Tuning) -> crate::tract::Tract {
    let mut t = crate::tract::Tract::new();
    t.jaw = true;
    t.motion = tuning;
    t
}

/// The tuning with the original's epilarynx (seven cells of 0.6, a step
/// to 1.1, not tied to the folds): today's defaults, set by name so that
/// the tests below hold whatever the defaults become.
fn original_epilarynx() -> crate::Tuning {
    let mut t = crate::Tuning::DEFAULT;
    t.anatomy.epilarynx_cells = 7.0;
    t.anatomy.epilarynx_diameter = 0.6;
    t.anatomy.epilarynx_flare = 1.1;
    t.anatomy.epilarynx_flare_cells = 0.0;
    t.anatomy.epilarynx_effort = 0.0;
    t
}

#[test]
fn epilarynx_original_numbers_give_the_original_cells() {
    // Seven cells of 0.6 and three of 1.1, exactly, as the original's tube.
    let t = tract_with(original_epilarynx());
    for i in 0..crate::tract::BLADE_START {
        let want = if i < 7 { 0.6f32 } else { 1.1 };
        let got = t.epilarynx(i).expect("a cell of the tube or its flare");
        assert_eq!(got.to_bits(), want.to_bits(), "cell {i}: {got}");
    }
    assert!(t.epilarynx(crate::tract::BLADE_START).is_none());
    // The classic engine keeps the original's whatever the anatomy says.
    let mut odd = original_epilarynx();
    odd.anatomy.epilarynx_cells = 4.0;
    odd.anatomy.epilarynx_diameter = 0.4;
    let mut classic = tract_with(odd);
    classic.jaw = false;
    assert!(classic.epilarynx(0).is_none());
}

#[test]
fn epilarynx_takes_its_length_area_and_flare() {
    let mut tuning = original_epilarynx();
    tuning.anatomy.epilarynx_cells = 4.5;
    tuning.anatomy.epilarynx_diameter = 0.5;
    tuning.anatomy.epilarynx_flare = 1.5;
    tuning.anatomy.epilarynx_flare_cells = 2.0;
    let t = tract_with(tuning);
    let d: Vec<f32> = (0..10).map(|i| t.epilarynx(i).unwrap()).collect();
    std::println!("{d:?}");
    for &v in &d[..4] {
        assert!((v - 0.5).abs() < 1e-6);
    }
    // Cell 4: half tube, half the start of the ramp; then the ramp, then
    // the flare.
    assert!(
        d[4] > 0.5 && d[4] < d[5] && d[5] < d[6] && d[6] < 1.5,
        "{d:?}"
    );
    for &v in &d[7..] {
        assert!((v - 1.5).abs() < 1e-6);
    }
}

#[test]
fn epilarynx_narrows_with_pressed_folds() {
    let mut tuning = original_epilarynx();
    tuning.anatomy.epilarynx_effort = 1.0;
    let mut t = tract_with(tuning);
    let area = |t: &crate::tract::Tract| t.epilarynx(0).unwrap().powi(2) / 0.36;
    t.tenseness = 0.65;
    assert!((area(&t) - 1.0).abs() < 1e-5);
    t.tenseness = 0.95;
    assert!((area(&t) - 0.5).abs() < 1e-4, "{}", area(&t));
    t.tenseness = 0.35;
    assert!((area(&t) - 2.0).abs() < 1e-4, "{}", area(&t));
    // Never past twice, never wider than the flare.
    t.tenseness = 0.0;
    assert!(area(&t) <= 2.0 + 1e-4);
    // Off, the folds change nothing.
    let mut off = tract_with(original_epilarynx());
    off.tenseness = 0.95;
    assert_eq!(off.epilarynx(0).unwrap().to_bits(), 0.6f32.to_bits());
}

/// The lowest root from `lo` Hz on of `A1·tan(kL1) = A2·cot(kL2)`: a tube
/// of length `l1` (cm) and area `a1`, closed at the glottis, opening into
/// one of `l2` and `a2` open at its far end, lossless (two tubes, Fant
/// 1960).
fn two_tube_root(a1: f32, l1: f32, a2: f32, l2: f32, lo: f32, hi: f32) -> f32 {
    let g = |hz: f32| {
        let k = 2.0 * PI * hz / C;
        a1 * (k * l1).sin() * (k * l2).sin() - a2 * (k * l1).cos() * (k * l2).cos()
    };
    let step = 1.0;
    let mut hz = lo;
    while hz < hi {
        if g(hz).signum() != g(hz + step).signum() {
            let (mut a, mut b) = (hz, hz + step);
            for _ in 0..30 {
                let m = f32::midpoint(a, b);
                if g(a).signum() == g(m).signum() {
                    a = m;
                } else {
                    b = m;
                }
            }
            return f32::midpoint(a, b);
        }
        hz += step;
    }
    f32::NAN
}

#[test]
fn epilarynx_tube_resonates_where_theory_says() {
    // A tube of 4 to 8 cells and 0.45 cm² opening, in a step (no flare
    // cells), into the rest of the tract at 6.1 cm² to the lips. Its own
    // resonance is the root of the two-tube equation nearest its quarter
    // wave c/4L (in one dimension the step has no end correction); the
    // lips' end correction is 8r/3π, as for the uniform tube. The engine
    // must put it within 3 %.
    for cells in [4.0f32, 5.0, 6.0, 7.0, 8.0] {
        let mut tuning = original_epilarynx();
        tuning.anatomy.epilarynx_cells = cells;
        tuning.anatomy.epilarynx_flare = 2.2;
        let t = tract_with(tuning);
        let mut d = [2.2f32; N];
        for (i, v) in d.iter_mut().enumerate().take(crate::tract::BLADE_START) {
            *v = t.epilarynx(i).unwrap();
        }
        let mut p = original();
        p.set_ideal(true);
        p.snap(&d, 0.0);
        let x = impulse(&mut p, 40_000);
        let found = peaks(&x, 8000.0);
        let dx = C * TICK;
        let (a1, a2) = (p.area_of(0.6), p.area_of(2.2));
        let l1 = cells * dx;
        #[allow(clippy::cast_precision_loss)]
        let l2 = (N as f32 - cells) * dx + 8.0 * sqrt(a2 / PI) / (3.0 * PI);
        let quarter = C / (4.0 * l1);
        let mut theory = f32::NAN;
        let mut from = 100.0;
        while from < 9000.0 {
            let r = two_tube_root(a1, l1, a2, l2, from, 9000.0);
            if !r.is_finite() {
                break;
            }
            if theory.is_nan() || (r - quarter).abs() < (theory - quarter).abs() {
                theory = r;
            }
            from = r + 2.0;
        }
        let (hz, bw) = found
            .iter()
            .copied()
            .min_by(|a, b| (a.0 - theory).abs().total_cmp(&(b.0 - theory).abs()))
            .expect("resonances");
        std::println!(
            "{cells} cells ({l1:.2} cm): c/4L {quarter:.0} Hz, two tubes {theory:.0} Hz, engine {hz:.0} Hz (bandwidth {bw:.0})"
        );
        let err = (hz - theory).abs() / theory;
        assert!(err < 0.03, "{cells} cells: {hz} against {theory}");
    }
}
