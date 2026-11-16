//! Gates for the babble map (`synthie_trombone::inverse`): solved shapes
//! are measured back through the real physical engine.
//!
//! ```sh
//! cargo test -p synthie-trombone --release --test inverse_gate -- --nocapture
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/spectrum.rs"]
mod spectrum;
#[path = "../examples/support/survey.rs"]
mod survey;

use survey::{Lcg, Meter};
use synthie_trombone::inverse::{self, Shape};

/// Tightest lip rounding and tongue narrowing of the physical vowels.
const MIN_LIPS: f32 = 0.65;
const MIN_NARROWING: f32 = 0.7;

/// Measure F1..F3 of each `(shape, tract length)` on all threads.
fn measure_all(jobs: &[(Shape, f32)]) -> Vec<Option<[f32; 3]>> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = jobs.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut m = Meter::new();
                    part.iter()
                        .map(|(shape, l)| m.measure(shape, *l))
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

/// Random targets, uniform on a log scale over the map's range, kept only
/// when `ok` accepts them.
fn targets(seed: u64, n: usize, ok: impl Fn(f32, f32) -> bool, scale: f32) -> Vec<(f32, f32)> {
    let mut rng = Lcg(seed);
    let (a, b) = (inverse::F1_RANGE, inverse::F2_RANGE);
    let mut out = Vec::new();
    while out.len() < n {
        let f1 = (a.0.ln() + (a.1.ln() - a.0.ln()) * rng.unit()).exp() * scale;
        let f2 = (b.0.ln() + (b.1.ln() - b.0.ln()) * rng.unit()).exp() * scale;
        if ok(f1, f2) {
            out.push((f1, f2));
        }
    }
    out
}

fn assert_no_hiss(s: &Shape) {
    assert!(
        s.tongue_index.is_finite() && s.tongue_diameter.is_finite(),
        "{s:?}"
    );
    if let Some(l) = s.lips {
        assert!(l >= MIN_LIPS - 1e-4, "lips too tight: {s:?}");
    }
    if let Some((at, d)) = s.narrowing {
        assert!(at.is_finite(), "{s:?}");
        assert!(d >= MIN_NARROWING - 1e-4, "narrowing too tight: {s:?}");
    }
}

/// Relative error of `got` against `want`.
fn err(got: f32, want: f32) -> f32 {
    (got / want - 1.0).abs()
}

/// Solve `targets` at `length`, measure, and return per-target errors
/// `(f1 error, f2 error)`; prints a summary and the misses.
fn solve_and_measure(
    name: &str,
    targets: &[(f32, f32)],
    length: f32,
    limit: f32,
) -> Vec<(f32, f32)> {
    let shapes: Vec<Shape> = targets
        .iter()
        .map(|&(f1, f2)| inverse::solve_for(f1, f2, None, None, length))
        .collect();
    for s in &shapes {
        assert_no_hiss(s);
    }
    let jobs: Vec<(Shape, f32)> = shapes.iter().map(|&s| (s, length)).collect();
    let got = measure_all(&jobs);
    let mut errs = Vec::new();
    for (i, (&(f1, f2), f)) in targets.iter().zip(&got).enumerate() {
        let (e1, e2) = f.map_or((f32::INFINITY, f32::INFINITY), |f| {
            (err(f[0], f1), err(f[1], f2))
        });
        if e1 > limit || e2 > limit {
            println!(
                "{name} miss #{i}: target ({f1:.0}, {f2:.0}) got {f:?}, shape {:?}",
                shapes[i]
            );
        }
        errs.push((e1, e2));
    }
    let n = errs.len() as f32;
    let in1 = errs.iter().filter(|e| e.0 <= limit).count();
    let in2 = errs.iter().filter(|e| e.1 <= limit).count();
    let both = errs.iter().filter(|e| e.0 <= limit && e.1 <= limit).count();
    let worst1 = errs.iter().fold(0.0f32, |m, e| m.max(e.0));
    let worst2 = errs.iter().fold(0.0f32, |m, e| m.max(e.1));
    let mut all: Vec<f32> = errs.iter().map(|e| e.0.max(e.1)).collect();
    all.sort_by(f32::total_cmp);
    println!(
        "{name}: within {:.0} %: F1 {in1}/{n} F2 {in2}/{n} both {both}/{n}; worst F1 {:.2} %, F2 {:.2} %; median {:.2} %",
        100.0 * limit,
        100.0 * worst1,
        100.0 * worst2,
        100.0 * all[all.len() / 2],
    );
    errs
}

#[test]
fn two_hundred_targets_land_within_three_percent() {
    let t = targets(0x5eed, 200, inverse::reachable, 1.0);
    let errs = solve_and_measure("gate", &t, 1.0, 0.03);
    let misses = errs.iter().filter(|e| e.0 > 0.03 || e.1 > 0.03).count();
    assert_eq!(misses, 0, "{misses} of 200 targets missed by more than 3 %");
}

#[test]
fn other_tract_lengths_land_within_five_percent() {
    for length in [0.7f32, 1.3] {
        let t = targets(
            0x00c0_ffee + (length * 10.0) as u64,
            60,
            |a, b| inverse::reachable_for(a, b, length),
            1.0 / length,
        );
        let errs = solve_and_measure(&format!("length {length}"), &t, length, 0.05);
        let misses = errs.iter().filter(|e| e.0 > 0.05 || e.1 > 0.05).count();
        assert_eq!(
            misses, 0,
            "length {length}: {misses} targets missed by more than 5 %"
        );
    }
}

#[test]
fn a_slow_glide_gives_a_smooth_shape_path() {
    // Around the vowel space: /i/-ish, /a/-ish, /u/-ish and back, through
    // points that every voice can reach.
    let waypoints = [
        (300.0f32, 2100.0f32),
        (600.0, 1500.0),
        (620.0, 1050.0),
        (380.0, 800.0),
        (300.0, 2100.0),
    ];
    for &(a, b) in &waypoints {
        assert!(
            inverse::reachable(a, b),
            "waypoint ({a}, {b}) not reachable"
        );
    }
    let steps = 250;
    let mut path: Vec<((f32, f32), Shape)> = Vec::new();
    let mut prev: Option<Shape> = None;
    for w in waypoints.windows(2) {
        let ((a0, b0), (a1, b1)) = (w[0], w[1]);
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let f1 = (a0.ln() + (a1.ln() - a0.ln()) * t).exp();
            let f2 = (b0.ln() + (b1.ln() - b0.ln()) * t).exp();
            let shape = inverse::solve(f1, f2, None, prev.as_ref());
            assert_no_hiss(&shape);
            path.push(((f1, f2), shape));
            prev = Some(shape);
        }
    }
    // Each target step is under 0.3 % in frequency. A smooth path changes
    // the tract a little per step, never by a jump to a distant shape. The
    // tract is what is compared: its settled diameters, cell by cell. (Shape
    // coordinates can leap where they do nothing: lips wider than about
    // 1.8 hardly touch the tract, so `None` and `Some(1.9)` are neighbours.)
    let tract = |s: &Shape| survey::voice(s, 1.0).diameters().to_vec();
    let mut worst = (0.0f32, 0);
    let mut worst_coordinate = (0.0f32, 0);
    let mut before = tract(&path[0].1);
    for (i, p) in path.windows(2).enumerate() {
        let after = tract(&p[1].1);
        let step = before
            .iter()
            .zip(&after)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        if step > worst.0 {
            worst = (step, i);
        }
        let (x, y) = (p[0].1.coordinates(), p[1].1.coordinates());
        let step = (0..5).fold(0.0f32, |m, d| m.max((x[d] - y[d]).abs()));
        if step > worst_coordinate.0 {
            worst_coordinate = (step, i);
        }
        before = after;
    }
    let (w, at) = worst;
    println!(
        "glide: {} steps; largest tract change {w:.4} (diameter) at {:?} -> {:?}; largest coordinate step {:.4} at step {}",
        path.len(),
        path[at],
        path[at + 1],
        worst_coordinate.0,
        worst_coordinate.1,
    );
    // A jump between distinct shapes moves some cell by 0.3 or more and a
    // coordinate by 0.25 or more; smooth tracking stays far below. Where
    // the formants hardly depend on a coordinate (a low tongue behind
    // tightly rounded lips, on the way from /u/ to /i/) smooth tracking
    // moves it fastest, by up to 0.075 per step, while no cell moves by
    // more than 0.09.
    assert!(w < 0.1, "the tract jumps by {w} at step {at}");
    assert!(
        worst_coordinate.0 < 0.1,
        "the shape jumps by {} in coordinates at step {}",
        worst_coordinate.0,
        worst_coordinate.1
    );
    // And it still hits the targets: measure every tenth point.
    let jobs: Vec<(Shape, f32)> = path.iter().step_by(10).map(|p| (p.1, 1.0)).collect();
    let got = measure_all(&jobs);
    for (p, f) in path.iter().step_by(10).zip(got) {
        let f = f.expect("three formants");
        let (f1, f2) = p.0;
        assert!(
            err(f[0], f1) < 0.03 && err(f[1], f2) < 0.03,
            "glide target ({f1:.0}, {f2:.0}) got {f:?}"
        );
    }
}

#[test]
fn unreachable_targets_are_clamped() {
    let far = [
        (100.0f32, 5000.0f32),
        (1500.0, 500.0),
        (200.0, 200.0),
        (2000.0, 3000.0),
        (250.0, 3500.0),
        (900.0, 900.0),
    ];
    let mut jobs = Vec::new();
    let mut want = Vec::new();
    for &(f1, f2) in &far {
        assert!(
            !inverse::reachable(f1, f2),
            "({f1}, {f2}) should be out of reach"
        );
        let (c1, c2) = inverse::nearest_reachable(f1, f2);
        assert!(
            inverse::reachable(c1, c2),
            "clamped ({c1}, {c2}) not reachable"
        );
        // The clamp moves the target to the edge, not somewhere arbitrary.
        let (a, b) = inverse::nearest_reachable(c1 * 1.01, c2 * 0.99);
        assert!(err(a, c1 * 1.01) < 0.03 && err(b, c2 * 0.99) < 0.03);
        let shape = inverse::solve(f1, f2, None, None);
        assert_no_hiss(&shape);
        jobs.push((shape, 1.0));
        want.push((c1, c2));
    }
    // Garbage in: still a sane shape.
    for s in [
        inverse::solve(f32::NAN, f32::INFINITY, Some(f32::NAN), None),
        inverse::solve(-5.0, 0.0, None, None),
        inverse::solve_for(500.0, 1500.0, None, None, f32::NAN),
    ] {
        assert_no_hiss(&s);
    }
    let got = measure_all(&jobs);
    for ((&(f1, f2), &(c1, c2)), f) in far.iter().zip(&want).zip(got) {
        let f = f.expect("three formants");
        println!("unreachable ({f1}, {f2}) -> clamped ({c1:.0}, {c2:.0}) -> measured {f:?}");
        assert!(
            err(f[0], c1) < 0.03 && err(f[1], c2) < 0.03,
            "({f1}, {f2}) clamped to ({c1:.0}, {c2:.0}) but measured {f:?}"
        );
    }
}

#[test]
fn f3_steers_among_equal_shapes() {
    // With F1 and F2 fixed, asking for a lower or higher F3 moves it that
    // way (or at least not the other way), without spoiling F1 and F2.
    let (f1, f2) = (450.0, 1400.0);
    assert!(inverse::reachable(f1, f2));
    let low = inverse::solve(f1, f2, Some(1900.0), None);
    let high = inverse::solve(f1, f2, Some(2900.0), None);
    let got = measure_all(&[(low, 1.0), (high, 1.0)]);
    let (l, h) = (got[0].expect("formants"), got[1].expect("formants"));
    println!("F3 low {l:?}, high {h:?}");
    assert!(l[2] < h[2], "F3 did not follow: {l:?} vs {h:?}");
    for f in [l, h] {
        assert!(err(f[0], f1) < 0.03 && err(f[1], f2) < 0.03, "{f:?}");
    }
}

// ----- the modal tongue body (`inverse::modal`) ----------------------------

/// Random targets over the modal map's reach.
fn modal_targets(seed: u64, n: usize, length: f32) -> Vec<(f32, f32)> {
    let mut rng = Lcg(seed);
    let (a, b) = (inverse::modal::F1_RANGE, inverse::modal::F2_RANGE);
    let mut out = Vec::new();
    while out.len() < n {
        let f1 = (a.0.ln() + (a.1.ln() - a.0.ln()) * rng.unit()).exp() / length;
        let f2 = (b.0.ln() + (b.1.ln() - b.0.ln()) * rng.unit()).exp() / length;
        if inverse::modal::reachable_for(f1, f2, length) {
            out.push((f1, f2));
        }
    }
    out
}

/// Solve `targets` on the modal map at `length`, measure them on the modal
/// body, and count the misses beyond `limit`.
fn modal_misses(name: &str, targets: &[(f32, f32)], length: f32, limit: f32) -> usize {
    let shapes: Vec<Shape> = targets
        .iter()
        .map(|&(f1, f2)| inverse::modal::solve_for(f1, f2, None, None, length))
        .collect();
    for s in &shapes {
        assert_no_hiss(s);
        assert!(s.modes.is_some(), "{s:?}");
    }
    let jobs: Vec<(Shape, f32)> = shapes.iter().map(|&s| (s, length)).collect();
    let got = measure_all(&jobs);
    let mut misses = 0;
    let mut worst = 0.0f32;
    for (i, (&(f1, f2), f)) in targets.iter().zip(&got).enumerate() {
        let e = f.map_or(f32::INFINITY, |f| err(f[0], f1).max(err(f[1], f2)));
        worst = worst.max(e);
        if e > limit {
            misses += 1;
            println!(
                "{name} miss #{i}: target ({f1:.0}, {f2:.0}) got {f:?}, shape {:?}",
                shapes[i]
            );
        }
    }
    println!(
        "{name}: {} of {} within {:.0} %, worst {:.2} %",
        targets.len() - misses,
        targets.len(),
        100.0 * limit,
        100.0 * worst
    );
    misses
}

#[test]
fn modal_map_targets_land_within_three_percent() {
    let t = modal_targets(0x5eed, 150, 1.0);
    let misses = modal_misses("modal gate", &t, 1.0, 0.03);
    assert_eq!(misses, 0, "{misses} of 150 targets missed by more than 3 %");
}

#[test]
fn modal_map_other_tract_lengths_land_within_five_percent() {
    for length in [0.7f32, 1.3] {
        let t = modal_targets(0x00c0_ffee + (length * 10.0) as u64, 50, length);
        let misses = modal_misses(&format!("modal, length {length}"), &t, length, 0.05);
        assert_eq!(misses, 0, "length {length}: {misses} missed by more than 5 %");
    }
}

#[test]
fn a_modal_shape_solves_on_the_modal_map_and_predicts_what_it_measures() {
    // A `near` shape of the modal body keeps `solve` on the modal map, and
    // the map's prediction of a solved shape is what the engine measures.
    let near = inverse::modal::solve(500.0, 1500.0, None, None);
    let s = inverse::solve(520.0, 1450.0, None, Some(&near));
    assert!(s.modes.is_some(), "{s:?}");
    let want = inverse::predict(&s).expect("a prediction");
    let got = measure_all(&[(s, 1.0)])[0].expect("formants");
    for k in 0..2 {
        assert!(err(got[k], want[k]) < 0.03, "{got:?} vs {want:?}");
    }
}
