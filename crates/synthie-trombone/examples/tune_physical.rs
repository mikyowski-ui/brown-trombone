//! Fit the physical engine's vowels to Peterson & Barney (1952).
//!
//! The physical tract has walls, losses and a real lip load, so the same
//! tongue positions give different formants than the classic engine. This
//! measures the tract's transfer function over a grid of tongue position,
//! tongue height and lip rounding, for several global cross-section scales,
//! and reports the best articulation per vowel at each scale.
//!
//! Among shapes that put F1 and F2 within 2 % of the target, the one with
//! the closest F3 wins.
//!
//! ```sh
//! cargo run -p synthie-trombone --example tune_physical --release
//! cargo run -p synthie-trombone --example tune_physical --release -- 1.0   # one scale
//! ```

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use synthie_trombone::{Constriction, Engine, SLOT_LIPS, SLOT_MAIN, Trombone};

/// (name, F1, F2, F3): Peterson & Barney (1952) men; /o/ from Hillenbrand
/// et al. (1995).
const TARGETS: &[(&str, [f32; 3])] = &[
    ("IY", [270.0, 2290.0, 3010.0]),
    ("IH", [390.0, 1990.0, 2550.0]),
    ("EH", [530.0, 1840.0, 2480.0]),
    ("AE", [660.0, 1720.0, 2410.0]),
    ("AH", [640.0, 1190.0, 2390.0]),
    ("AA", [730.0, 1090.0, 2440.0]),
    ("AO", [570.0, 840.0, 2410.0]),
    ("UH", [440.0, 1020.0, 2240.0]),
    ("UW", [300.0, 870.0, 2240.0]),
    ("OH", [497.0, 910.0, 2459.0]),
    ("ER", [490.0, 1350.0, 1690.0]),
    // Schwa: the neutral tract.
    ("AX", [500.0, 1500.0, 2500.0]),
];

/// Narrowest vowel lip opening and tongue channel searched (touch units).
/// Tighter ones would reach the formants too, but they would hiss.
const MIN_LIPS: f32 = 0.65;
const MIN_NARROW: f32 = 0.7;

const LIPS: [Option<f32>; 6] = [
    None,
    Some(1.3),
    Some(1.1),
    Some(0.95),
    Some(0.8),
    Some(0.65),
];

#[derive(Clone, Copy)]
struct Shape {
    index: f32,
    diameter: f32,
    lips: Option<f32>,
    /// A second narrowing along the tongue, `(cell, diameter)`.
    narrow: Option<(f32, f32)>,
    f: [f32; 3],
}

type Candidate = (f32, f32, Option<f32>, Option<(f32, f32)>);

fn measure(scale: f32, (index, diameter, lips, narrow): Candidate) -> Option<Shape> {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    t.set_area_scale(scale);
    t.set_tongue(index, diameter);
    t.set_constriction(SLOT_LIPS, lips.map(|d| Constriction::new(41.0, d)));
    t.set_constriction(SLOT_MAIN, narrow.map(|(i, d)| Constriction::new(i, d)));
    t.settle();
    let step = 4.0;
    let mag = spectrum::transfer(&t, 0.08, 4200.0, step);
    let r = spectrum::resonances(&mag, step);
    // Refine each peak by a parabola through its neighbours.
    let refine = |hz: f32| {
        let k = (hz / step) as usize;
        let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
        hz + step * 0.5 * (a - c) / (a - 2.0 * b + c)
    };
    Some(Shape {
        index,
        diameter,
        lips,
        narrow,
        f: [
            refine(r.first()?.hz),
            refine(r.get(1)?.hz),
            refine(r.get(2)?.hz),
        ],
    })
}

fn survey(scale: f32) -> Vec<Shape> {
    let mut grid: Vec<Candidate> = Vec::new();
    for i in 0..=34 {
        for d in 0..=29 {
            for &l in &LIPS {
                grid.push((12.0 + i as f32 * 0.5, 2.05 + d as f32 * 0.05, l, None));
            }
        }
    }
    // Close vowels need a longer, tighter channel than the tongue body
    // alone makes: add a narrowing along the tongue, on a coarser grid.
    for i in 0..=17 {
        for d in 0..=11 {
            for &l in &LIPS {
                for at in [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0] {
                    for width in [0.7, 0.85, 1.0, 1.2] {
                        grid.push((12.0 + i as f32, 2.05 + d as f32 * 0.1, l, Some((at, width))));
                    }
                }
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = grid.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = grid
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .filter_map(|&c| measure(scale, c))
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

/// How far a shape is from a target: F1 and F2 first; once both are
/// within 1.5 %, F3 decides.
fn cost(s: &Shape, t: [f32; 3]) -> f32 {
    let e12 = (s.f[0] / t[0]).ln().abs().max((s.f[1] / t[1]).ln().abs());
    let e3 = (s.f[2] / t[2]).ln().abs();
    if e12 < 0.015 { e3 } else { 1.0 + e12 }
}

/// Small deterministic generator for the local search.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Hill-climb from `start` with shrinking random steps.
fn refine(scale: f32, start: Shape, target: [f32; 3], seed: u64) -> Shape {
    let mut rng = Lcg(seed);
    let mut best = start;
    let mut best_cost = cost(&best, target);
    for round in 0..420 {
        let reach = (1.0 - round as f32 / 440.0).max(0.05);
        let index = (best.index + 1.5 * reach * rng.unit()).clamp(12.0, 29.0);
        let diameter = (best.diameter + 0.25 * reach * rng.unit()).clamp(2.05, 3.5);
        let lips = best
            .lips
            .map(|l| (l + 0.15 * reach * rng.unit()).clamp(MIN_LIPS, 1.45));
        let narrow = best.narrow.map(|(at, w)| {
            (
                (at + 2.5 * reach * rng.unit()).clamp(14.0, 34.0),
                (w + 0.15 * reach * rng.unit()).clamp(MIN_NARROW, 1.4),
            )
        });
        let Some(s) = measure(scale, (index, diameter, lips, narrow)) else {
            continue;
        };
        let c = cost(&s, target);
        if c < best_cost {
            best = s;
            best_cost = c;
        }
    }
    best
}

fn main() {
    let scales: Vec<f32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let scales = if scales.is_empty() {
        vec![0.6, 0.8, 1.0, 1.25, 1.6]
    } else {
        scales
    };
    for scale in scales {
        let shapes = survey(scale);
        println!("area scale {scale}: {} shapes measured", shapes.len());
        let mut worst = 0.0f32;
        let mut f3_total = 0.0f32;
        // Refine the best few coarse shapes of every vowel, in parallel.
        let mut jobs = Vec::new();
        for (v, &(_, t)) in TARGETS.iter().enumerate() {
            let mut ranked: Vec<&Shape> = shapes.iter().collect();
            ranked.sort_by(|a, b| cost(a, t).total_cmp(&cost(b, t)));
            // Start from the best shape of each distinct lip setting and
            // narrowing place, so the climbs explore different basins.
            let mut seen: Vec<(Option<u32>, Option<u32>)> = Vec::new();
            for s in ranked {
                let key = (
                    s.lips.map(|l| (l * 100.0) as u32),
                    s.narrow.map(|(at, _)| at as u32),
                );
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                jobs.push((v, *s, (v * 100 + seen.len()) as u64 + 1));
                if seen.len() >= 24 {
                    break;
                }
            }
        }
        let refined: Vec<(usize, Shape)> = std::thread::scope(|sc| {
            let handles: Vec<_> = jobs
                .iter()
                .map(|&(v, start, seed)| {
                    sc.spawn(move || (v, refine(scale, start, TARGETS[v].1, seed)))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("worker panicked"))
                .collect()
        });
        for (v, &(name, t)) in TARGETS.iter().enumerate() {
            let err12 = |s: &Shape| (s.f[0] / t[0]).ln().abs().max((s.f[1] / t[1]).ln().abs());
            let err3 = |s: &Shape| (s.f[2] / t[2]).ln().abs();
            let best = refined
                .iter()
                .filter(|(i, _)| *i == v)
                .map(|(_, s)| s)
                .min_by(|a, b| cost(a, t).total_cmp(&cost(b, t)));
            let Some(b) = best else { continue };
            worst = worst.max(err12(b));
            f3_total += err3(b);
            println!(
                "  {name}: tongue ({:.2}, {:.3}) lips {:?} narrow {:?} -> {:.0}/{:.0}/{:.0} (target {:.0}/{:.0}/{:.0}; F1 {:+.1}% F2 {:+.1}% F3 {:+.1}%)",
                b.index,
                b.diameter,
                b.lips,
                b.narrow,
                b.f[0],
                b.f[1],
                b.f[2],
                t[0],
                t[1],
                t[2],
                100.0 * (b.f[0] / t[0] - 1.0),
                100.0 * (b.f[1] / t[1] - 1.0),
                100.0 * (b.f[2] / t[2] - 1.0),
            );
        }
        println!(
            "  worst F1/F2 error {:.1}%, mean F3 error {:.1}%",
            100.0 * worst,
            100.0 * f3_total / TARGETS.len() as f32
        );
    }
}
