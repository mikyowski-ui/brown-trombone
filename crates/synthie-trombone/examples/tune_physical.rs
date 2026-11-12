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
];

const LIPS: [Option<f32>; 7] = [
    None,
    Some(1.3),
    Some(1.1),
    Some(0.95),
    Some(0.8),
    Some(0.65),
    Some(0.55),
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
                    for width in [0.6, 0.7, 0.85, 1.0, 1.2] {
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
        for &(name, t) in TARGETS {
            let err12 = |s: &Shape| (s.f[0] / t[0]).ln().abs().max((s.f[1] / t[1]).ln().abs());
            let err3 = |s: &Shape| (s.f[2] / t[2]).ln().abs();
            let close: Vec<&Shape> = shapes.iter().filter(|s| err12(s) < 0.02).collect();
            let best = if close.is_empty() {
                shapes.iter().min_by(|a, b| err12(a).total_cmp(&err12(b)))
            } else {
                close.into_iter().min_by(|a, b| err3(a).total_cmp(&err3(b)))
            };
            let Some(b) = best else { continue };
            worst = worst.max(err12(b));
            f3_total += err3(b);
            println!(
                "  {name}: tongue ({:.1}, {:.2}) lips {:?} narrow {:?} -> {:.0}/{:.0}/{:.0} (target {:.0}/{:.0}/{:.0}; F1 {:+.1}% F2 {:+.1}% F3 {:+.1}%)",
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
