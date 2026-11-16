//! Generate the babble map, `src/inverse_table.rs`: the formants of the
//! physical engine over a grid of vowel shapes, for
//! [`synthie_trombone::inverse`].
//!
//! 1. Measures F1, F2 and F3 (peaks of the tract's transfer function, as
//!    `tune_physical` does) at every node of a five-dimensional grid of
//!    shapes: tongue position, tongue height, lip opening, narrowing place
//!    and narrowing opening (see `Shape::coordinates`).
//! 2. Checks every grid cell: measures points inside it and compares them
//!    with what interpolating its corners predicts. Cells where the two
//!    disagree (formants that jump or bend sharply) are marked rough, and
//!    the solver stays out of them.
//! 3. Draws the reachable (F1, F2) region from the cells that pass.
//! 4. Measures how formants scale with tract length (0.6 to 1.4) and fits
//!    the correction to `1 / length` scaling.
//!
//! ```sh
//! cargo run -p synthie-trombone --example babble --release
//! ```
//!
//! Uses every CPU thread; takes a couple of minutes.
//!
//! `-- --modal` measures the babble map of the modal tongue body instead
//! (`src/inverse_modes_table.rs`, `inverse::modal`), on the tract the
//! command line configures: `--lengthened` (the wave's lip horn and larynx,
//! the tract the modal body is switched on with), `--profile FILE` on top.
//! The cosine map is always the default tract's.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop
)]

#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/survey.rs"]
mod survey;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use survey::{Lcg, Meter};
use synthie_trombone::inverse::Shape;

/// Grid nodes along each coordinate.
const DIMS: [usize; 5] = [11, 7, 9, 14, 8];
/// Node combinations of narrowing place and opening; "no narrowing" is one.
const JOINT: usize = DIMS[3] * (DIMS[4] - 1) + 1;
/// Points measured inside every cell to check it.
const CHECKS: usize = 3;
/// A cell is rough when interpolation misses F1 or F2 by more than this
/// (ln ratio) at a check point...
const ROUGH_F12: f32 = 0.025;
/// ...or F3 by more than this.
const ROUGH_F3: f32 = 0.08;
/// Corners of a cell whose formant ranges come closer than this (ln ratio)
/// mean that a formant changes names inside it.
const LABEL_GAP: f32 = -0.05;
/// Reachability grid cell size, ln Hz.
const REACH_STEP: f32 = 0.022;
/// Tract lengths of the size survey.
const LENGTHS: [f32; 9] = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];
/// Shapes in the size survey.
const SIZE_SHAPES: usize = 1500;
/// Bins of native formant frequency in the size correction.
const SIZE_BINS: usize = 10;
/// Seconds of impulse response analysed for the table and the size survey
/// (the reference measurement)...
const FULL: f32 = 0.08;
/// ...and for the cell checks, which need less precision (within 0.2 %).
const QUICK: f32 = 0.04;
/// Largest formant code; 1023 marks an unmeasured node.
const TOP: f32 = 1022.0;

/// [`par_map`], or its result from an earlier run: with the environment
/// variable `BABBLE_CACHE` set to a directory, measurements are kept there
/// (per kind and count) and read back instead of being measured again.
/// Only for working on the analysis; delete the files after changing how
/// shapes are measured or what the grid is.
fn cached<T: Sync>(
    kind: &str,
    items: &[T],
    seconds: f32,
    f: impl Fn(&mut Meter, &T) -> Option<[f32; 3]> + Sync,
) -> Vec<Option<[f32; 3]>> {
    let path = std::env::var_os("BABBLE_CACHE")
        .map(|d| std::path::PathBuf::from(d).join(format!("{kind}-{}.bin", items.len())));
    if let Some(bytes) = path.as_ref().and_then(|p| std::fs::read(p).ok())
        && bytes.len() == items.len() * 12
    {
        println!(
            "  (read from {})",
            path.as_ref()
                .map_or_else(String::new, |p| p.display().to_string())
        );
        return bytes
            .chunks(12)
            .map(|c| {
                let v = [0, 1, 2].map(|k| {
                    f32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
                });
                v[0].is_finite().then_some(v)
            })
            .collect();
    }
    let out = par_map(items, seconds, f);
    if let Some(p) = path {
        let bytes: Vec<u8> = out
            .iter()
            .flat_map(|v| v.unwrap_or([f32::NAN; 3]))
            .flat_map(f32::to_le_bytes)
            .collect();
        let _ = std::fs::write(p, bytes);
    }
    out
}

/// Measure `items` on every thread, each with its own [`Meter`].
/// `seconds` of impulse response are analysed (see `Meter::with_seconds`).
fn par_map<T: Sync, R: Send + Clone>(
    items: &[T],
    seconds: f32,
    f: impl Fn(&mut Meter, &T) -> R + Sync,
) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, R)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut meter = Meter::with_seconds(seconds);
                    let mut out = Vec::new();
                    loop {
                        let start = next.fetch_add(32, Ordering::Relaxed);
                        if start >= items.len() {
                            break;
                        }
                        for (i, item) in items.iter().enumerate().skip(start).take(32) {
                            out.push((i, f(&mut meter, item)));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    let mut out: Vec<Option<R>> = vec![None; items.len()];
    for (i, r) in parts.into_iter().flatten() {
        out[i] = Some(r);
    }
    out.into_iter()
        .map(|r| r.expect("every item measured"))
        .collect()
}

/// ln F1, F2, F3 of `x` (shape coordinates) at `length`.
fn ln_formants(meter: &mut Meter, x: &[f32; 5], length: f32) -> Option<[f32; 3]> {
    meter
        .measure(&Shape::from_coordinates(*x), length)
        .filter(|f| f.iter().all(|v| v.is_finite() && *v > 0.0))
        .map(|f| f.map(f32::ln))
}

/// Grid index of node `i` (with "no narrowing" stored once).
fn node_index(i: [usize; 5]) -> usize {
    let j = if i[4] == DIMS[4] - 1 {
        JOINT - 1
    } else {
        i[3] * (DIMS[4] - 1) + i[4]
    };
    ((i[0] * DIMS[1] + i[1]) * DIMS[2] + i[2]) * JOINT + j
}

/// Coordinates of every stored node, in storage order.
fn nodes() -> Vec<[f32; 5]> {
    let c = |k: usize, i: usize| i as f32 / (DIMS[k] - 1) as f32;
    let mut out = Vec::new();
    for i0 in 0..DIMS[0] {
        for i1 in 0..DIMS[1] {
            for i2 in 0..DIMS[2] {
                for j in 0..JOINT {
                    let (i3, i4) = if j == JOINT - 1 {
                        ((DIMS[3] - 1) / 2, DIMS[4] - 1)
                    } else {
                        (j / (DIMS[4] - 1), j % (DIMS[4] - 1))
                    };
                    out.push([c(0, i0), c(1, i1), c(2, i2), c(3, i3), c(4, i4)]);
                }
            }
        }
    }
    out
}

/// Cell base indices in storage order of the rough bits.
fn cells() -> Vec<[usize; 5]> {
    let mut out = Vec::new();
    for c0 in 0..DIMS[0] - 1 {
        for c1 in 0..DIMS[1] - 1 {
            for c2 in 0..DIMS[2] - 1 {
                for c3 in 0..DIMS[3] - 1 {
                    for c4 in 0..DIMS[4] - 1 {
                        out.push([c0, c1, c2, c3, c4]);
                    }
                }
            }
        }
    }
    out
}

/// The 32 corner values of `cell`, or `None` if one is unmeasured.
fn corners(table: &[Option<[f32; 3]>], cell: [usize; 5]) -> Option<[[f32; 3]; 32]> {
    let mut out = [[0.0; 3]; 32];
    for (corner, o) in out.iter_mut().enumerate() {
        let mut i = cell;
        for (d, v) in i.iter_mut().enumerate() {
            *v += (corner >> d) & 1;
        }
        *o = table[node_index(i)]?;
    }
    Some(out)
}

/// Multilinear interpolation of corner values at `frac` inside a cell.
fn interpolate(c: &[[f32; 3]; 32], frac: [f32; 5]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (corner, v) in c.iter().enumerate() {
        let mut w = 1.0;
        for (d, f) in frac.iter().enumerate() {
            w *= if (corner >> d) & 1 == 1 { *f } else { 1.0 - f };
        }
        for k in 0..3 {
            out[k] += w * v[k];
        }
    }
    out
}

/// Coordinates of `frac` inside `cell`.
fn inside(cell: [usize; 5], frac: [f32; 5]) -> [f32; 5] {
    [0, 1, 2, 3, 4].map(|d| (cell[d] as f32 + frac[d]) / (DIMS[d] - 1) as f32)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+_";

/// A byte string literal of `chars`, wrapped at 100 columns.
fn literal(chars: &[u8]) -> String {
    let mut s = String::from("b\"\\\n");
    for line in chars.chunks(100) {
        s.push_str(std::str::from_utf8(line).expect("ASCII"));
        s.push_str("\\\n");
    }
    s.truncate(s.len() - 2);
    s.push('"');
    s
}

/// Bits packed six to a base-64 character.
fn bits(b: &[bool]) -> Vec<u8> {
    b.chunks(6)
        .map(|c| {
            let v = c
                .iter()
                .enumerate()
                .fold(0usize, |v, (i, &on)| v | (usize::from(on) << i));
            B64[v]
        })
        .collect()
}

/// Convex hull of 2-D points (monotone chain), counter-clockwise.
fn hull(mut p: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut h: Vec<(f32, f32)> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let iter: Box<dyn Iterator<Item = &(f32, f32)>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in iter {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

fn inside_hull(h: &[(f32, f32)], q: (f32, f32)) -> bool {
    h.len() >= 3
        && (0..h.len()).all(|i| {
            let (a, b) = (h[i], h[(i + 1) % h.len()]);
            (b.0 - a.0) * (q.1 - a.1) - (b.1 - a.1) * (q.0 - a.0) >= 0.0
        })
}

/// Quantile `q` of `v` (sorted in place).
fn quantile(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get(((v.len().max(1) - 1) as f32 * q) as usize)
        .copied()
        .unwrap_or(0.0)
}

/// Least-squares fit of `y` against `x` (both slices of the same length)
/// with hat functions on `SIZE_BINS` points over `lo..hi`, smoothed by a
/// small penalty on second differences.
fn fit_bins(x: &[f32], y: &[f32], lo: f32, hi: f32) -> [f32; SIZE_BINS] {
    let n = SIZE_BINS;
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];
    for (&x, &y) in x.iter().zip(y) {
        let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0) * (n - 1) as f32;
        let i = (t as usize).min(n - 2);
        let u = f64::from(t - i as f32);
        let w = [(i, 1.0 - u), (i + 1, u)];
        for &(p, wp) in &w {
            b[p] += wp * f64::from(y);
            for &(q, wq) in &w {
                a[p][q] += wp * wq;
            }
        }
    }
    let smooth = 1.0e-2 * x.len() as f64 / n as f64;
    for i in 1..n - 1 {
        let d = [(i - 1, 1.0), (i, -2.0), (i + 1, 1.0)];
        for &(p, wp) in &d {
            for &(q, wq) in &d {
                a[p][q] += smooth * wp * wq;
            }
        }
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1.0e-9;
    }
    // Gaussian elimination.
    for p in 0..n {
        for r in p + 1..n {
            let m = a[r][p] / a[p][p];
            for c in p..n {
                a[r][c] -= m * a[p][c];
            }
            b[r] -= m * b[p];
        }
    }
    let mut out = [0.0f32; SIZE_BINS];
    for p in (0..n).rev() {
        let mut s = b[p];
        for c in p + 1..n {
            s -= a[p][c] * f64::from(out[c]);
        }
        out[p] = (s / a[p][p]) as f32;
    }
    out
}

fn hat(row: &[f32; SIZE_BINS], x: f32, lo: f32, hi: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0) * (SIZE_BINS - 1) as f32;
    let i = (t as usize).min(SIZE_BINS - 2);
    let u = t - i as f32;
    row[i] + (row[i + 1] - row[i]) * u
}

fn main() {
    if std::env::args().any(|a| a == "--modal") {
        modal_main();
        return;
    }
    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let mut src = String::new();

    // The lips start to narrow the lip cells at a rounding of 1.8, where
    // the formants bend; that must be a grid line.
    let knee = Shape {
        tongue_index: 20.0,
        tongue_diameter: 3.0,
        lips: Some(1.8),
        narrowing: None,
        pharynx: 0.0,
        modes: None,
    }
    .coordinates()[2]
        * (DIMS[2] - 1) as f32;
    assert!(
        (knee - knee.round()).abs() < 1e-3,
        "lip rounding 1.8 must fall on a grid line of the lip coordinate"
    );

    // 1. Nodes.
    let coords = nodes();
    println!(
        "measuring {} grid nodes on {threads} threads...",
        coords.len()
    );
    let measured = cached("nodes", &coords, FULL, |m, x| ln_formants(m, x, 1.0));
    let unmeasured = measured.iter().filter(|v| v.is_none()).count();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for f in measured.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(f[k]);
            hi[k] = hi[k].max(f[k]);
        }
    }
    for k in 0..3 {
        lo[k] = ((lo[k] - 0.005) * 1000.0).floor() / 1000.0;
        hi[k] = ((hi[k] + 0.005) * 1000.0).ceil() / 1000.0;
    }
    // Quantise, and use the quantised values from here on, as the solver
    // will.
    let mut codes = Vec::with_capacity(coords.len() * 5);
    let table: Vec<Option<[f32; 3]>> = measured
        .iter()
        .map(|f| {
            let (v, back) = match f {
                None => (1023u32, None),
                Some(f) => {
                    let mut v = 0u32;
                    let mut back = [0.0f32; 3];
                    for k in 0..3 {
                        let q = ((f[k] - lo[k]) / (hi[k] - lo[k]) * TOP)
                            .round()
                            .clamp(0.0, TOP) as u32;
                        v |= q << (10 * k);
                        back[k] = lo[k] + (hi[k] - lo[k]) * q as f32 / TOP;
                    }
                    (v, Some(back))
                }
            };
            for c in 0..5 {
                codes.push(B64[((v >> (6 * c)) & 63) as usize]);
            }
            back
        })
        .collect();
    println!(
        "  {} nodes, {unmeasured} without three formants, {:.1} s",
        coords.len(),
        started.elapsed().as_secs_f32()
    );

    // 2. Cell checks.
    let all_cells = cells();
    let mut rng = Lcg(0x00ba_bb1e);
    let mut probes: Vec<(usize, [f32; 5])> = Vec::new();
    for (n, &cell) in all_cells.iter().enumerate() {
        if corners(&table, cell).is_none() {
            continue;
        }
        for p in 0..CHECKS {
            // The centre, and points that often lie on a face of the
            // cell: the solver may settle on one, at a limit of the space.
            let frac = if p == 0 {
                [0.5; 5]
            } else {
                [0; 5].map(|_| {
                    let r = rng.unit();
                    if r < 0.15 {
                        0.0
                    } else if r < 0.3 {
                        1.0
                    } else {
                        rng.range(0.1, 0.9)
                    }
                })
            };
            probes.push((n, frac));
        }
    }
    println!(
        "checking {} cells at {} points...",
        all_cells.len(),
        probes.len()
    );
    let truth = cached("checks", &probes, QUICK, |m, &(n, frac)| {
        ln_formants(m, &inside(all_cells[n], frac), 1.0)
    });
    // Why cells are rough: unmeasured, formant labels that change places
    // between corners, a check point that failed, one that missed.
    let mut why = [0usize; 4];
    let mut rough: Vec<bool> = all_cells
        .iter()
        .map(|&cell| {
            let Some(c) = corners(&table, cell) else {
                why[0] += 1;
                return true;
            };
            // A formant that fades out lets the next one take its name: one
            // corner's F2 is then another corner's F3. Interpolating across
            // that is meaningless.
            let top = |k: usize| c.iter().fold(f32::MIN, |m, v| m.max(v[k]));
            let bottom = |k: usize| c.iter().fold(f32::MAX, |m, v| m.min(v[k]));
            let swapped = top(0) > bottom(1) + LABEL_GAP || top(1) > bottom(2) + LABEL_GAP;
            if swapped {
                why[1] += 1;
            }
            swapped
        })
        .collect();
    let mut worst12 = vec![0.0f32; all_cells.len()];
    for (&(n, frac), t) in probes.iter().zip(&truth) {
        if rough[n] {
            continue;
        }
        let Some(t) = t else {
            rough[n] = true;
            why[2] += 1;
            continue;
        };
        let c = corners(&table, all_cells[n]).expect("checked above");
        let p = interpolate(&c, frac);
        let e12 = (p[0] - t[0]).abs().max((p[1] - t[1]).abs());
        let e3 = (p[2] - t[2]).abs();
        worst12[n] = worst12[n].max(e12);
        if e12 > ROUGH_F12 || e3 > ROUGH_F3 {
            rough[n] = true;
            why[3] += 1;
        }
    }
    let n_rough = rough.iter().filter(|&&r| r).count();
    println!(
        "  rough cells: {} unmeasured, {} with formants changing places, {} failed checks, {} missed checks",
        why[0], why[1], why[2], why[3]
    );
    let mut errs: Vec<f32> = worst12.clone();
    println!(
        "  {n_rough} of {} cells rough ({:.1} %); worst F1/F2 check error per cell: median {:.2} %, p90 {:.2} %, p99 {:.2} %; {:.1} s",
        all_cells.len(),
        100.0 * n_rough as f32 / all_cells.len() as f32,
        100.0 * quantile(&mut errs, 0.5),
        100.0 * quantile(&mut errs, 0.9),
        100.0 * quantile(&mut errs, 0.99),
        started.elapsed().as_secs_f32()
    );

    // 3. Reachable region.
    let mut lo12 = [f32::INFINITY; 2];
    let mut hi12 = [f32::NEG_INFINITY; 2];
    let hulls: Vec<Vec<(f32, f32)>> = all_cells
        .iter()
        .zip(&rough)
        .filter(|(_, r)| !**r)
        .map(|(&cell, _)| {
            let c = corners(&table, cell).expect("smooth cells are measured");
            for v in &c {
                for k in 0..2 {
                    lo12[k] = lo12[k].min(v[k]);
                    hi12[k] = hi12[k].max(v[k]);
                }
            }
            hull(c.iter().map(|v| (v[0], v[1])).collect())
        })
        .collect();
    let range_hz = |k: usize| {
        (
            ((lo12[k].exp() / 10.0).floor() * 10.0),
            ((hi12[k].exp() / 10.0).ceil() * 10.0),
        )
    };
    let (r1, r2) = (range_hz(0), range_hz(1));
    let na = ((r1.1.ln() - r1.0.ln()) / REACH_STEP).ceil() as usize;
    let nb = ((r2.1.ln() - r2.0.ln()) / REACH_STEP).ceil() as usize;
    let centre = |i: usize, n: usize, r: (f32, f32)| {
        r.0.ln() + (r.1.ln() - r.0.ln()) * (i as f32 + 0.5) / n as f32
    };
    let mut raw = vec![false; na * nb];
    for h in &hulls {
        let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(a, b) in h {
            a0 = a0.min(a);
            a1 = a1.max(a);
            b0 = b0.min(b);
            b1 = b1.max(b);
        }
        let cell_of = |v: f32, r: (f32, f32), n: usize| {
            ((v - r.0.ln()) / (r.1.ln() - r.0.ln()) * n as f32).clamp(0.0, n as f32 - 1.0) as usize
        };
        for i in cell_of(a0, r1, na)..=cell_of(a1, r1, na) {
            for j in cell_of(b0, r2, nb)..=cell_of(b1, r2, nb) {
                if !raw[i * nb + j] && inside_hull(h, (centre(i, na, r1), centre(j, nb, r2))) {
                    raw[i * nb + j] = true;
                }
            }
        }
    }
    // Erode by one cell, so the edge is safely inside.
    let at = |m: &[bool], i: isize, j: isize| {
        i >= 0
            && j >= 0
            && (i as usize) < na
            && (j as usize) < nb
            && m[i as usize * nb + j as usize]
    };
    let mut reach = vec![false; na * nb];
    for i in 0..na as isize {
        for j in 0..nb as isize {
            reach[i as usize * nb + j as usize] = at(&raw, i, j)
                && at(&raw, i - 1, j)
                && at(&raw, i + 1, j)
                && at(&raw, i, j - 1)
                && at(&raw, i, j + 1);
        }
    }
    // Keep the largest connected part.
    let mut label = vec![0usize; na * nb];
    let mut part_sizes = vec![0usize];
    for start in 0..na * nb {
        if !reach[start] || label[start] != 0 {
            continue;
        }
        let id = part_sizes.len();
        let mut stack = vec![start];
        label[start] = id;
        let mut size = 0;
        while let Some(p) = stack.pop() {
            size += 1;
            let (i, j) = ((p / nb) as isize, (p % nb) as isize);
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                if at(&reach, i + di, j + dj) {
                    let q = (i + di) as usize * nb + (j + dj) as usize;
                    if label[q] == 0 {
                        label[q] = id;
                        stack.push(q);
                    }
                }
            }
        }
        part_sizes.push(size);
    }
    let main_part = (1..part_sizes.len())
        .max_by_key(|&i| part_sizes[i])
        .unwrap_or(0);
    for (r, l) in reach.iter_mut().zip(&label) {
        *r = *r && *l == main_part;
    }
    println!(
        "  reachable: {} of {na}×{nb} cells over F1 {:.0}–{:.0} Hz, F2 {:.0}–{:.0} Hz ({} parts before cleanup)",
        reach.iter().filter(|&&r| r).count(),
        r1.0,
        r1.1,
        r2.0,
        r2.1,
        part_sizes.len() - 1
    );
    for i in (0..na).rev().step_by(2) {
        let row: String = (0..nb)
            .map(|j| if reach[i * nb + j] { '#' } else { '.' })
            .collect();
        println!("  {:5.0} {row}", centre(i, na, r1).exp());
    }

    // 4. Tract length.
    let smooth: Vec<usize> = (0..all_cells.len()).filter(|&n| !rough[n]).collect();
    let mut size_probes: Vec<([f32; 5], f32)> = Vec::new();
    let mut size_shapes = Vec::new();
    for _ in 0..SIZE_SHAPES {
        let n = smooth[(rng.unit() * smooth.len() as f32) as usize % smooth.len()];
        let x = inside(all_cells[n], [0; 5].map(|_| rng.unit()));
        size_shapes.push(x);
        for &l in &LENGTHS {
            size_probes.push((x, l));
        }
    }
    println!(
        "measuring {SIZE_SHAPES} shapes at {} tract lengths...",
        LENGTHS.len()
    );
    let sized = cached("sizes", &size_probes, FULL, |m, (x, l)| {
        ln_formants(m, x, *l)
    });
    let native_at = LENGTHS
        .iter()
        .position(|&l| (l - 1.0).abs() < 1e-6)
        .expect("1.0 is listed");
    let mut correction = [[[0.0f32; SIZE_BINS]; LENGTHS.len()]; 3];
    let mut report = String::new();
    for (li, &l) in LENGTHS.iter().enumerate() {
        let _ = write!(report, "  length {l:.1}:");
        for k in 0..3 {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for s in 0..SIZE_SHAPES {
                let row = &sized[s * LENGTHS.len()..(s + 1) * LENGTHS.len()];
                if let (Some(n), Some(f)) = (row[native_at], row[li]) {
                    xs.push(n[k]);
                    ys.push(f[k] + l.ln() - n[k]);
                }
            }
            correction[k][li] = fit_bins(&xs, &ys, lo[k], hi[k]);
            let mut raw: Vec<f32> = ys.iter().map(|y| y.abs()).collect();
            let mut left: Vec<f32> = xs
                .iter()
                .zip(&ys)
                .map(|(&x, &y)| (y - hat(&correction[k][li], x, lo[k], hi[k])).abs())
                .collect();
            let _ = write!(
                report,
                "  F{}  1/L err p95 {:4.1} % max {:4.1} %, corrected p95 {:4.2} % max {:4.2} % |",
                k + 1,
                100.0 * quantile(&mut raw, 0.95),
                100.0 * quantile(&mut raw, 1.0),
                100.0 * quantile(&mut left, 0.95),
                100.0 * quantile(&mut left, 1.0),
            );
        }
        report.push('\n');
    }
    print!("{report}");

    // Write the table.
    let _ = writeln!(
        src,
        "//! Babble map table, generated by `examples/babble.rs`; do not edit.\n\
         //! Regenerate with `cargo run -p synthie-trombone --example babble --release`.\n\
         //!\n\
         //! Measured {} grid nodes, {unmeasured} without three formants; {n_rough} of {} cells \
         rough.\n\
         //! How formants follow tract length (ln ratio to `native / length`):\n//!\n//! ```text",
        coords.len(),
        all_cells.len(),
    );
    for line in report.lines() {
        let _ = writeln!(src, "//! {}", line.trim_start());
    }
    let _ = writeln!(
        src,
        "//! ```\n\n#![allow(clippy::unreadable_literal, clippy::excessive_precision)]\n"
    );
    let _ = writeln!(src, "/// Grid nodes along each shape coordinate.");
    let _ = writeln!(src, "pub(super) const DIMS: [usize; 5] = {DIMS:?};");
    let _ = writeln!(src, "/// `ln` Hz of formant code 0, per formant.");
    let _ = writeln!(src, "pub(super) const F_LO: [f32; 3] = {lo:?};");
    let _ = writeln!(src, "/// `ln` Hz of formant code 1022, per formant.");
    let _ = writeln!(src, "pub(super) const F_HI: [f32; 3] = {hi:?};");
    let _ = writeln!(
        src,
        "/// Formants of every node: five base-64 characters, low bits first,\n\
         /// holding three ten-bit codes (F1 lowest); F1 code 1023 = unmeasured."
    );
    let _ = writeln!(src, "pub(super) static NODES: &[u8] = {};", literal(&codes));
    let _ = writeln!(src, "/// One bit per grid cell, six per character: rough.");
    let _ = writeln!(
        src,
        "pub(super) static ROUGH: &[u8] = {};",
        literal(&bits(&rough))
    );
    let _ = writeln!(src, "/// F1 range of the reachability grid, Hz.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_F1: (f32, f32) = ({:?}, {:?});",
        r1.0, r1.1
    );
    let _ = writeln!(src, "/// F2 range of the reachability grid, Hz.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_F2: (f32, f32) = ({:?}, {:?});",
        r2.0, r2.1
    );
    let _ = writeln!(src, "/// Reachability cells along F1 and F2.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_SIZE: [usize; 2] = [{na}, {nb}];"
    );
    let _ = writeln!(
        src,
        "/// One bit per reachability cell (F1 major), six per character."
    );
    let _ = writeln!(
        src,
        "pub(super) static REACH: &[u8] = {};",
        literal(&bits(&reach))
    );
    let _ = writeln!(src, "/// Tract lengths of the size correction.");
    let _ = writeln!(
        src,
        "pub(super) const SIZE_LENGTHS: [f32; {}] = {LENGTHS:?};",
        LENGTHS.len()
    );
    let _ = writeln!(
        src,
        "/// `ln (formant · length / native formant)` per formant, tract length\n\
         /// and native formant (bins over `F_LO..F_HI`).\n\
         #[rustfmt::skip]"
    );
    let _ = writeln!(
        src,
        "pub(super) const SIZE_CORRECTION: [[[f32; {SIZE_BINS}]; {}]; 3] = [",
        LENGTHS.len()
    );
    for k in correction {
        let _ = writeln!(src, "    [");
        for row in k {
            let row: Vec<String> = row.iter().map(|v| format!("{v:.5}")).collect();
            let _ = writeln!(src, "        [{}],", row.join(", "));
        }
        let _ = writeln!(src, "    ],");
    }
    let _ = writeln!(src, "];");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/inverse_table.rs");
    std::fs::write(path, &src).expect("write src/inverse_table.rs");
    println!(
        "wrote {path}: {} bytes, {:.1} s in all",
        src.len(),
        started.elapsed().as_secs_f32()
    );
}

// ----- the modal body ------------------------------------------------------

/// Grid nodes along each coordinate of the modal map: the body's three
/// numbers, the lips (the knee of the lips, coordinate 0.75, on a grid
/// line).
const MODAL_DIMS: [usize; 4] = [11, 11, 11, 9];
/// Modal shapes in the size survey (the cosine map's correction is checked
/// on them).
const MODAL_SIZE_SHAPES: usize = 400;

/// Node `i` of the modal grid in storage order.
fn modal_index(i: [usize; 4]) -> usize {
    let d = MODAL_DIMS;
    ((i[0] * d[1] + i[1]) * d[2] + i[2]) * d[3] + i[3]
}

fn modal_cells() -> Vec<[usize; 4]> {
    let d = MODAL_DIMS;
    let mut out = Vec::new();
    for a in 0..d[0] - 1 {
        for b in 0..d[1] - 1 {
            for c in 0..d[2] - 1 {
                for e in 0..d[3] - 1 {
                    out.push([a, b, c, e]);
                }
            }
        }
    }
    out
}

fn modal_corners(table: &[Option<[f32; 3]>], cell: [usize; 4]) -> Option<[[f32; 3]; 16]> {
    let mut out = [[0.0; 3]; 16];
    for (corner, o) in out.iter_mut().enumerate() {
        let mut i = cell;
        for (d, v) in i.iter_mut().enumerate() {
            *v += (corner >> d) & 1;
        }
        *o = table[modal_index(i)]?;
    }
    Some(out)
}

fn modal_interpolate(c: &[[f32; 3]; 16], frac: [f32; 4]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (corner, v) in c.iter().enumerate() {
        let mut w = 1.0;
        for (d, f) in frac.iter().enumerate() {
            w *= if (corner >> d) & 1 == 1 { *f } else { 1.0 - f };
        }
        for k in 0..3 {
            out[k] += w * v[k];
        }
    }
    out
}

fn modal_inside(cell: [usize; 4], frac: [f32; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|d| (cell[d] as f32 + frac[d]) / (MODAL_DIMS[d] - 1) as f32)
}

/// ln F1, F2, F3 of modal coordinates `x` at `length`.
fn modal_ln_formants(meter: &mut Meter, x: &[f32; 4], length: f32) -> Option<[f32; 3]> {
    meter
        .measure(&synthie_trombone::inverse::modal::from_coordinates(*x), length)
        .filter(|f| f.iter().all(|v| v.is_finite() && *v > 0.0))
        .map(|f| f.map(f32::ln))
}

/// The babble map of the modal body, `src/inverse_modes_table.rs`: the same
/// steps as the cosine map's over the modal grid, and instead of a size
/// correction of its own, how well the cosine map's serves modal shapes.
fn modal_main() {
    use synthie_trombone::inverse;
    let started = Instant::now();
    // The tract it is measured on: `--lengthened`, `--profile FILE`.
    let base = survey::base_from_args();
    let solved_for: Vec<String> = synthie_trombone::Tuning::FIELDS
        .iter()
        .filter(|f| f.name != "tongue_model")
        .filter_map(|f| {
            let v = base.get(f.name)?;
            (v.to_bits() != f.default.to_bits()).then(|| format!("{} {v}", f.name))
        })
        .collect();
    let d = MODAL_DIMS;
    let knee = inverse::modal::coordinates(&Shape {
        lips: Some(1.8),
        ..inverse::modal::from_coordinates([0.5; 4])
    })[3] * (d[3] - 1) as f32;
    assert!(
        (knee - knee.round()).abs() < 1e-3,
        "lip rounding 1.8 must fall on a grid line of the lip coordinate"
    );
    // 1. Nodes.
    let c = |k: usize, i: usize| i as f32 / (d[k] - 1) as f32;
    let mut coords = Vec::new();
    for a in 0..d[0] {
        for b in 0..d[1] {
            for e in 0..d[2] {
                for l in 0..d[3] {
                    coords.push([c(0, a), c(1, b), c(2, e), c(3, l)]);
                }
            }
        }
    }
    println!("measuring {} modal grid nodes...", coords.len());
    let measured = par_map(&coords, FULL, |m, x| modal_ln_formants(m, x, 1.0));
    let unmeasured = measured.iter().filter(|v| v.is_none()).count();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for f in measured.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(f[k]);
            hi[k] = hi[k].max(f[k]);
        }
    }
    for k in 0..3 {
        lo[k] = ((lo[k] - 0.005) * 1000.0).floor() / 1000.0;
        hi[k] = ((hi[k] + 0.005) * 1000.0).ceil() / 1000.0;
    }
    let mut codes = Vec::with_capacity(coords.len() * 5);
    let table: Vec<Option<[f32; 3]>> = measured
        .iter()
        .map(|f| {
            let (v, back) = match f {
                None => (1023u32, None),
                Some(f) => {
                    let mut v = 0u32;
                    let mut back = [0.0f32; 3];
                    for k in 0..3 {
                        let q = ((f[k] - lo[k]) / (hi[k] - lo[k]) * TOP)
                            .round()
                            .clamp(0.0, TOP) as u32;
                        v |= q << (10 * k);
                        back[k] = lo[k] + (hi[k] - lo[k]) * q as f32 / TOP;
                    }
                    (v, Some(back))
                }
            };
            for c in 0..5 {
                codes.push(B64[((v >> (6 * c)) & 63) as usize]);
            }
            back
        })
        .collect();
    println!(
        "  {unmeasured} without three formants, {:.1} s",
        started.elapsed().as_secs_f32()
    );

    // 2. Cell checks.
    let all_cells = modal_cells();
    let mut rng = Lcg(0x00ba_bb1e);
    let mut probes: Vec<(usize, [f32; 4])> = Vec::new();
    for (n, &cell) in all_cells.iter().enumerate() {
        if modal_corners(&table, cell).is_none() {
            continue;
        }
        for p in 0..CHECKS {
            let frac = if p == 0 {
                [0.5; 4]
            } else {
                [0; 4].map(|_| {
                    let r = rng.unit();
                    if r < 0.15 {
                        0.0
                    } else if r < 0.3 {
                        1.0
                    } else {
                        rng.range(0.1, 0.9)
                    }
                })
            };
            probes.push((n, frac));
        }
    }
    println!(
        "checking {} cells at {} points...",
        all_cells.len(),
        probes.len()
    );
    let truth = par_map(&probes, QUICK, |m, &(n, frac)| {
        modal_ln_formants(m, &modal_inside(all_cells[n], frac), 1.0)
    });
    let mut rough: Vec<bool> = all_cells
        .iter()
        .map(|&cell| {
            let Some(c) = modal_corners(&table, cell) else {
                return true;
            };
            let top = |k: usize| c.iter().fold(f32::MIN, |m, v| m.max(v[k]));
            let bottom = |k: usize| c.iter().fold(f32::MAX, |m, v| m.min(v[k]));
            top(0) > bottom(1) + LABEL_GAP || top(1) > bottom(2) + LABEL_GAP
        })
        .collect();
    for (&(n, frac), t) in probes.iter().zip(&truth) {
        if rough[n] {
            continue;
        }
        let Some(t) = t else {
            rough[n] = true;
            continue;
        };
        let c = modal_corners(&table, all_cells[n]).expect("checked above");
        let p = modal_interpolate(&c, frac);
        if (p[0] - t[0]).abs().max((p[1] - t[1]).abs()) > ROUGH_F12 || (p[2] - t[2]).abs() > ROUGH_F3
        {
            rough[n] = true;
        }
    }
    let n_rough = rough.iter().filter(|&&r| r).count();
    println!(
        "  {n_rough} of {} cells rough ({:.1} %); {:.1} s",
        all_cells.len(),
        100.0 * n_rough as f32 / all_cells.len() as f32,
        started.elapsed().as_secs_f32()
    );

    // 3. Reachable region.
    let mut lo12 = [f32::INFINITY; 2];
    let mut hi12 = [f32::NEG_INFINITY; 2];
    let hulls: Vec<Vec<(f32, f32)>> = all_cells
        .iter()
        .zip(&rough)
        .filter(|(_, r)| !**r)
        .map(|(&cell, _)| {
            let c = modal_corners(&table, cell).expect("smooth cells are measured");
            for v in &c {
                for k in 0..2 {
                    lo12[k] = lo12[k].min(v[k]);
                    hi12[k] = hi12[k].max(v[k]);
                }
            }
            hull(c.iter().map(|v| (v[0], v[1])).collect())
        })
        .collect();
    let range_hz = |k: usize| {
        (
            ((lo12[k].exp() / 10.0).floor() * 10.0),
            ((hi12[k].exp() / 10.0).ceil() * 10.0),
        )
    };
    let (r1, r2) = (range_hz(0), range_hz(1));
    let na = ((r1.1.ln() - r1.0.ln()) / REACH_STEP).ceil() as usize;
    let nb = ((r2.1.ln() - r2.0.ln()) / REACH_STEP).ceil() as usize;
    let centre = |i: usize, n: usize, r: (f32, f32)| {
        r.0.ln() + (r.1.ln() - r.0.ln()) * (i as f32 + 0.5) / n as f32
    };
    let mut raw = vec![false; na * nb];
    for h in &hulls {
        let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(a, b) in h {
            a0 = a0.min(a);
            a1 = a1.max(a);
            b0 = b0.min(b);
            b1 = b1.max(b);
        }
        let cell_of = |v: f32, r: (f32, f32), n: usize| {
            ((v - r.0.ln()) / (r.1.ln() - r.0.ln()) * n as f32).clamp(0.0, n as f32 - 1.0) as usize
        };
        for i in cell_of(a0, r1, na)..=cell_of(a1, r1, na) {
            for j in cell_of(b0, r2, nb)..=cell_of(b1, r2, nb) {
                if !raw[i * nb + j] && inside_hull(h, (centre(i, na, r1), centre(j, nb, r2))) {
                    raw[i * nb + j] = true;
                }
            }
        }
    }
    let at = |m: &[bool], i: isize, j: isize| {
        i >= 0
            && j >= 0
            && (i as usize) < na
            && (j as usize) < nb
            && m[i as usize * nb + j as usize]
    };
    let mut reach = vec![false; na * nb];
    for i in 0..na as isize {
        for j in 0..nb as isize {
            reach[i as usize * nb + j as usize] = at(&raw, i, j)
                && at(&raw, i - 1, j)
                && at(&raw, i + 1, j)
                && at(&raw, i, j - 1)
                && at(&raw, i, j + 1);
        }
    }
    println!(
        "  reachable: {} of {na}×{nb} cells over F1 {:.0}–{:.0} Hz, F2 {:.0}–{:.0} Hz",
        reach.iter().filter(|&&r| r).count(),
        r1.0,
        r1.1,
        r2.0,
        r2.1
    );
    for i in (0..na).rev().step_by(2) {
        let row: String = (0..nb)
            .map(|j| if reach[i * nb + j] { '#' } else { '.' })
            .collect();
        println!("  {:5.0} {row}", centre(i, na, r1).exp());
    }

    // 4. Tract length: how far the cosine map's correction is off for
    // modal shapes.
    let smooth: Vec<usize> = (0..all_cells.len()).filter(|&n| !rough[n]).collect();
    let mut size_probes: Vec<([f32; 4], f32)> = Vec::new();
    for _ in 0..MODAL_SIZE_SHAPES {
        let n = smooth[(rng.unit() * smooth.len() as f32) as usize % smooth.len()];
        let x = modal_inside(all_cells[n], [0; 4].map(|_| rng.unit()));
        for &l in &LENGTHS {
            size_probes.push((x, l));
        }
    }
    let sized = par_map(&size_probes, FULL, |m, (x, l)| modal_ln_formants(m, x, *l));
    let native_at = LENGTHS
        .iter()
        .position(|&l| (l - 1.0).abs() < 1e-6)
        .expect("1.0 is listed");
    let mut report = String::new();
    for (li, &l) in LENGTHS.iter().enumerate() {
        let _ = write!(report, "length {l:.1}:");
        for k in 0..3 {
            let mut left = Vec::new();
            for s in 0..MODAL_SIZE_SHAPES {
                let row = &sized[s * LENGTHS.len()..(s + 1) * LENGTHS.len()];
                if let (Some(n), Some(f)) = (row[native_at], row[li]) {
                    let want = inverse::formants_at_length(
                        [n[0].exp(), n[1].exp(), n[2].exp()],
                        l,
                    )[k];
                    left.push((f[k] - want.ln()).abs());
                }
            }
            let _ = write!(
                report,
                "  F{} p95 {:4.2} % max {:4.2} %",
                k + 1,
                100.0 * quantile(&mut left, 0.95),
                100.0 * quantile(&mut left, 1.0)
            );
        }
        report.push('\n');
    }
    println!("The cosine map's size correction on modal shapes, error:\n{report}");

    let mut src = String::new();
    let _ = writeln!(
        src,
        "//! Babble map table of the modal tongue body, generated by\n\
         //! `examples/babble.rs`; do not edit. Regenerate with\n\
         //! `cargo run -p synthie-trombone --example babble --release -- --modal --lengthened`.\n\
         //!\n\
         //! Measured on the tract {}.\n\
         //!\n\
         //! Measured {} grid nodes, {unmeasured} without three formants; {n_rough} of {} cells\n\
         //! rough. The cosine map's size correction, on modal shapes:\n//!\n//! ```text",
        if solved_for.is_empty() {
            "of the defaults".to_string()
        } else {
            format!("with {}", solved_for.join(", "))
        },
        coords.len(),
        all_cells.len(),
    );
    for line in report.lines() {
        let _ = writeln!(src, "//! {}", line.trim_start());
    }
    let _ = writeln!(
        src,
        "//! ```\n\n#![allow(clippy::unreadable_literal, clippy::excessive_precision)]\n"
    );
    let _ = writeln!(
        src,
        "/// Grid nodes along each coordinate: the body's three numbers, the lips."
    );
    let _ = writeln!(src, "pub(super) const DIMS: [usize; 4] = {MODAL_DIMS:?};");
    let _ = writeln!(src, "/// `ln` Hz of formant code 0, per formant.");
    let _ = writeln!(src, "pub(super) const F_LO: [f32; 3] = {lo:?};");
    let _ = writeln!(src, "/// `ln` Hz of formant code 1022, per formant.");
    let _ = writeln!(src, "pub(super) const F_HI: [f32; 3] = {hi:?};");
    let _ = writeln!(
        src,
        "/// Formants of every node: five base-64 characters, low bits first,\n\
         /// holding three ten-bit codes (F1 lowest); F1 code 1023 = unmeasured."
    );
    let _ = writeln!(src, "pub(super) static NODES: &[u8] = {};", literal(&codes));
    let _ = writeln!(src, "/// One bit per grid cell, six per character: rough.");
    let _ = writeln!(
        src,
        "pub(super) static ROUGH: &[u8] = {};",
        literal(&bits(&rough))
    );
    let _ = writeln!(src, "/// F1 range of the reachability grid, Hz.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_F1: (f32, f32) = ({:?}, {:?});",
        r1.0, r1.1
    );
    let _ = writeln!(src, "/// F2 range of the reachability grid, Hz.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_F2: (f32, f32) = ({:?}, {:?});",
        r2.0, r2.1
    );
    let _ = writeln!(src, "/// Reachability cells along F1 and F2.");
    let _ = writeln!(
        src,
        "pub(super) const REACH_SIZE: [usize; 2] = [{na}, {nb}];"
    );
    let _ = writeln!(
        src,
        "/// One bit per reachability cell (F1 major), six per character."
    );
    let _ = writeln!(
        src,
        "pub(super) static REACH: &[u8] = {};",
        literal(&bits(&reach))
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/inverse_modes_table.rs");
    std::fs::write(path, &src).expect("write src/inverse_modes_table.rs");
    println!(
        "wrote {path}: {} bytes, {:.1} s in all",
        src.len(),
        started.elapsed().as_secs_f32()
    );
}
