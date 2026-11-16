//! The babble map of the modal tongue body (`AnatomyConfig::tongue_model`
//! 1): formants back to the body's three numbers and the lips, as
//! [`super::solve`] does for the cosine body. The table
//! (`src/inverse_modes_table.rs`) is measured by
//! `cargo run -p synthie-trombone --example babble --release -- --modal`
//! over a grid of the three numbers (each over [`super::mode_range`]) and
//! the lip rounding; the size correction is the cosine map's (measured on
//! modal shapes by the same run: see the table's header).

use super::{
    LIPS_KNEE_AT, OPEN, Shape, TONGUE_DIAMETER, TONGUE_INDEX, bit, finite_or, from_unit_lips,
    locate, sextet, to_native, unit_lips,
};
use crate::math::{exp, ln};

#[path = "inverse_modes_table.rs"]
mod table;

/// Coordinates of the modal map: the body's three numbers, the lips.
const D: usize = 4;
/// Corners of a grid cell.
const CORNERS: usize = 1 << D;
/// Code of a node whose formants could not be measured.
const INVALID: u32 = 1023;
/// Largest code of a measured formant.
const TOP: f32 = 1022.0;
/// Pull towards the neutral body with lips at the knee (see
/// [`super::solve`]'s), and towards the previous shape.
const EASE: f32 = 3.0e-4;
const STAY: f32 = 1.0e-3;
/// Weight of F3 against F1 and F2 when it is asked for.
const F3_WEIGHT: f32 = 0.3;
/// Starting points taken from the table.
const STARTS: usize = 8;

/// Range of F1, Hz, that the modal map covers.
pub const F1_RANGE: (f32, f32) = table::REACH_F1;
/// Range of F2, Hz, that the modal map covers.
pub const F2_RANGE: (f32, f32) = table::REACH_F2;

/// Coordinate of body number `k` with value `q`, `0‥1` over
/// [`super::mode_range`].
fn unit_mode(k: usize, q: f32) -> f32 {
    let (lo, hi) = super::mode_range(k);
    ((q - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn from_unit_mode(k: usize, x: f32) -> f32 {
    let (lo, hi) = super::mode_range(k);
    lo + (hi - lo) * x.clamp(0.0, 1.0)
}

/// The relaxed shape: the neutral body, lips at the knee.
fn relaxed() -> [f32; D] {
    [unit_mode(0, 0.0), unit_mode(1, 0.0), unit_mode(2, 0.0), LIPS_KNEE_AT]
}

/// `shape` as the modal map's four coordinates: its three numbers (or its
/// cosine tongue's projection), and the lip coordinate of
/// [`Shape::coordinates`].
#[must_use]
pub fn coordinates(shape: &Shape) -> [f32; D] {
    let q = shape
        .modes
        .unwrap_or_else(|| super::project_tongue(shape.tongue_index, shape.tongue_diameter));
    [
        unit_mode(0, q[0]),
        unit_mode(1, q[1]),
        unit_mode(2, q[2]),
        shape.lips.map_or(1.0, unit_lips),
    ]
}

/// The shape at modal coordinates `x`: its body numbers, no narrowing, the
/// root at rest, and as its cosine tongue the place and height whose
/// projection is nearest the body (what the app's tongue handle shows).
#[must_use]
pub fn from_coordinates(x: [f32; D]) -> Shape {
    let q = [0, 1, 2].map(|k| from_unit_mode(k, x[k]));
    let (tongue_index, tongue_diameter) = nearest_tongue(q);
    Shape {
        tongue_index,
        tongue_diameter,
        lips: (x[3] < OPEN).then(|| from_unit_lips(x[3])),
        narrowing: None,
        pharynx: 0.0,
        modes: Some(q),
    }
}

/// The cosine tongue whose projection onto the modes is nearest `q`
/// (searched on a grid of place and height).
#[allow(clippy::cast_precision_loss)]
fn nearest_tongue(q: [f32; 3]) -> (f32, f32) {
    let mut best = (f32::INFINITY, (20.0, 2.75));
    for a in 0..=34 {
        let index = TONGUE_INDEX.0 + (TONGUE_INDEX.1 - TONGUE_INDEX.0) * a as f32 / 34.0;
        for b in 0..=15 {
            let d = TONGUE_DIAMETER.0 + (TONGUE_DIAMETER.1 - TONGUE_DIAMETER.0) * b as f32 / 15.0;
            let p = super::project_tongue(index, d);
            let e = (0..3).map(|k| (p[k] - q[k]) * (p[k] - q[k])).sum::<f32>();
            if e < best.0 {
                best = (e, (index, d));
            }
        }
    }
    best.1
}

/// `ln` of F1, F2 and F3 at a grid node, or `None` if it was not measured.
#[allow(clippy::cast_precision_loss)]
fn node(i: [usize; D]) -> Option<[f32; 3]> {
    let d = table::DIMS;
    let n = ((i[0] * d[1] + i[1]) * d[2] + i[2]) * d[3] + i[3];
    let bytes = table::NODES.get(5 * n..5 * n + 5)?;
    let v = bytes
        .iter()
        .rev()
        .fold(0u32, |acc, &c| (acc << 6) | sextet(c));
    if v & 1023 == INVALID {
        return None;
    }
    let mut f = [0.0f32; 3];
    for (k, f) in f.iter_mut().enumerate() {
        let q = ((v >> (10 * k)) & 1023) as f32;
        *f = table::F_LO[k] + (table::F_HI[k] - table::F_LO[k]) * q / TOP;
    }
    Some(f)
}

/// Whether a grid cell is rough (see the cosine map's).
fn rough(c: [usize; D]) -> bool {
    let d = table::DIMS;
    let n = ((c[0] * (d[1] - 1) + c[1]) * (d[2] - 1) + c[2]) * (d[3] - 1) + c[3];
    bit(table::ROUGH, n)
}

struct Eval {
    f: [f32; 3],
    g: [[f32; D]; 3],
}

/// The table's formants at `x`, interpolated between the corners of the
/// cell holding it, with their derivatives; `None` in an unmeasured or
/// rough cell.
#[allow(clippy::cast_precision_loss)]
fn eval(x: &[f32; D]) -> Option<Eval> {
    let mut base = [0usize; D];
    let mut frac = [0.0f32; D];
    for d in 0..D {
        (base[d], frac[d]) = locate(x[d], table::DIMS[d]);
    }
    if rough(base) {
        return None;
    }
    let mut out = Eval {
        f: [0.0; 3],
        g: [[0.0; D]; 3],
    };
    for corner in 0..CORNERS {
        let mut i = base;
        let mut w = 1.0f32;
        let mut part = [1.0f32; D];
        for d in 0..D {
            let up = (corner >> d) & 1 == 1;
            i[d] += usize::from(up);
            let wd = if up { frac[d] } else { 1.0 - frac[d] };
            w *= wd;
            for (e, p) in part.iter_mut().enumerate() {
                if e != d {
                    *p *= wd;
                }
            }
        }
        let f = node(i)?;
        for k in 0..3 {
            out.f[k] += w * f[k];
            for d in 0..D {
                let sign = if (corner >> d) & 1 == 1 { 1.0 } else { -1.0 };
                out.g[k][d] += sign * part[d] * f[k] * (table::DIMS[d] - 1) as f32;
            }
        }
    }
    Some(out)
}

/// The modal map's estimate of the F1, F2 and F3 of a shape of the modal
/// body (its numbers, or its cosine tongue's projection; its narrowing is
/// not counted), Hz, at the native tract length.
#[must_use]
pub fn predict(shape: &Shape) -> Option<[f32; 3]> {
    eval(&coordinates(shape)).map(|e| e.f.map(exp))
}

struct Goal {
    t: [f32; 3],
    w: [f32; 3],
    near: Option<[f32; D]>,
    relaxed: [f32; D],
}

impl Goal {
    fn cost(&self, x: &[f32; D], f: &[f32; 3]) -> f32 {
        let mut c = 0.0;
        for k in 0..3 {
            let r = f[k] - self.t[k];
            c += self.w[k] * r * r;
        }
        for d in 0..D {
            let r = x[d] - self.relaxed[d];
            c += EASE * r * r;
            if let Some(n) = self.near {
                let r = x[d] - n[d];
                c += STAY * r * r;
            }
        }
        c
    }

    /// Damped Gauss-Newton from `x` (as the cosine map's).
    fn refine(&self, mut x: [f32; D]) -> Option<([f32; D], f32)> {
        let mut e = eval(&x)?;
        let mut cost = self.cost(&x, &e.f);
        let mut damping = 1.0e-3f32;
        let pull = EASE + if self.near.is_some() { STAY } else { 0.0 };
        for _ in 0..40 {
            let mut h = [[0.0f32; D]; D];
            let mut b = [0.0f32; D];
            for k in 0..3 {
                let r = e.f[k] - self.t[k];
                for d in 0..D {
                    b[d] += self.w[k] * r * e.g[k][d];
                    for c in 0..D {
                        h[d][c] += self.w[k] * e.g[k][d] * e.g[k][c];
                    }
                }
            }
            for d in 0..D {
                b[d] += EASE * (x[d] - self.relaxed[d]);
                if let Some(n) = self.near {
                    b[d] += STAY * (x[d] - n[d]);
                }
                h[d][d] += pull;
            }
            let mut moved = false;
            while damping < 1.0e4 {
                let mut a = h;
                let mut rhs = b.map(|v| -v);
                for d in 0..D {
                    a[d][d] += damping * (h[d][d] + 1.0e-3);
                    let stuck = (x[d] <= 0.0 && b[d] > 0.0) || (x[d] >= 1.0 && b[d] < 0.0);
                    if stuck {
                        for c in 0..D {
                            a[d][c] = 0.0;
                            a[c][d] = 0.0;
                        }
                        a[d][d] = 1.0;
                        rhs[d] = 0.0;
                    }
                }
                let Some(step) = solve_d(a, rhs) else { break };
                let mut y = x;
                for d in 0..D {
                    y[d] = (x[d] + step[d]).clamp(0.0, 1.0);
                }
                if let Some(ey) = eval(&y) {
                    let cy = self.cost(&y, &ey.f);
                    if cy < cost {
                        let size = (0..D).fold(0.0f32, |m, d| m.max((y[d] - x[d]).abs()));
                        x = y;
                        e = ey;
                        cost = cy;
                        damping = (damping * 0.3).max(1.0e-6);
                        moved = size > 1.0e-6;
                        break;
                    }
                }
                damping *= 8.0;
            }
            if !moved {
                break;
            }
        }
        Some((x, cost))
    }
}

fn solve_d(mut a: [[f32; D]; D], mut b: [f32; D]) -> Option<[f32; D]> {
    for p in 0..D {
        let pivot = a[p][p];
        if pivot.abs() < 1.0e-12 || !pivot.is_finite() {
            return None;
        }
        for r in p + 1..D {
            let m = a[r][p] / pivot;
            for c in p..D {
                a[r][c] -= m * a[p][c];
            }
            b[r] -= m * b[p];
        }
    }
    let mut x = [0.0f32; D];
    for p in (0..D).rev() {
        let mut s = b[p];
        for c in p + 1..D {
            s -= a[p][c] * x[c];
        }
        x[p] = s / a[p][p];
    }
    Some(x)
}

/// A point just inside a cell next to grid node `i` that is not rough.
#[allow(clippy::cast_precision_loss)]
fn start_at(i: [usize; D]) -> Option<[f32; D]> {
    let d = table::DIMS;
    for around in 0..CORNERS {
        let mut cell = [0usize; D];
        let mut x = [0.0f32; D];
        let mut fits = true;
        for k in 0..D {
            let below = (around >> k) & 1 == 1;
            if (below && i[k] == 0) || (!below && i[k] == d[k] - 1) {
                fits = false;
                break;
            }
            cell[k] = if below { i[k] - 1 } else { i[k] };
            let nudge = if below { -0.01 } else { 0.01 };
            x[k] = ((i[k] as f32 + nudge) / (d[k] - 1) as f32).clamp(0.0, 1.0);
        }
        if fits && !rough(cell) && eval(&x).is_some() {
            return Some(x);
        }
    }
    None
}

/// Grid nodes with the lowest cost, at most one per neighbourhood.
#[allow(clippy::cast_precision_loss)]
fn candidates(goal: &Goal) -> [Option<(f32, [usize; D])>; STARTS] {
    let d = table::DIMS;
    let mut best: [Option<(f32, [usize; D])>; STARTS] = [None; STARTS];
    let coord = |k: usize, i: usize| i as f32 / (d[k] - 1) as f32;
    for i0 in 0..d[0] {
        for i1 in 0..d[1] {
            for i2 in 0..d[2] {
                for i3 in 0..d[3] {
                    let i = [i0, i1, i2, i3];
                    let Some(f) = node(i) else { continue };
                    let x = [0, 1, 2, 3].map(|k| coord(k, i[k]));
                    let c = goal.cost(&x, &f);
                    let close = |j: &[usize; D]| (0..D).all(|k| i[k].abs_diff(j[k]) <= 1);
                    let slot = if let Some(near) =
                        best.iter_mut().find(|b| b.is_some_and(|(_, j)| close(&j)))
                    {
                        Some(near)
                    } else if let Some(free) = best.iter_mut().find(|b| b.is_none()) {
                        Some(free)
                    } else {
                        best.iter_mut().max_by(|a, b| {
                            a.map_or(0.0, |v| v.0).total_cmp(&b.map_or(0.0, |v| v.0))
                        })
                    };
                    if let Some(slot) = slot
                        && slot.is_none_or(|(old, _)| c < old)
                    {
                        *slot = Some((c, i));
                    }
                }
            }
        }
    }
    best
}

/// [`super::solve`] on the modal body: the shape whose formants at the
/// native tract length are closest to F1 `f1` and F2 `f2` (Hz) and, if
/// given, F3 `f3`; the most relaxed one (the neutral body, lips at the
/// knee), or with `near` the one closest to it.
#[must_use]
pub fn solve(f1: f32, f2: f32, f3: Option<f32>, near: Option<&Shape>) -> Shape {
    let (f1, f2) = nearest_reachable(finite_or(f1, 500.0), finite_or(f2, 1500.0));
    let f3 = f3.filter(|v| v.is_finite() && *v > 0.0);
    let goal = Goal {
        t: [ln(f1), ln(f2), ln(f3.unwrap_or(2500.0))],
        w: [1.0, 1.0, if f3.is_some() { F3_WEIGHT } else { 0.0 }],
        near: near.map(coordinates),
        relaxed: relaxed(),
    };
    let mut best: Option<([f32; D], f32)> = goal.near.and_then(|n| goal.refine(n));
    for (_, i) in candidates(&goal).into_iter().flatten() {
        let Some(found) = start_at(i).and_then(|x| goal.refine(x)) else {
            continue;
        };
        if best.is_none_or(|b| found.1 < b.1) {
            best = Some(found);
        }
    }
    from_coordinates(best.map_or_else(|| near.map_or(relaxed(), coordinates), |(x, _)| x))
}

/// [`solve`] for a voice of `tract_length`, with the cosine map's size
/// correction.
#[must_use]
pub fn solve_for(
    f1: f32,
    f2: f32,
    f3: Option<f32>,
    near: Option<&Shape>,
    tract_length: f32,
) -> Shape {
    let l = super::sane_length(tract_length);
    let f1 = to_native(0, finite_or(f1, 500.0 / l), l);
    let f2 = to_native(1, finite_or(f2, 1500.0 / l), l);
    let f3 = f3
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| to_native(2, v, l));
    solve(f1, f2, f3, near)
}

/// [`reachable`] for a voice of `tract_length`.
#[must_use]
pub fn reachable_for(f1: f32, f2: f32, tract_length: f32) -> bool {
    let l = super::sane_length(tract_length);
    reachable(to_native(0, f1, l), to_native(1, f2, l))
}

/// [`nearest_reachable`] for a voice of `tract_length`.
#[must_use]
pub fn nearest_reachable_for(f1: f32, f2: f32, tract_length: f32) -> (f32, f32) {
    let l = super::sane_length(tract_length);
    let (a, b) = nearest_reachable(
        to_native(0, finite_or(f1, 500.0 / l), l),
        to_native(1, finite_or(f2, 1500.0 / l), l),
    );
    (super::from_native(0, a, l), super::from_native(1, b, l))
}

fn axis(hz: f32, range: (f32, f32), cells: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let n = cells as f32;
    (ln(hz) - ln(range.0)) / (ln(range.1) - ln(range.0)) * n
}

fn axis_hz(pos: f32, range: (f32, f32), cells: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let n = cells as f32;
    exp(ln(range.0) + (ln(range.1) - ln(range.0)) * pos / n)
}

fn reach_cell(a: usize, b: usize) -> bool {
    bit(table::REACH, a * table::REACH_SIZE[1] + b)
}

/// Whether the modal body can make F1 `f1` and F2 `f2` (Hz) at the native
/// tract length.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn reachable(f1: f32, f2: f32) -> bool {
    if !(f1.is_finite() && f2.is_finite() && f1 > 0.0 && f2 > 0.0) {
        return false;
    }
    let [na, nb] = table::REACH_SIZE;
    let a = axis(f1, F1_RANGE, na);
    let b = axis(f2, F2_RANGE, nb);
    #[allow(clippy::cast_precision_loss)]
    let inside = (0.0..na as f32).contains(&a) && (0.0..nb as f32).contains(&b);
    inside && reach_cell(a as usize, b as usize)
}

/// The reachable point of the modal body nearest to `(f1, f2)` (Hz, on a
/// logarithmic scale); the target itself if it is reachable.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn nearest_reachable(f1: f32, f2: f32) -> (f32, f32) {
    let f1 = finite_or(f1, 500.0);
    let f2 = finite_or(f2, 1500.0);
    if reachable(f1, f2) {
        return (f1, f2);
    }
    let [na, nb] = table::REACH_SIZE;
    let a = axis(f1, F1_RANGE, na);
    let b = axis(f2, F2_RANGE, nb);
    let sa = (ln(F1_RANGE.1) - ln(F1_RANGE.0)) / na as f32;
    let sb = (ln(F2_RANGE.1) - ln(F2_RANGE.0)) / nb as f32;
    let mut best: Option<(f32, f32, f32)> = None;
    for i in 0..na {
        for j in 0..nb {
            if !reach_cell(i, j) {
                continue;
            }
            let pa = a.clamp(i as f32 + 0.02, i as f32 + 0.98);
            let pb = b.clamp(j as f32 + 0.02, j as f32 + 0.98);
            let dist = ((pa - a) * sa) * ((pa - a) * sa) + ((pb - b) * sb) * ((pb - b) * sb);
            if best.is_none_or(|(d, _, _)| dist < d) {
                best = Some((dist, pa, pb));
            }
        }
    }
    best.map_or((f1, f2), |(_, pa, pb)| {
        (axis_hz(pa, F1_RANGE, na), axis_hz(pb, F2_RANGE, nb))
    })
}
