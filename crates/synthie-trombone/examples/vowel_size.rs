//! Generate `src/vowel_size_table.rs`: how each physical vowel's shape
//! changes with the speaker's size, so that its formants scale exactly as
//! `1 / length`.
//!
//! The native shapes (`PHYSICAL_VOWELS` in `src/phoneme.rs`) were fitted at
//! tract length 1.0. Used unchanged on a shorter or longer tract their F1
//! drifts from `native / length` by up to 8.5 % at length 0.6, because the
//! yielding walls, the side cavities and the lip load do not scale with
//! the tube. For every vowel and every row of a grid of tract lengths this
//!
//! 1. starts from the shape of the row next to it, nearer to 1.0 (at first
//!    the native shape), stepped on along the way the shape has been
//!    moving;
//! 2. refines it against the real engine (damped Gauss-Newton on measured
//!    formants, with derivatives by finite differences) until F1 and F2 are
//!    the native ones divided by the length, and F3 as near as the other
//!    two allow, staying close to the neighbouring row so that the rows
//!    make a smooth path. If that misses, it also tries the babble map's
//!    answer (`inverse::solve_for`, started near the same shape) and keeps
//!    the better one. Lips and narrowing are never tighter than the
//!    tightest native vowel's, so no shape hisses;
//! 3. verifies the table as the library will read it (quantised, and
//!    interpolated between rows) at the rows, between them, and at the
//!    lengths the tests use, measuring each shape through the engine.
//!
//! ```sh
//! cargo run -p synthie-trombone --example vowel_size --release
//! ```
//!
//! Uses every CPU thread; takes under a minute.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/survey.rs"]
mod survey;

use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use survey::Meter;
use synthie_trombone::Phoneme;
use synthie_trombone::inverse::{self, Shape};

/// The physical vowels, in the order of `PHYSICAL_VOWELS`.
const VOWELS: [Phoneme; 14] = [
    Phoneme::Iy,
    Phoneme::Ih,
    Phoneme::Eh,
    Phoneme::Ae,
    Phoneme::Ah,
    Phoneme::Aa,
    Phoneme::Ao,
    Phoneme::Uh,
    Phoneme::Uw,
    Phoneme::Ow,
    Phoneme::Er,
    Phoneme::Ax,
    Phoneme::W,
    Phoneme::Y,
];
/// Tract length of the first table row...
const FIRST: f32 = 0.5;
/// ...the step between rows...
const STEP: f32 = 0.1;
/// ...and their number: 0.5 to 1.6, all that `set_tract_length` allows.
const ROWS: usize = 12;
/// The row at the native length.
const NATIVE_ROW: usize = 5;
/// Weight of F3 against F1 and F2 (on squared `ln` errors).
const F3_WEIGHT: f32 = 0.3;
/// Pull towards the neighbouring row's shape, per squared coordinate:
/// small next to a formant error (moving a coordinate by 0.1 costs as much
/// as a 0.3 % error), so it only chooses among shapes that reach the
/// target about equally well, and among those the nearest.
const STAY: f32 = 1.0e-3;
/// Pull towards the shape the two rows before predict (going on in a
/// straight line), per squared coordinate: keeps the path from bending.
const BEND: f32 = 3.0e-3;
/// Finite-difference step, in shape coordinates.
const H: f32 = 0.008;
/// Tongue body positions and heights the engine allows.
const TONGUE_INDEX: (f32, f32) = (12.0, 29.0);
const TONGUE_DIAMETER: (f32, f32) = (2.05, 3.5);
/// Lip rounding searched: from the tightest vowel lips to lips that hardly
/// touch the tract.
const LIPS: (f32, f32) = (MIN_LIPS, 3.3);
/// Where the narrowing may sit: along the tongue, up to the teeth (the
/// babble map stops at 34; /i/ in a long tract wants a little more).
const NARROWING_AT: (f32, f32) = (12.0, 38.0);
/// Narrowing openings searched.
const NARROWING: (f32, f32) = (MIN_NARROWING, 3.0);
/// What a touch diameter loses before it becomes a tract diameter.
const TOUCH: f32 = 0.3;
/// F1 and F2 errors at which a row counts as solved...
const GOOD: f32 = 1.5e-3;
/// ...and beyond which the babble map's answer is tried too.
const TRY_MAP: f32 = 5.0e-3;
/// The tests' limit for F1 and F2 (relative error).
const LIMIT: f32 = 0.03;
/// Tightest lips and narrowing of the native vowels.
const MIN_LIPS: f32 = 0.65;
const MIN_NARROWING: f32 = 0.7;

/// Tract length of row `k`, computed as the library does.
fn row_length(k: usize) -> f32 {
    FIRST + STEP * k as f32
}

fn native(p: Phoneme) -> Shape {
    p.physical_shape(1.0).expect("a physical vowel")
}

/// The five raw numbers of a shape: tongue index, tongue diameter, lips
/// (`0` when spread), narrowing cell and narrowing diameter.
fn raw(s: &Shape) -> [f32; 5] {
    let n = s.narrowing.expect("physical vowels narrow the tongue");
    [
        s.tongue_index,
        s.tongue_diameter,
        s.lips.unwrap_or(0.0),
        n.0,
        n.1,
    ]
}

/// A shape as five coordinates, each `0‥1` over the ranges searched:
/// tongue index, tongue diameter, lips, narrowing place, narrowing
/// opening; the openings on a logarithmic scale. Spread lips are `1`.
fn to_x(s: &Shape) -> [f32; 5] {
    let lin = |v: f32, r: (f32, f32)| (v - r.0) / (r.1 - r.0);
    let log = |v: f32, r: (f32, f32)| {
        ((v - TOUCH).ln() - (r.0 - TOUCH).ln()) / ((r.1 - TOUCH).ln() - (r.0 - TOUCH).ln())
    };
    let n = s.narrowing.expect("physical vowels narrow the tongue");
    [
        lin(s.tongue_index, TONGUE_INDEX),
        lin(s.tongue_diameter, TONGUE_DIAMETER),
        s.lips.map_or(1.0, |l| log(l, LIPS)),
        lin(n.0, NARROWING_AT),
        log(n.1, NARROWING),
    ]
    .map(|v| v.clamp(0.0, 1.0))
}

/// The shape at coordinates `x` (see [`to_x`]); lips only if `rounded`.
fn from_x(x: &[f32; 5], rounded: bool) -> Shape {
    let lin = |v: f32, r: (f32, f32)| r.0 + (r.1 - r.0) * v.clamp(0.0, 1.0);
    let log = |v: f32, r: (f32, f32)| {
        let (lo, hi) = ((r.0 - TOUCH).ln(), (r.1 - TOUCH).ln());
        (TOUCH + (lo + (hi - lo) * v.clamp(0.0, 1.0)).exp()).max(r.0)
    };
    Shape {
        tongue_index: lin(x[0], TONGUE_INDEX),
        tongue_diameter: lin(x[1], TONGUE_DIAMETER),
        lips: rounded.then(|| log(x[2], LIPS)),
        narrowing: Some((lin(x[3], NARROWING_AT), log(x[4], NARROWING))),
    }
}

/// What one row of one vowel aims for.
struct Goal {
    /// `ln` of the target formants.
    t: [f32; 3],
    /// Coordinates of the neighbouring row, to stay near...
    near: [f32; 5],
    /// ...and where the rows before point, not to bend away from.
    ahead: [f32; 5],
    /// Which coordinates may move.
    free: [bool; 5],
    rounded: bool,
    length: f32,
}

impl Goal {
    fn residuals(&self, f: [f32; 3]) -> [f32; 3] {
        [
            f[0].ln() - self.t[0],
            f[1].ln() - self.t[1],
            F3_WEIGHT.sqrt() * (f[2].ln() - self.t[2]),
        ]
    }

    fn cost(&self, x: &[f32; 5], r: &[f32; 3]) -> f32 {
        let mut c: f32 = r.iter().map(|v| v * v).sum();
        for d in 0..5 {
            c += STAY * (x[d] - self.near[d]) * (x[d] - self.near[d])
                + BEND * (x[d] - self.ahead[d]) * (x[d] - self.ahead[d]);
        }
        c
    }

    fn eval(&self, m: &mut Meter, x: &[f32; 5]) -> Option<([f32; 3], f32)> {
        let f = m.measure(&from_x(x, self.rounded), self.length)?;
        let r = self.residuals(f);
        Some((r, self.cost(x, &r)))
    }

    /// Damped Gauss-Newton from `x`, on measured formants. Returns the
    /// coordinates reached, their residuals and cost.
    fn refine(&self, m: &mut Meter, mut x: [f32; 5]) -> Option<([f32; 5], [f32; 3], f32)> {
        for d in 0..5 {
            x[d] = x[d].clamp(0.0, 1.0);
        }
        let (mut r, mut cost) = self.eval(m, &x)?;
        let mut damping = 1.0e-3f32;
        for _ in 0..20 {
            if r[0].abs() < 0.3 * GOOD && r[1].abs() < 0.3 * GOOD {
                break;
            }
            // Derivatives, stepping into the allowed box.
            let mut j = [[0.0f32; 5]; 3];
            for d in 0..5 {
                if !self.free[d] {
                    continue;
                }
                let h = if x[d] + H <= 1.0 { H } else { -H };
                let mut y = x;
                y[d] += h;
                let Some((ry, _)) = self.eval(m, &y) else {
                    continue;
                };
                for k in 0..3 {
                    j[k][d] = (ry[k] - r[k]) / h;
                }
            }
            let mut a = [[0.0f32; 5]; 5];
            let mut b = [0.0f32; 5];
            for d in 0..5 {
                for k in 0..3 {
                    b[d] += j[k][d] * r[k];
                    for c in 0..5 {
                        a[d][c] += j[k][d] * j[k][c];
                    }
                }
                b[d] += STAY * (x[d] - self.near[d]) + BEND * (x[d] - self.ahead[d]);
                a[d][d] += STAY + BEND;
            }
            let mut moved = false;
            while damping < 1.0e3 {
                let mut lhs = a;
                let mut rhs = b.map(|v| -v);
                for d in 0..5 {
                    lhs[d][d] += damping * (a[d][d] + 1.0e-4);
                    let stuck =
                        !self.free[d] || (x[d] <= 0.0 && b[d] > 0.0) || (x[d] >= 1.0 && b[d] < 0.0);
                    if stuck {
                        for c in 0..5 {
                            lhs[d][c] = 0.0;
                            lhs[c][d] = 0.0;
                        }
                        lhs[d][d] = 1.0;
                        rhs[d] = 0.0;
                    }
                }
                let Some(step) = solve5(lhs, rhs) else { break };
                let mut y = x;
                for d in 0..5 {
                    y[d] = (x[d] + step[d]).clamp(0.0, 1.0);
                }
                if let Some((ry, cy)) = self.eval(m, &y)
                    && cy < cost
                {
                    let size = (0..5).fold(0.0f32, |s, d| s.max((y[d] - x[d]).abs()));
                    x = y;
                    r = ry;
                    cost = cy;
                    damping = (damping * 0.3).max(1.0e-6);
                    moved = size > 1.0e-5;
                    break;
                }
                damping *= 6.0;
            }
            if !moved {
                break;
            }
        }
        Some((x, r, cost))
    }
}

/// Solve the 5×5 system `a · x = b` by Gaussian elimination.
fn solve5(mut a: [[f32; 5]; 5], mut b: [f32; 5]) -> Option<[f32; 5]> {
    for p in 0..5 {
        let pivot = a[p][p];
        if pivot.abs() < 1.0e-12 || !pivot.is_finite() {
            return None;
        }
        for r in p + 1..5 {
            let m = a[r][p] / pivot;
            for c in p..5 {
                a[r][c] -= m * a[p][c];
            }
            b[r] -= m * b[p];
        }
    }
    let mut x = [0.0f32; 5];
    for p in (0..5).rev() {
        let mut s = b[p];
        for c in p + 1..5 {
            s -= a[p][c] * x[c];
        }
        x[p] = s / a[p][p];
    }
    Some(x)
}

/// One solved row.
#[derive(Clone, Copy)]
struct Solved {
    x: [f32; 5],
    /// Relative errors of F1, F2, F3 against `native / length`.
    err: [f32; 3],
    /// Whether the babble map's start won.
    from_map: bool,
}

/// Solve the rows of vowel `p` from the native one outwards, `up` towards
/// longer tracts or down towards shorter ones.
fn chain(m: &mut Meter, p: Phoneme, f_native: [f32; 3], up: bool) -> Vec<(usize, Solved)> {
    let s0 = native(p);
    let x0 = to_x(&s0);
    let rounded = s0.lips.is_some();
    // /a/ in a small body keeps its throat: the narrowing that would lift
    // F1 the last 2 % sits deep in the pharynx, and tightening it there
    // makes the vowel louder, not only higher (by 0.4 dB at length 0.6,
    // which upsets the balance of /s/ against /a/ that the physical gates
    // hold). Its tongue still moves.
    let throat = p == Phoneme::Aa && !up;
    let free = [true, true, rounded, !throat, !throat];
    let rows: Vec<usize> = if up {
        (NATIVE_ROW + 1..ROWS).collect()
    } else {
        (0..NATIVE_ROW).rev().collect()
    };
    let (mut before, mut last) = (x0, x0);
    let mut out = Vec::new();
    for k in rows {
        let length = row_length(k);
        let target = f_native.map(|f| f / length);
        // Carry on the way the shape has been moving.
        let mut ahead = [0.0f32; 5];
        for d in 0..5 {
            ahead[d] = if free[d] {
                (2.0 * last[d] - before[d]).clamp(0.0, 1.0)
            } else {
                last[d]
            };
        }
        let goal = Goal {
            t: target.map(f32::ln),
            near: last,
            ahead,
            free,
            rounded,
            length,
        };
        let mut best = goal.refine(m, ahead).map(|(x, r, c)| (x, r, c, false));
        let miss = |r: &[f32; 3]| r[0].abs().max(r[1].abs());
        if best.is_none_or(|b| miss(&b.1) > TRY_MAP) {
            let near = from_x(&last, rounded);
            let guess =
                inverse::solve_for(target[0], target[1], Some(target[2]), Some(&near), length);
            let mut g = to_x(&guess);
            if !rounded {
                g[2] = 1.0;
            }
            if guess.narrowing.is_none() {
                g[4] = 1.0;
            }
            if let Some((x, r, c)) = goal.refine(m, g)
                && best.is_none_or(|b| c < b.2)
            {
                best = Some((x, r, c, true));
            }
        }
        let (x, r, _, from_map) = best.expect("a measurable shape");
        let err = [
            r[0].exp() - 1.0,
            r[1].exp() - 1.0,
            (r[2] / F3_WEIGHT.sqrt()).exp() - 1.0,
        ];
        out.push((k, Solved { x, err, from_map }));
        before = last;
        last = x;
    }
    out
}

/// Run `f` on every item on all threads, each with its own [`Meter`].
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&mut Meter, &T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, R)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut meter = Meter::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let r = f(&mut meter, item);
                    out.lock().expect("lock").push((i, r));
                }
            });
        }
    });
    let mut out = out.into_inner().expect("lock");
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, r)| r).collect()
}

/// The shape the library makes of the table at `length`, with the same
/// arithmetic as `vowel_size::at_length`.
fn interpolate(native: &Shape, rows: &[[i16; 5]; ROWS], length: f32) -> Shape {
    if length.to_bits() == 1.0f32.to_bits() {
        return *native;
    }
    let last = (ROWS - 1) as f32;
    let t = ((length - FIRST) / STEP).clamp(0.0, last);
    let a = (t as usize).min(ROWS - 2);
    let s = t - a as f32;
    let d = |k: usize| {
        let (lo, hi) = (f32::from(rows[a][k]), f32::from(rows[a + 1][k]));
        (lo + (hi - lo) * s) * 1.0e-3
    };
    let n = native.narrowing.expect("narrowing");
    Shape {
        tongue_index: native.tongue_index + d(0),
        tongue_diameter: native.tongue_diameter + d(1),
        lips: native.lips.map(|l| (l + d(2)).max(MIN_LIPS)),
        narrowing: Some((n.0 + d(3), (n.1 + d(4)).max(MIN_NARROWING))),
    }
}

/// Relative errors of `f` against `native / length`.
fn errors(f: Option<[f32; 3]>, f_native: [f32; 3], length: f32) -> [f32; 3] {
    f.map_or([f32::INFINITY; 3], |f| {
        [0, 1, 2].map(|k| f[k] / (f_native[k] / length) - 1.0)
    })
}

fn main() {
    let started = Instant::now();

    // Native formants, measured as the tests measure them.
    let natives: Vec<[f32; 3]> = par_map(&VOWELS, |_, &p| {
        survey::measure(&native(p), 1.0).expect("native formants")
    });
    for (p, f) in VOWELS.iter().zip(&natives) {
        println!("{p:>2} native {:.0} / {:.0} / {:.0} Hz", f[0], f[1], f[2]);
    }

    // Solve every vowel's rows, outwards from the native one.
    let jobs: Vec<(usize, bool)> = (0..VOWELS.len())
        .flat_map(|v| [(v, false), (v, true)])
        .collect();
    let chains = par_map(&jobs, |m, &(v, up)| chain(m, VOWELS[v], natives[v], up));
    let mut solved: Vec<[Option<Solved>; ROWS]> = vec![[None; ROWS]; VOWELS.len()];
    for (&(v, _), rows) in jobs.iter().zip(chains) {
        for (k, s) in rows {
            solved[v][k] = Some(s);
        }
    }
    println!("solved in {:.1} s", started.elapsed().as_secs_f32());

    // Quantise to thousandths of the raw shape numbers.
    let mut table = vec![[[0i16; 5]; ROWS]; VOWELS.len()];
    for (v, &p) in VOWELS.iter().enumerate() {
        let base = raw(&native(p));
        for k in 0..ROWS {
            let Some(s) = solved[v][k] else { continue };
            let shape = from_x(&s.x, native(p).lips.is_some());
            let r = raw(&shape);
            for d in 0..5 {
                let delta = if d == 2 && native(p).lips.is_none() {
                    0.0
                } else {
                    r[d] - base[d]
                };
                table[v][k][d] = (delta * 1000.0).round() as i16;
            }
        }
    }

    // Verify: rows, half-way between them, and the tests' lengths.
    let mut lengths: Vec<f32> = (0..ROWS).map(row_length).collect();
    lengths.extend((0..ROWS - 1).map(|k| row_length(k) + 0.5 * STEP));
    lengths.extend([0.85, 0.93, 1.15]);
    lengths.sort_by(f32::total_cmp);
    lengths.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    let checks: Vec<(usize, f32)> = (0..VOWELS.len())
        .flat_map(|v| lengths.iter().map(move |&l| (v, l)))
        .collect();
    let measured = par_map(&checks, |m, &(v, l)| {
        let p = VOWELS[v];
        let before = m.measure(&native(p), l);
        let after = m.measure(&interpolate(&native(p), &table[v], l), l);
        (errors(before, natives[v], l), errors(after, natives[v], l))
    });

    let mut report = String::new();
    let _ = writeln!(
        report,
        "Worst error against native / length over all {} vowels (F1, F2, F3):",
        VOWELS.len()
    );
    // The same, shorter, for the table's header: table rows only.
    let mut summary = String::from(
        "Largest F1 / F2 / F3 error against native / length, of all vowels:\n\
         length: unchanged shapes -> these shapes\n",
    );
    let mut misses = Vec::new();
    // Per vowel: (first length, last length, largest F1 error) missing it.
    let mut miss_span: Vec<Option<(f32, f32, f32)>> = vec![None; VOWELS.len()];
    for &l in &lengths {
        let mut before = [0.0f32; 3];
        let mut after = [0.0f32; 3];
        let mut worst = [(0.0f32, Phoneme::Aa); 3];
        for (&(v, cl), (b, a)) in checks.iter().zip(&measured) {
            if cl.to_bits() != l.to_bits() {
                continue;
            }
            for k in 0..3 {
                before[k] = before[k].max(b[k].abs());
                if a[k].abs() > worst[k].0 {
                    worst[k] = (a[k].abs(), VOWELS[v]);
                }
                after[k] = after[k].max(a[k].abs());
            }
            if a[0].abs() > LIMIT || a[1].abs() > LIMIT {
                let m = miss_span[v].get_or_insert((l, l, 0.0));
                m.1 = l;
                if a[0].abs() > m.2.abs() {
                    m.2 = a[0];
                }
                misses.push(format!(
                    "{} at length {l:.2}: F1 {:+.2} %, F2 {:+.2} %",
                    VOWELS[v],
                    100.0 * a[0],
                    100.0 * a[1]
                ));
            }
        }
        let _ = writeln!(
            report,
            "length {l:.2}: unchanged shapes {:4.1} / {:4.1} / {:4.1} %, sized shapes {:4.2} / {:4.2} / {:4.2} % ({} / {} / {})",
            100.0 * before[0],
            100.0 * before[1],
            100.0 * before[2],
            100.0 * after[0],
            100.0 * after[1],
            100.0 * after[2],
            worst[0].1,
            worst[1].1,
            worst[2].1,
        );
        if (0..ROWS).any(|k| row_length(k).to_bits() == l.to_bits()) {
            let _ = writeln!(
                summary,
                "{l:.1}: {:4.1} / {:3.1} / {:3.1} % -> {:4.2} / {:4.2} / {:4.2} %",
                100.0 * before[0],
                100.0 * before[1],
                100.0 * before[2],
                100.0 * after[0],
                100.0 * after[1],
                100.0 * after[2],
            );
        }
    }
    for (v, m) in miss_span.iter().enumerate() {
        if let Some((from, to, worst)) = m {
            let _ = writeln!(
                summary,
                "{} misses {:.0} % from length {from:.2} to {to:.2} (F1 up to {:+.1} %)",
                VOWELS[v],
                100.0 * LIMIT,
                100.0 * worst
            );
        }
    }
    if misses.is_empty() {
        let _ = writeln!(
            report,
            "Every vowel within {:.0} % for F1 and F2 at every length checked.",
            100.0 * LIMIT
        );
    } else {
        let _ = writeln!(report, "Missing {:.0} %:", 100.0 * LIMIT);
        for m in &misses {
            let _ = writeln!(report, "  {m}");
        }
    }
    let from_map: Vec<String> = solved
        .iter()
        .enumerate()
        .flat_map(|(v, rows)| {
            rows.iter().enumerate().filter_map(move |(k, s)| {
                s.filter(|s| s.from_map)
                    .map(|_| format!("{} {:.1}", VOWELS[v], row_length(k)))
            })
        })
        .collect();
    if !from_map.is_empty() {
        let _ = writeln!(
            report,
            "Started from the babble map: {}",
            from_map.join(", ")
        );
    }
    print!("{report}");
    for (v, rows) in solved.iter().enumerate() {
        for (k, s) in rows.iter().enumerate() {
            if let Some(s) = s {
                println!(
                    "  {:>2} {:.1}: F1 {:+.2} % F2 {:+.2} % F3 {:+.2} %  {:?}",
                    VOWELS[v],
                    row_length(k),
                    100.0 * s.err[0],
                    100.0 * s.err[1],
                    100.0 * s.err[2],
                    table[v][k]
                );
            }
        }
    }

    // Write the table.
    let mut src = String::new();
    let _ = writeln!(
        src,
        "//! Vowel shapes by tract length, generated by `examples/vowel_size.rs`;\n\
         //! do not edit. Regenerate with\n\
         //! `cargo run -p synthie-trombone --example vowel_size --release`.\n\
         //!\n\
         //! ```text"
    );
    for line in summary.lines() {
        let _ = writeln!(src, "//! {line}");
    }
    let _ = writeln!(
        src,
        "//! ```\n\nuse crate::phoneme::Phoneme;\n\n\
         /// Tract length of the first row.\n\
         pub(super) const FIRST: f32 = {FIRST:?};\n\
         /// Tract length between rows.\n\
         pub(super) const STEP: f32 = {STEP:?};\n\
         /// Per vowel and row: how much tongue index, tongue diameter, lips,\n\
         /// narrowing cell and narrowing diameter differ from the native\n\
         /// shape, in thousandths.\n\
         #[rustfmt::skip]\n\
         pub(super) static DELTAS: &[(Phoneme, [[i16; 5]; {ROWS}])] = &["
    );
    for (v, p) in VOWELS.iter().enumerate() {
        let _ = writeln!(src, "    (Phoneme::{p:?}, [");
        for rows in table[v].chunks(4) {
            let rows: Vec<String> = rows
                .iter()
                .map(|row| {
                    let cells: Vec<String> = row.iter().map(i16::to_string).collect();
                    format!("[{}]", cells.join(","))
                })
                .collect();
            let _ = writeln!(src, "        {},", rows.join(", "));
        }
        let _ = writeln!(src, "    ]),");
    }
    let _ = writeln!(src, "];");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vowel_size_table.rs");
    std::fs::write(path, &src).expect("write src/vowel_size_table.rs");
    println!(
        "wrote {path}: {} bytes, {:.1} s in all",
        src.len(),
        started.elapsed().as_secs_f32()
    );
}
