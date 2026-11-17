//! Babble map: from formants back to a vowel shape of the physical engine.
//!
//! [`solve`] answers "which tongue, lips and narrowing give F1 = 500 Hz and
//! F2 = 1500 Hz?". It looks the answer up in a table of measured formants
//! that `examples/babble.rs` generates: the physical engine's transfer
//! function, measured over a grid of vowel shapes. Between grid points the
//! formants are interpolated, so answers are not tied to the grid.
//!
//! Many shapes share the same F1 and F2. Among them the solver prefers the
//! most relaxed one, and when it is given the previous shape it stays
//! close to that one, so a target that moves slowly gives a shape that
//! moves smoothly. Lips and narrowing are never tighter than the tightest
//! [physical vowels](crate::Phoneme), so solved shapes do not hiss.
//!
//! The engine reaches F1 from about 250 to 760 Hz and F2 from about 550 to
//! 2700 Hz, in the vowel-triangle shaped region that [`reachable`] knows;
//! [`nearest_reachable`] clamps a target into it. Measured back through the
//! engine, solved shapes land within about 2 % of the F1 and F2 asked for
//! (`tests/inverse_gate.rs`). For other tract lengths use [`solve_for`].
//!
//! Everything here is allocation-free and works without `std`.
//!
//! ```
//! use synthie_trombone::{Engine, Trombone, inverse};
//!
//! let mut voice = Trombone::new(48_000.0);
//! voice.set_engine(Engine::Physical);
//! let mut shape = inverse::solve(500.0, 1500.0, None, None);
//! voice.set_shape(&shape);
//! // Gliding: pass the last shape, so the next one stays close to it.
//! shape = inverse::solve(510.0, 1480.0, None, Some(&shape));
//! voice.set_shape(&shape);
//! ```

// Grid and matrix arithmetic over small fixed-size arrays.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::math::{exp, ln};

#[path = "inverse_table.rs"]
mod table;

#[path = "inverse_wave_table.rs"]
mod wave_table;

#[path = "inverse_modal.rs"]
pub mod modal;

/// Lip cell that [`Shape::lips`] narrows.
pub const LIP_CELL: f32 = 41.0;

/// Tongue body positions.
const TONGUE_INDEX: (f32, f32) = (12.0, 29.0);
/// Tongue body heights.
const TONGUE_DIAMETER: (f32, f32) = (2.05, 3.5);
/// Lip rounding, touch diameter: from as narrow as a vowel gets without
/// hissing (the rounding of /u/) to lips so wide that they no longer shape
/// the tract at all, the same as no rounding.
const LIPS: (f32, f32) = (0.65, 3.3);
/// Lip rounding at which the lips start to narrow the lip cells (whose rest
/// diameter is 1.5); wider rounding only trims the cells behind them. The
/// formants bend sharply here, so it sits on a grid line of the babble
/// map: at lip coordinate [`LIPS_KNEE_AT`].
const LIPS_KNEE: f32 = 1.8;
/// Lip coordinate of [`LIPS_KNEE`].
const LIPS_KNEE_AT: f32 = 0.75;
/// Where along the tract a narrowing can sit: from the lower pharynx (a
/// tongue root pulled back narrows it for /a/, which is what lifts F1 past
/// 700 Hz) to behind the teeth.
const NARROWING_AT: (f32, f32) = (8.0, 34.0);
/// Narrowing, touch diameter: from the narrowest that does not hiss (that
/// of /i/) to wider than the tract ever is.
const NARROWING: (f32, f32) = (0.7, 3.0);
/// What a touch diameter loses before it becomes a tract diameter.
const TOUCH: f32 = 0.3;
/// Coordinates this close to `1` mean "no rounding" or "no narrowing".
const OPEN: f32 = 1.0 - 1.0e-4;

/// Range of F1, Hz, that the babble map of the default tract covers (see
/// [`reachable`] and [`Map::f1_range`]).
pub const F1_RANGE: (f32, f32) = Map::DEFAULT.f1_range();
/// Range of F2, Hz, that the babble map of the default tract covers (see
/// [`reachable`] and [`Map::f2_range`]).
pub const F2_RANGE: (f32, f32) = Map::DEFAULT.f2_range();

/// A vowel shape of the physical engine; hold it with
/// [`Trombone::set_shape`](crate::Trombone::set_shape).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Tongue body position (cell index).
    pub tongue_index: f32,
    /// Tongue body height (diameter under the tongue).
    pub tongue_diameter: f32,
    /// Lip rounding: touch diameter at the lips, `None` for spread lips.
    pub lips: Option<f32>,
    /// Narrowing along the tongue, `(cell, touch diameter)`, or `None`.
    pub narrowing: Option<(f32, f32)>,
    /// The tongue root: `ln` of how much it widens the middle of the
    /// pharynx (positive drawn forward, as in the tense vowels; negative
    /// pulled back, as in /ɑ/), `0.0` at rest. It belongs to the tongue
    /// body: consonants made on a vowel keep its root. The babble map is
    /// measured with the root at rest: [`solve`] leaves it there and
    /// [`predict`] knows only such shapes. The physical vowels of
    /// [`vowel_targets`](crate::vowel_targets) are solved with it free,
    /// against the engine itself.
    pub pharynx: f32,
    /// The tongue body as the three numbers of the measured modes (see
    /// [`AnatomyConfig::tongue_model`](crate::AnatomyConfig::tongue_model)),
    /// or `None`. On a tract with the modal body they are the body, and the
    /// tongue, its root and the narrowing are not used; `None` there
    /// projects the cosine tongue `(tongue_index, tongue_diameter)` onto the
    /// modes. The cosine body ignores them.
    pub modes: Option<[f32; 3]>,
}

// ----- the modal body ----------------------------------------------------

/// The range of each of the modal body's three numbers
/// ([`Shape::modes`]) a tract takes: the measured vowels' range, widened by
/// half its span either way.
#[must_use]
pub fn mode_range(k: usize) -> (f32, f32) {
    crate::tract::mode_range(k.min(2))
}

/// The modal body's numbers of the eleven measured vowels (Story 2008, one
/// man, MRI), by their IPA symbol: `i ɪ e ɛ æ ʌ ɑ ɔ o ʊ u`.
#[must_use]
pub fn measured_modes(ipa: &str) -> Option<[f32; 3]> {
    crate::tract_modes::MEASURED
        .iter()
        .find(|(v, _)| *v == ipa)
        .map(|&(_, q)| q)
}

/// The modal body's numbers nearest to the cosine tongue `(index,
/// diameter)` (least squares over the cosine's cells): what a tract with the
/// modal body makes of a tongue set by place and height.
#[must_use]
pub fn project_tongue(index: f32, diameter: f32) -> [f32; 3] {
    crate::tract::project_tongue(
        index.clamp(TONGUE_INDEX.0, TONGUE_INDEX.1),
        diameter.clamp(TONGUE_DIAMETER.0, TONGUE_DIAMETER.1),
    )
}

/// Position of `v` in `range`, `0‥1`.
fn unit(v: f32, range: (f32, f32)) -> f32 {
    ((v - range.0) / (range.1 - range.0)).clamp(0.0, 1.0)
}

/// Position of a touch diameter in `range`, `0‥1`, on a scale where equal
/// steps change the opening by equal ratios.
fn unit_log(v: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = (ln(range.0 - TOUCH), ln(range.1 - TOUCH));
    ((ln((v - TOUCH).max(1.0e-3)) - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn from_unit(x: f32, range: (f32, f32)) -> f32 {
    range.0 + (range.1 - range.0) * x.clamp(0.0, 1.0)
}

fn from_unit_log(x: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = (ln(range.0 - TOUCH), ln(range.1 - TOUCH));
    TOUCH + exp(lo + (hi - lo) * x.clamp(0.0, 1.0))
}

/// Lip coordinate of a lip rounding: logarithmic below and above
/// [`LIPS_KNEE`], which is at [`LIPS_KNEE_AT`].
fn unit_lips(v: f32) -> f32 {
    if v < LIPS_KNEE {
        LIPS_KNEE_AT * unit_log(v, (LIPS.0, LIPS_KNEE))
    } else {
        LIPS_KNEE_AT + (1.0 - LIPS_KNEE_AT) * unit_log(v, (LIPS_KNEE, LIPS.1))
    }
}

fn from_unit_lips(x: f32) -> f32 {
    if x < LIPS_KNEE_AT {
        from_unit_log(x / LIPS_KNEE_AT, (LIPS.0, LIPS_KNEE))
    } else {
        from_unit_log(
            (x - LIPS_KNEE_AT) / (1.0 - LIPS_KNEE_AT),
            (LIPS_KNEE, LIPS.1),
        )
    }
}

impl Shape {
    /// The shape as five coordinates, each `0‥1`: tongue position, tongue
    /// height, lip opening, narrowing place and narrowing opening. Openings
    /// are on a logarithmic scale, and `1` there means none. Values outside
    /// the ranges the babble map covers are clamped into them.
    #[must_use]
    pub fn coordinates(&self) -> [f32; 5] {
        let (at, narrowing) = match self.narrowing {
            Some((at, d)) => (unit(at, NARROWING_AT), unit_log(d, NARROWING)),
            None => (0.5, 1.0),
        };
        [
            unit(self.tongue_index, TONGUE_INDEX),
            unit(self.tongue_diameter, TONGUE_DIAMETER),
            self.lips.map_or(1.0, unit_lips),
            at,
            narrowing,
        ]
    }

    /// The shape at the given [`coordinates`](Self::coordinates); each one
    /// is clamped to `0‥1`.
    #[must_use]
    pub fn from_coordinates(x: [f32; 5]) -> Self {
        Self {
            tongue_index: from_unit(x[0], TONGUE_INDEX),
            tongue_diameter: from_unit(x[1], TONGUE_DIAMETER),
            lips: (x[2] < OPEN).then(|| from_unit_lips(x[2])),
            narrowing: (x[4] < OPEN).then(|| {
                (
                    from_unit(x[3], NARROWING_AT),
                    from_unit_log(x[4], NARROWING),
                )
            }),
            pharynx: 0.0,
            modes: None,
        }
    }
}

// ----- the table ---------------------------------------------------------

/// Value of one base-64 character of the table.
fn sextet(c: u8) -> u32 {
    u32::from(match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    })
}

/// Code of a node whose formants could not be measured.
const INVALID: u32 = 1023;
/// Largest code of a measured formant.
const TOP: f32 = 1022.0;

/// Grid nodes along each shape coordinate, the same in both maps.
const DIMS: [usize; 5] = table::DIMS;
const _: () = {
    let mut k = 0;
    while k < 5 {
        assert!(wave_table::DIMS[k] == DIMS[k]);
        k += 1;
    }
};

/// Nodes along the narrowing place and opening together: the last opening
/// is "no narrowing", where the place does not matter, stored once.
const JOINT: usize = DIMS[3] * (DIMS[4] - 1) + 1;

/// The babble map of the cosine tongue body on one tract
/// ([`TractModel`](crate::TractModel)): the formants measured over the grid
/// of shapes, the rough cells, the reachable region and the size
/// correction, as `examples/babble.rs` measured them on that tract
/// (`src/inverse_table.rs` for the original tract, `src/inverse_wave_table.rs`
/// for the anatomical one). Every vowel shape is solved against the tract it
/// is made on, so a voice reads the map of its own tract
/// ([`Map::for_tuning`]); the free functions of this module read that of
/// the default tract ([`Map::DEFAULT`]).
#[derive(Debug)]
pub struct Map {
    f_lo: [f32; 3],
    f_hi: [f32; 3],
    nodes: &'static [u8],
    rough: &'static [u8],
    reach_f1: (f32, f32),
    reach_f2: (f32, f32),
    reach_size: [usize; 2],
    reach: &'static [u8],
    size_lengths: &'static [f32; 9],
    size_correction: &'static [[[f32; 10]; 9]; 3],
}

impl Map {
    /// The map of the original tract (`TractModel::Original`).
    pub const ORIGINAL: Map = Map {
        f_lo: table::F_LO,
        f_hi: table::F_HI,
        nodes: table::NODES,
        rough: table::ROUGH,
        reach_f1: table::REACH_F1,
        reach_f2: table::REACH_F2,
        reach_size: table::REACH_SIZE,
        reach: table::REACH,
        size_lengths: &table::SIZE_LENGTHS,
        size_correction: &table::SIZE_CORRECTION,
    };

    /// The map of the anatomical tract (`TractModel::Anatomical`).
    pub const ANATOMICAL: Map = Map {
        f_lo: wave_table::F_LO,
        f_hi: wave_table::F_HI,
        nodes: wave_table::NODES,
        rough: wave_table::ROUGH,
        reach_f1: wave_table::REACH_F1,
        reach_f2: wave_table::REACH_F2,
        reach_size: wave_table::REACH_SIZE,
        reach: wave_table::REACH,
        size_lengths: &wave_table::SIZE_LENGTHS,
        size_correction: &wave_table::SIZE_CORRECTION,
    };

    /// The map of the tract of [`Tuning::DEFAULT`](crate::Tuning::DEFAULT).
    pub const DEFAULT: &'static Map = Map::of(crate::TractModel::of(&crate::Tuning::DEFAULT));

    /// The map of `tract`.
    #[must_use]
    pub const fn of(tract: crate::TractModel) -> &'static Map {
        match tract {
            crate::TractModel::Original => &Map::ORIGINAL,
            crate::TractModel::Anatomical => &Map::ANATOMICAL,
        }
    }

    /// The map of the tract a voice of `tuning` has
    /// ([`TractModel::of`](crate::TractModel::of)).
    #[must_use]
    pub const fn for_tuning(tuning: &crate::Tuning) -> &'static Map {
        Map::of(crate::TractModel::of(tuning))
    }

    /// Range of F1, Hz, that the map covers (see [`Map::reachable`]).
    #[must_use]
    pub const fn f1_range(&self) -> (f32, f32) {
        self.reach_f1
    }

    /// Range of F2, Hz, that the map covers (see [`Map::reachable`]).
    #[must_use]
    pub const fn f2_range(&self) -> (f32, f32) {
        self.reach_f2
    }

    /// `ln` of F1, F2 and F3 at a grid node, or `None` if it was not
    /// measured.
    #[allow(clippy::cast_precision_loss)]
    fn node(&self, i: [usize; 5]) -> Option<[f32; 3]> {
        let [_, n1, n2, _, n4] = DIMS;
        let j = if i[4] == n4 - 1 {
            JOINT - 1
        } else {
            i[3] * (n4 - 1) + i[4]
        };
        let at = 5 * (((i[0] * n1 + i[1]) * n2 + i[2]) * JOINT + j);
        let bytes = self.nodes.get(at..at + 5)?;
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
            *f = self.f_lo[k] + (self.f_hi[k] - self.f_lo[k]) * q / TOP;
        }
        Some(f)
    }

    /// Whether a grid cell's interior disagrees with its corners (the
    /// formants jump or bend there), so it is not used.
    fn rough(&self, c: [usize; 5]) -> bool {
        let d = DIMS;
        let n = (((c[0] * (d[1] - 1) + c[1]) * (d[2] - 1) + c[2]) * (d[3] - 1) + c[3]) * (d[4] - 1)
            + c[4];
        bit(self.rough, n)
    }
}

/// Bit `n` of a base-64 bit field.
fn bit(field: &[u8], n: usize) -> bool {
    field
        .get(n / 6)
        .is_some_and(|&c| (sextet(c) >> (n % 6)) & 1 == 1)
}

/// Interpolated `ln` formants at a point and their derivatives with respect
/// to the coordinates.
struct Eval {
    f: [f32; 3],
    g: [[f32; 5]; 3],
}

/// Cell holding coordinate `x` along a dimension of `n` nodes, and the
/// position inside it, `0‥1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn locate(x: f32, n: usize) -> (usize, f32) {
    let t = x.clamp(0.0, 1.0) * (n - 1) as f32;
    let i = (t as usize).min(n - 2);
    (i, t - i as f32)
}

/// The table's formants at `x`, interpolated between the 32 corners of the
/// cell holding it; `None` in an unmeasured or rough cell.
#[allow(clippy::cast_precision_loss)]
fn eval(map: &Map, x: &[f32; 5]) -> Option<Eval> {
    let mut base = [0usize; 5];
    let mut frac = [0.0f32; 5];
    for d in 0..5 {
        (base[d], frac[d]) = locate(x[d], DIMS[d]);
    }
    if map.rough(base) {
        return None;
    }
    let mut out = Eval {
        f: [0.0; 3],
        g: [[0.0; 5]; 3],
    };
    for corner in 0..32usize {
        let mut i = base;
        let mut w = 1.0f32;
        // Weight without dimension d, for the derivative along d.
        let mut part = [1.0f32; 5];
        for d in 0..5 {
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
        let f = map.node(i)?;
        for k in 0..3 {
            out.f[k] += w * f[k];
            for d in 0..5 {
                let sign = if (corner >> d) & 1 == 1 { 1.0 } else { -1.0 };
                out.g[k][d] += sign * part[d] * f[k] * (DIMS[d] - 1) as f32;
            }
        }
    }
    Some(out)
}

/// The babble map's estimate of a shape's F1, F2 and F3, Hz, at the native
/// tract length; `None` where the map has no data (a few extreme shapes)
/// and for a shape whose tongue root is not at rest (the map is measured
/// with it at rest, and what the root does depends on the whole shape).
///
/// On the default tract's map ([`Map::predict`]).
#[must_use]
pub fn predict(shape: &Shape) -> Option<[f32; 3]> {
    Map::DEFAULT.predict(shape)
}

impl Map {
    /// [`predict`] on this map.
    #[must_use]
    pub fn predict(&self, shape: &Shape) -> Option<[f32; 3]> {
        if shape.modes.is_some() {
            // A shape of the modal body: its own map.
            return modal::predict(shape);
        }
        if shape.pharynx.abs() > f32::EPSILON {
            return None;
        }
        eval(self, &shape.coordinates()).map(|e| e.f.map(exp))
    }
}

// ----- solving -----------------------------------------------------------

/// Pull towards a relaxed shape: tongue centred, lips at the knee, no
/// narrowing. Tiny, so it only decides between shapes that reach the
/// target equally well.
const EASE: f32 = 3.0e-4;
/// Pull towards the previous shape, when there is one. Larger than
/// [`EASE`], so it wins that decision, yet small next to a formant error.
const STAY: f32 = 1.0e-3;
/// The relaxed shape, as coordinates. Its lips are at [`LIPS_KNEE`], not
/// spread: rounding a little from spread lips narrows only the cells behind
/// them and raises F2, while rounding past the knee lowers it, so from
/// spread lips a falling F2 would have to jump to rounded ones once the
/// tongue runs out of reach. From the knee it can go either way smoothly.
const RELAXED: [f32; 5] = [0.5, 0.5, LIPS_KNEE_AT, 0.5, 1.0];
/// Weight of F3 against F1 and F2 when it is asked for.
const F3_WEIGHT: f32 = 0.3;
/// Starting points taken from the table.
const STARTS: usize = 8;

/// What the solver aims for: `ln` formants, their weights, and the shape to
/// stay near.
struct Goal<'a> {
    map: &'a Map,
    t: [f32; 3],
    w: [f32; 3],
    near: Option<[f32; 5]>,
}

impl Goal<'_> {
    /// Cost of shape `x` with interpolated formants `f`.
    fn cost(&self, x: &[f32; 5], f: &[f32; 3]) -> f32 {
        let mut c = 0.0;
        for k in 0..3 {
            let r = f[k] - self.t[k];
            c += self.w[k] * r * r;
        }
        for d in 0..5 {
            let r = x[d] - RELAXED[d];
            c += EASE * r * r;
            if let Some(n) = self.near {
                let r = x[d] - n[d];
                c += STAY * r * r;
            }
        }
        c
    }

    /// Damped Gauss-Newton descent from `x`, kept inside `0‥1` and out of
    /// rough cells. Returns the shape reached and its cost.
    fn refine(&self, mut x: [f32; 5]) -> Option<([f32; 5], f32)> {
        let mut e = eval(self.map, &x)?;
        let mut cost = self.cost(&x, &e.f);
        let mut damping = 1.0e-3f32;
        let pull = EASE + if self.near.is_some() { STAY } else { 0.0 };
        for _ in 0..40 {
            let mut h = [[0.0f32; 5]; 5];
            let mut b = [0.0f32; 5];
            for k in 0..3 {
                let r = e.f[k] - self.t[k];
                for d in 0..5 {
                    b[d] += self.w[k] * r * e.g[k][d];
                    for c in 0..5 {
                        h[d][c] += self.w[k] * e.g[k][d] * e.g[k][c];
                    }
                }
            }
            for d in 0..5 {
                b[d] += EASE * (x[d] - RELAXED[d]);
                if let Some(n) = self.near {
                    b[d] += STAY * (x[d] - n[d]);
                }
                h[d][d] += pull;
            }
            let mut moved = false;
            while damping < 1.0e4 {
                let mut a = h;
                let mut rhs = b.map(|v| -v);
                for d in 0..5 {
                    a[d][d] += damping * (h[d][d] + 1.0e-3);
                    // A coordinate pressed against its limit stays there.
                    let stuck = (x[d] <= 0.0 && b[d] > 0.0) || (x[d] >= 1.0 && b[d] < 0.0);
                    if stuck {
                        for c in 0..5 {
                            a[d][c] = 0.0;
                            a[c][d] = 0.0;
                        }
                        a[d][d] = 1.0;
                        rhs[d] = 0.0;
                    }
                }
                let Some(step) = solve5(a, rhs) else { break };
                let mut y = x;
                for d in 0..5 {
                    y[d] = (x[d] + step[d]).clamp(0.0, 1.0);
                }
                if let Some(ey) = eval(self.map, &y) {
                    let cy = self.cost(&y, &ey.f);
                    if cy < cost {
                        let size = (0..5).fold(0.0f32, |m, d| m.max((y[d] - x[d]).abs()));
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

/// Solve the symmetric positive definite 5×5 system `a · x = b`.
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

/// A point just inside a cell next to grid node `i`, in the first cell
/// around it that is not rough; `None` if all of them are.
#[allow(clippy::cast_precision_loss)]
fn start_at(map: &Map, i: [usize; 5]) -> Option<[f32; 5]> {
    let d = DIMS;
    for around in 0..32usize {
        let mut cell = [0usize; 5];
        let mut x = [0.0f32; 5];
        let mut fits = true;
        for k in 0..5 {
            let below = (around >> k) & 1 == 1;
            if (below && i[k] == 0) || (!below && i[k] == d[k] - 1) {
                fits = false;
                break;
            }
            cell[k] = if below { i[k] - 1 } else { i[k] };
            let nudge = if below { -0.01 } else { 0.01 };
            x[k] = ((i[k] as f32 + nudge) / (d[k] - 1) as f32).clamp(0.0, 1.0);
        }
        if fits && !map.rough(cell) && eval(map, &x).is_some() {
            return Some(x);
        }
    }
    None
}

/// Grid nodes with the lowest cost, at most one per neighbourhood.
#[allow(clippy::cast_precision_loss)]
fn candidates(goal: &Goal) -> [Option<(f32, [usize; 5])>; STARTS] {
    let d = DIMS;
    let mut best: [Option<(f32, [usize; 5])>; STARTS] = [None; STARTS];
    let coord = |k: usize, i: usize| i as f32 / (d[k] - 1) as f32;
    for i0 in 0..d[0] {
        for i1 in 0..d[1] {
            for i2 in 0..d[2] {
                for i3 in 0..d[3] {
                    for i4 in 0..d[4] {
                        // "No narrowing" is stored once, at the middle place.
                        if i4 == d[4] - 1 && i3 != (d[3] - 1) / 2 {
                            continue;
                        }
                        let i = [i0, i1, i2, i3, i4];
                        let Some(f) = goal.map.node(i) else { continue };
                        let x = [0, 1, 2, 3, 4].map(|k| coord(k, i[k]));
                        let c = goal.cost(&x, &f);
                        offer(&mut best, c, i);
                    }
                }
            }
        }
    }
    best
}

/// Keep `(c, i)` among the best if it beats them; a node next to a kept
/// one replaces it only when it is better.
fn offer(best: &mut [Option<(f32, [usize; 5])>; STARTS], c: f32, i: [usize; 5]) {
    let close = |j: &[usize; 5]| (0..5).all(|k| i[k].abs_diff(j[k]) <= 1);
    let slot = if let Some(near) = best.iter_mut().find(|b| b.is_some_and(|(_, j)| close(&j))) {
        Some(near)
    } else if let Some(free) = best.iter_mut().find(|b| b.is_none()) {
        Some(free)
    } else {
        best.iter_mut()
            .max_by(|a, b| a.map_or(0.0, |v| v.0).total_cmp(&b.map_or(0.0, |v| v.0)))
    };
    if let Some(slot) = slot
        && slot.is_none_or(|(old, _)| c < old)
    {
        *slot = Some((c, i));
    }
}

/// Replace a non-finite formant by a neutral one.
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        fallback
    }
}

/// The shape whose formants at the native tract length are closest to F1
/// `f1` and F2 `f2` (Hz) and, if given, F3 `f3`.
///
/// A target outside the [`reachable`] region is first moved to
/// the nearest reachable point. F1 and F2 are matched first; F3 only
/// chooses among shapes that match them about equally well.
///
/// Many shapes share the same formants. Without `near` the solver picks the
/// most relaxed one: tongue nearest the middle, lips nearest a loose
/// rounding (1.8), least narrowing. With `near` (usually the shape it returned last
/// time) it picks the one closest to `near`, so a target that moves slowly
/// gives a shape that moves smoothly.
///
/// A `near` shape of the modal body ([`Shape::modes`]) solves on the modal
/// body ([`modal::solve`]).
///
/// On the default tract's map ([`Map::solve`]).
#[must_use]
pub fn solve(f1: f32, f2: f32, f3: Option<f32>, near: Option<&Shape>) -> Shape {
    Map::DEFAULT.solve(f1, f2, f3, near)
}

impl Map {
    /// [`solve`] on this map.
    #[must_use]
    pub fn solve(&self, f1: f32, f2: f32, f3: Option<f32>, near: Option<&Shape>) -> Shape {
        solve_on(self, f1, f2, f3, near)
    }
}

fn solve_on(map: &Map, f1: f32, f2: f32, f3: Option<f32>, near: Option<&Shape>) -> Shape {
    if near.is_some_and(|s| s.modes.is_some()) {
        return modal::solve(f1, f2, f3, near);
    }
    let (f1, f2) = map.nearest_reachable(finite_or(f1, 500.0), finite_or(f2, 1500.0));
    let f3 = f3.filter(|v| v.is_finite() && *v > 0.0);
    let goal = Goal {
        map,
        t: [ln(f1), ln(f2), ln(f3.unwrap_or(2500.0))],
        w: [1.0, 1.0, if f3.is_some() { F3_WEIGHT } else { 0.0 }],
        near: near.map(Shape::coordinates),
    };
    let mut best: Option<([f32; 5], f32)> = goal.near.and_then(|n| goal.refine(n));
    for (_, i) in candidates(&goal).into_iter().flatten() {
        let Some(found) = start_at(map, i).and_then(|x| goal.refine(x)) else {
            continue;
        };
        if best.is_none_or(|b| found.1 < b.1) {
            best = Some(found);
        }
    }
    best.map_or_else(
        || near.copied().unwrap_or(Shape::from_coordinates(RELAXED)),
        |(x, _)| Shape::from_coordinates(x),
    )
}

// ----- tract length ------------------------------------------------------

/// A tract length as [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)
/// takes it.
fn sane_length(length: f32) -> f32 {
    if length.is_finite() {
        length.clamp(0.5, 1.6)
    } else {
        1.0
    }
}

/// `ln` of the size correction for formant `k` (0 for F1) whose value at
/// the native length is `ln_native`, at tract length `length`: how far
/// the formant is from `native / length`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn correction(map: &Map, k: usize, ln_native: f32, length: f32) -> f32 {
    let lengths = map.size_lengths;
    let n = lengths.len();
    let l = length.clamp(lengths[0], lengths[n - 1]);
    let a = lengths
        .iter()
        .rposition(|&v| v <= l)
        .unwrap_or(0)
        .min(n - 2);
    let s = (l - lengths[a]) / (lengths[a + 1] - lengths[a]);
    let row = &map.size_correction[k];
    let bins = row[0].len();
    let t = ((ln_native - map.f_lo[k]) / (map.f_hi[k] - map.f_lo[k])).clamp(0.0, 1.0)
        * (bins - 1) as f32;
    let b = (t as usize).min(bins - 2);
    let u = t - b as f32;
    let at = |r: &[f32]| r[b] + (r[b + 1] - r[b]) * u;
    at(&row[a]) + (at(&row[a + 1]) - at(&row[a])) * s
}

/// Formant `k` (0 for F1) of a tract of `length`, Hz, for a shape whose
/// formant at the native length is `native`, with the size correction of
/// `map`.
fn from_native_on(map: &Map, k: usize, native: f32, length: f32) -> f32 {
    native / length * exp(correction(map, k, ln(native), length))
}

/// The native-length formant `k` that becomes `hz` at `length`.
fn to_native_on(map: &Map, k: usize, hz: f32, length: f32) -> f32 {
    let mut native = hz * length;
    for _ in 0..4 {
        native = hz * length * exp(-correction(map, k, ln(native), length));
    }
    native
}

/// [`from_native_on`] with the original tract's size correction, which the
/// modal map uses (`inverse::modal`, measured on modal shapes by the run
/// that made it).
fn from_native(k: usize, native: f32, length: f32) -> f32 {
    from_native_on(&Map::ORIGINAL, k, native, length)
}

/// [`to_native_on`] with the original tract's size correction (see
/// [`from_native`]).
fn to_native(k: usize, hz: f32, length: f32) -> f32 {
    to_native_on(&Map::ORIGINAL, k, hz, length)
}

/// Formants of a tract of `length`, Hz, for a shape whose formants at the
/// native length are `native`.
///
/// Formants scale nearly as `1 / length`; the yielding walls of the
/// physical tract add a resonance that does not scale, which lifts low
/// formants of short tracts and lowers those of long ones by a few per
/// cent. This includes that correction, measured by `examples/babble.rs`
/// on the default tract ([`Map::formants_at_length`]).
#[must_use]
pub fn formants_at_length(native: [f32; 3], tract_length: f32) -> [f32; 3] {
    Map::DEFAULT.formants_at_length(native, tract_length)
}

/// [`solve`] for a voice whose tract length is `tract_length` (as given to
/// [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)):
/// the formants are converted to the native length first, with the
/// correction described at [`formants_at_length`]. On the default tract's
/// map ([`Map::solve_for`]).
#[must_use]
pub fn solve_for(
    f1: f32,
    f2: f32,
    f3: Option<f32>,
    near: Option<&Shape>,
    tract_length: f32,
) -> Shape {
    Map::DEFAULT.solve_for(f1, f2, f3, near, tract_length)
}

// ----- reachability ------------------------------------------------------

/// Position of `hz` along a reachability axis spanning `range`, in cells.
fn axis(hz: f32, range: (f32, f32), cells: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let n = cells as f32;
    (ln(hz) - ln(range.0)) / (ln(range.1) - ln(range.0)) * n
}

/// Hz at `pos` cells along a reachability axis spanning `range`.
fn axis_hz(pos: f32, range: (f32, f32), cells: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let n = cells as f32;
    exp(ln(range.0) + (ln(range.1) - ln(range.0)) * pos / n)
}

/// Whether the physical engine can make F1 `f1` and F2 `f2` (Hz) at the
/// native tract length, so that [`solve`] meets them.
///
/// The reachable region lies inside [`F1_RANGE`] × [`F2_RANGE`]; it is
/// stored as a grid of about 2.5 % per cell, a little inside the edge of
/// what the engine can do. On the default tract's map ([`Map::reachable`]).
#[must_use]
pub fn reachable(f1: f32, f2: f32) -> bool {
    Map::DEFAULT.reachable(f1, f2)
}

/// The reachable point (see [`reachable`]) nearest to F1 `f1` and F2 `f2`,
/// Hz, with distance measured on a logarithmic frequency scale; the target
/// itself if it is reachable. On the default tract's map
/// ([`Map::nearest_reachable`]).
#[must_use]
pub fn nearest_reachable(f1: f32, f2: f32) -> (f32, f32) {
    Map::DEFAULT.nearest_reachable(f1, f2)
}

/// [`reachable`] for a voice whose tract length is `tract_length`.
#[must_use]
pub fn reachable_for(f1: f32, f2: f32, tract_length: f32) -> bool {
    Map::DEFAULT.reachable_for(f1, f2, tract_length)
}

/// [`nearest_reachable`] for a voice whose tract length is `tract_length`.
#[must_use]
pub fn nearest_reachable_for(f1: f32, f2: f32, tract_length: f32) -> (f32, f32) {
    Map::DEFAULT.nearest_reachable_for(f1, f2, tract_length)
}

impl Map {
    /// [`formants_at_length`] with this map's size correction.
    #[must_use]
    pub fn formants_at_length(&self, native: [f32; 3], tract_length: f32) -> [f32; 3] {
        let l = sane_length(tract_length);
        [0, 1, 2].map(|k| from_native_on(self, k, native[k], l))
    }

    /// [`solve_for`] on this map.
    #[must_use]
    pub fn solve_for(
        &self,
        f1: f32,
        f2: f32,
        f3: Option<f32>,
        near: Option<&Shape>,
        tract_length: f32,
    ) -> Shape {
        let l = sane_length(tract_length);
        let f1 = to_native_on(self, 0, finite_or(f1, 500.0 / l), l);
        let f2 = to_native_on(self, 1, finite_or(f2, 1500.0 / l), l);
        let f3 = f3
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| to_native_on(self, 2, v, l));
        self.solve(f1, f2, f3, near)
    }

    /// Whether reachability cell `(a, b)` (F1, F2) is reachable.
    fn reach_cell(&self, a: usize, b: usize) -> bool {
        bit(self.reach, a * self.reach_size[1] + b)
    }

    /// [`reachable`] on this map.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn reachable(&self, f1: f32, f2: f32) -> bool {
        if !(f1.is_finite() && f2.is_finite() && f1 > 0.0 && f2 > 0.0) {
            return false;
        }
        let [na, nb] = self.reach_size;
        let a = axis(f1, self.reach_f1, na);
        let b = axis(f2, self.reach_f2, nb);
        #[allow(clippy::cast_precision_loss)]
        let inside = (0.0..na as f32).contains(&a) && (0.0..nb as f32).contains(&b);
        inside && self.reach_cell(a as usize, b as usize)
    }

    /// [`nearest_reachable`] on this map.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nearest_reachable(&self, f1: f32, f2: f32) -> (f32, f32) {
        let f1 = finite_or(f1, 500.0);
        let f2 = finite_or(f2, 1500.0);
        if self.reachable(f1, f2) {
            return (f1, f2);
        }
        let (r1, r2) = (self.reach_f1, self.reach_f2);
        let [na, nb] = self.reach_size;
        let a = axis(f1, r1, na);
        let b = axis(f2, r2, nb);
        // Cells per unit of ln frequency, to measure distance in ln Hz.
        let sa = (ln(r1.1) - ln(r1.0)) / na as f32;
        let sb = (ln(r2.1) - ln(r2.0)) / nb as f32;
        let mut best: Option<(f32, f32, f32)> = None;
        for i in 0..na {
            for j in 0..nb {
                if !self.reach_cell(i, j) {
                    continue;
                }
                // Nearest point of the cell, a hair inside it.
                let pa = a.clamp(i as f32 + 0.02, i as f32 + 0.98);
                let pb = b.clamp(j as f32 + 0.02, j as f32 + 0.98);
                let dist = ((pa - a) * sa) * ((pa - a) * sa) + ((pb - b) * sb) * ((pb - b) * sb);
                if best.is_none_or(|(d, _, _)| dist < d) {
                    best = Some((dist, pa, pb));
                }
            }
        }
        best.map_or((f1, f2), |(_, pa, pb)| {
            (axis_hz(pa, r1, na), axis_hz(pb, r2, nb))
        })
    }

    /// [`reachable_for`] on this map.
    #[must_use]
    pub fn reachable_for(&self, f1: f32, f2: f32, tract_length: f32) -> bool {
        let l = sane_length(tract_length);
        self.reachable(to_native_on(self, 0, f1, l), to_native_on(self, 1, f2, l))
    }

    /// [`nearest_reachable_for`] on this map.
    #[must_use]
    pub fn nearest_reachable_for(&self, f1: f32, f2: f32, tract_length: f32) -> (f32, f32) {
        let l = sane_length(tract_length);
        let (a, b) = self.nearest_reachable(
            to_native_on(self, 0, finite_or(f1, 500.0 / l), l),
            to_native_on(self, 1, finite_or(f2, 1500.0 / l), l),
        );
        (from_native_on(self, 0, a, l), from_native_on(self, 1, b, l))
    }
}
