//! Generate `src/vowel_size_table.rs`: the shape of every physical vowel,
//! glide and diphthong end, for every body size, solved from the formant
//! targets in `src/vowel_targets.rs`.
//!
//! To move a vowel, change its F1, F2 or F3 in `TARGETS` there and run
//!
//! ```sh
//! cargo run -p synthie-trombone --example vowel_size --release
//! ```
//!
//! then check with `cargo test -p synthie-trombone --release --test
//! vowel_size_gate`.
//!
//! There are four tables, by tongue body and tract
//! (`vowel_targets::lengthened`), each one command:
//!
//! ```sh
//! cargo run -p synthie-trombone --example vowel_size --release                  # cosine, default tract
//! cargo run -p synthie-trombone --example vowel_size --release -- --modal       # modal body (tongue_model 1)
//! cargo run -p synthie-trombone --example vowel_size --release -- --lengthened --both
//! #   both bodies for the tract the wave is switched on with (lip_horn 1.2,
//! #   larynx_travel 1, rounded_larynx 0.5); `--profile FILE` lays a voice
//! #   profile's tuning on top (the epilarynx's numbers, say), and the table
//! #   is written where a voice of that tuning reads it
//! ```
//!
//! `--both` solves the cosine and the modal table in one run. Solve the
//! modal babble map for the same tract first (`examples/babble.rs --modal`):
//! the modal solver starts from its answers. Uses every CPU thread; takes a minute or two. The babble
//! map (`examples/babble.rs`) only needs regenerating when the engine's
//! acoustics change, not when a target does.
//!
//! For every target this
//!
//! 1. solves the shape at the native tract length (1.0): starts from the
//!    articulators' natural place for the sound (the classic engine's
//!    tongue and lips) and from the babble map's answers (`inverse::solve`,
//!    with and without F3), refines each against the real
//!    engine (damped Gauss-Newton on measured formants, derivatives by
//!    finite differences) until F1 and F2 are on the target and F3 as near
//!    as the other two allow, and keeps the best. Rounded targets keep
//!    rounded lips and spread ones spread lips; lips and narrowing are
//!    never tighter than the tightest vowels may be without hissing;
//! 2. solves the other rows (0.5 to 1.6) outwards from the native one, each
//!    starting from the row next to it nearer to 1.0, stepped on along the
//!    way the shape has been moving, and staying close to it so that the
//!    rows make a smooth path. If that misses, it also tries the babble
//!    map's answer (`inverse::solve_for`) and keeps the better one;
//! 3. verifies the table as the library will read it (quantised, and
//!    interpolated between rows) at the rows, between them, and at the
//!    lengths the tests use, measuring each shape through the engine, and
//!    reports how far apart the lax and tense vowels are.

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
use synthie_trombone::inverse::{self, Shape};
use synthie_trombone::vowel_targets::{self, Lips, TARGETS, Target, VowelKey};
use synthie_trombone::{Articulation, Phoneme};

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
/// Shape coordinates: tongue index and diameter, lips, narrowing place and
/// opening, tongue root.
const D: usize = 6;
/// Pull towards the neighbouring row's shape, per squared coordinate:
/// small next to a formant error (moving a coordinate by 0.1 costs as much
/// as a 0.3 % error), so it only chooses among shapes that reach the
/// target about equally well, and among those the nearest.
const STAY: f32 = 1.0e-3;
/// Pull towards the shape the two rows before predict (going on in a
/// straight line), per squared coordinate: keeps the path from bending.
const BEND: f32 = 3.0e-3;
/// How hard the native shape is pulled towards the articulators' natural
/// place for it (see [`prior`]), per coordinate, relative to [`STAY`] and
/// [`BEND`]: tongue and lips have a natural place; the narrowing is there
/// to place F3 and is let go wherever that needs it.
const NATURAL_PULL: [f32; D] = [1.0, 1.0, 1.0, 0.1, 0.1, ROOT_PULL];
/// How hard the tongue root is pulled towards its natural setting
/// ([`LAX_ROOT`]): about half as hard as the tongue, so that it moves where
/// the formants need it and otherwise stays where the vowel's class puts
/// it.
const ROOT_PULL: f32 = 0.5;
/// The tongue roots the engine allows (`synthie_trombone::PHARYNX`), as
/// the shape coordinates span them; [`root_range`] says which part of it a
/// shape may take.
const PHARYNX: (f32, f32) = (-1.2, 0.3);
/// Which tongue roots `key` may take: the lax /ɪ ʌ ʊ/ and the ends of their
/// glides pull it back (a retracted root, a narrower pharynx) by up to
/// [`LAX_ROOT_MIN`]; every other shape keeps it at rest, so a tense vowel's
/// root lies forward of a lax one's. The root belongs to the tongue body
/// and stays under the consonants around the vowel
/// (`Articulation::physical_root`).
///
/// What was measured (round 6, every gate run on each): solved free on
/// both sides for every vowel, /u/ and schwa took a root pulled back as far
/// as the engine allows (-0.9 and -1.2) and /i/ one drawn forward to the
/// limit, and twelve gates failed (the voice bar and the onset pitch kick
/// of /p t/ in "a pa", /ð/ after schwa, /ŋ/ after /i/, the hiss
/// regulator, a sung /u/ heard and copied, the glides, the table's
/// smoothness). Drawn forward for the tense /i u eɪ oʊ/ (by up to 0.3 or
/// 0.15) it moved the listener's step into /i/ (27 to 39 and 91 ms), the
/// mimic's copy of /i/, /ð/ after /u/ and the nasal notch after /i/ out of
/// their gates, and the recognizer heard the vowels no better than with the
/// tense roots at rest. Pulled back for /ɛ/ it moved either the listener's
/// step into /ɛ/ (at -0.3) or the mimic's copy of the /æ/ sung after it
/// (-0.35 to -0.6) out of theirs: /ɛ/ keeps the narrowing low in its
/// pharynx that does the same for its F1.
///
/// With /ɪ ʌ ʊ/ alone every gate passed and the recognizer's vowel error in
/// the LibriSpeech texts fell 2.8 points [0.3, 5.7], but the tuning
/// sentences gained nothing (phoneme error +0.1 [-1.7, 1.8], vowel lab
/// +0.5 [-2.1, 2.9]), the carrier phrases lost (phoneme error +3.5
/// [1.3, 5.6] tuning, +1.7 [0.6, 3.1] held out) and the held-out Harvard
/// vowels too (+2.9 [0.0, 6.4]). So [`LAX_ROOTS`] is off: set it and
/// regenerate to hear it.
fn root_range(key: VowelKey) -> (f32, f32) {
    use Phoneme as P;
    match key {
        VowelKey::Vowel(P::Ih | P::Ah | P::Uh) | VowelKey::End(P::Ih | P::Ah | P::Uh)
            if LAX_ROOTS =>
        {
            (LAX_ROOT_MIN, 0.0)
        }
        _ => (0.0, 0.0),
    }
}

/// Whether the lax vowels' roots are solved (see [`root_range`]).
const LAX_ROOTS: bool = false;

/// Furthest back a lax vowel's root may be pulled.
const LAX_ROOT_MIN: f32 = -0.5;
/// Natural tongue root of the lax vowels: pulled back, narrowing the
/// pharynx.
const LAX_ROOT: f32 = -0.25;

/// Whether the root of `key` is solved.
fn root_free(key: VowelKey) -> bool {
    let (lo, hi) = root_range(key);
    hi > lo
}
/// Finite-difference step, in shape coordinates.
const H: f32 = 0.008;
/// Tongue body positions and heights the engine allows.
const TONGUE_INDEX: (f32, f32) = (12.0, 29.0);
const TONGUE_DIAMETER: (f32, f32) = (2.05, 3.5);
/// Lip rounding of rounded vowels: from the tightest that does not hiss to
/// lips that still round (rest lips are 1.8 in these units).
const LIPS: (f32, f32) = (MIN_LIPS, 1.7);
/// Lips a rounded vowel starts from when the babble map leaves them spread.
const START_LIPS: f32 = 1.2;
/// Where the narrowing may sit: from the lower pharynx, as in the babble
/// map, up to the teeth (/i/ in a long tract wants a little more than the
/// map's 34).
const NARROWING_AT: (f32, f32) = (10.0, 38.0);
/// The same on the modal body: from the top of the epilarynx (its first
/// cell, `tract_modes::FROM`). /ɔ/ reached F1 only to within 3.7 % at the
/// small sizes with its narrowing no lower than cell 10 (the cosine body's
/// /ɑ ɔ/ narrow the lower pharynx too: the piriform sinuses, a closed side
/// branch of some 2 cm³ hung below it, pull the F1 of an open vowel's
/// small back cavity down by a quarter otherwise).
const MODAL_NARROWING_AT: (f32, f32) = (7.0, 38.0);

/// Where the narrowing may sit for `model`.
fn narrowing_at(model: Model) -> (f32, f32) {
    match model {
        Model::Cosine => NARROWING_AT,
        Model::Modal => MODAL_NARROWING_AT,
    }
}
/// Lowest narrowing place for all but the open vowels...
const ORAL: f32 = 14.0;
/// ...and the F1 (adult man, Hz) from which a vowel counts as open.
const THROAT_F1: f32 = 640.0;
/// Narrowing openings searched.
const NARROWING: (f32, f32) = (MIN_NARROWING, 3.0);
/// What a touch diameter loses before it becomes a tract diameter.
const TOUCH: f32 = 0.3;
/// F1 and F2 errors at which a row counts as solved...
const GOOD: f32 = 1.5e-3;
/// ...and beyond which the babble map's answer is tried too.
const TRY_MAP: f32 = 5.0e-3;
/// ...and taken instead of the shape that carries on from the row before
/// only if that one misses by more than this.
const SWITCH: f32 = 0.015;
/// The tests' limit for F1 and F2 (relative error).
const LIMIT: f32 = 0.03;
/// Tightest lips and narrowing a vowel may have without hissing.
const MIN_LIPS: f32 = 0.65;
const MIN_NARROWING: f32 = 0.7;
/// Lax and tense vowels the recognizer confuses, reported side by side.
const PAIRS: [(Phoneme, Phoneme); 4] = [
    (Phoneme::Ih, Phoneme::Iy),
    (Phoneme::Uh, Phoneme::Uw),
    (Phoneme::Eh, Phoneme::Ae),
    (Phoneme::Ah, Phoneme::Aa),
];

/// Which tongue body the shapes are solved for (`AnatomyConfig::tongue_model`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    /// The cosine tongue: `src/vowel_size_table.rs`.
    Cosine,
    /// The measured modes: `src/vowel_modes_table.rs`.
    Modal,
}

/// What one target's shapes are made of.
#[derive(Clone, Copy)]
struct Kind {
    model: Model,
    /// Rounded lips (else spread).
    rounded: bool,
    /// A narrowing along the tongue of its own (the modal body: only /ɚ/,
    /// whose third formant three modes cannot bring down to 1700 Hz).
    narrowed: bool,
}

impl Kind {
    fn of(model: Model, t: &Target) -> Self {
        Self {
            model,
            rounded: t.lips == Lips::Rounded,
            narrowed: model == Model::Cosine || t.key == VowelKey::Vowel(Phoneme::Er),
        }
    }
}

/// Pull of the modal body's three numbers towards those the speaker
/// measured for the vowel (or the blend of them that stands for a target he
/// did not say, see [`measured_prior`]), per squared coordinate, relative to
/// [`STAY`] and [`BEND`] as [`NATURAL_PULL`] is: a step of `0.1` in a
/// coordinate (a fifth of the measured range of that number) costs what a
/// 0.6 % error of F1 or F2 does.
const MEASURED_PULL: f32 = 1.0;
/// F1 or F2 error (relative) of a native shape of the modal body beyond
/// which it is solved again with a narrowing along the tongue of its own.
const NEEDS_NARROWING: f32 = 0.01;

/// The cosine table just solved in this run (`--both`), whose tongues the
/// modal shapes of the same tract carry.
static SOLVED_COSINE: Mutex<Option<Vec<(VowelKey, [[i32; D]; ROWS])>>> = Mutex::new(None);

/// `shape` with the cosine tongue a modal shape of `key` carries at
/// `length` in the library (the cosine table of the same tract, which the
/// jaw follows, `Tract::jaw_opening`): that of the cosine table solved just
/// before in this run, or else the library's. A cosine shape as it is.
fn carrying(shape: Shape, key: VowelKey, length: f32) -> Shape {
    if shape.modes.is_none() {
        return shape;
    }
    let solved = SOLVED_COSINE.lock().expect("lock").as_ref().and_then(|rows| {
        rows.iter()
            .find(|(k, _)| *k == key)
            .map(|(_, r)| interpolate(Model::Cosine, r, length))
    });
    let cosine = solved.unwrap_or_else(|| {
        let mut t = survey::base();
        t.anatomy.tongue_model = 0.0;
        articulation(key)
            .physical_shape_with(length, &t)
            .expect("a vowel shape")
    });
    Shape {
        tongue_index: cosine.tongue_index,
        tongue_diameter: cosine.tongue_diameter,
        ..shape
    }
}

/// Tract length of row `k`, computed as the library does.
fn row_length(k: usize) -> f32 {
    FIRST + STEP * k as f32
}

/// The articulation that stands for `key`.
fn articulation(key: VowelKey) -> Articulation {
    key.articulation().expect("a vowel shape")
}

/// The six raw numbers of a shape as the table stores them: for the cosine
/// body tongue index, tongue diameter, lips (`0` when spread), narrowing
/// cell, narrowing diameter and tongue root; for the modal body its three
/// numbers, lips, narrowing cell and diameter (`0` and `3` for none).
fn raw(s: &Shape) -> [f32; D] {
    if let Some(q) = s.modes {
        let (at, d) = s.narrowing.unwrap_or((0.0, NARROWING.1));
        return [q[0], q[1], q[2], s.lips.unwrap_or(0.0), at, d];
    }
    let n = s.narrowing.expect("physical vowels narrow the tongue");
    [
        s.tongue_index,
        s.tongue_diameter,
        s.lips.unwrap_or(0.0),
        n.0,
        n.1,
        s.pharynx,
    ]
}

/// Coordinate `k` of the modal body (`0‥1` over the range the tract takes)
/// sits at position [`MODE_AT`]`[k]` of the six: the lips and the narrowing
/// keep the places they have for the cosine body.
const MODE_AT: [usize; 3] = [0, 1, 5];

/// A shape as six coordinates, each `0‥1` over the ranges searched: for the
/// cosine body tongue index, tongue diameter, lips, narrowing place,
/// narrowing opening, tongue root; for the modal body its first two
/// numbers, lips, narrowing place and opening, its third number (see
/// [`MODE_AT`]). Openings are on a logarithmic scale. Spread lips are `1`,
/// no narrowing the widest one.
fn to_x(s: &Shape) -> [f32; D] {
    let lin = |v: f32, r: (f32, f32)| (v - r.0) / (r.1 - r.0);
    let log = |v: f32, r: (f32, f32)| {
        ((v - TOUCH).ln() - (r.0 - TOUCH).ln()) / ((r.1 - TOUCH).ln() - (r.0 - TOUCH).ln())
    };
    let (at, narrow) = s.narrowing.unwrap_or((23.0, NARROWING.1));
    let mut x = [
        lin(s.tongue_index, TONGUE_INDEX),
        lin(s.tongue_diameter, TONGUE_DIAMETER),
        s.lips.map_or(1.0, |l| log(l, LIPS)),
        lin(
            at,
            narrowing_at(if s.modes.is_some() { Model::Modal } else { Model::Cosine }),
        ),
        log(narrow, NARROWING),
        lin(s.pharynx, PHARYNX),
    ];
    if let Some(q) = s.modes {
        for k in 0..3 {
            x[MODE_AT[k]] = lin(q[k], inverse::mode_range(k));
        }
    }
    x.map(|v| v.clamp(0.0, 1.0))
}

/// The shape at coordinates `x` (see [`to_x`]) of `kind`.
fn from_x(x: &[f32; D], kind: Kind) -> Shape {
    let lin = |v: f32, r: (f32, f32)| r.0 + (r.1 - r.0) * v.clamp(0.0, 1.0);
    let log = |v: f32, r: (f32, f32)| {
        let (lo, hi) = ((r.0 - TOUCH).ln(), (r.1 - TOUCH).ln());
        (TOUCH + (lo + (hi - lo) * v.clamp(0.0, 1.0)).exp()).max(r.0)
    };
    let lips = kind.rounded.then(|| log(x[2], LIPS));
    let narrowing = kind
        .narrowed
        .then(|| (lin(x[3], narrowing_at(kind.model)), log(x[4], NARROWING)));
    if kind.model == Model::Modal {
        let q = [0, 1, 2].map(|k| lin(x[MODE_AT[k]], inverse::mode_range(k)));
        // The cosine tongue is not used by a modal tract; the projection's
        // inverse is not needed.
        return Shape {
            tongue_index: 20.0,
            tongue_diameter: 2.5,
            lips,
            narrowing,
            pharynx: 0.0,
            modes: Some(q),
        };
    }
    Shape {
        tongue_index: lin(x[0], TONGUE_INDEX),
        tongue_diameter: lin(x[1], TONGUE_DIAMETER),
        lips,
        narrowing,
        pharynx: lin(x[5], PHARYNX),
        modes: None,
    }
}

/// A start for target `t` from shape `s`: lips made to agree with the
/// target's.
fn start_from(s: &Shape, t: &Target) -> [f32; D] {
    let mut x = to_x(s);
    match t.lips {
        Lips::Spread => x[2] = 1.0,
        Lips::Rounded if s.lips.is_none_or(|l| l > LIPS.1) => {
            x[2] = to_x(&Shape {
                lips: Some(START_LIPS),
                ..*s
            })[2];
        }
        Lips::Rounded => {}
    }
    x
}

/// What one row of one vowel aims for.
struct Goal {
    /// `ln` of the target formants.
    t: [f32; 3],
    /// Coordinates of the neighbouring row, to stay near...
    near: [f32; D],
    /// ...and where the rows before point, not to bend away from.
    ahead: [f32; D],
    /// Which coordinates may move.
    free: [bool; D],
    /// How hard each coordinate is pulled towards `near` and `ahead`.
    pull: [f32; D],
    /// Lowest value of each coordinate (the highest is 1).
    lo: [f32; D],
    /// Highest value of each coordinate.
    hi: [f32; D],
    kind: Kind,
    /// The vowel shape it is.
    key: VowelKey,
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

    fn cost(&self, x: &[f32; D], r: &[f32; 3]) -> f32 {
        let mut c: f32 = r.iter().map(|v| v * v).sum();
        for d in 0..D {
            c += self.pull[d]
                * (STAY * (x[d] - self.near[d]) * (x[d] - self.near[d])
                    + BEND * (x[d] - self.ahead[d]) * (x[d] - self.ahead[d]));
        }
        c
    }

    fn eval(&self, m: &mut Meter, x: &[f32; D]) -> Option<([f32; 3], f32)> {
        let f = m.measure(&carrying(from_x(x, self.kind), self.key, self.length), self.length)?;
        let r = self.residuals(f);
        Some((r, self.cost(x, &r)))
    }

    /// Damped Gauss-Newton from `x`, on measured formants. Returns the
    /// coordinates reached, their residuals and cost.
    fn refine(&self, m: &mut Meter, mut x: [f32; D]) -> Option<([f32; D], [f32; 3], f32)> {
        for d in 0..D {
            x[d] = x[d].clamp(self.lo[d], self.hi[d]);
        }
        let (mut r, mut cost) = self.eval(m, &x)?;
        let mut damping = 1.0e-3f32;
        for _ in 0..20 {
            if r[0].abs() < 0.3 * GOOD && r[1].abs() < 0.3 * GOOD {
                break;
            }
            // Derivatives, stepping into the allowed box.
            let mut j = [[0.0f32; D]; 3];
            for d in 0..D {
                if !self.free[d] {
                    continue;
                }
                let h = if x[d] + H <= self.hi[d] { H } else { -H };
                let mut y = x;
                y[d] += h;
                let Some((ry, _)) = self.eval(m, &y) else {
                    continue;
                };
                for k in 0..3 {
                    j[k][d] = (ry[k] - r[k]) / h;
                }
            }
            let mut a = [[0.0f32; D]; D];
            let mut b = [0.0f32; D];
            for d in 0..D {
                for k in 0..3 {
                    b[d] += j[k][d] * r[k];
                    for c in 0..D {
                        a[d][c] += j[k][d] * j[k][c];
                    }
                }
                b[d] +=
                    self.pull[d] * (STAY * (x[d] - self.near[d]) + BEND * (x[d] - self.ahead[d]));
                a[d][d] += self.pull[d] * (STAY + BEND);
            }
            let mut moved = false;
            while damping < 1.0e3 {
                let mut lhs = a;
                let mut rhs = b.map(|v| -v);
                for d in 0..D {
                    lhs[d][d] += damping * (a[d][d] + 1.0e-4);
                    let stuck = !self.free[d]
                        || (x[d] <= self.lo[d] && b[d] > 0.0)
                        || (x[d] >= self.hi[d] && b[d] < 0.0);
                    if stuck {
                        for c in 0..D {
                            lhs[d][c] = 0.0;
                            lhs[c][d] = 0.0;
                        }
                        lhs[d][d] = 1.0;
                        rhs[d] = 0.0;
                    }
                }
                let Some(step) = solve_d(lhs, rhs) else { break };
                let mut y = x;
                for d in 0..D {
                    y[d] = (x[d] + step[d]).clamp(self.lo[d], self.hi[d]);
                }
                if let Some((ry, cy)) = self.eval(m, &y)
                    && cy < cost
                {
                    let size = (0..D).fold(0.0f32, |s, d| s.max((y[d] - x[d]).abs()));
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

/// Solve the D×D system `a · x = b` by Gaussian elimination.
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

/// One solved row.
#[derive(Clone, Copy)]
struct Solved {
    x: [f32; D],
    /// Relative errors of F1, F2, F3 against the target.
    err: [f32; 3],
    /// Whether the babble map's start won.
    from_map: bool,
}

/// Relative errors from `ln` residuals.
fn rel(r: &[f32; 3]) -> [f32; 3] {
    [
        r[0].exp() - 1.0,
        r[1].exp() - 1.0,
        (r[2] / F3_WEIGHT.sqrt()).exp() - 1.0,
    ]
}

/// Where the articulators of target `t` would be by phonetics: the classic
/// engine's tongue and lips for the sound (hand-drawn after the original's
/// tract diagram), its constriction as the narrowing, the target's
/// rounding. Of the many shapes that make the same formants the solver
/// keeps the one nearest to this, so that /u/ keeps a high back tongue
/// and does not make its formants with a pharynx pinched near the folds
/// (which a real voice's folds would feel: such shapes sound different
/// from their closed-glottis formants).
///
/// On the modal body the body is the speaker's measured one for the vowel
/// (see [`measured_prior`]); lips and narrowing as on the cosine body.
fn prior(kind: Kind, t: &Target) -> [f32; D] {
    let mut x = cosine_prior(t);
    if kind.model == Model::Modal {
        // The cosine prior's narrowing place, on the modal range of places.
        let at = NARROWING_AT.0 + (NARROWING_AT.1 - NARROWING_AT.0) * x[3];
        x[3] = ((at - MODAL_NARROWING_AT.0) / (MODAL_NARROWING_AT.1 - MODAL_NARROWING_AT.0))
            .clamp(0.0, 1.0);
        let q = measured_prior(t);
        for k in 0..3 {
            let (lo, hi) = inverse::mode_range(k);
            x[MODE_AT[k]] = ((q[k] - lo) / (hi - lo)).clamp(0.0, 1.0);
        }
        if !kind.narrowed {
            x[3] = 0.5;
            x[4] = 1.0;
        } else if t.key != VowelKey::Vowel(Phoneme::Er) {
            // A vowel that needs a narrowing on the modal body starts from
            // the one the cosine body has.
            let s = articulation(t.key)
                .physical_shape(1.0)
                .expect("a vowel shape");
            // (Read as a modal shape: the modal range of places.)
            let n = to_x(&Shape {
                modes: Some([0.0; 3]),
                ..s
            });
            x[3] = n[3];
            x[4] = n[4];
        }
    }
    x
}

/// The measured vowel (IPA, as `inverse::measured_modes` names it) that
/// stands for `key`, if the speaker said it.
fn measured_vowel(key: VowelKey) -> Option<&'static str> {
    use Phoneme as P;
    Some(match key {
        VowelKey::Vowel(P::Iy) => "i",
        VowelKey::Vowel(P::Ih) => "ɪ",
        VowelKey::Start(P::Ey) => "e",
        VowelKey::Vowel(P::Eh) => "ɛ",
        VowelKey::Vowel(P::Ae) => "æ",
        VowelKey::Vowel(P::Ah) => "ʌ",
        VowelKey::Vowel(P::Aa) => "ɑ",
        VowelKey::Vowel(P::Ao) => "ɔ",
        VowelKey::Start(P::Ow) => "o",
        VowelKey::Vowel(P::Uh) => "ʊ",
        VowelKey::Vowel(P::Uw) => "u",
        _ => return None,
    })
}

/// The modal body the solver starts target `t` from and pulls it towards:
/// the speaker's measured one for the eleven vowels he said; for the other
/// shapes (glides, the ends of diphthongs and lax glides, schwa, the
/// fronted /u/ and schwa, /ɚ/) a blend of his, each weighted by how near
/// its target's formants are to this one's (a Gaussian of 10 % in `ln` F1
/// and F2, F3 at a third of their weight), so that they stay inside the
/// shapes he makes.
fn measured_prior(t: &Target) -> [f32; 3] {
    if let Some(v) = measured_vowel(t.key) {
        return inverse::measured_modes(v).expect("a measured vowel");
    }
    let mut sum = [0.0f32; 3];
    let mut weight = 0.0f32;
    for m in TARGETS {
        let Some(v) = measured_vowel(m.key) else {
            continue;
        };
        let d2: f32 = (0..3)
            .map(|k| {
                let w = if k == 2 { F3_WEIGHT } else { 1.0 };
                w * (t.formants[k].ln() - m.formants[k].ln()).powi(2)
            })
            .sum();
        let w = (-d2 / (2.0 * 0.1f32.powi(2))).exp();
        let q = inverse::measured_modes(v).expect("a measured vowel");
        for k in 0..3 {
            sum[k] += w * q[k];
        }
        weight += w;
    }
    sum.map(|s| s / weight.max(1e-12))
}

/// [`prior`] on the cosine body.
fn cosine_prior(t: &Target) -> [f32; D] {
    let a = articulation(t.key);
    let (index, diameter) = a.tongue.expect("a vowel tongue");
    let lips = match t.lips {
        Lips::Spread => None,
        Lips::Rounded => Some(a.lips.map_or(START_LIPS, |c| c.diameter).min(LIPS.1)),
    };
    let narrowing = a.constriction.map(|c| (c.index, c.diameter));
    let mut x = to_x(&Shape {
        tongue_index: index,
        tongue_diameter: diameter,
        lips,
        narrowing,
        pharynx: if root_free(t.key) {
            natural_root(t.key)
        } else {
            0.0
        },
        modes: None,
    });
    if narrowing.is_none() {
        x[3] = 0.5;
        x[4] = 1.0;
    }
    x
}

/// The tongue root a vowel shape has by phonetics: pulled back for the
/// lax vowels whose root is solved, at rest for the others (see
/// [`root_range`]).
fn natural_root(key: VowelKey) -> f32 {
    let (lo, hi) = root_range(key);
    if lo < 0.0 && hi <= 0.0 {
        LAX_ROOT.max(lo)
    } else {
        0.0
    }
}

/// Lowest coordinates target `t` may take: only open vowels (F1 of
/// [`THROAT_F1`] or more) narrow the pharynx below [`ORAL`]. A narrowing
/// that low, close to the folds, loads them: the sound's F1 then rises
/// above the tract's own, by 5 % for /e/ and /æ/ (as the listener hears
/// them), where a narrowing along the tongue leaves it alone. Open vowels
/// need it all the same, to reach their F1.
fn lowest(model: Model, t: &Target) -> [f32; D] {
    let mut lo = [0.0; D];
    if t.formants[0] < THROAT_F1 {
        let (a, b) = narrowing_at(model);
        lo[3] = (ORAL - a) / (b - a);
    }
    if model == Model::Cosine {
        lo[5] = root_x(root_range(t.key).0);
    }
    lo
}

/// Highest coordinates target `t` may take: the root no further forward
/// than its class allows (see [`root_range`]).
fn highest(model: Model, t: &Target) -> [f32; D] {
    let mut hi = [1.0; D];
    if model == Model::Cosine {
        hi[5] = root_x(root_range(t.key).1);
    }
    hi
}

/// Which coordinates of target `t` are solved.
fn free(kind: Kind, key: VowelKey) -> [bool; D] {
    match kind.model {
        Model::Cosine => [true, true, kind.rounded, true, true, root_free(key)],
        Model::Modal => [true, true, kind.rounded, kind.narrowed, kind.narrowed, true],
    }
}

/// The coordinate of tongue root `r`.
fn root_x(r: f32) -> f32 {
    ((r - PHARYNX.0) / (PHARYNX.1 - PHARYNX.0)).clamp(0.0, 1.0)
}

/// Solve the native row of target `t`: from the articulators' natural
/// place for it and the babble map's answers, keep the best, measured by
/// formants and nearness to that place. (Not from the shape the table held
/// before: the table is then the same however often it is regenerated.)
fn native_row(m: &mut Meter, kind: Kind, t: &Target) -> Solved {
    let f = vowel_targets::formants(t.key, 1.0).expect("a target");
    let model = kind.model;
    let home = prior(kind, t);
    let mut starts: Vec<([f32; D], bool)> = vec![(home, false)];
    // The modal body starts from the speaker's own shape and from the
    // modal babble map's answers (`inverse::modal`), its narrowing, if it
    // has one, the cosine body's.
    for f3 in [Some(f[2]), None] {
        let near = from_x(&home, kind);
        let (a, b) = match model {
            Model::Cosine => (
                inverse::solve(f[0], f[1], f3, Some(&near)),
                inverse::solve(f[0], f[1], f3, None),
            ),
            Model::Modal => (
                inverse::modal::solve(f[0], f[1], f3, Some(&near)),
                inverse::modal::solve(f[0], f[1], f3, None),
            ),
        };
        for s in [a, b] {
            let mut x = start_from(&s, t);
            if model == Model::Modal {
                x[3] = home[3];
                x[4] = home[4];
            }
            starts.push((x, true));
        }
    }
    let lo = lowest(model, t);
    let starts: Vec<([f32; D], bool)> = starts
        .into_iter()
        .map(|(mut x, m)| {
            let hi = highest(model, t);
            for d in 0..D {
                x[d] = x[d].clamp(lo[d], hi[d]);
            }
            (x, m)
        })
        .collect();
    let mut pull = NATURAL_PULL;
    if model == Model::Modal {
        for k in MODE_AT {
            pull[k] = MEASURED_PULL;
        }
    }
    let goal = Goal {
        t: f.map(f32::ln),
        near: home,
        ahead: home,
        free: free(kind, t.key),
        pull,
        lo: lowest(model, t),
        hi: highest(model, t),
        kind,
        key: t.key,
        length: 1.0,
    };
    let mut best: Option<([f32; D], f32, bool)> = None;
    for (x, from_map) in starts {
        let Some((x, _, cost)) = goal.refine(m, x) else {
            continue;
        };
        if best.is_none_or(|b| cost < b.1) {
            best = Some((x, cost, from_map));
        }
    }
    // The pull towards the natural place has chosen the shape; now put the
    // formants on target from there, pulled only towards where it is.
    let (x, _, from_map) = best.expect("a measurable shape");
    let polish = Goal {
        near: x,
        ahead: x,
        pull: [1.0; D],
        ..goal
    };
    let (x, r, _) = polish.refine(m, x).expect("a measurable shape");
    Solved {
        x,
        err: rel(&r),
        from_map,
    }
}

/// Solve the rows of target `t` outwards from the native one `x0`, `up`
/// towards longer tracts or down towards shorter ones.
fn chain(m: &mut Meter, kind: Kind, t: &Target, x0: [f32; D], up: bool) -> Vec<(usize, Solved)> {
    let model = kind.model;
    // /a/ in a small body keeps its throat: the narrowing that would lift
    // F1 the last 2 % sits deep in the pharynx, and tightening it there
    // makes the vowel louder, not only higher (by 0.4 dB at length 0.6,
    // which upsets the balance of /s/ against /a/ that the physical gates
    // hold). Its tongue still moves.
    let throat = t.key == VowelKey::Vowel(Phoneme::Aa) && !up;
    let mut free = free(kind, t.key);
    if throat {
        free[3] = false;
        free[4] = false;
    }
    let rows: Vec<usize> = if up {
        (NATIVE_ROW + 1..ROWS).collect()
    } else {
        (0..NATIVE_ROW).rev().collect()
    };
    let (mut before, mut last) = (x0, x0);
    let mut out = Vec::new();
    for k in rows {
        let length = row_length(k);
        let target = vowel_targets::formants(t.key, length).expect("a target");
        // Carry on the way the shape has been moving.
        let mut ahead = [0.0f32; D];
        for d in 0..D {
            ahead[d] = if free[d] {
                (2.0 * last[d] - before[d]).clamp(0.0, highest(model, t)[d])
            } else {
                last[d]
            };
        }
        let goal = Goal {
            t: target.map(f32::ln),
            near: last,
            ahead,
            free,
            pull: [1.0; D],
            lo: lowest(model, t),
            hi: highest(model, t),
            kind,
            key: t.key,
            length,
        };
        let mut best = goal.refine(m, ahead).map(|(x, r, c)| (x, r, c, false));
        let miss = |r: &[f32; 3]| r[0].abs().max(r[1].abs());
        if best.is_none_or(|b| miss(&b.1) > TRY_MAP) {
            // (A modal `near` solves on the modal map.)
            let near = from_x(&last, kind);
            let guess =
                inverse::solve_for(target[0], target[1], Some(target[2]), Some(&near), length);
            let mut g = start_from(&guess, t);
            for d in 0..D {
                if !free[d] || (model == Model::Modal && (d == 3 || d == 4)) {
                    g[d] = last[d];
                }
            }
            // A different shape breaks the path the rows make, so it is
            // taken only when the path cannot go on well enough.
            if let Some((x, r, c)) = goal.refine(m, g)
                && best.is_none_or(|b| c < b.2 && miss(&b.1) > SWITCH)
            {
                best = Some((x, r, c, true));
            }
        }
        let (x, r, _, from_map) = best.expect("a measurable shape");
        out.push((
            k,
            Solved {
                x,
                err: rel(&r),
                from_map,
            },
        ));
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
/// arithmetic as `vowel_size::shape`.
fn interpolate(model: Model, rows: &[[i32; D]; ROWS], length: f32) -> Shape {
    let (a, s) = if length.to_bits() == 1.0f32.to_bits() {
        (NATIVE_ROW, 0.0)
    } else {
        let t = ((length - FIRST) / STEP).clamp(0.0, (ROWS - 1) as f32);
        let a = (t as usize).min(ROWS - 2);
        (a, t - a as f32)
    };
    let v = |k: usize| {
        let lo = rows[a][k] as f32;
        let hi = rows.get(a + 1).map_or(lo, |r| r[k] as f32);
        (lo + (hi - lo) * s) / 1000.0
    };
    if model == Model::Modal {
        // As `vowel_size::modal_shape` reads it.
        return Shape {
            tongue_index: 20.0,
            tongue_diameter: 2.5,
            lips: (rows[a][3] > 0).then(|| v(3).max(MIN_LIPS)),
            narrowing: (rows[a][5] < 3000).then(|| (v(4), v(5).max(MIN_NARROWING))),
            pharynx: 0.0,
            modes: Some([v(0), v(1), v(2)]),
        };
    }
    Shape {
        tongue_index: v(0).clamp(TONGUE_INDEX.0, TONGUE_INDEX.1),
        tongue_diameter: v(1).clamp(TONGUE_DIAMETER.0, TONGUE_DIAMETER.1),
        lips: (rows[a][2] > 0).then(|| v(2).max(MIN_LIPS)),
        narrowing: Some((v(3), v(4).max(MIN_NARROWING))),
        pharynx: v(5).clamp(PHARYNX.0, PHARYNX.1),
        modes: None,
    }
}

/// Relative errors of `f` against the target of `key` at `length`.
fn errors(f: Option<[f32; 3]>, key: VowelKey, length: f32) -> [f32; 3] {
    let t = vowel_targets::formants(key, length).expect("a target");
    f.map_or([f32::INFINITY; 3], |f| [0, 1, 2].map(|k| f[k] / t[k] - 1.0))
}

/// The Rust name of a key, as the table writes it.
fn key_src(key: VowelKey) -> String {
    match key {
        VowelKey::Vowel(p) => format!("Vowel(Phoneme::{p:?})"),
        VowelKey::Start(p) => format!("Start(Phoneme::{p:?})"),
        VowelKey::End(p) => format!("End(Phoneme::{p:?})"),
        VowelKey::Fronted(p) => format!("Fronted(Phoneme::{p:?})"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = survey::base_from_args();
    let models: &[Model] = if args.iter().any(|a| a == "--both") {
        &[Model::Cosine, Model::Modal]
    } else if args.iter().any(|a| a == "--modal") {
        &[Model::Modal]
    } else {
        &[Model::Cosine]
    };
    for &model in models {
        run(model, &base);
    }
}

/// Solve, verify and write the table of `model` for the tract `base`
/// configures (see `survey::base_from_args`).
fn run(model: Model, base: &synthie_trombone::Tuning) {
    let started = Instant::now();
    let targets: Vec<Target> = TARGETS.to_vec();
    let first = |t: &Target| Kind::of(model, t);

    // How far apart the lax and tense vowels are now, for the report.
    let pair_now = par_map(&PAIRS, |m, &(a, b)| {
        let fa = m.measure(&a.physical_shape(1.0).expect("a vowel"), 1.0);
        let fb = m.measure(&b.physical_shape(1.0).expect("a vowel"), 1.0);
        fa.zip(fb)
    });

    // The speaker's own shapes on the engine, before anything is solved.
    if model == Model::Modal {
        let measured: Vec<&Target> = targets
            .iter()
            .filter(|t| measured_vowel(t.key).is_some())
            .collect();
        let at_measured = par_map(&measured, |m, t| {
            m.measure(&carrying(from_x(&prior(first(t), t), first(t)), t.key, 1.0), 1.0)
        });
        println!("The measured shapes on the engine (lips as the cosine body's), against the targets:");
        for (t, f) in measured.iter().zip(&at_measured) {
            let e = errors(*f, t.key, 1.0);
            println!(
                "  {:>4} ({}): F1 {:+5.1} % F2 {:+5.1} % F3 {:+5.1} %",
                t.key.to_string(),
                measured_vowel(t.key).unwrap_or("?"),
                100.0 * e[0],
                100.0 * e[1],
                100.0 * e[2]
            );
        }
    }

    // 1. The native rows.
    let mut natives = par_map(&targets, |m, t| native_row(m, first(t), t));
    let mut kinds: Vec<Kind> = targets.iter().map(first).collect();
    if model == Model::Modal {
        // A vowel three modes and the lips leave short of its F1 or F2 takes
        // a narrowing along the tongue as well, starting from the cosine
        // body's.
        let short: Vec<usize> = (0..targets.len())
            .filter(|&v| !kinds[v].narrowed && natives[v].err[..2].iter().any(|e| e.abs() > NEEDS_NARROWING))
            .collect();
        let again = par_map(&short, |m, &v| {
            native_row(m, Kind { narrowed: true, ..kinds[v] }, &targets[v])
        });
        for (&v, s) in short.iter().zip(again) {
            let miss = |s: &Solved| s.err[0].abs().max(s.err[1].abs());
            println!(
                "{:>4} needs a narrowing: F1/F2 off by {:.2} % without, {:.2} % with",
                targets[v].key.to_string(),
                100.0 * miss(&natives[v]),
                100.0 * miss(&s)
            );
            if miss(&s) < miss(&natives[v]) {
                natives[v] = s;
                kinds[v].narrowed = true;
            }
        }
    }
    let kind = |t: &Target| kinds[targets.iter().position(|u| u.key == t.key).expect("a target")];
    for (t, s) in targets.iter().zip(&natives) {
        println!(
            "{:>4} native: F1 {:+.2} % F2 {:+.2} % F3 {:+.2} %{}  {:?}",
            t.key.to_string(),
            100.0 * s.err[0],
            100.0 * s.err[1],
            100.0 * s.err[2],
            if s.from_map { " (map)" } else { "" },
            from_x(&s.x, kind(t))
        );
    }
    if model == Model::Modal {
        println!("How far each solved body ends from the speaker's (coefficients, unit RMS over his vowels):");
        for (t, s) in targets.iter().zip(&natives) {
            let q = from_x(&s.x, kind(t)).modes.expect("modal");
            let p = measured_prior(t);
            println!(
                "  {:>4}{}: {:+.2} {:+.2} {:+.2} from {:+.2} {:+.2} {:+.2} (moved {:+.2} {:+.2} {:+.2})",
                t.key.to_string(),
                if measured_vowel(t.key).is_some() { " *" } else { "  " },
                q[0], q[1], q[2], p[0], p[1], p[2],
                q[0] - p[0], q[1] - p[1], q[2] - p[2]
            );
        }
    }

    // 2. Every other row, outwards from the native one.
    let jobs: Vec<(usize, bool)> = (0..targets.len())
        .flat_map(|v| [(v, false), (v, true)])
        .collect();
    let chains = par_map(&jobs, |m, &(v, up)| {
        chain(m, kinds[v], &targets[v], natives[v].x, up)
    });
    let mut solved: Vec<[Option<Solved>; ROWS]> = vec![[None; ROWS]; targets.len()];
    for (v, s) in natives.iter().enumerate() {
        solved[v][NATIVE_ROW] = Some(*s);
    }
    for (&(v, _), rows) in jobs.iter().zip(chains) {
        for (k, s) in rows {
            solved[v][k] = Some(s);
        }
    }
    println!("solved in {:.1} s", started.elapsed().as_secs_f32());

    // Quantise to thousandths of the raw shape numbers.
    let mut table = vec![[[0i32; D]; ROWS]; targets.len()];
    for (v, t) in targets.iter().enumerate() {
        for k in 0..ROWS {
            let s = solved[v][k].expect("every row solved");
            let r = raw(&from_x(&s.x, kind(t)));
            for d in 0..D {
                table[v][k][d] = (r[d] * 1000.0).round() as i32;
            }
        }
    }

    // 3. Verify: rows, half-way between them, and the tests' lengths.
    let mut lengths: Vec<f32> = (0..ROWS).map(row_length).collect();
    lengths.extend((0..ROWS - 1).map(|k| row_length(k) + 0.5 * STEP));
    lengths.extend([0.85, 0.93, 0.97, 1.03, 1.15]);
    lengths.sort_by(f32::total_cmp);
    lengths.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    let checks: Vec<(usize, f32)> = (0..targets.len())
        .flat_map(|v| lengths.iter().map(move |&l| (v, l)))
        .collect();
    let measured = par_map(&checks, |m, &(v, l)| {
        let f = m.measure(&carrying(interpolate(model, &table[v], l), targets[v].key, l), l);
        (f, errors(f, targets[v].key, l))
    });

    let mut report = String::new();
    let _ = writeln!(
        report,
        "Worst error against the targets over all {} shapes (F1, F2, F3):",
        targets.len()
    );
    // The same, shorter, for the table's header: table rows only.
    let mut summary =
        String::from("Largest F1 / F2 / F3 error against the targets, of all shapes:\n");
    let mut misses = Vec::new();
    for &l in &lengths {
        let mut worst = [(0.0f32, targets[0].key); 3];
        for (&(v, cl), (_, a)) in checks.iter().zip(&measured) {
            if cl.to_bits() != l.to_bits() {
                continue;
            }
            for k in 0..3 {
                if a[k].abs() > worst[k].0 {
                    worst[k] = (a[k].abs(), targets[v].key);
                }
            }
            if a[0].abs() > LIMIT || a[1].abs() > LIMIT {
                misses.push(format!(
                    "{} at length {l:.2}: F1 {:+.2} %, F2 {:+.2} %",
                    targets[v].key,
                    100.0 * a[0],
                    100.0 * a[1]
                ));
            }
        }
        let _ = writeln!(
            report,
            "length {l:.2}: {:4.2} / {:4.2} / {:4.2} % ({} / {} / {})",
            100.0 * worst[0].0,
            100.0 * worst[1].0,
            100.0 * worst[2].0,
            worst[0].1,
            worst[1].1,
            worst[2].1,
        );
        if (0..ROWS).any(|k| row_length(k).to_bits() == l.to_bits()) {
            let _ = writeln!(
                summary,
                "{l:.1}: {:4.2} / {:4.2} / {:5.2} %",
                100.0 * worst[0].0,
                100.0 * worst[1].0,
                100.0 * worst[2].0,
            );
        }
    }
    if misses.is_empty() {
        let _ = writeln!(
            report,
            "Every shape within {:.0} % for F1 and F2 at every length checked.",
            100.0 * LIMIT
        );
    } else {
        let _ = writeln!(report, "Missing {:.0} %:", 100.0 * LIMIT);
        for m in &misses {
            let _ = writeln!(report, "  {m}");
        }
    }
    // Per shape: the worst F1, F2 and F3 errors at the gates' lengths.
    let _ = writeln!(report, "Per shape, worst F1 / F2 / F3 error, %, at length:");
    let gate_lengths = [0.6f32, 0.8, 1.0, 1.2, 1.4];
    let _ = write!(report, "      ");
    for l in gate_lengths {
        let _ = write!(report, "{l:^18.1}");
    }
    let _ = writeln!(report);
    for (v, t) in targets.iter().enumerate() {
        let _ = write!(report, "{:>5} ", t.key.to_string());
        for l in gate_lengths {
            let i = checks
                .iter()
                .position(|&(cv, cl)| cv == v && (cl - l).abs() < 1e-4)
                .expect("a checked length");
            let e = measured[i].1;
            let _ = write!(
                report,
                "{:+5.1} {:+5.1} {:+5.1} ",
                100.0 * e[0],
                100.0 * e[1],
                100.0 * e[2]
            );
        }
        let _ = writeln!(report);
    }
    // Lax and tense, before and after.
    let at_native = |p: Phoneme| {
        let v = targets
            .iter()
            .position(|t| t.key == VowelKey::Vowel(p))
            .expect("a vowel");
        let i = checks
            .iter()
            .position(|&(cv, cl)| cv == v && cl.to_bits() == 1.0f32.to_bits())
            .expect("the native length");
        measured[i].0.expect("formants")
    };
    let _ = writeln!(report, "Lax against tense at length 1.0, ΔF1 / ΔF2, Hz:");
    for (&(a, b), now) in PAIRS.iter().zip(&pair_now) {
        let (fa, fb) = (at_native(a), at_native(b));
        let ta = vowel_targets::formants(VowelKey::Vowel(a), 1.0).expect("a target");
        let tb = vowel_targets::formants(VowelKey::Vowel(b), 1.0).expect("a target");
        let before = now.map_or_else(
            || "unmeasured".to_string(),
            |(x, y)| format!("{:+4.0} / {:+5.0}", y[0] - x[0], y[1] - x[1]),
        );
        let _ = writeln!(
            report,
            "  {a}-{b}: before {before}, now {:+4.0} / {:+5.0} (target {:+4.0} / {:+5.0})",
            fb[0] - fa[0],
            fb[1] - fa[1],
            tb[0] - ta[0],
            tb[1] - ta[1],
        );
    }
    let from_map: Vec<String> = solved
        .iter()
        .enumerate()
        .flat_map(|(v, rows)| {
            let key = targets[v].key;
            rows.iter().enumerate().filter_map(move |(k, s)| {
                s.filter(|s| s.from_map)
                    .map(|_| format!("{key} {:.1}", row_length(k)))
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

    // Write the table.
    let mut src = String::new();
    match model {
        Model::Cosine => {
            let _ = writeln!(
                src,
                "//! Vowel shapes by tract length, solved from `src/vowel_targets.rs` by\n\
                 //! `examples/vowel_size.rs`; do not edit. Regenerate with\n\
                 //! `cargo run -p synthie-trombone --example vowel_size --release`.\n\
                 //!\n\
                 //! ```text"
            );
        }
        Model::Modal => {
            let _ = writeln!(
                src,
                "//! Vowel shapes of the modal tongue body, by tract length (with\n\
                 //! `AnatomyConfig::tongue_model` at 1), solved from `src/vowel_targets.rs` by\n\
                 //! `examples/vowel_size.rs`, starting from and pulled towards the\n\
                 //! measured vowels' own bodies (`src/tract_modes.rs`); do not edit.\n\
                 //! Regenerate with\n\
                 //! `cargo run -p synthie-trombone --example vowel_size --release -- --modal`\n\
                 //! (or `-- --lengthened --both` for the lengthened tract).\n\
                 //!\n\
                 //! ```text"
            );
        }
    }
    for line in summary.lines() {
        let _ = writeln!(src, "//! {line}");
    }
    // The numbers of the tract it is solved for, where they are not the
    // defaults (the default tract's table has none).
    let changed: Vec<String> = synthie_trombone::Tuning::FIELDS
        .iter()
        .filter(|f| f.name != "tongue_model")
        .filter_map(|f| {
            let v = base.get(f.name)?;
            (v.to_bits() != f.default.to_bits()).then(|| format!("`{}` {v}", f.name))
        })
        .collect();
    if !changed.is_empty() {
        let _ = writeln!(src, "//! Solved for: {}.", changed.join(", "));
    }
    let _ = writeln!(
        src,
        "//! ```\n\n\
         use crate::phoneme::Phoneme;\n\
         use crate::vowel_targets::VowelKey::{{self, End, Fronted, Start, Vowel}};\n\n\
         /// Tract length of the first row.\n\
         pub(super) const FIRST: f32 = {FIRST:?};\n\
         /// Tract length between rows.\n\
         pub(super) const STEP: f32 = {STEP:?};\n\
         /// The row at the native length.\n\
         pub(super) const NATIVE: usize = {NATIVE_ROW};\n\
         /// Per vowel shape and row: {columns}, in thousandths.\n\
         #[rustfmt::skip]\n\
         pub(super) static SHAPES: &[(VowelKey, [[i32; {D}]; {ROWS}])] = &[",
        columns = match model {
            Model::Cosine => {
                "tongue index, tongue diameter, lips (`0`\n\
                 /// when spread), narrowing cell, narrowing diameter and tongue\n\
                 /// root"
            }
            Model::Modal => {
                "the body's three numbers, lips (`0`\n\
                 /// when spread), narrowing cell and narrowing diameter (`3000`\n\
                 /// for none)"
            }
        }
    );
    for (v, t) in targets.iter().enumerate() {
        let _ = writeln!(src, "    ({}, [", key_src(t.key));
        for rows in table[v].chunks(4) {
            let rows: Vec<String> = rows
                .iter()
                .map(|row| {
                    let cells: Vec<String> = row.iter().map(i32::to_string).collect();
                    format!("[{}]", cells.join(","))
                })
                .collect();
            let _ = writeln!(src, "        {},", rows.join(", "));
        }
        let _ = writeln!(src, "    ]),");
    }
    let _ = writeln!(src, "];");
    let path = match (model, vowel_targets::lengthened(base)) {
        (Model::Cosine, false) => concat!(env!("CARGO_MANIFEST_DIR"), "/src/vowel_size_table.rs"),
        (Model::Cosine, true) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/vowel_size_wave_table.rs")
        }
        (Model::Modal, false) => concat!(env!("CARGO_MANIFEST_DIR"), "/src/vowel_modes_table.rs"),
        (Model::Modal, true) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/vowel_modes_wave_table.rs")
        }
    };
    std::fs::write(path, &src).expect("write the table");
    if model == Model::Cosine {
        let rows: Vec<(VowelKey, [[i32; D]; ROWS])> =
            targets.iter().map(|t| t.key).zip(table.iter().copied()).collect();
        *SOLVED_COSINE.lock().expect("lock") = Some(rows);
    }
    println!(
        "wrote {path}: {} bytes, {:.1} s in all",
        src.len(),
        started.elapsed().as_secs_f32()
    );
}
