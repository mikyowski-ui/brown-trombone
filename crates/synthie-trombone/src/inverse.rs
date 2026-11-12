//! Babble map: from formants back to a vocal tract shape.

use crate::math::{exp, ln};

/// Lip cell that [`Shape::lips`] narrows.
pub const LIP_CELL: f32 = 41.0;

/// What a touch diameter loses before it becomes a tract diameter.
const TOUCH: f32 = 0.3;
/// Tongue body positions.
const TONGUE_INDEX: (f32, f32) = (12.0, 29.0);
/// Tongue body heights.
const TONGUE_DIAMETER: (f32, f32) = (2.05, 3.5);
/// Lip rounding, touch diameter: from as narrow as a vowel gets without
/// hissing to lips that no longer shape the tract.
const LIPS: (f32, f32) = (0.65, 2.3);
/// Where along the tongue a narrowing can sit.
const NARROWING_AT: (f32, f32) = (14.0, 34.0);
/// Narrowing, touch diameter: from the narrowest that does not hiss to
/// wider than the tract ever is.
const NARROWING: (f32, f32) = (0.7, 3.0);
/// Coordinates this close to `1` mean "no rounding" or "no narrowing".
const OPEN: f32 = 1.0 - 1.0e-4;

/// A vowel shape of the physical engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Tongue body position (cell index).
    pub tongue_index: f32,
    /// Tongue body height (diameter under the tongue).
    pub tongue_diameter: f32,
    /// Lip rounding: touch diameter at the lips, `None` for spread lips.
    pub lips: Option<f32>,
    /// Narrowing along the tongue, `(cell, touch diameter)`.
    pub narrowing: Option<(f32, f32)>,
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

impl Shape {
    /// The shape as five coordinates, each `0‥1`: tongue position, tongue
    /// height, lip opening, narrowing place and narrowing opening. Openings
    /// are on a logarithmic scale, and `1` there means none.
    #[must_use]
    pub fn coordinates(&self) -> [f32; 5] {
        let (at, narrowing) = match self.narrowing {
            Some((at, d)) => (unit(at, NARROWING_AT), unit_log(d, NARROWING)),
            None => (0.5, 1.0),
        };
        [
            unit(self.tongue_index, TONGUE_INDEX),
            unit(self.tongue_diameter, TONGUE_DIAMETER),
            self.lips.map_or(1.0, |d| unit_log(d, LIPS)),
            at,
            narrowing,
        ]
    }

    /// The shape at the given [`coordinates`](Self::coordinates).
    #[must_use]
    pub fn from_coordinates(x: [f32; 5]) -> Self {
        Self {
            tongue_index: from_unit(x[0], TONGUE_INDEX),
            tongue_diameter: from_unit(x[1], TONGUE_DIAMETER),
            lips: (x[2] < OPEN).then(|| from_unit_log(x[2], LIPS)),
            narrowing: (x[4] < OPEN)
                .then(|| (from_unit(x[3], NARROWING_AT), from_unit_log(x[4], NARROWING))),
        }
    }
}
