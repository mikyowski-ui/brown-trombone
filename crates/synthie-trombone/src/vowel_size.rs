//! Physical vowel shapes for speakers of every size.
//!
//! Each vowel of the physical engine aims for the formants of its
//! [target](crate::vowel_targets), scaled to the speaker's size: a shorter
//! tract should raise every formant by `1 / length`, as a smaller person's
//! are. The tube itself does, but the yielding walls, the side cavities and
//! the load of the lips do not scale with it, so one shape cannot serve
//! every size. `examples/vowel_size.rs` solved, for a grid of tract
//! lengths, the shape that puts each vowel's F1 and F2 (and as far as
//! possible F3) on its target, and wrote them to `vowel_size_table.rs`;
//! here the shapes are interpolated between the grid rows. That is a few
//! additions and multiplications, so it is cheap enough for the audio
//! thread.
//!
//! Lips and narrowing are never tighter than [`MIN_LIPS`] and
//! [`MIN_NARROWING`], so no shape hisses.

use crate::inverse::Shape;
use crate::tract::{PHARYNX, TONGUE_DIAMETER, TONGUE_INDEX};
use crate::vowel_targets::{TARGET_COUNT, VowelKey, position};

#[path = "vowel_size_table.rs"]
mod table;

/// A physical vowel shape: tongue body `(index, diameter)`, lip rounding,
/// the narrowing along the tongue `(cell, diameter)` and the tongue root.
pub(crate) type VowelShape = ((f32, f32), Option<f32>, (f32, f32), f32);

/// Tightest lip rounding a vowel may have (that of /u/); tighter lips
/// would hiss.
pub(crate) const MIN_LIPS: f32 = 0.65;
/// Tightest narrowing along the tongue a vowel may have (that of /i/).
pub(crate) const MIN_NARROWING: f32 = 0.7;

/// Table entries are in thousandths.
const PER_UNIT: f32 = 1000.0;

/// Narrowing of a shape that has none: wide open, half-way along the
/// tongue.
const NO_NARROWING: (f32, f32) = (24.0, 3.0);

/// Vowel shapes a voice holds in place of the table's, one per
/// [target](crate::vowel_targets::TARGETS), `None` where the table's
/// shape is used (see
/// [`Trombone::set_vowel_shape`](crate::Trombone::set_vowel_shape)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OwnShapes([Option<VowelShape>; TARGET_COUNT]);

impl OwnShapes {
    /// Every shape from the table.
    pub(crate) const NONE: Self = Self([None; TARGET_COUNT]);

    /// Hold `shape` for `key` (or the table's again, with `None`); `false`
    /// if `key` has no target.
    pub(crate) fn set(&mut self, key: VowelKey, shape: Option<&Shape>) -> bool {
        let Some(k) = position(key) else {
            return false;
        };
        self.0[k] = shape.map(|s| {
            (
                (
                    s.tongue_index
                        .clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
                    s.tongue_diameter
                        .clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
                ),
                s.lips.map(|d| d.max(MIN_LIPS)),
                s.narrowing
                    .map_or(NO_NARROWING, |(at, d)| (at, d.max(MIN_NARROWING))),
                root(s.pharynx),
            )
        });
        true
    }

    /// The shape held for `key`, if one is.
    fn get(&self, key: VowelKey) -> Option<VowelShape> {
        position(key).and_then(|k| self.0[k])
    }
}

impl Default for OwnShapes {
    fn default() -> Self {
        Self::NONE
    }
}

/// A tongue root inside what the engine allows; at rest if not finite.
fn root(pharynx: f32) -> f32 {
    if pharynx.is_finite() {
        pharynx.clamp(*PHARYNX.start(), *PHARYNX.end())
    } else {
        0.0
    }
}

/// [`shape`], unless `own` holds a shape of its own for `key`.
pub(crate) fn shape_in(key: VowelKey, tract_length: f32, own: &OwnShapes) -> Option<VowelShape> {
    own.get(key).or_else(|| shape(key, tract_length))
}

/// The shape of `key` at `tract_length`, or `None` for a key the table
/// does not know. Lengths outside the table are clamped into it; at
/// exactly `1.0` the native row is returned as it is. The tongue and its
/// root stay where the engine lets them go, and lips and narrowing are
/// never tighter than [`MIN_LIPS`] and [`MIN_NARROWING`].
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn shape(key: VowelKey, tract_length: f32) -> Option<VowelShape> {
    let (_, rows) = table::SHAPES.iter().find(|(k, _)| *k == key)?;
    // Row below the length, and how far towards the next one it lies.
    let (row, frac) = if tract_length.to_bits() == 1.0f32.to_bits() || !tract_length.is_finite() {
        (table::NATIVE, 0.0)
    } else {
        let last = (rows.len() - 1) as f32;
        let pos = ((tract_length - table::FIRST) / table::STEP).clamp(0.0, last);
        let row = (pos as usize).min(rows.len() - 2);
        (row, pos - row as f32)
    };
    let v = |k: usize| {
        let lo = rows[row][k] as f32;
        let hi = rows.get(row + 1).map_or(lo, |r| r[k] as f32);
        // Divided, not multiplied by 0.001: a row then decodes to exactly
        // the number written in it.
        (lo + (hi - lo) * frac) / PER_UNIT
    };
    Some((
        (
            v(0).clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
            v(1).clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
        ),
        // Spread lips are stored as 0.
        (rows[row][2] > 0).then(|| v(2).max(MIN_LIPS)),
        (v(3), v(4).max(MIN_NARROWING)),
        root(v(5)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_covers_the_supported_lengths() {
        let rows = table::SHAPES[0].1.len();
        #[allow(clippy::cast_precision_loss)]
        let end = table::FIRST + table::STEP * (rows - 1) as f32;
        assert!(table::FIRST <= 0.5 + 1e-6 && end >= 1.6 - 1e-6);
        #[allow(clippy::cast_precision_loss)]
        let native = table::FIRST + table::STEP * table::NATIVE as f32;
        assert!((native - 1.0).abs() < 1e-6);
    }

    #[test]
    fn every_target_has_shapes() {
        for t in crate::vowel_targets::TARGETS {
            assert!(shape(t.key, 1.0).is_some(), "{}", t.key);
        }
    }
}
