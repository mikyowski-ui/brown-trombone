//! Physical vowel shapes for speakers of every size.
//!
//! A shorter tract should raise every formant by `1 / length`, as a smaller
//! person's are. The tube itself does, but the yielding walls, the side
//! cavities and the load of the lips do not scale with it, so the vowel
//! shapes fitted at the native length drift: F1 by up to 8.5 % at length
//! 0.6. `examples/vowel_size.rs` measured, for a grid of tract lengths, how
//! each vowel's shape has to change so that its F1 and F2 (and as far as
//! possible F3) are the native ones divided by the length; here those
//! changes are interpolated between the grid rows. That is a few additions
//! and multiplications, so it is cheap enough for the audio thread.
//!
//! At the native length the shape is exactly the native one. Lips and
//! narrowing never get tighter than those of the tightest native vowels, so
//! no shape hisses. That limits the close rounded /u/ and /w/ in a long
//! tract: their F1 stays above `native / length`, by 3 % at length 1.2 and
//! 6 % at 1.4. And /a/ in a short tract keeps its throat, and so 2 % of
//! its F1 drift at length 0.6: the narrowing that would lift F1 the rest of
//! the way also makes the vowel louder.

use crate::phoneme::Phoneme;
use crate::tract::{TONGUE_DIAMETER, TONGUE_INDEX};

#[path = "vowel_size_table.rs"]
mod table;

/// A physical vowel shape: tongue body `(index, diameter)`, lip rounding,
/// and the narrowing along the tongue `(cell, diameter)`.
pub(crate) type VowelShape = ((f32, f32), Option<f32>, (f32, f32));

/// Tightest lip rounding of any physical vowel (that of /u/); tighter
/// lips would hiss.
pub(crate) const MIN_LIPS: f32 = 0.65;
/// Tightest narrowing along the tongue of any physical vowel (that of
/// /i/ and /u/).
pub(crate) const MIN_NARROWING: f32 = 0.7;

/// Value of a table entry, in thousandths.
const MILLI: f32 = 1.0e-3;

/// The shape `vowel` takes at `tract_length`, given its shape `native`
/// at the native length. Lengths outside the table are clamped into it;
/// at exactly `1.0`, or for a vowel the table does not know, the native
/// shape is returned unchanged. The tongue stays where the engine lets it
/// go, and lips and narrowing are never tighter than [`MIN_LIPS`] and
/// [`MIN_NARROWING`].
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn at_length(vowel: Phoneme, native: VowelShape, tract_length: f32) -> VowelShape {
    if tract_length.to_bits() == 1.0f32.to_bits() || !tract_length.is_finite() {
        return native;
    }
    let Some((_, rows)) = table::DELTAS.iter().find(|(q, _)| *q == vowel) else {
        return native;
    };
    // Row below the length, and how far towards the next one it lies.
    let last = (rows.len() - 1) as f32;
    let pos = ((tract_length - table::FIRST) / table::STEP).clamp(0.0, last);
    let row = (pos as usize).min(rows.len() - 2);
    let frac = pos - row as f32;
    let d = |k: usize| {
        let (lo, hi) = (f32::from(rows[row][k]), f32::from(rows[row + 1][k]));
        (lo + (hi - lo) * frac) * MILLI
    };
    let ((index, diameter), lips, (at, narrow)) = native;
    (
        (
            (index + d(0)).clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
            (diameter + d(1)).clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
        ),
        lips.map(|l| (l + d(2)).max(MIN_LIPS)),
        (at + d(3), (narrow + d(4)).max(MIN_NARROWING)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_covers_the_supported_lengths() {
        let rows = table::DELTAS[0].1.len();
        #[allow(clippy::cast_precision_loss)]
        let end = table::FIRST + table::STEP * (rows - 1) as f32;
        assert!(table::FIRST <= 0.5 + 1e-6 && end >= 1.6 - 1e-6);
    }
}
