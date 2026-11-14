//! The tilt of the voice source: a first-order shelf with its corner where
//! the glottal return phase of a modal voice puts it (about 500 Hz for a
//! return phase of 4 % of a 110 Hz period), `depth` dB down at the top
//! against the bottom, unity at [`PIVOT`] times the corner.
//!
//! The folds of this model shut abruptly, and the flow through them
//! ripples under the tract's own pressure: the derivative of the glottal
//! flow has its 2 to 4 kHz band some 20 dB under its 0 to 1 kHz band,
//! where a Liljencrants–Fant source of a modal male voice (Rd 1.0 to 1.3:
//! open quotient 0.7 to 0.8, return phase 4 to 5 % of the period) has 33
//! to 37 dB. Softer contact, edges that meet along their length, or a
//! low-pass on the area or on the flow the tract takes up did not change
//! that (round 8): the pressure the tract sends back gave it back. So the
//! tilt is put on the voice as it leaves, where the tract is linear and
//! the shelf on the radiated voice is the shelf on the glottal flow; and
//! every noise source in the tract is run through the shelf's inverse
//! before it enters, so that what the tract makes of frication, bursts'
//! hiss and breath leaves it as without the tilt. Both are on only as far
//! as the folds are asked to sing (`Phys::radiate`, [`Tilt::blend`]): the
//! flow of a voiceless burst or of the breath before a phrase is not the
//! voice.

use crate::math::{cos, exp, powf, sqrt};
use core::f32::consts::PI;

/// Where the shelf passes the voice unchanged, as a multiple of its
/// corner (500 Hz at native size). Unity at 0 Hz made a held /a/ 4.9 dB
/// quieter (6.3 dB in its peak sample), and the LibriSpeech reader's /s z/
/// would then have been 2 dB quieter against his vowels than the engine's;
/// unity at 750 Hz kept /a/ as loud, but put 5 dB more under its voice
/// bar and the flow of its voiced bursts. At the corner /a/ is 1.9 dB
/// quieter (3.5 dB in its peak) and the engine's /s z/ come to 1.3 dB
/// under the vowels beside them in sentences (the reader's: 0.4; 3.3
/// before the tilt).
pub(crate) const PIVOT: f32 = 1.0;

/// Coefficients of `y = b0·x + b1·x' + a1·y'` (primes: the last tick).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shelf {
    b0: f32,
    b1: f32,
    a1: f32,
}

impl Shelf {
    /// Passes everything unchanged.
    pub(crate) const FLAT: Self = Self {
        b0: 1.0,
        b1: 0.0,
        a1: 0.0,
    };

    /// The shelf (`inverse == false`) or its inverse, for ticks of `dt`
    /// seconds: a pole at `corner` Hz and a zero `depth` dB above it
    /// (matched z-transform), unity gain at `PIVOT · corner`.
    pub(crate) fn new(corner: f32, depth: f32, dt: f32, inverse: bool) -> Self {
        if depth <= 0.0 || corner <= 0.0 {
            return Self::FLAT;
        }
        let zero = corner * powf(10.0, depth / 20.0);
        let zp = exp(-2.0 * PI * corner * dt);
        let zz = exp(-2.0 * PI * zero * dt);
        // |1 - a·e^(-jw)| at the pivot.
        let w = 2.0 * PI * PIVOT * corner * dt;
        let mag = |a: f32| sqrt(1.0 - 2.0 * a * cos(w) + a * a);
        let k = mag(zp) / mag(zz);
        if inverse {
            Self {
                b0: 1.0 / k,
                b1: -zp / k,
                a1: zz,
            }
        } else {
            Self {
                b0: k,
                b1: -k * zz,
                a1: zp,
            }
        }
    }
}

/// The state of one stream through a [`Shelf`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Tilt {
    x: f32,
    y: f32,
}

impl Tilt {
    #[inline]
    pub(crate) fn run(&mut self, s: &Shelf, x: f32) -> f32 {
        let y = s.b0 * x + s.b1 * self.x + s.a1 * self.y;
        self.x = x;
        self.y = y;
        y
    }

    /// `x` through the shelf as far as `on` (`0‥1`) says, from not at all
    /// to wholly; the shelf's state follows `x` throughout.
    #[inline]
    pub(crate) fn blend(&mut self, s: &Shelf, x: f32, on: f32) -> f32 {
        let y = self.run(s, x);
        if on <= 0.0 { x } else { x + (y - x) * on }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn the_inverse_undoes_the_shelf_and_flat_is_exact() {
        let dt = 1.0 / 96_000.0;
        let (fwd, inv) = (
            Shelf::new(500.0, 18.0, dt, false),
            Shelf::new(500.0, 18.0, dt, true),
        );
        let (mut a, mut b) = (Tilt::default(), Tilt::default());
        let mut seed = 1u32;
        for _ in 0..10_000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let x = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
            let y = a.run(&fwd, b.run(&inv, x));
            assert!((y - x).abs() < 1e-4, "{x} -> {y}");
        }
        let mut f = Tilt::default();
        for x in [0.3f32, -1.7, 2.5e-9, 0.0] {
            assert_eq!(f.run(&Shelf::FLAT, x).to_bits(), x.to_bits());
        }
        // `depth` down at the top against the bottom, unity at the pivot.
        let mut d = Tilt::default();
        let mut bottom = 0.0;
        for _ in 0..20_000 {
            bottom = d.run(&fwd, 1.0);
        }
        let mut n = Tilt::default();
        let (mut sign, mut top) = (1.0, 0.0);
        for _ in 0..20_000 {
            top = n.run(&fwd, sign);
            sign = -sign;
        }
        let db = |x: f32| 20.0 * crate::math::ln(x.abs()) / core::f32::consts::LN_10;
        let fall = db(top) - db(bottom);
        assert!((fall + 18.0).abs() < 0.5, "{fall} dB");
        let hz = PIVOT * 500.0;
        let mut p = Tilt::default();
        let mut peak = 0.0f32;
        for k in 0..96_000 {
            #[allow(clippy::cast_precision_loss)]
            let x = crate::math::sin(2.0 * PI * hz * k as f32 * dt);
            let y = p.run(&fwd, x);
            if k > 48_000 {
                peak = peak.max(y.abs());
            }
        }
        assert!((peak - 1.0).abs() < 0.01, "{peak} at the pivot");
    }
}
