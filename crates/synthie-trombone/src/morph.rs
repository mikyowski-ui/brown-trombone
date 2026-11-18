//! Continuous anatomical morphing: one voice as another body would
//! produce it.
//!
//! A [`MorphTarget`] names four things about the body, each against the
//! voice it is laid on: how long the vocal tract is, how low the larynx
//! rests, how thick the vocal folds are and how wide the throat opens
//! above the tube over the folds. It says nothing about how the voice
//! speaks (its pitch contour, timing and effort), so the same sentence, by
//! the rules or by a neural prosody, can be spoken by a smaller, a larger,
//! a younger or a more mature body.
//!
//! A morph is used in two ways. Laid over a voice's settings before it
//! speaks ([`VoiceSettings::with_morph`](crate::VoiceSettings::with_morph))
//! it is simply part of the voice. Given to a voice that is sounding
//! ([`Trombone::set_morph`](crate::Trombone::set_morph)) the body glides
//! there: the physical engine keeps the waves in its tube, cell for cell,
//! and the folds keep vibrating, so the formants and the voice's colour
//! move without a click or a break.

use crate::tuning::Tuning;

/// A body to morph a voice towards (see the [module](self)).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct MorphTarget {
    /// The vocal tract's length against the voice's own, `0.75‥1.25`
    /// (`1` unchanged): every resonance moves by its inverse, and the
    /// folds scale with the body.
    pub tract_scale: f32,
    /// How far the larynx rests lower than the voice's, cm at the native
    /// size, `−0.3‥0.8` (`0` unchanged; added to
    /// [`AnatomyConfig::larynx_depth`](crate::AnatomyConfig::larynx_depth)).
    pub larynx_depth: f32,
    /// The folds' size against the body's, `0.6‥1.6` of the voice's own
    /// (`1` unchanged; it multiplies
    /// [`AnatomyConfig::fold_thickness`](crate::AnatomyConfig::fold_thickness)).
    pub fold_thickness: f32,
    /// The width of the lower pharynx the tube above the folds opens
    /// into, `0.6‥1.6` of the voice's own (`1` unchanged; it multiplies
    /// [`AnatomyConfig::epilarynx_flare`](crate::AnatomyConfig::epilarynx_flare)).
    pub epilarynx_flare: f32,
}

impl Default for MorphTarget {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

impl MorphTarget {
    /// The morph that leaves a voice as it is, bit for bit.
    pub const NEUTRAL: Self = Self {
        tract_scale: 1.0,
        larynx_depth: 0.0,
        fold_thickness: 1.0,
        epilarynx_flare: 1.0,
    };

    /// The whole body `scale` times the voice's (`0.75‥1.25`): a longer or
    /// shorter tract, with the folds that go with it. The pitch a voice is
    /// asked to speak at is not the body's: see
    /// [`Persona::size`](crate::Persona::size) for a size that lowers the
    /// pitch too.
    #[must_use]
    pub fn sized(scale: f32) -> Self {
        Self {
            tract_scale: scale,
            ..Self::NEUTRAL
        }
        .clamped()
    }

    /// The same speaker `amount` more mature (`−1‥1`; `0` unchanged): at
    /// `1` the tract is 5 % longer with the larynx resting 0.3 cm lower
    /// (7 % of length together on a small voice), the lower pharynx a
    /// quarter wider, and the folds 5 % smaller against the body, which is
    /// to say they stay as they were while the tract grows. Negative
    /// amounts go the other way, towards a juvenile of the same voice. The
    /// numbers are those that turn the `bright` voice into a grown
    /// woman's: measured, they lower every formant by 7 % and the sound's
    /// level above 1 kHz against that below by 2 to 3 dB (see the README,
    /// "Morphing between bodies").
    #[must_use]
    pub fn matured(amount: f32) -> Self {
        let a = if amount.is_finite() {
            amount.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        Self {
            tract_scale: 1.0 + MATURE_TRACT * a,
            larynx_depth: MATURE_LARYNX * a,
            fold_thickness: 1.0 + MATURE_FOLDS * a,
            epilarynx_flare: 1.0 + MATURE_FLARE * a,
        }
        .clamped()
    }

    /// This morph and then `other` on top of it.
    #[must_use]
    pub fn then(self, other: Self) -> Self {
        Self {
            tract_scale: self.tract_scale * other.tract_scale,
            larynx_depth: self.larynx_depth + other.larynx_depth,
            fold_thickness: self.fold_thickness * other.fold_thickness,
            epilarynx_flare: self.epilarynx_flare * other.epilarynx_flare,
        }
        .clamped()
    }

    /// The morph with every number inside its range (a number that is not
    /// finite goes back to neutral).
    #[must_use]
    pub fn clamped(self) -> Self {
        let within = |x: f32, lo: f32, hi: f32, neutral: f32| {
            if x.is_finite() {
                x.clamp(lo, hi)
            } else {
                neutral
            }
        };
        Self {
            tract_scale: within(self.tract_scale, 0.75, 1.25, 1.0),
            larynx_depth: within(self.larynx_depth, -0.3, 0.8, 0.0),
            fold_thickness: within(self.fold_thickness, 0.6, 1.6, 1.0),
            epilarynx_flare: within(self.epilarynx_flare, 0.6, 1.6, 1.0),
        }
    }

    /// Whether this morph leaves a voice as it is.
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        *self == Self::NEUTRAL
    }

    /// A tuning with this morph's larynx, folds and throat laid over it
    /// (the tract's length is the voice's, not the tuning's).
    #[must_use]
    pub fn over(&self, tuning: &Tuning) -> Tuning {
        let mut t = *tuning;
        let a = &mut t.anatomy;
        a.larynx_depth = (a.larynx_depth + self.larynx_depth).clamp(-0.3, 0.8);
        a.fold_thickness = (a.fold_thickness * self.fold_thickness).clamp(0.6, 1.6);
        a.epilarynx_flare = (a.epilarynx_flare * self.epilarynx_flare).clamp(0.6, 2.2);
        t
    }

    /// One step from this morph towards `target`: no number moves by more
    /// than its share of `reach` (the tract's scale by `reach` itself, the
    /// folds and the throat alike, the larynx by twice that in cm), and
    /// all of them arrive together.
    #[must_use]
    pub(crate) fn toward(self, target: Self, reach: f32) -> Self {
        let d = [
            target.tract_scale - self.tract_scale,
            0.5 * (target.larynx_depth - self.larynx_depth),
            target.fold_thickness - self.fold_thickness,
            target.epilarynx_flare - self.epilarynx_flare,
        ];
        let far = d.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        if far <= reach {
            return target;
        }
        let k = reach / far;
        Self {
            tract_scale: self.tract_scale + k * d[0],
            larynx_depth: self.larynx_depth + 2.0 * k * d[1],
            fold_thickness: self.fold_thickness + k * d[2],
            epilarynx_flare: self.epilarynx_flare + k * d[3],
        }
    }
}

/// What [`MorphTarget::matured`] at `1` adds: the tract's length (a share
/// of it), the larynx's rest (cm), the folds' size and the lower pharynx's
/// width (shares).
const MATURE_TRACT: f32 = 0.05;
const MATURE_LARYNX: f32 = 0.3;
const MATURE_FOLDS: f32 = -0.05;
const MATURE_FLARE: f32 = 0.25;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_changes_nothing() {
        let t = Tuning::DEFAULT;
        assert_eq!(MorphTarget::NEUTRAL.over(&t), t);
        assert!(MorphTarget::matured(0.0).is_neutral());
        assert!(MorphTarget::sized(1.0).is_neutral());
        assert!(MorphTarget::NEUTRAL.then(MorphTarget::NEUTRAL).is_neutral());
    }

    #[test]
    fn a_morph_stays_in_its_range() {
        let wild = MorphTarget {
            tract_scale: 9.0,
            larynx_depth: f32::NAN,
            fold_thickness: -1.0,
            epilarynx_flare: f32::INFINITY,
        }
        .clamped();
        assert_eq!(wild.tract_scale, 1.25);
        assert_eq!(wild.larynx_depth, 0.0);
        assert_eq!(wild.fold_thickness, 0.6);
        assert_eq!(wild.epilarynx_flare, 1.0);
    }

    #[test]
    fn steps_arrive_together() {
        let to = MorphTarget {
            tract_scale: 1.2,
            larynx_depth: 0.4,
            fold_thickness: 0.9,
            epilarynx_flare: 1.1,
        };
        let mut m = MorphTarget::NEUTRAL;
        let mut steps = 0;
        while m != to {
            let next = m.toward(to, 0.01);
            assert!((next.tract_scale - m.tract_scale).abs() <= 0.0101);
            m = next;
            steps += 1;
            assert!(steps < 100);
        }
        // A fifth of the way in steps of a hundredth (rounding may add one).
        assert!((20..=21).contains(&steps), "{steps}");
    }
}
