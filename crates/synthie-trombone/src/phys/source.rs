//! What drives the glottis: lung pressure and a glottal area.
//!
//! This is the *kinematic* source: the area between the folds follows a
//! prescribed pulse at the requested pitch, and the flow through it is
//! whatever lung pressure and the tract allow. Unlike the classic engine's
//! fixed flow pulse, the flow here already feels the tract above it (it
//! skews and ripples with the vowel) and the lungs below it.
//!
//! The controls are the classic ones, reinterpreted:
//!
//! * tenseness → how tightly the folds are pressed together (open quotient
//!   and the leak that stays open through the cycle);
//! * the voicing gate → adducted (vibrating) or pulled apart (breathing);
//! * anything that needs air — voicing or a consonant — keeps the lungs
//!   pushing.

use super::CM_H2O;
use crate::glottis::Glottis;
use crate::math::cos;
use core::f32::consts::PI;

/// Lung pressure while speaking, dyn/cm².
const LUNG_PRESSURE: f32 = 8.0 * CM_H2O;
/// Seconds for lung pressure to build or fall.
const LUNG_TIME: f32 = 0.03;
/// Seconds for the folds to come together or part.
const ADDUCTION_TIME: f32 = 0.012;
/// Peak area swept by the vibrating folds, cm², native size.
const VIBRATION_AREA: f32 = 0.14;
/// Standing leak of a fully breathy voice, cm².
const BREATHY_LEAK: f32 = 0.12;
/// Glottal opening for voiceless sounds and breathing, cm².
const ABDUCTED_AREA: f32 = 0.25;
/// Point in the open phase where the area peaks.
const PEAK: f32 = 0.62;

/// Kinematic glottal source.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    phase: f32,
    lung: f32,
    lung_target: f32,
    leak: f32,
    leak_target: f32,
    f0: f32,
    f1: f32,
    tense0: f32,
    tense1: f32,
    amp0: f32,
    amp1: f32,
    scale: f32,
}

impl Source {
    pub(crate) fn new() -> Self {
        Self {
            phase: 0.0,
            lung: 0.0,
            lung_target: 0.0,
            leak: ABDUCTED_AREA,
            leak_target: ABDUCTED_AREA,
            f0: 140.0,
            f1: 140.0,
            tense0: 0.6,
            tense1: 0.6,
            amp0: 0.0,
            amp1: 0.0,
            scale: 1.0,
        }
    }

    pub(crate) fn reset(&mut self) {
        let scale = self.scale;
        *self = Self::new();
        self.scale = scale;
    }

    /// Fold size follows the tract: `len` is the relative tract length.
    pub(crate) fn set_size(&mut self, len: f32) {
        self.scale = len * len;
    }

    /// Read the controls for the coming block. `constricted` says a
    /// consonant is being held, which needs breath even without voice.
    pub(crate) fn begin_block(&mut self, g: &Glottis, constricted: bool) {
        let (f0, f1) = g.block_frequency();
        let (t0, t1) = g.block_tenseness();
        self.f0 = f0;
        self.f1 = f1;
        self.tense0 = t0.clamp(0.0, 1.0);
        self.tense1 = t1.clamp(0.0, 1.0);
        self.amp0 = self.amp1;
        self.amp1 = g.intensity;

        let voicing = g.voiced || g.always_voice;
        let breathing = voicing || constricted || g.intensity > 0.01;
        self.lung_target = if breathing { LUNG_PRESSURE } else { 0.0 };
        self.leak_target = self.scale
            * if voicing {
                let slack = 1.0 - self.tense1;
                BREATHY_LEAK * slack * slack
            } else {
                ABDUCTED_AREA
            };
    }

    /// One tick: `(glottal area cm², lung pressure dyn/cm²)`.
    #[inline]
    pub(crate) fn tick(&mut self, lambda: f32, dt: f32) -> (f32, f32) {
        self.lung += (self.lung_target - self.lung) * (dt / LUNG_TIME);
        self.leak += (self.leak_target - self.leak) * (dt / ADDUCTION_TIME);

        let f = self.f0 + (self.f1 - self.f0) * lambda;
        self.phase += f * dt;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        let tense = self.tense0 + (self.tense1 - self.tense0) * lambda;
        let amp = self.amp0 + (self.amp1 - self.amp0) * lambda;

        // Pressed folds stay shut for most of the cycle, slack ones barely
        // close; fully slack folds do not vibrate at all (a whisper).
        let open_quotient = 0.85 - 0.45 * tense;
        let x = self.phase / open_quotient;
        let pulse = if x >= 1.0 {
            0.0
        } else if x < PEAK {
            0.5 * (1.0 - cos(PI * x / PEAK))
        } else {
            cos(0.5 * PI * (x - PEAK) / (1.0 - PEAK))
        };
        let engage = (tense / 0.3).min(1.0);
        let engage = engage * engage * (3.0 - 2.0 * engage);
        let area = self.leak + self.scale * VIBRATION_AREA * amp * engage * pulse;
        (area, self.lung)
    }
}
