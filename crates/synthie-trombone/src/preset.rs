//! Voice character: [`VoiceSettings`] and ready-made [`Preset`]s.

use crate::{Engine, Trombone, Tuning};

/// Everything that makes one speaker sound different from another.
///
/// Apply it to a [`Trombone`] with [`apply`](Self::apply); the speech
/// front end takes its intonation from [`pitch`](Self::pitch),
/// [`expressiveness`](Self::expressiveness), [`rate`](Self::rate) and
/// [`nasality`](Self::nasality). [`Talker`](crate::Talker) does both.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct VoiceSettings {
    /// Speaking pitch in Hz (the presets span 62 to 360).
    pub pitch: f32,
    /// Vocal tract length relative to the model's native tract (about
    /// 17 cm); shorter raises every formant and sounds like a smaller
    /// speaker (`0.86` bright, `0.72` child). `0.5‥1.6`.
    pub tract_length: f32,
    /// Glottal tenseness: `0` whisper, `0.4` breathy, `0.65` modal, `0.9`
    /// pressed.
    pub tenseness: f32,
    /// Pitch movement on accents and phrase ends: `0` monotone, `1` lively.
    pub expressiveness: f32,
    /// Speaking rate multiplier.
    pub rate: f32,
    /// Vibrato depth, relative pitch.
    pub vibrato: f32,
    /// Vibrato rate in Hz.
    pub vibrato_rate: f32,
    /// Random pitch jitter: `0` robotic, `1` natural, `2+` shaky.
    pub jitter: f32,
    /// Constant velum opening during vowels: a nasal, whiny voice. `0‥1`.
    pub nasality: f32,
    /// The sound model: Pink Trombone's own, or the physical one.
    pub engine: Engine,
    /// How the voice turns sounds into speech (see [`Tuning`]).
    pub tuning: Tuning,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Preset::Deep.settings()
    }
}

/// A voice described the way a person would describe it, instead of by
/// its parameters. Applied on top of any voice with
/// [`VoiceSettings::with_persona`]; [`Persona::NEUTRAL`] changes nothing.
///
/// Each dial moves several parameters together, as they move together in
/// people:
///
/// * **size:** body size relative to the base voice. A smaller speaker has
///   a shorter tract (every resonance higher) and a higher pitch.
/// * **effort:** from murmuring (`0`) to calling out (`1`): lung pressure,
///   firmer folds, a slightly higher and livelier pitch, quicker
///   articulators.
/// * **breathiness:** from pressed (`0`) to breathy (`1`): how much air
///   escapes past the folds.
/// * **clarity:** from casual (`0`) to careful (`1`) speech: articulators
///   start earlier and move faster, sounds are held longer, stressed
///   vowels stand out more, closures are firmer.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Persona {
    /// Body size relative to the base voice, `0.6‥1.4` (`1` unchanged).
    pub size: f32,
    /// Vocal effort, `0‥1` (`0.5` unchanged).
    pub effort: f32,
    /// Breathiness, `0‥1` (`0.35` unchanged).
    pub breathiness: f32,
    /// Clarity of articulation, `0‥1` (`0.5` unchanged).
    pub clarity: f32,
}

impl Persona {
    /// The persona that leaves a voice as it is.
    pub const NEUTRAL: Self = Self {
        size: 1.0,
        effort: 0.5,
        breathiness: 0.35,
        clarity: 0.5,
    };
}

impl Default for Persona {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

impl VoiceSettings {
    /// The same voice as a different person would produce it (see
    /// [`Persona`]). Every factor is written as `1 + k·(dial − neutral)`,
    /// so the neutral persona returns the voice bit for bit.
    #[must_use]
    pub fn with_persona(mut self, persona: &Persona) -> Self {
        let finite = |x: f32, neutral: f32| if x.is_finite() { x } else { neutral };
        let size = finite(persona.size, 1.0).clamp(0.6, 1.4);
        let effort = finite(persona.effort, 0.5).clamp(0.0, 1.0) - 0.5;
        let breath = finite(persona.breathiness, 0.35).clamp(0.0, 1.0) - 0.35;
        let clarity = finite(persona.clarity, 0.5).clamp(0.0, 1.0) - 0.5;

        // Size: the tract scales with the body; pitch falls faster than
        // length grows (folds lengthen and thicken), about as the built-in
        // voices do between child and giant.
        self.tract_length = (self.tract_length * size).clamp(0.5, 1.6);
        self.pitch *= crate::math::powf(size, -2.5);

        // Effort: lung pressure doubles and halves across the dial's range
        // and a bit more; the folds press together, pitch rises and moves
        // more, the articulators hurry.
        let t = &mut self.tuning;
        t.anatomy.lung_pressure *= crate::math::powf(2.0, 1.6 * effort);
        self.pitch *= 1.0 + 0.24 * effort;
        self.expressiveness *= 1.0 + 0.6 * effort;
        t.articulation.tongue_tip.stiffness *= 1.0 + 0.2 * effort;

        // Breathiness and effort both act on how firmly the folds close.
        // A whisper has no voice to press or loosen and stays a whisper.
        let firmer = 0.3 * effort - 0.6 * breath;
        if firmer != 0.0 && self.tenseness >= 0.12 {
            self.tenseness = (self.tenseness + firmer).clamp(0.15, 0.95);
        }

        // Clarity: earlier, faster, longer, firmer.
        t.articulation.lead *= 1.0 + 0.8 * clarity;
        t.articulation.tongue_tip.stiffness *= 1.0 + 0.4 * clarity;
        let consonants = 1.0 + 0.3 * clarity;
        t.prosody.stop_length *= consonants;
        t.prosody.nasal_length *= consonants;
        t.prosody.fricative_length *= consonants;
        t.prosody.voiced_fricative_length *= consonants;
        t.prosody.approximant_length *= consonants;
        t.prosody.aspirate_length *= consonants;
        let vowels = 1.0 + 0.2 * clarity;
        t.prosody.vowel_length *= vowels;
        t.prosody.diphthong_length *= vowels;
        t.prosody.stress_stretch *= 1.0 + 0.2 * clarity;
        t.articulation.closure_overshoot *= 1.0 + 0.4 * clarity;
        t.articulation.fricative_gap -= 0.06 * clarity;

        self.tuning = self.tuning.clamped();
        self
    }
}

impl VoiceSettings {
    /// The same voice in another body (see
    /// [`MorphTarget`](crate::MorphTarget)): the tract's length scaled, and
    /// the larynx's rest, the folds' size and the lower pharynx's width
    /// laid into the tuning. The pitch, the timing and the effort stay the
    /// voice's; the neutral morph returns it bit for bit. A voice that is
    /// already sounding glides instead:
    /// [`Trombone::set_morph`](crate::Trombone::set_morph).
    #[must_use]
    pub fn with_morph(mut self, morph: &crate::MorphTarget) -> Self {
        let morph = morph.clamped();
        if morph.is_neutral() {
            return self;
        }
        self.tract_length = (self.tract_length * morph.tract_scale).clamp(0.5, 1.6);
        self.tuning = morph.over(&self.tuning).clamped();
        self
    }

    /// Configure `voice` with these settings (voice quality only; pitch is
    /// set per segment while speaking).
    pub fn apply(&self, voice: &mut Trombone) {
        voice.set_engine(self.engine);
        voice.set_tuning(&self.tuning);
        voice.set_tract_length(self.tract_length);
        voice.set_tenseness(self.tenseness);
        voice.set_vibrato(self.vibrato, self.vibrato_rate);
        voice.set_jitter(self.jitter);
        voice.set_frequency(self.pitch);
    }

    /// Intonation settings for the speech compiler.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn prosody(&self) -> crate::speech::Prosody {
        crate::speech::Prosody {
            pitch: self.pitch,
            expressiveness: self.expressiveness,
            rate: self.rate,
            tenseness: self.tenseness,
            nasality: self.nasality,
            tuning: self.tuning,
            engine: self.engine,
        }
    }

    /// The same voice at a different pitch, keeping everything else.
    #[must_use]
    pub fn with_pitch(mut self, hz: f32) -> Self {
        self.pitch = hz;
        self
    }

    /// The same voice on another sound model.
    #[must_use]
    pub fn with_engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// The same voice speaking faster (`> 1`) or slower.
    #[must_use]
    pub fn with_rate(mut self, rate: f32) -> Self {
        self.rate = rate;
        self
    }

    /// The same voice with the living voice's breath and creak (wave 3,
    /// physical engine) set: breaths drawn `intake` cmH2O deep in the
    /// pauses between its sentences, and before the utterance too where
    /// [`ProsodyConfig::breath_first`](crate::ProsodyConfig::breath_first)
    /// is on
    /// ([`ProsodyConfig::breath_intake`](crate::ProsodyConfig::breath_intake);
    /// `0` switches the breaths off), and statements ending `fry` (`0‥1`) of
    /// the way into creak
    /// ([`ProsodyConfig::final_fry`](crate::ProsodyConfig::final_fry); `0`
    /// ends them in modal voice). `with_breath(0.0, 0.0)` switches both off,
    /// `with_breath(5.0, 1.0)` is the depth and share
    /// [`Tuning::with_living_voice`](crate::Tuning::with_living_voice) sets.
    /// Text with bracketed phonemes and songs never take a breath. For a
    /// [`Talker`](crate::Talker) and the stream's text
    /// ([`VoiceStream::set_prosody`](crate::stream::VoiceStream::set_prosody)
    /// with [`prosody`](Self::prosody)).
    #[must_use]
    pub fn with_breath(mut self, intake: f32, fry: f32) -> Self {
        self.tuning.prosody.breath_intake = intake.clamp(0.0, 6.0);
        self.tuning.prosody.final_fry = fry.clamp(0.0, 1.0);
        self
    }
}

/// Ready-made voices. Start from one and tweak the [`VoiceSettings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Preset {
    /// Low and full: the model's native tract at about 105 Hz.
    Deep,
    /// Higher pitch, shorter tract, livelier intonation.
    Bright,
    /// Child: high pitch, short tract.
    Child,
    /// Old: breathy, slow, with a tremor.
    Elder,
    /// Huge, slow and deep: a long tract and a low pitch.
    Giant,
    /// Small, squeaky, fast and excitable: cartoon sidekicks and gremlins.
    Critter,
    /// Monotone, steady, pressed.
    Robot,
    /// No voicing, only breath.
    Whisper,
    /// A computer singing: bright, pressed, perfectly steady pitch, no
    /// vibrato. Made for [`sing`](crate::sing) with hard pitch steps.
    Android,
}

impl Preset {
    /// Every preset.
    pub const ALL: [Preset; 9] = [
        Preset::Deep,
        Preset::Bright,
        Preset::Child,
        Preset::Elder,
        Preset::Giant,
        Preset::Critter,
        Preset::Robot,
        Preset::Whisper,
        Preset::Android,
    ];

    /// Lowercase name, e.g. `"critter"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Preset::Deep => "deep",
            Preset::Bright => "bright",
            Preset::Child => "child",
            Preset::Elder => "elder",
            Preset::Giant => "giant",
            Preset::Critter => "critter",
            Preset::Robot => "robot",
            Preset::Whisper => "whisper",
            Preset::Android => "android",
        }
    }

    /// Look a preset up by name (case-insensitive).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(name))
    }

    /// The preset's settings.
    #[must_use]
    #[allow(clippy::too_many_lines)] // a table
    pub const fn settings(self) -> VoiceSettings {
        let base = VoiceSettings {
            pitch: 105.0,
            tract_length: 1.0,
            tenseness: 0.65,
            expressiveness: 0.5,
            rate: 1.0,
            vibrato: 0.005,
            vibrato_rate: 5.5,
            jitter: 1.0,
            nasality: 0.0,
            engine: Engine::Classic,
            tuning: Tuning::DEFAULT,
        };
        match self {
            Preset::Deep => base,
            Preset::Bright => VoiceSettings {
                pitch: 210.0,
                tract_length: 0.86,
                tenseness: 0.6,
                expressiveness: 0.7,
                rate: 1.05,
                vibrato: 0.006,
                vibrato_rate: 5.8,
                ..base
            },
            Preset::Child => VoiceSettings {
                pitch: 290.0,
                tract_length: 0.72,
                tenseness: 0.6,
                expressiveness: 0.85,
                rate: 1.1,
                vibrato_rate: 6.0,
                jitter: 1.2,
                nasality: 0.05,
                ..base
            },
            Preset::Elder => VoiceSettings {
                pitch: 118.0,
                tract_length: 1.02,
                tenseness: 0.42,
                expressiveness: 0.45,
                rate: 0.8,
                vibrato: 0.03,
                vibrato_rate: 5.0,
                jitter: 2.5,
                nasality: 0.1,
                ..base
            },
            Preset::Giant => VoiceSettings {
                pitch: 62.0,
                tract_length: 1.4,
                tenseness: 0.75,
                expressiveness: 0.4,
                rate: 0.75,
                vibrato: 0.004,
                vibrato_rate: 4.0,
                jitter: 0.8,
                ..base
            },
            Preset::Critter => VoiceSettings {
                pitch: 360.0,
                tract_length: 0.6,
                tenseness: 0.75,
                expressiveness: 1.4,
                rate: 1.4,
                vibrato: 0.012,
                vibrato_rate: 7.5,
                jitter: 1.5,
                nasality: 0.15,
                ..base
            },
            Preset::Robot => VoiceSettings {
                pitch: 95.0,
                tract_length: 0.95,
                tenseness: 0.85,
                expressiveness: 0.0,
                rate: 0.95,
                vibrato: 0.0,
                vibrato_rate: 0.0,
                jitter: 0.0,
                ..base
            },
            Preset::Android => VoiceSettings {
                pitch: 220.0,
                tract_length: 0.84,
                tenseness: 0.8,
                expressiveness: 0.2,
                rate: 1.0,
                vibrato: 0.0,
                vibrato_rate: 0.0,
                jitter: 0.0,
                nasality: 0.06,
                ..base
            },
            Preset::Whisper => VoiceSettings {
                pitch: 140.0,
                tract_length: 0.9,
                tenseness: 0.0,
                expressiveness: 0.3,
                rate: 0.9,
                vibrato: 0.0,
                ..base
            },
        }
    }
}

impl From<Preset> for VoiceSettings {
    fn from(p: Preset) -> Self {
        p.settings()
    }
}

impl core::fmt::Display for Preset {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_neutral_persona_changes_nothing() {
        for preset in Preset::ALL {
            let voice = preset.settings();
            assert_eq!(voice.with_persona(&Persona::NEUTRAL), voice, "{preset}");
        }
    }

    #[test]
    fn each_dial_moves_what_it_says() {
        let base = Preset::Deep.settings();
        let with = |p: Persona| base.with_persona(&p);
        let n = Persona::NEUTRAL;

        let small = with(Persona { size: 0.7, ..n });
        assert!(small.tract_length < base.tract_length && small.pitch > 1.5 * base.pitch);
        let large = with(Persona { size: 1.3, ..n });
        assert!(large.tract_length > base.tract_length && large.pitch < base.pitch);

        let loud = with(Persona { effort: 1.0, ..n });
        let soft = with(Persona { effort: 0.0, ..n });
        assert!(
            loud.tuning.anatomy.lung_pressure > 13.0 && soft.tuning.anatomy.lung_pressure < 5.0
        );
        assert!(loud.tenseness > base.tenseness && soft.tenseness < base.tenseness);
        assert!(loud.pitch > soft.pitch);

        let breathy = with(Persona {
            breathiness: 1.0,
            ..n
        });
        let pressed = with(Persona {
            breathiness: 0.0,
            ..n
        });
        assert!(breathy.tenseness < base.tenseness && pressed.tenseness > base.tenseness);

        let careful = with(Persona { clarity: 1.0, ..n });
        let casual = with(Persona { clarity: 0.0, ..n });
        assert!(careful.tuning.articulation.lead > casual.tuning.articulation.lead);
        assert!(
            careful.tuning.articulation.tongue_tip.stiffness
                > casual.tuning.articulation.tongue_tip.stiffness
        );
        assert!(careful.tuning.prosody.stop_length > casual.tuning.prosody.stop_length);
        assert!(careful.tuning.prosody.vowel_length > casual.tuning.prosody.vowel_length);
    }

    #[test]
    fn extreme_dials_stay_inside_every_range() {
        for persona in [
            Persona {
                size: 0.0,
                effort: 9.0,
                breathiness: -3.0,
                clarity: 7.0,
            },
            Persona {
                size: f32::NAN,
                effort: f32::INFINITY,
                breathiness: 2.0,
                clarity: -1.0,
            },
        ] {
            for preset in Preset::ALL {
                let voice = preset.settings().with_persona(&persona);
                assert_eq!(voice.tuning, voice.tuning.clamped(), "{preset}");
                assert!(voice.pitch.is_finite() && voice.pitch > 0.0);
                assert!((0.5..=1.6).contains(&voice.tract_length));
                assert!((0.0..=0.95).contains(&voice.tenseness));
            }
        }
    }
}
