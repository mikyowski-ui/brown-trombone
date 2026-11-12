//! [`Tuning`]: the sixty-one numbers that decide how a voice turns sounds into
//! speech, in one place.
//!
//! They used to be constants scattered through the speech rules, the
//! articulators, the noise sources and the vocal folds. Gathered here they
//! can be saved with a voice, set by a [`Persona`](crate::Persona), or
//! tuned by a program that listens to the result.
//!
//! [`Tuning::DEFAULT`] holds the values the engines were built and tested
//! with; with it, nothing sounds different from before this struct existed.
//! Every field has a range a real speaker could plausibly occupy
//! ([`Tuning::FIELDS`]); [`Tuning::clamped`] keeps a profile inside it, so a
//! bad file cannot break the engine.
//!
//! The fields fall in two groups, tuned separately:
//!
//! * **A — how sounds are made:** the breath and the folds, the noise of
//!   jets, and how wide consonant gaps are.
//! * **B — how sounds follow one another:** how long each kind of sound
//!   lasts, how early the articulators start for the next one, and how fast
//!   they move.

/// Which tuning pass a field belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// How sounds are made: source, noise, consonant gaps.
    A,
    /// How sounds follow one another: timing and speeds.
    B,
}

/// One field of [`Tuning`]: its name, default and allowed range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    /// The field's name, as in the struct and in profile files.
    pub name: &'static str,
    /// Value in [`Tuning::DEFAULT`].
    pub default: f32,
    /// Smallest allowed value.
    pub lo: f32,
    /// Largest allowed value.
    pub hi: f32,
    /// The tuning pass it belongs to.
    pub pass: Pass,
}

/// Keep `x` in `lo..=hi`; anything not finite becomes the default.
fn within(x: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        default
    }
}

macro_rules! tuning {
    ($( $(#[$doc:meta])* $name:ident = $default:literal, $lo:literal ..= $hi:literal, $pass:ident; )*) => {
        /// How a voice turns sounds into speech (see the [module](self)).
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(default))]
        pub struct Tuning {
            $( $(#[$doc])* pub $name: f32, )*
        }

        impl Tuning {
            /// The values the engines were built and tested with.
            pub const DEFAULT: Self = Self { $( $name: $default, )* };

            /// Every field with its default and allowed range, in order.
            pub const FIELDS: &'static [Field] = &[
                $( Field {
                    name: stringify!($name),
                    default: $default,
                    lo: $lo,
                    hi: $hi,
                    pass: Pass::$pass,
                }, )*
            ];

            /// The same tuning with every field inside its allowed range.
            #[must_use]
            pub fn clamped(mut self) -> Self {
                $( self.$name = within(self.$name, $lo, $hi, $default); )*
                self
            }

            /// A field's value by name.
            #[must_use]
            pub fn get(&self, name: &str) -> Option<f32> {
                match name {
                    $( stringify!($name) => Some(self.$name), )*
                    _ => None,
                }
            }

            /// Set a field by name; `false` if there is no such field.
            pub fn set(&mut self, name: &str, value: f32) -> bool {
                match name {
                    $( stringify!($name) => self.$name = value, )*
                    _ => return false,
                }
                true
            }
        }
    };
}

tuning! {
    // ----- A: the breath and the folds ---------------------------------

    /// Lung pressure while speaking, cmH2O. Louder and brighter with more.
    lung_pressure = 8.0, 4.0..=16.0, A;
    /// Seconds for the folds to come together or part.
    adduction_time = 0.008, 0.003..=0.02, A;
    /// How far past touching the folds are pressed to start the voice, cm.
    attack_depth = 0.04, 0.02..=0.06, A;
    /// How long that press is held once there is pressure behind it, s.
    attack_hold = 0.003, 0.001..=0.01, A;
    /// Rest half-width of the vibrating edges in modal voice, cm.
    modal_gap = 0.012, 0.004..=0.014, A;
    /// Gap left between the cartilages in a fully breathy voice, cm².
    posterior_gap = 0.25, 0.05..=0.5, A;
    /// Glottal opening for voiceless sounds and breathing, cm².
    abducted_area = 0.25, 0.1..=0.5, A;

    // ----- A: noise -----------------------------------------------------

    /// Share of a free jet's dynamic pressure that returns as sound.
    jet_efficiency = 0.057, 0.015..=0.2, A;
    /// The same for a jet that strikes the teeth, relative to a free jet.
    teeth_gain = 0.28, 0.07..=1.0, A;
    /// Efficiency of the jet between the folds (breath noise).
    glottal_efficiency = 0.043, 0.01..=0.15, A;
    /// Frequency above which turbulence has less and less energy, Hz.
    hiss_corner = 7000.0, 2500.0..=10000.0, A;

    // ----- A: gaps ------------------------------------------------------

    /// Added to the opening of every fricative (touch-diameter units).
    fricative_gap = 0.0, -0.1..=0.15, A;
    /// Added to the opening of every approximant (`l r w y`).
    approximant_gap = 0.0, -0.2..=0.3, A;
    /// Velar port fully open, cm² (how strongly nasals couple the nose).
    nasal_port = 0.8, 0.3..=1.6, A;

    // ----- A: the larynx and the walls in consonants (physical engine) --

    /// Seconds for the arytenoids to open or close the glottis behind the
    /// folds (much slower than the folds meeting).
    abduction_time = 0.02, 0.008..=0.05, A;
    /// Glottal opening added behind the folds for a voiceless stop or
    /// fricative and its aspiration, cm² at native size (on top of
    /// `abducted_area`).
    voiceless_spread = 0.12, 0.0..=0.4, A;
    /// Opening left behind the vibrating folds in a voiced fricative, cm²
    /// at native size: air for the hiss without stopping the voice.
    voiced_fricative_spread = 0.02, 0.0..=0.12, A;
    /// Share by which the fold cover slackens in a voiced stop (it lowers
    /// the pressure the folds need to keep going).
    voiced_stop_slack = 0.08, 0.0..=0.25, A;
    /// Rate at which the pharynx and cheeks are actively widened during a
    /// voiced closure, cm³/s; the air the voice sends into the shut mouth
    /// goes there, so the voice lasts through the closure.
    cavity_expansion = 150.0, 0.0..=300.0, A;
    /// Pitch raised at the start of a vowel after a voiceless consonant, Hz
    /// at native size (stiff folds after the glottis was spread).
    onset_kick = 12.0, 0.0..=25.0, A;
    /// Seconds over which that raise decays.
    kick_time = 0.03, 0.01..=0.08, A;

    // ----- B: timing ----------------------------------------------------

    /// Head start the articulators get on each sound, s (physical engine).
    lead = 0.03, 0.0..=0.06, B;
    /// The head start out of a closure, s: short, so stops stay shut. A
    /// little over one articulation block (10.7 ms), so the release falls
    /// before the next sound starts whatever the phase of the blocks.
    release_lead = 0.011, 0.0..=0.03, B;
    /// Length of an unstressed vowel, s.
    vowel_length = 0.085, 0.06..=0.14, B;
    /// Length of a diphthong, s.
    diphthong_length = 0.15, 0.11..=0.22, B;
    /// Length of a stop's closure, s.
    stop_length = 0.06, 0.04..=0.1, B;
    /// Length of a nasal, s.
    nasal_length = 0.065, 0.045..=0.11, B;
    /// Length of a voiceless fricative, s.
    fricative_length = 0.1, 0.07..=0.15, B;
    /// Length of a voiced fricative, s.
    voiced_fricative_length = 0.07, 0.05..=0.12, B;
    /// Length of an approximant, s.
    approximant_length = 0.055, 0.04..=0.1, B;
    /// Length of an `h`, s.
    aspirate_length = 0.06, 0.04..=0.1, B;
    /// Breath after a voiceless stop's burst, before the vowel, s (classic
    /// engine; the physical one holds the glottis open for `vot_*`).
    aspiration = 0.035, 0.015..=0.07, B;
    /// How much longer a stressed vowel is than an unstressed one.
    stress_stretch = 1.5, 1.2..=1.9, B;

    // ----- B: voicing timing (physical engine) -------------------------

    /// How long the glottis stays open after a /p/ is released before a
    /// stressed vowel, s (the voice follows a few milliseconds later).
    /// The physical engine uses these three in place of `aspiration`.
    vot_labial = 0.035, 0.015..=0.08, B;
    /// The same after /t/, s.
    vot_alveolar = 0.045, 0.015..=0.08, B;
    /// The same after /k/, s.
    vot_velar = 0.047, 0.015..=0.09, B;
    /// How long before a voiceless consonant the voice is let go and the
    /// glottis starts to open, s: with the tract, which starts closing
    /// `lead` ahead, so the voice stops as the tract shuts.
    devoicing_lead = 0.012, 0.0..=0.03, B;
    /// Share of that hold before an unstressed vowel.
    unstressed_vot = 0.5, 0.2..=1.0, B;
    /// How much longer the closure of a voiceless stop is than a voiced
    /// one's.
    fortis_closure = 1.25, 1.0..=1.6, B;
    /// Length of a vowel before a voiceless consonant that ends its
    /// syllable, relative to before a sonorant (pre-fortis clipping).
    prefortis_clip = 0.85, 0.6..=1.0, B;
    /// Length of a vowel before a voiced obstruent that ends its syllable,
    /// relative to before a sonorant.
    prelenis_stretch = 1.2, 1.0..=1.5, B;

    // ----- B: speeds ----------------------------------------------------

    /// Natural frequency of the articulators, rad/s, at speed 1 (physical
    /// engine). Higher is nimbler.
    articulation_rate = 40.0, 25.0..=70.0, B;
    /// How much faster articulators close than open.
    closing_ratio = 2.0, 1.0..=3.0, B;
    /// Speed at which a closure lets go, as a share of its full swing.
    release_speed = 0.37, 0.1..=0.8, B;
    /// How far past contact a closing articulator aims (diameter units).
    closure_overshoot = 0.3, 0.1..=0.5, B;

    // ----- B: two speeds of articulator (physical engine) --------------

    /// Natural frequency of the tongue body and the jaw, rad/s at speed 1
    /// (physical engine). They are heavy and slow (a time constant of
    /// about 40 ms in speech), so the vowel shape glides from vowel to
    /// vowel underneath the consonants; `articulation_rate` is that of the
    /// light, quick tongue tip, blade and lips that make the consonants on
    /// top of it.
    body_rate = 11.4, 6.0..=25.0, B;
    /// Natural frequency of the back of the tongue making a closure or a
    /// gap (/k g ŋ/), rad/s at speed 1: between the two.
    dorsum_rate = 20.0, 10.0..=40.0, B;

    // ----- A: voiced sibilants (physical engine) ------------------------

    /// Opening left behind the vibrating folds in a word-final /z ʒ/, cm²
    /// at native size: more than `voiced_fricative_spread`. English lets
    /// them go partly voiceless at the end of a word, and their hiss has to
    /// be as plain as the voice under it (a final /z/ with too little air
    /// is heard as /ð/); /v ð/ are weak, and too much air makes them breathy.
    voiced_sibilant_spread = 0.05, 0.0..=0.12, A;

    // ----- B: voiced sibilants (physical engine) ------------------------

    /// Length of /z ʒ/, s: longer than the weak /v ð/
    /// (`voiced_fricative_length`), as in English; a short /z/ is heard as
    /// /ð/ or /d/.
    voiced_sibilant_length = 0.095, 0.06..=0.14, B;

    // ----- A: the breath over a phrase (physical engine) ---------------

    /// Share of the lung pressure lost between the start and the end of a
    /// phrase: the speaker's breath running down, which lowers the level
    /// and the pitch together.
    phrase_decline = 0.15, 0.0..=0.4, A;
    /// Share of the lung pressure let go over the last vowel of a
    /// statement: the voice dies away instead of being cut off (half of it
    /// before a comma).
    final_drop = 0.5, 0.0..=0.9, A;
    /// Share of the lung pressure a phrase that opens on a vowel or a
    /// sonorant starts from, rising to all of it over `phrase_onset`: about
    /// what the folds need to start, so the voice grows in from its first
    /// cycles without the first sound being lost.
    onset_pressure = 0.4, 0.0..=1.0, A;

    // ----- A: the hiss of fricatives (physical engine) -----------------

    /// Corner below which the jets in the mouth make less and less noise,
    /// Hz at the native size: their eddies are small and fast, and a
    /// speaker's /s/ has little energy below 3 kHz.
    hiss_floor = 1500.0, 400.0..=3000.0, A;
    /// Lip opening of /ʃ ʒ tʃ dʒ/ (diameter units): rounded, but open.
    postalveolar_lips = 1.8, 0.8..=2.5, A;
    /// How much wider the gap of /s ʃ/ is as it forms (twice that as it
    /// opens again), diameter units.
    frication_widening = 0.15, 0.0..=0.3, A;

    // ----- B: the hiss of fricatives (physical engine) -----------------

    /// Share of /s ʃ/ at each end spent in that wider gap.
    frication_ramp = 0.25, 0.0..=0.4, B;

    // ----- B: intonation (physical engine) ------------------------------

    /// Rise of the pitch on a stressed vowel, semitones at expressiveness
    /// 1 (a speaker's accents are 2 to 5).
    accent_size = 10.0, 0.0..=16.0, B;
    /// Seconds over which an accent sinks back after its vowel.
    accent_decay = 0.2, 0.05..=0.5, B;
    /// Fall of the pitch over a phrase, semitones at expressiveness 1.
    declination = 6.0, 0.0..=12.0, B;
    /// Fall across the last vowel of a statement, semitones at
    /// expressiveness 1.
    final_fall = 8.0, 0.0..=16.0, B;

    // ----- B: the breath over a phrase (physical engine) ---------------

    /// Seconds over which the lungs build up their pressure at the start of
    /// a phrase, from nothing: the first sound swells in rather than
    /// switching on.
    phrase_onset = 0.15, 0.04..=0.3, B;
}

impl Default for Tuning {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_lie_inside_their_ranges() {
        for f in Tuning::FIELDS {
            assert!(f.lo < f.hi, "{}", f.name);
            assert!((f.lo..=f.hi).contains(&f.default), "{}", f.name);
        }
        assert_eq!(Tuning::DEFAULT.clamped(), Tuning::DEFAULT);
    }

    #[test]
    fn two_passes_of_twenty_eight_and_thirty_three() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (28, 33));
    }

    #[test]
    fn fields_are_reachable_by_name() {
        let mut t = Tuning::DEFAULT;
        for f in Tuning::FIELDS {
            assert_eq!(t.get(f.name), Some(f.default), "{}", f.name);
            assert!(t.set(f.name, f.hi));
            assert_eq!(t.get(f.name), Some(f.hi));
        }
        assert!(!t.set("no_such_field", 1.0));
        assert_eq!(t.get("no_such_field"), None);
    }

    #[test]
    fn clamping_tames_garbage() {
        let mut t = Tuning::DEFAULT;
        t.lung_pressure = f32::NAN;
        t.lead = 99.0;
        t.modal_gap = -1.0;
        let c = t.clamped();
        assert!((c.lung_pressure - 8.0).abs() < f32::EPSILON);
        assert!((c.lead - 0.06).abs() < f32::EPSILON);
        assert!((c.modal_gap - 0.004).abs() < f32::EPSILON);
    }
}
