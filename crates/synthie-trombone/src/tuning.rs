//! [`Tuning`]: the forty-five numbers that decide how a voice turns sounds into
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
//! The fields fall in three groups, tuned separately:
//!
//! * **A — how sounds are made:** the breath and the folds, the noise of
//!   jets, and how wide consonant gaps are.
//! * **B — how sounds follow one another:** how long each kind of sound
//!   lasts, how early the articulators start for the next one, and how fast
//!   they move.
//! * **C — vowels one by one (physical engine):** how long each vowel lasts
//!   against the others, how far into it the tongue starts to drift, and
//!   how much of a diphthong is its first vowel. The classic engine keeps
//!   every vowel alike.

/// Which tuning pass a field belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// How sounds are made: source, noise, consonant gaps.
    A,
    /// How sounds follow one another: timing and speeds.
    B,
    /// How long each vowel lasts and how it moves (physical engine).
    C,
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
    hiss_corner = 4500.0, 2500.0..=8000.0, A;

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
    /// The head start out of a closure, s: short, so stops stay shut.
    release_lead = 0.008, 0.0..=0.03, B;
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
    vot_labial = 0.04, 0.015..=0.08, B;
    /// The same after /t/, s.
    vot_alveolar = 0.045, 0.015..=0.08, B;
    /// The same after /k/, s.
    vot_velar = 0.05, 0.015..=0.09, B;
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

    // ----- C: vowel lengths ----------------------------------------------
    //
    // English vowels have lengths of their own. In the hVd words of
    // Hillenbrand et al. (1995) the men's lax /ɪ ɛ ʌ ʊ/ last about 0.81 of
    // their mean and the long tense /æ ɑ ɔ ɝ/ about 1.17 (/i u/ about 1).
    // Running speech holds these apart less than words said one by one, so
    // the factors here are half-way between 1 and those ratios (the full
    // ratios also shortened the vowels around stops enough to move their
    // bursts and voice onsets past `tests/physical_gates.rs`). Of the
    // diphthongs, /aɪ aʊ ɔɪ/ are longer than /eɪ oʊ/ (Peterson & Lehiste
    // 1960).

    /// Length of the lax vowels /ɪ ɛ ʌ ʊ/ against `vowel_length` (physical
    /// engine).
    lax_length = 0.905, 0.5..=1.5, C;
    /// Length of the long tense vowels /æ ɑ ɔ ɝ/ against `vowel_length`
    /// (physical engine).
    long_length = 1.085, 0.7..=2.0, C;
    /// Length of /aɪ aʊ ɔɪ/ against `diphthong_length` (physical engine).
    wide_diphthong_length = 1.05, 0.7..=2.0, C;

    // ----- C: movement ----------------------------------------------------

    /// Share of a vowel at whose end its tongue heads for where the vowel
    /// drifts to (its end shape; physical engine). `0` holds every vowel
    /// still.
    vowel_drift = 0.0, 0.0..=0.7, C;
    /// Share of a diphthong spent on the vowel it starts from, before it
    /// glides to its end (physical engine; the classic engine holds 55 %).
    diphthong_nucleus = 0.55, 0.2..=0.8, C;
    /// Speed of the tongue and lips while a diphthong glides or a vowel
    /// drifts, against their other movements (physical engine). Below `1`
    /// the glide takes all of its time, as people's do, instead of
    /// arriving early and holding the end, which sounds like two vowels.
    glide_rate = 0.25, 0.15..=1.0, C;
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
    fn two_passes_of_twenty_one_and_twenty_four() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (21, 24));
    }

    #[test]
    fn vowel_lengths_keep_their_order() {
        let t = Tuning::DEFAULT;
        assert!(t.lax_length < 1.0 && t.long_length > 1.0 && t.wide_diphthong_length > 1.0);
        let c = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::C).count();
        assert_eq!(c, 6);
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
