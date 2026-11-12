//! [`Tuning`]: the thirty numbers that decide how a voice turns sounds into
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
    /// Breath after a voiceless stop's burst, before the vowel, s.
    aspiration = 0.035, 0.015..=0.07, B;
    /// How much longer a stressed vowel is than an unstressed one.
    stress_stretch = 1.5, 1.2..=1.9, B;

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

    // ----- C: each vowel's length, against `vowel_length` -----------------
    //
    // English vowels have lengths of their own. These are the men's mean
    // durations of Hillenbrand et al. (1995) over their mean (233 ms for
    // the ten monophthongs, 266 ms for /eɪ oʊ/): lax /ɪ ɛ ʌ ʊ/ about 0.8,
    // tense /æ ɑ ɔ ɝ/ 1.1 to 1.2. They did not record schwa, /aɪ aʊ ɔɪ/:
    // schwa is made short (0.7), and those three a little longer than /eɪ
    // oʊ/, as Peterson & Lehiste (1960) found them.

    /// Length of /i/ ("beat") against `vowel_length` (physical engine).
    iy_length = 1.04, 0.5..=2.0, C;
    /// Length of /ɪ/ ("bit") against `vowel_length` (physical engine).
    ih_length = 0.82, 0.5..=2.0, C;
    /// Length of /ɛ/ ("bet") against `vowel_length` (physical engine).
    eh_length = 0.81, 0.5..=2.0, C;
    /// Length of /æ/ ("bat") against `vowel_length` (physical engine).
    ae_length = 1.19, 0.5..=2.0, C;
    /// Length of /ʌ/ ("but") against `vowel_length` (physical engine).
    ah_length = 0.81, 0.5..=2.0, C;
    /// Length of /ɑ/ ("father") against `vowel_length` (physical engine).
    aa_length = 1.14, 0.5..=2.0, C;
    /// Length of /ɔ/ ("thought") against `vowel_length` (physical engine).
    ao_length = 1.21, 0.5..=2.0, C;
    /// Length of /ʊ/ ("book") against `vowel_length` (physical engine).
    uh_length = 0.82, 0.5..=2.0, C;
    /// Length of /u/ ("boot") against `vowel_length` (physical engine).
    uw_length = 1.02, 0.5..=2.0, C;
    /// Length of /ɝ/ ("bird") against `vowel_length` (physical engine).
    er_length = 1.13, 0.5..=2.0, C;
    /// Length of schwa ("about") against `vowel_length` (physical engine).
    ax_length = 0.7, 0.5..=2.0, C;

    // ----- C: each diphthong's length, against `diphthong_length` ---------

    /// Length of /eɪ/ ("bait") against `diphthong_length` (physical engine).
    ey_length = 0.95, 0.5..=2.0, C;
    /// Length of /aɪ/ ("bite") against `diphthong_length` (physical engine).
    ay_length = 1.05, 0.5..=2.0, C;
    /// Length of /oʊ/ ("boat") against `diphthong_length` (physical engine).
    ow_length = 0.95, 0.5..=2.0, C;
    /// Length of /aʊ/ ("bout") against `diphthong_length` (physical engine).
    aw_length = 1.1, 0.5..=2.0, C;
    /// Length of /ɔɪ/ ("boy") against `diphthong_length` (physical engine).
    oy_length = 1.1, 0.5..=2.0, C;

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
    fn two_passes_of_fourteen_and_sixteen() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (14, 16));
    }

    #[test]
    fn every_vowel_has_a_length_in_pass_c() {
        for p in crate::Phoneme::ALL {
            if matches!(p.manner(), crate::Manner::Vowel | crate::Manner::Diphthong) {
                let f = Tuning::FIELDS
                    .iter()
                    .find(|f| {
                        f.name
                            .strip_suffix("_length")
                            .is_some_and(|v| v.eq_ignore_ascii_case(p.symbol()))
                    })
                    .unwrap_or_else(|| panic!("{p}"));
                assert_eq!(f.pass, Pass::C, "{p}");
            }
        }
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
