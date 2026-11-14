//! [`Tuning`]: the seventy-three numbers that decide how a voice turns sounds into
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
//! * **C — vowels (physical engine):** how long vowels are held, and each
//!   kind against the others. The classic engine keeps every vowel alike.

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
        /// How a voice turns sounds into speech: seventy-three numbers in three
        /// passes (A: how sounds are made, B: how they follow one another,
        /// C: vowels), each with a range a real speaker could plausibly
        /// occupy ([`Tuning::FIELDS`]).
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
    teeth_gain = 0.38, 0.07..=1.0, A;
    /// Efficiency of the jet between the folds (breath noise).
    glottal_efficiency = 0.043, 0.01..=0.15, A;
    /// Frequency above which turbulence has less and less energy, Hz (that
    /// of a jet on the teeth: `teeth_corner`).
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

    // ----- C: vowels (physical engine) ---------------------------------

    /// How much longer every vowel and diphthong is held on the physical
    /// engine than `vowel_length` and `diphthong_length` say (those are the
    /// classic engine's). Its tongue body is heavy and slow
    /// (`body_rate`), and a vowel held for the classic engine's time ends
    /// before the body has reached it.
    vowel_hold = 1.18, 1.0..=1.6, C;

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
    onset_pressure = 0.5, 0.0..=1.0, A;

    // ----- A: the hiss of fricatives (physical engine) -----------------

    /// Corner below which a jet on the teeth (/s ʃ/) makes less and less
    /// noise, Hz at the native size. Raising it takes the low end out of
    /// /s/ (a speaker's has little below 3 kHz), but also blurs /i/, whose
    /// narrow front lets a little of that jet through.
    hiss_floor = 400.0, 200.0..=3000.0, A;
    /// Frequency above which the noise of a jet on the teeth has less and
    /// less energy, Hz at the native size: a speaker's /s/ peaks at 5 to 8
    /// kHz.
    teeth_corner = 5500.0, 3000.0..=10000.0, A;
    /// Lip opening of /ʃ ʒ tʃ dʒ/ (diameter units): rounded, but open.
    postalveolar_lips = 1.2, 0.8..=2.5, A;
    /// How much wider the gap of /s ʃ/ is as it forms (twice that as it
    /// opens again), diameter units.
    frication_widening = 0.15, 0.0..=0.3, A;

    // ----- B: the hiss of fricatives (physical engine) -----------------

    /// Share of /s ʃ/ at each end spent in that wider gap. Off by default:
    /// at 0.25 the hiss rises over some 25 ms as a speaker's does, but the
    /// recognizer heard 2 to 6 points more phoneme errors in sentences and
    /// carrier phrases.
    frication_ramp = 0.0, 0.0..=0.4, B;

    // ----- B: intonation (physical engine) ------------------------------

    /// Rise of the pitch on a stressed vowel, semitones at expressiveness
    /// 1 (a speaker's accents are 2 to 5).
    accent_size = 6.0, 0.0..=16.0, B;
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

    // ----- A: nasals, dentals and stops (physical engine) ---------------

    /// Narrowest the mouth between the velum and a nasal's closure may be,
    /// diameter units: the back of the tongue drops so that the closed
    /// mouth is a side branch of the nose with its antiresonance where a
    /// speaker's is (near 1 kHz for /m/, 1.5 to 2 kHz for /n/), whatever
    /// the vowel's body around it. Below about 1.0 a central or back
    /// vowel's hump pulls /n/ down towards /m/'s, and the notch /n/ leaves
    /// in the murmur grows shallow.
    nasal_cavity = 1.6, 0.6..=2.0, A;

    // ----- B: nasals, dentals and stops (physical engine) ---------------

    /// How long before the end of a nasal followed by an obstruent the
    /// velum starts to shut, s. The velum is slow (some 50 ms to shut), and
    /// a speaker raises it before the mouth opens: shut late, the first
    /// half of the /d/ in "and" was still nasal, built no pressure and was
    /// heard as /ŋ/ or not at all, and the /ð/ in "on the" was nasal.
    velum_lead = 0.03, 0.0..=0.05, B;
    /// Shortest complete closure of a voiceless stop (/p t k/), s: the
    /// mouth and the nose shut, before the release is let go. In a fast
    /// voice, or in a cluster where the articulators are sent on to the
    /// next sound as the closure forms, a closure of a few milliseconds is
    /// not heard as a stop.
    closure_floor = 0.022, 0.0..=0.04, B;
    /// How long the voice goes on once a voiceless stop has shut the
    /// mouth, s, before the folds are let go: a speaker's voicing dies away
    /// over the first 10 to 20 ms of a /p t k/ closure as the mouth fills.
    /// Cut off as the mouth shut, the level fell by 25 to 30 dB in 10 ms at
    /// every voiceless closure, where a speaker's falls by 13.
    voicing_tail = 0.012, 0.0..=0.03, B;

    // ----- A: sibilants (physical engine) -------------------------------
    /// Gap left between the incisors while /s z/ are made, as the jaw
    /// raises the lower ones to the upper (articulation diameter units:
    /// 0.6 is about 0.45 cm² at the native size). The jet from the tongue
    /// tip strikes the teeth through it, and the small cavity it closes off
    /// behind them is what gives /s/ its peak at 5 to 8 kHz. Narrower is
    /// brighter and quieter.
    sibilant_teeth = 0.6, 0.3..=1.2, A;

    // ----- C: lax vowels (physical engine) -------------------------------

    /// Length of a stressed lax vowel /ɪ ɛ ʌ ʊ/ against a tense vowel's
    /// (/i u/) in the same place. A speaker reading LibriSpeech holds his
    /// stressed lax vowels for some 60 % of his tense ones. At 0.62 the
    /// recognizer heard the vowels better (vowel lab held-out −2.7 points)
    /// but the sentences of the tuning material worse (6 points of phoneme
    /// accuracy, Whisper 8.5 of words) than at the round 4 value, 0.905.
    /// The owner chose 0.62 by ear and asked in round 6 for a milder 0.82
    /// for the stressed lax vowels ("sells", "punch"; the unstressed ones
    /// keep `unstressed_lax_length`). At 0.82 the recognizer heard the
    /// stressed /ɪ/ of the LibriSpeech texts as /eɪ/ 14 times in 54 (2 at
    /// 0.62): in-sentence vowel error +9.4 points [5.9, 13.2], with the
    /// sentences' phoneme error and Whisper words unchanged. So it stays at
    /// 0.62, and `profiles/stressed-lax.json` has 0.82 for listening.
    lax_length = 0.62, 0.5..=1.0, C;
    /// Share of a lax vowel held on its own shape before it glides towards
    /// schwa (to its end shape in `vowel_targets`) for the rest.
    lax_nucleus = 0.67, 0.4..=1.0, C;
    /// The same as `lax_length` for an unstressed lax vowel ("it", "-ing"),
    /// which the stress already makes shorter.
    unstressed_lax_length = 0.62, 0.5..=1.0, C;

    // ----- C: vowel lengths and glides by stress (round 6) ---------------

    /// Shortest the held part of a stressed lax vowel may be, seconds (its
    /// nucleus: `lax_nucleus` of its length, before its glide towards
    /// schwa), however fast the speech; `0` for none. The owner asked for
    /// 60 ms with the milder stressed length (`profiles/stressed-lax.json`),
    /// where of the 136 stressed lax vowels of the judge's sentences it
    /// lengthens none at the default rate and 39 at 1.2 times the rate (at
    /// `lax_length` 0.62: 39 already at the default rate). Off by default:
    /// with it and `ae_nucleus` on, the held-out sentences' phoneme error
    /// rose 1.9 points [-0.1, 4.2] and the vowel lab's held-out vowel error
    /// 1.3 [-0.3, 3.0].
    lax_floor = 0.0, 0.0..=0.09, C;
    /// Share of a stressed /æ/ held on its own shape before it glides
    /// towards schwa for the rest ("planks", "background"); `1.0` (the
    /// default) holds it to the end. At 0.55 it ends about a fifth of the
    /// way to schwa (`tests/vowel_glide_gate.rs`; at 0.67, the lax vowels'
    /// share, 13 %), and the recognizer heard more /æ/ as /ɛ/. In
    /// `profiles/stressed-lax.json`.
    ae_nucleus = 1.0, 0.4..=1.0, C;
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
    fn two_passes_of_thirty_one_and_thirty_six() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (31, 36));
    }

    #[test]
    fn vowels_are_held_longer_on_the_physical_engine() {
        let hold = Tuning::FIELDS.iter().find(|f| f.name == "vowel_hold");
        assert!(hold.is_some_and(|f| f.default > 1.0 && f.lo >= 1.0));
        let c = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::C).count();
        assert_eq!(c, 6);
        // Lax vowels are never longer than tense ones.
        for name in ["lax_length", "unstressed_lax_length"] {
            let f = Tuning::FIELDS.iter().find(|f| f.name == name);
            assert!(f.is_some_and(|f| f.default <= 1.0 && f.hi <= 1.0), "{name}");
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
