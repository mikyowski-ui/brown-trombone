//! [`Tuning`]: the numbers that decide how a voice turns sounds into
//! speech, in one place, in three parts by what each number physically is.
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
//! A [`Tuning`] is three parts, each a plain struct of numbers:
//!
//! * [`AnatomyConfig`] (`tuning.anatomy`), **the body:** its tissues and its
//!   sources. The lungs' pressure, the folds' constants, the voice's source
//!   tilt and breath noise, the jets and their hiss, the nose's port, the
//!   walls that yield. The anatomical tract (MRI-derived shape modes, a
//!   larynx and lips that change length, an epilarynx) adds to it, and the
//!   living voice (breaths, fry, non-verbal sounds) to its source.
//! * [`ArticulationConfig`] (`tuning.articulation`), **how sounds are made
//!   and joined:** gaps and places, the glottis's openings in consonants,
//!   the lengths of closures and releases, the articulators' rates and
//!   leads, how far vowels glide into their neighbours, and the limits on
//!   following a speaker's schedule. Its [`TierDynamics`] for the six tiers
//!   of the gesture score are where the articulators' springs live (see
//!   [`ArticulationConfig::tongue_body`]).
//! * [`ProsodyConfig`] (`tuning.prosody`), **the rhythm and melody of a
//!   phrase:** the rules' lengths of sounds, lengths by stress and position,
//!   rhythm, pauses, declination and accents, the breath over a phrase and
//!   its decline, and tenseness and level by syllable. The living voice's
//!   breaths add to it.
//!
//! Every field also belongs to one of three tuning passes, as before the
//! split ([`Pass`]); the passes cut across the parts.
//!
//! A field has one name, unique across the three parts ([`Field::name`]);
//! profiles use it either flat, `{"tuning": {"voice_tilt": 14}}`, or within
//! its part, `{"tuning": {"anatomy": {"voice_tilt": 14}}}` (see
//! [`VoiceSettings::with_profile`](crate::VoiceSettings::with_profile)).
//! Six fields live under another name in the struct than in profiles,
//! as the springs of their tiers: `articulation_rate` is
//! `articulation.tongue_tip.stiffness`, `lips_rate`
//! `articulation.lips_and_jaw.stiffness`, `body_rate`
//! `articulation.tongue_body.stiffness`, `body_damping`
//! `articulation.tongue_body.damping`, `larynx_rate`
//! `articulation.larynx_and_pitch.stiffness` and `larynx_damping`
//! `articulation.larynx_and_pitch.damping`.

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

/// Which of the three parts of a [`Tuning`] a field lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// [`AnatomyConfig`]: the body, its tissues and its sources.
    Anatomy,
    /// [`ArticulationConfig`]: how sounds are made and joined.
    Articulation,
    /// [`ProsodyConfig`]: lengths, rhythm, pitch and breath over a phrase.
    Prosody,
}

impl Part {
    /// The three, in the order of [`Tuning::FIELDS`].
    pub const ALL: [Part; 3] = [Part::Anatomy, Part::Articulation, Part::Prosody];

    /// The part's name, as the field of [`Tuning`] and the key of a
    /// profile's nested form: `anatomy`, `articulation`, `prosody`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Part::Anatomy => "anatomy",
            Part::Articulation => "articulation",
            Part::Prosody => "prosody",
        }
    }

    /// The part of that name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }
}

/// One field of [`Tuning`]: its name, default and allowed range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    /// The field's name in profile files, unique across the three parts;
    /// the struct's own name but for the three springs of
    /// [`ArticulationConfig`]'s tiers (see [`Tuning`]).
    pub name: &'static str,
    /// Value in [`Tuning::DEFAULT`].
    pub default: f32,
    /// Smallest allowed value.
    pub lo: f32,
    /// Largest allowed value.
    pub hi: f32,
    /// The tuning pass it belongs to.
    pub pass: Pass,
    /// The part of [`Tuning`] it lives in.
    pub part: Part,
}

/// How one tier of articulators moves towards its targets: a second-order
/// spring and its weight against the other gestures on the tier.
///
/// The six tiers of [`ArticulationConfig`] (`glottal_adduction`,
/// `larynx_and_pitch`, `velum`, `tongue_body`, `tongue_tip`,
/// `lips_and_jaw`) each have one. Today's engine reads six of their
/// numbers, which are [`Tuning::FIELDS`] under names of their own
/// (`body_rate`, `body_damping`, `articulation_rate`, `lips_rate`,
/// `larynx_rate`, `larynx_damping`); the others are
/// filled with what today's articulators do where that is a spring, and are
/// kept for the gesture engine: they are not in [`Tuning::FIELDS`], so not
/// in profiles, until something reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierDynamics {
    /// Natural frequency (ω\_n) of the tier's spring, rad/s at speed 1 (the
    /// voice's articulation speed scales it).
    pub stiffness: f32,
    /// Damping ratio ζ: `1` critically damped, below it a little overshoot.
    pub damping: f32,
    /// Weight of the tier's gesture where gestures on the tier overlap
    /// (`1` for every tier: today nothing blends).
    pub dominance: f32,
}

impl TierDynamics {
    /// A tier's spring and dominance.
    #[must_use]
    pub const fn new(stiffness: f32, damping: f32, dominance: f32) -> Self {
        Self {
            stiffness,
            damping,
            dominance,
        }
    }
}

/// Keep `x` in `lo..=hi`; anything not finite becomes the default.
fn within(x: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        default
    }
}

/// Builds the three parts and [`Tuning`] from one table. Each part lists
/// its plain fields (name, default, range, pass), then its tiers (fields of
/// type [`TierDynamics`] with their defaults), then `aliases`: numbers
/// inside a tier that are fields of [`Tuning::FIELDS`] under a name of
/// their own (range and pass; the default is the tier's).
macro_rules! tuning {
    ($(
        $(#[$sdoc:meta])*
        $part:ident: $Struct:ident, $Part:ident {
            $( $(#[$doc:meta])* $name:ident = $default:literal, $lo:literal ..= $hi:literal, $pass:ident; )*
        } tiers {
            $( $(#[$tdoc:meta])* $tier:ident = $tdefault:expr; )*
        } aliases {
            $( $(#[$adoc:meta])* $key:ident => $atier:ident . $aprop:ident, $alo:literal ..= $ahi:literal, $apass:ident; )*
        }
    )*) => {
        $(
            $(#[$sdoc])*
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct $Struct {
                $( $(#[$doc])* pub $name: f32, )*
                $( $(#[$tdoc])* pub $tier: TierDynamics, )*
            }

            impl $Struct {
                /// The values the engines were built and tested with.
                pub const DEFAULT: Self = Self {
                    $( $name: $default, )*
                    $( $tier: $tdefault, )*
                };

                $(
                    $(#[$adoc])*
                    #[doc = ""]
                    #[doc = concat!(
                        "The field `", stringify!($key), "` of profiles and [`Tuning::FIELDS`]: `",
                        stringify!($atier), ".", stringify!($aprop), "`."
                    )]
                    #[must_use]
                    pub const fn $key(&self) -> f32 {
                        self.$atier.$aprop
                    }
                )*
            }

            impl Default for $Struct {
                fn default() -> Self {
                    Self::DEFAULT
                }
            }
        )*

        /// How a voice turns sounds into speech, in three parts: the body
        /// ([`AnatomyConfig`]), how sounds are made and joined
        /// ([`ArticulationConfig`]) and the rhythm and melody of a phrase
        /// ([`ProsodyConfig`]); each number with a range a real speaker could
        /// plausibly occupy ([`Tuning::FIELDS`]).
        ///
        /// A field has one name, unique across the three parts
        /// ([`Field::name`]); profiles use it either flat,
        /// `{"tuning": {"voice_tilt": 14}}`, or within its part,
        /// `{"tuning": {"anatomy": {"voice_tilt": 14}}}` (see
        /// [`VoiceSettings::with_profile`](crate::VoiceSettings::with_profile)).
        /// Six fields live under another name in the struct than in
        /// profiles, as the springs of their tiers: `articulation_rate` is
        /// `articulation.tongue_tip.stiffness`, `lips_rate`
        /// `articulation.lips_and_jaw.stiffness`, `body_rate`
        /// `articulation.tongue_body.stiffness`, `body_damping`
        /// `articulation.tongue_body.damping`, `larynx_rate`
        /// `articulation.larynx_and_pitch.stiffness` and `larynx_damping`
        /// `articulation.larynx_and_pitch.damping`.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Tuning {
            $(
                #[doc = concat!("The [`", stringify!($Struct), "`] part.")]
                pub $part: $Struct,
            )*
        }

        impl Tuning {
            /// The values the engines were built and tested with.
            pub const DEFAULT: Self = Self { $( $part: $Struct::DEFAULT, )* };

            /// Every field with its default, allowed range, pass and part:
            /// the anatomy's, then the articulation's, then the prosody's.
            pub const FIELDS: &'static [Field] = &[
                $(
                    $( Field {
                        name: stringify!($name),
                        default: $default,
                        lo: $lo,
                        hi: $hi,
                        pass: Pass::$pass,
                        part: Part::$Part,
                    }, )*
                    $( Field {
                        name: stringify!($key),
                        default: $Struct::DEFAULT.$atier.$aprop,
                        lo: $alo,
                        hi: $ahi,
                        pass: Pass::$apass,
                        part: Part::$Part,
                    }, )*
                )*
            ];

            /// The same tuning with every field inside its allowed range.
            #[must_use]
            pub fn clamped(mut self) -> Self {
                $(
                    $( self.$part.$name = within(self.$part.$name, $lo, $hi, $default); )*
                    $(
                        self.$part.$atier.$aprop = within(
                            self.$part.$atier.$aprop,
                            $alo,
                            $ahi,
                            $Struct::DEFAULT.$atier.$aprop,
                        );
                    )*
                )*
                self
            }

            /// A field's value by its name (as in profiles).
            #[must_use]
            pub fn get(&self, name: &str) -> Option<f32> {
                match name {
                    $(
                        $( stringify!($name) => Some(self.$part.$name), )*
                        $( stringify!($key) => Some(self.$part.$atier.$aprop), )*
                    )*
                    _ => None,
                }
            }

            /// Set a field by its name (as in profiles); `false` if there is
            /// no such field.
            pub fn set(&mut self, name: &str, value: f32) -> bool {
                match name {
                    $(
                        $( stringify!($name) => self.$part.$name = value, )*
                        $( stringify!($key) => self.$part.$atier.$aprop = value, )*
                    )*
                    _ => return false,
                }
                true
            }
        }
    };
}

tuning! {
    /// The body: its tissues and its sources (see [`Tuning`]). The
    /// lungs, the folds, the voice's source tilt and breath noise, the jets
    /// and their hiss, the nose's port, the walls and the tube above the
    /// folds.
    anatomy: AnatomyConfig, Anatomy {
        // ----- the breath ------------------------------------------------------------

        /// Lung pressure while speaking, cmH2O. Louder and brighter with more.
        lung_pressure = 8.0, 4.0..=16.0, A;

        // ----- the folds -------------------------------------------------------------

        /// How far past touching the folds are pressed to start the voice, cm.
        attack_depth = 0.04, 0.02..=0.06, A;
        /// How long that press is held once there is pressure behind it, s.
        attack_hold = 0.003, 0.001..=0.01, A;
        /// Rest half-width of the vibrating edges in modal voice, cm.
        modal_gap = 0.012, 0.004..=0.014, A;
        /// Gap left between the cartilages in a fully breathy voice, cm².
        posterior_gap = 0.25, 0.05..=0.5, A;
        /// Share by which the fold cover slackens in a voiced stop (it lowers
        /// the pressure the folds need to keep going).
        voiced_stop_slack = 0.08, 0.0..=0.25, A;
        /// Cycle-to-cycle jitter of the folds (physical engine): the spread
        /// of each cycle's tension about its course, per cent, at the voice's
        /// natural jitter (`VoiceSettings::jitter` 1; it grows with it, up to
        /// twice). RAP in the LibriSpeech texts 0.23 %, the reader 0.25 %. The
        /// pitch's slow wander does not change the length of one cycle against
        /// the next.
        cycle_jitter = 1.0, 0.0..=2.0, A;
        /// Depth of a random movement of the pitch at 6 to 10 Hz, as a share
        /// of it at the voice's natural jitter (physical engine): the movement
        /// within a vowel that the slow wander of 2 to 4 Hz, a straight line
        /// across most vowels, does not give.
        pitch_flutter = 0.02, 0.0..=0.06, A;

        // ----- the voice's source tilt and softness ----------------------------------

        /// How much darker the voice is at the top than at the bottom, dB
        /// (physical engine): a first-order shelf on the voice as it leaves,
        /// falling from `tilt_corner` (where it is unity) and levelling off
        /// this far down, on while the folds are asked to sing (see
        /// `phys::tilt`). Every noise source in the tract takes the inverse on
        /// the way in, so frication, the hiss of a burst and the breath are
        /// heard as without it. It stands for the glottal return phase these
        /// folds lack: their flow derivative has 2-4 kHz some 20 dB under 0-1
        /// kHz, a modal Liljencrants-Fant source 33 to 37. `0` switches it off.
        voice_tilt = 18.0, 0.0..=30.0, A;
        /// Corner of that shelf at native size, Hz (it scales with the
        /// inverse of the voice's size, as the glottal pulse does).
        tilt_corner = 500.0, 200.0..=2000.0, A;
        /// A second section of the voice's source tilt, dB (physical engine):
        /// from `tilt_top_corner` on the voice falls this much further, or for
        /// a negative value rises back as far, unity at 0 Hz; the noise
        /// sources take its inverse too. `0` leaves the tilt one shelf. Off:
        /// a gentler shelf with a steeper top (`voice_tilt` 10, this 8 dB at
        /// 2.5 kHz) brightened the accented vowels' 1-2 and 2-4 kHz bands by
        /// 0.9 and 1.0 dB (2.4 dB brighter than the LibriSpeech reader's in
        /// 2-4 kHz), left 4-7.5 kHz where it was, and gave the same phoneme
        /// error and Whisper words within a point (25.9 % and 68.7 % over the
        /// 30 held-out sentences, against 25.1 % and 68.3 %).
        tilt_top = 0.0, -12.0..=18.0, A;
        /// Corner of that second section at native size, Hz (it scales with
        /// the inverse of the voice's size, as `tilt_corner` does).
        tilt_top_corner = 3000.0, 1000.0..=6000.0, A;
        /// Corner of the one-pole low-pass on the voice at full softness, Hz
        /// (physical engine; see `Trombone::set_fold_softness`). At softness
        /// `s` the corner is this over `s`. With the default softnesses the
        /// nasals, liquids and unaccented vowels of the LibriSpeech texts come
        /// out 6.1, 4.0 and 0.1 dB darker in 2-4 kHz against 0-1 kHz than the
        /// accented vowels (the reader: 5.6, 5.1 and 0.6; at 2500 Hz 6.7, 4.6
        /// and 0.4).
        soft_corner = 3000.0, 1000.0..=8000.0, A;

        // ----- breath noise ----------------------------------------------------------

        /// Efficiency of the jet between the folds (breath noise).
        glottal_efficiency = 0.043, 0.01..=0.15, A;
        /// Corner below which the breath noise of the glottal jet has less
        /// and less energy while the folds are asked to sing, Hz at native size
        /// (physical engine; two one-pole high-passes on top of the breath's
        /// own shaping; `0` for none). The jet through the narrow slit of
        /// vibrating folds makes higher-pitched eddies than that of an open
        /// glottis in /h/. Without it the engine's accented vowels carried
        /// breath noise some 21 dB under their 0-1 kHz band in 2-4 kHz, where
        /// the LibriSpeech reader's whole 2-4 kHz band is 29 dB under it (and
        /// his 4-7.5 kHz band 26 dB under: his voice's noise lies higher).
        /// 3 kHz in round 9; 5 kHz since round 10, with `voiced_breath_gain`.
        voiced_breath_corner = 5000.0, 0.0..=8000.0, A;
        /// Level of the breath noise of vibrating folds against that of an
        /// open glottis at the same flow, dB (physical engine; on as far as the
        /// folds are asked to sing, after the high-passes at
        /// `voiced_breath_corner`). The accented vowels' 4-7.5 kHz band is
        /// this noise, not the voice's harmonics (lifting those above 5 kHz by
        /// 12 dB moved it by 0.0 dB): at 0 dB and 3 kHz it stood 28.6 dB under
        /// 0-1 kHz, 2.9 dB darker than the LibriSpeech reader's; at 6 dB and
        /// 5 kHz 25.7, his, with 1-2 and 2-4 kHz as they were (20.0 and 27.4
        /// dB under; his 20.9 and 28.9).
        voiced_breath_gain = 6.0, -12.0..=12.0, A;

        // ----- jets and hiss ---------------------------------------------------------

        /// Share of a free jet's dynamic pressure that returns as sound.
        jet_efficiency = 0.057, 0.015..=0.2, A;
        /// The same for a jet that strikes the teeth, relative to a free jet.
        teeth_gain = 0.38, 0.07..=1.0, A;
        /// Frequency above which turbulence has less and less energy, Hz (that
        /// of a jet on the teeth: `teeth_corner`).
        hiss_corner = 4500.0, 2500.0..=8000.0, A;
        /// Corner below which a jet on the teeth (/s ʃ/) makes less and less
        /// noise, Hz at the native size. Raising it takes the low end out of
        /// /s/ (a speaker's has little below 3 kHz), but also blurs /i/, whose
        /// narrow front lets a little of that jet through.
        hiss_floor = 400.0, 200.0..=3000.0, A;
        /// Frequency above which the noise of a jet on the teeth has less and
        /// less energy, Hz at the native size: a speaker's /s/ peaks at 5 to 8
        /// kHz.
        teeth_corner = 5500.0, 3000.0..=10000.0, A;
        /// Strength of the sound the jet of /f v/ makes where it strikes the
        /// edges of the upper incisors and the upper lip (physical engine): the
        /// face its eddies load, in areas of the lip gap (see `Phys::set_edge`).
        /// At 0 the hiss of /f/ was the free jet's alone, 27 to 36 dB under the
        /// vowel beside it in sentences, where the LibriSpeech reader's is 16
        /// dB under, and Whisper heard "makes fine" as "makes pine" (7 of 24
        /// /f/ words right in short sentences). At 3 a steady /f/ is 10 dB
        /// under a steady /s/ (the reader's /f/ is 10.5 dB under his /s/), 11
        /// dB louder than the free jet alone; in sentences it is some 22 dB
        /// under its vowel, its pressure still building, and flat from 2 to 7
        /// kHz. At 7 it matched the reader against the vowel, but a steady /f/
        /// was only 3 dB under /s/ and, once the hand-over out of /s/ kept the
        /// pressure up, as loud as the /s/ before it.
        labial_edge = 3.0, 0.0..=40.0, A;
        /// The same for the jet of /θ ð/ out of the gap between the tongue tip
        /// and the edges of the upper incisors. Held low: the reader's /ð/ has
        /// some 9 dB more hiss above 2 kHz than the engine's without it, but
        /// at 5 to 7 the recognizer starts to hear /ð/ as /z/; at 3 it hears
        /// /ð/ as /l/ less often and nothing else changes beyond noise.
        dental_edge = 3.0, 0.0..=40.0, A;

        // ----- the nose --------------------------------------------------------------

        /// Velar port fully open, cm² (how strongly nasals couple the nose).
        nasal_port = 0.8, 0.3..=1.6, A;

        // ----- the walls -------------------------------------------------------------

        /// Rate at which the pharynx and cheeks are actively widened during a
        /// voiced closure, cm³/s; the air the voice sends into the shut mouth
        /// goes there, so the voice lasts through the closure.
        cavity_expansion = 150.0, 0.0..=300.0, A;

        // ----- the tract's length (wave 2A; each `0` off) ---------------------------

        /// Length of the fully protruded lip tube, cm at native size (physical
        /// engine; it scales with the tract): with it above `0` the lips' cell
        /// where rounding is centred (41) is a slide one tick long plus the
        /// protrusion, of the lip opening's area, and the lumped inertance of a
        /// lip tube of up to two cells in front of the opening is not used. The
        /// protrusion is a state of the tract moved by the lips' spring towards
        /// this times the rounding slot's protrusion (see `Tract::lengths`).
        /// `0` keeps the lumped lip tube exactly. The owner's value is 1.2.
        lip_horn = 0.0, 0.0..=2.0, A;
        /// Share of the larynx's height the lower pharynx follows (physical
        /// engine): with it above `0` cells 8 and 9 are one slide, two ticks
        /// long at rest, stretched by a lowered larynx (up to 0.8 cm) and
        /// shortened by a raised one (at most by one tick, 0.36 cm at native
        /// size), the piriform sinuses at its lower end where they are today;
        /// the air the moving larynx sweeps is taken up there (the piston).
        /// `0` leaves the two cells as cells; `1` gives the larynx its full
        /// travel.
        larynx_travel = 0.0, 0.0..=1.0, A;

        // ----- the epilarynx (physical engine) ---------------------------------------

        /// Length of the epilaryngeal tube, the narrow tube just above the
        /// folds, in tract cells from the glottis (0.3645 cm each at the native
        /// size: 4 are 1.46 cm, 7 are 2.55 cm, 8 are 2.92 cm; physical engine;
        /// see `Tract::epilarynx`). The default is the original's seven cells.
        /// Opening in a step into a tract of 6.1 cm², the tube rings at 5.9 kHz
        /// at four cells, 4.8 at five, 3.85 at six, 3.5 at seven and 2.84 at
        /// eight (the two-tube equation within 2 %; the quarter wave `c / 4L`
        /// is 6.0, 4.8, 4.0, 3.43 and 3.0 kHz: in one dimension the step has
        /// no end correction). A fraction makes the cell at the top part tube
        /// and part flare, by area, which moves smoothly but is not a length:
        /// at 7.5 cells the tube rang at 3.48 kHz, where a tube of 7.5 cells
        /// rings at 3.34 (lengths between cells need a sliding joint). In the
        /// vowels the original tube's resonance is F4, 3.3 to 3.9 kHz (mean
        /// 3.52, F3 2.63); four cells put it near 4 kHz. A cluster at 2.8 to
        /// 3.2 kHz needs eight cells of 0.25 cm² opening into 3.6 cm²
        /// (`epilarynx_diameter` 0.45, `epilarynx_flare` 1.7): the tube rings
        /// at 2.9 to 3.2 kHz in every vowel and the transfer function's 3 kHz
        /// band rises 8 dB against 0-1 kHz (`tests/epilarynx_gate.rs`). With
        /// the vowel shapes solved again for each, over the 30 held-out
        /// sentences that tube gave phoneme error 30.2 % and Whisper words
        /// 66.1 % (the original 26.3 and 74.3), four cells 24.1 and 75.7 with
        /// vowels heard wrong in sentences 20.3 % (24.1), but /ɑ æ/ short of
        /// their F1 with the tongue at its lowest.
        epilarynx_cells = 7.0, 3.0..=9.0, A;
        /// Width of that tube, articulation diameter units (physical engine;
        /// the area is `1.257·d²` cm² at the native size: the default 0.6 is
        /// 0.45 cm², 0.45 is 0.25 cm², 0.4 is 0.2 cm², 0.8 is 0.8 cm²). At
        /// 0.2 cm² the folds took twice as long to start at 90 Hz (190 ms
        /// against 95) with four times the period jitter; at 0.25 cm² and
        /// wider they start as the original's.
        epilarynx_diameter = 0.6, 0.35..=1.0, A;
        /// Width of the lower pharynx the tube opens into, from its top to
        /// where the tongue's shape begins (cell 10), diameter units (physical
        /// engine; the default 1.1 is 1.52 cm², 3.4 times the tube's area;
        /// 1.7 is 3.6 cm², eight times).
        epilarynx_flare = 1.1, 0.6..=2.2, A;
        /// Cells over which the tube widens into the flare (physical engine):
        /// `0` a step, as the original's tube; more, a ramp linear in area
        /// starting at the tube's top, cut off at cell 10.
        epilarynx_flare_cells = 0.0, 0.0..=6.0, A;
        /// How far the tube narrows as the folds press together and widens as
        /// they slacken (physical engine): its area is multiplied by
        /// `2^(−this · (tenseness − 0.65) / 0.3)`, kept within 0.5 to 2 times
        /// and never wider than the flare. Tenseness carries a voice's effort
        /// (`Persona::effort` adds 0.3 per unit of the dial) and the laxness of
        /// its light syllables (8 %: 13 % more area in a function word's vowel
        /// at 1, which moves the tube's resonance by some 25 Hz). At 1, from
        /// tenseness 0.5 to 0.95 a tube of four cells rings from 3.85 to 4.0
        /// kHz, and the eight-cell tube that gives the cluster gives it in 4
        /// to 12 of the 31 vowel shapes. `0` switches it off.
        epilarynx_effort = 0.0, 0.0..=2.0, A;

        // ----- the tongue body from measured modes (wave 2, stage 2C) ----------------

        /// The shape of the tongue body on the physical engine: `0` the
        /// cosine of Pink Trombone (a hump slid along the tube by its place and
        /// height), `1` a measured neutral shape and three principal modes of
        /// eleven vowels of one man from MRI (Story 2008; `src/tract_modes.rs`)
        /// from the top of the epilarynx to the lips, the vowels solved on it
        /// (`src/vowel_modes_table.rs`; on the lengthened tract, see
        /// `vowel_targets::lengthened`, `src/vowel_modes_wave_table.rs`). The jaw
        /// follows the cosine tongue the shapes carry. The classic engine keeps
        /// the cosine.
        tongue_model = 0.0, 0.0..=1.0, A;
    } tiers {
    } aliases {
    }

    /// How sounds are made and joined (see [`Tuning`]): gaps and
    /// places, the glottis in consonants, closures and releases, leads,
    /// the articulators' speeds, coarticulation and the limits of
    /// following a speaker's schedule.
    ///
    /// The six tiers of the gesture score each have a [`TierDynamics`]:
    /// the springs of the articulators live there, filled with what today's
    /// tract does (`tongue_body` and `tongue_tip` are read; see their
    /// numbers' own fields below).
    articulation: ArticulationConfig, Articulation {
        // ----- the glottis in consonants (the glottal adduction tier) ----------------

        /// Seconds for the folds to come together or part.
        adduction_time = 0.008, 0.003..=0.02, A;
        /// Seconds for the arytenoids to open or close the glottis behind the
        /// folds (much slower than the folds meeting).
        abduction_time = 0.02, 0.008..=0.05, A;
        /// Seconds for the arytenoids to bring the glottis back together
        /// behind the folds once a voiceless sound is over (physical engine;
        /// opening it takes `abduction_time`, and the shorter of the two
        /// applies). With both at 20 ms the spread glottis was still closing
        /// as the vowel after a voiceless consonant began, and the vowel took
        /// 19 ms from its voicing onset to come within 3 dB of its peak (the
        /// LibriSpeech reader's: 7.5 ms), growing for 40 ms.
        spread_closing = 0.012, 0.004..=0.05, A;
        /// Glottal opening for voiceless sounds and breathing, cm².
        abducted_area = 0.25, 0.1..=0.5, A;
        /// Glottal opening added behind the folds for a voiceless stop or
        /// fricative and its aspiration, cm² at native size (on top of
        /// `abducted_area`).
        voiceless_spread = 0.12, 0.0..=0.4, A;
        /// Opening left behind the vibrating folds in a voiced fricative, cm²
        /// at native size: air for the hiss without stopping the voice.
        voiced_fricative_spread = 0.02, 0.0..=0.12, A;
        /// Opening left behind the vibrating folds in a word-final /z ʒ/, cm²
        /// at native size: more than `voiced_fricative_spread`. English lets
        /// them go partly voiceless at the end of a word, and their hiss has to
        /// be as plain as the voice under it (a final /z/ with too little air
        /// is heard as /ð/); /v ð/ are weak, and too much air makes them breathy.
        voiced_sibilant_spread = 0.05, 0.0..=0.12, A;
        /// Pitch raised at the start of a vowel after a voiceless consonant, Hz
        /// at native size (stiff folds after the glottis was spread).
        onset_kick = 12.0, 0.0..=25.0, A;
        /// Seconds over which that raise decays.
        kick_time = 0.03, 0.01..=0.08, A;

        // ----- gaps and places -------------------------------------------------------

        /// Added to the opening of every fricative (touch-diameter units).
        fricative_gap = 0.0, -0.1..=0.15, A;
        /// Added to the opening of every approximant (`l r w y`).
        approximant_gap = 0.0, -0.2..=0.3, A;
        /// Added to the opening the tongue leaves beside its tip for /l/,
        /// diameter units (physical engine). At the classic engine's opening
        /// the first formant of /l/ stood near 500 Hz, where a speaker's has
        /// 350 to 400, and an /l/ running into /ɪ/ or /u/ was heard as the
        /// start of a diphthong: "slid" as "slayed", "blue" as "b-uh-ue".
        lateral_gap = -0.12, -0.3..=0.1, A;
        /// Cells in front of its place over which the side channels of an /l/
        /// run (physical engine; see `tract::carve_lateral`): the air of a
        /// lateral leaves round the sides of a tongue whose tip touches the
        /// ridge, between them and the teeth. At 3 the channel ends at the
        /// front of the ridge; at 1 (a pinch at the place) the /l/ after a
        /// consonant was heard as /r/ in 8 of 27 short phrases, at 3 and 4 in
        /// none (at 4, into the teeth's cell, the lips of "a blue" were 0.36
        /// apart 10 ms after the release, where they open to 0.63).
        lateral_channel = 3.0, 1.0..=5.0, A;
        /// Gap left between the incisors while /s z/ are made, as the jaw
        /// raises the lower ones to the upper (articulation diameter units:
        /// 0.6 is about 0.45 cm² at the native size). The jet from the tongue
        /// tip strikes the teeth through it, and the small cavity it closes off
        /// behind them is what gives /s/ its peak at 5 to 8 kHz. Narrower is
        /// brighter and quieter.
        sibilant_teeth = 0.6, 0.3..=1.2, A;
        /// Lip opening of /ʃ ʒ tʃ dʒ/ (diameter units): rounded, but open.
        postalveolar_lips = 1.2, 0.8..=2.5, A;
        /// How much wider the gap of /s ʃ/ is as it forms (twice that as it
        /// opens again), diameter units.
        frication_widening = 0.15, 0.0..=0.3, A;
        /// Narrowest the mouth between the velum and a nasal's closure may be,
        /// diameter units: the back of the tongue drops so that the closed
        /// mouth is a side branch of the nose with its antiresonance where a
        /// speaker's is (near 1 kHz for /m/, 1.5 to 2 kHz for /n/), whatever
        /// the vowel's body around it. Below about 1.0 a central or back
        /// vowel's hump pulls /n/ down towards /m/'s, and the notch /n/ leaves
        /// in the murmur grows shallow.
        nasal_cavity = 1.6, 0.6..=2.0, A;
        /// How far the velum opens for /n/, against /m ŋ/ (1: as far). Lowered
        /// less, the murmur of /n/ grew weaker and was lost to other sounds
        /// (at 0.7, 86 of 94 /n/ before consonants heard right against 85,
        /// with fewer heard as /ŋ/ but more as nothing or as /l/), so it opens
        /// as far by default.
        alveolar_velum = 1.0, 0.5..=1.0, A;
        /// How far past contact a closing articulator aims (diameter units).
        closure_overshoot = 0.3, 0.1..=0.5, B;

        // ----- leads and hand-overs --------------------------------------------------

        /// Head start the articulators get on each sound, s (physical engine).
        lead = 0.03, 0.0..=0.06, B;
        /// The head start out of a closure, s: short, so stops stay shut. A
        /// little over one articulation block (10.7 ms), so the release falls
        /// before the next sound starts whatever the phase of the blocks.
        release_lead = 0.011, 0.0..=0.03, B;
        /// How long before the end of a nasal followed by an obstruent the
        /// velum starts to shut, s. The velum is slow (some 50 ms to shut), and
        /// a speaker raises it before the mouth opens: shut late, the first
        /// half of the /d/ in "and" was still nasal, built no pressure and was
        /// heard as /ŋ/ or not at all, and the /ð/ in "on the" was nasal.
        velum_lead = 0.03, 0.0..=0.05, B;
        /// Out of /s z ʃ ʒ/ into /f v/: how long before the tongue would let go
        /// of the sibilant (`lead` before its end) the lower lip sets off for
        /// the teeth and the tongue starts to let go, s (physical engine).
        labial_lead = 0.04, 0.0..=0.08, B;
        /// How long the lower lip then takes to come up to the teeth, s: its
        /// target moves from the lip's opening to the gap of /f v/ over this
        /// time, a slower gesture than a lip closing on its own.
        labial_closing = 0.048, 0.01..=0.08, B;
        /// How long the tongue takes to let go of the sibilant meanwhile, s:
        /// its gap widens steadily to just past the widest that hisses, and
        /// only then is let go altogether. 0 switches the hand-over off: the
        /// tongue lets go at once and the lip shuts behind it. (0.06 until the
        /// control block went from 10.7 to 2.7 ms; the gap then followed its
        /// ramp more closely and the /s/ fell a little sooner.)
        sibilant_exit = 0.07, 0.0..=0.1, B;
        /// How long before an alveolar tap begins its tip sets off, s
        /// (physical engine), in place of `lead`: the tap is a quick throw of
        /// the tip, which reaches the ridge from a vowel in some 10 to 18 ms.
        /// Sent with the usual head start (`lead`, 0.03), it shut the mouth 10
        /// to 15 ms before the tap was due, for 15 to 25 ms, and the level
        /// dipped 12 to 29 dB.
        tap_lead = 0.015, 0.0..=0.04, B;
        /// How long before an /l/'s tip lets go of it into a vowel the tongue
        /// body sets off from the /l/'s own body for the vowel's, s (physical
        /// engine): the end of the /l/ is made on the vowel's body. `0`
        /// releases tip and body together, as until round 8: the tip lets go
        /// while the body is still the /l/'s, low and back (with the tip let
        /// go it is an /ɑ/, F1 710 Hz), and the first formant rose some 130 Hz
        /// above both the /l/'s and the vowel's between them ("slid": 470,
        /// 585, 455 Hz), heard as /aɪ/: "slid" as "slide", and 7 of 15 /l ɪ/
        /// in short phrases as "l-ay". At 0.01 it runs flat from the /l/ into
        /// the vowel (470, 455, 480 Hz) and no /l ɪ/ is heard so; at 0.02 the
        /// recognizer heard more /l/ before /ɪ/ as /r/ and the /ɪ/ as /i/.
        /// Raising the /l/'s own body instead (to (20, 2.2..2.4)) took the rise
        /// down to 55 Hz, but the /l/ was heard as /w/ or /r/ in sentences and
        /// a /z/ after it at the end of a phrase no longer hissed.
        lateral_release = 0.01, 0.0..=0.04, B;
        /// How long after a /b p g k/ in front of an /l r/ (as in "blue",
        /// "play", "glue", "cream") opens the mouth the liquid's tongue tip is
        /// held back, s: the release goes into the liquid's tongue body without
        /// its narrowing, so a burst and a stretch of the stop's own formant
        /// transition are heard before the liquid takes over. `0` sends the
        /// liquid with the release, as before.
        liquid_transition = 0.018, 0.0..=0.04, B;
        /// Share of /s ʃ/ at each end spent in that wider gap. Off by default:
        /// at 0.25 the hiss rises over some 25 ms as a speaker's does, but the
        /// recognizer heard 2 to 6 points more phoneme errors in sentences and
        /// carrier phrases.
        frication_ramp = 0.0, 0.0..=0.4, B;

        // ----- closures and releases -------------------------------------------------

        /// Shortest complete closure of a voiceless stop (/p t k/), s: the
        /// mouth and the nose shut, before the release is let go. In a fast
        /// voice, or in a cluster where the articulators are sent on to the
        /// next sound as the closure forms, a closure of a few milliseconds is
        /// not heard as a stop.
        closure_floor = 0.022, 0.0..=0.04, B;
        /// How much longer the closure of a voiceless stop is than a voiced
        /// one's.
        fortis_closure = 1.25, 1.0..=1.6, B;
        /// How long the voice goes on once a voiceless stop has shut the
        /// mouth, s, before the folds are let go: a speaker's voicing dies away
        /// over the first 10 to 20 ms of a /p t k/ closure as the mouth fills.
        /// Cut off as the mouth shut, the level fell by 25 to 30 dB in 10 ms at
        /// every voiceless closure, where a speaker's falls by 13.
        voicing_tail = 0.012, 0.0..=0.03, B;
        /// Breath after a voiceless stop's burst, before the vowel, s (classic
        /// engine; the physical one holds the glottis open for `vot_*`).
        aspiration = 0.035, 0.015..=0.07, B;
        /// How long the glottis stays open after a /p/ is released before a
        /// stressed vowel, s (the voice follows a few milliseconds later).
        /// The physical engine uses these three in place of `aspiration`.
        vot_labial = 0.035, 0.015..=0.08, B;
        /// The same after /t/, s.
        vot_alveolar = 0.045, 0.015..=0.08, B;
        /// The same after /k/, s.
        vot_velar = 0.047, 0.015..=0.09, B;
        /// Share of that hold before an unstressed vowel.
        unstressed_vot = 0.5, 0.2..=1.0, B;
        /// How long before a voiceless consonant the voice is let go and the
        /// glottis starts to open, s: with the tract, which starts closing
        /// `lead` ahead, so the voice stops as the tract shuts.
        devoicing_lead = 0.012, 0.0..=0.03, B;
        /// Length of an alveolar tap, s (physical engine): a /t d/ between a
        /// vowel and an unstressed one ("slid on", "get it", "water", "city")
        /// is one voiced segment, the tongue tip thrown against the ridge and
        /// let go, with no closure held, no pressure built behind it, no burst
        /// and no breath after it (see `speech::taps`). At 0.02 the mouth is
        /// shut for 3 to 7 ms and the level dips 6 to 15 dB under the vowels
        /// beside it in 25 ms frames (the LibriSpeech reader's taps: 5 to 13
        /// dB, voiced throughout). `0` makes them stops again.
        tap_length = 0.02, 0.0..=0.05, B;

        // ----- coarticulation: vowels gliding into their neighbours ------------------

        /// Share of a lax vowel held on its own shape before it glides towards
        /// schwa (to its end shape in `vowel_targets`) for the rest.
        lax_nucleus = 0.67, 0.4..=1.0, C;
        /// Share of a stressed /æ/ held on its own shape before it glides
        /// towards schwa for the rest ("planks", "background"); `1.0` (the
        /// default) holds it to the end. At 0.55 it ends about a fifth of the
        /// way to schwa (`tests/vowel_glide_gate.rs`; at 0.67, the lax vowels'
        /// share, 13 %), and the recognizer heard more /æ/ as /ɛ/ (with it
        /// the 30 held-out sentences' phoneme error was 24.9 % against 23.5 %
        /// and vowel error in the LibriSpeech texts 30.1 % against 28.3 %). A
        /// stressed /æ/ reaches its formants without it ("planks": 229 ms).
        ae_nucleus = 1.0, 0.4..=1.0, C;
        /// Share of an /u/ after /t d s z/ held on its fronted shape (F2 1600
        /// Hz against the plain /u/'s 1000) before it glides back towards the
        /// plain /u/ for the rest (physical engine; see `speech::fronts_u`):
        /// the LibriSpeech reader's /u/ after /t d/ starts at 1700 to 2200 Hz
        /// and ends some 400 Hz lower. `1.0` holds it to the end (held, the
        /// recognizer heard 3 to 6 of 20 such /u/ as /i/), `0` makes every /u/
        /// the plain back one, as until round 9.
        fronted_nucleus = 0.5, 0.0..=1.0, C;
        /// Share of a vowel before a dark /l/ (one that closes its syllable)
        /// over which the vowel glides towards /ʊ/ (physical engine; see
        /// `speech::darken_before_coda_l`): the back of the tongue rises for
        /// the /l/ during the vowel. `0` makes the vowel on its own shape up to
        /// the /l/, as until round 11. In the LibriSpeech texts, 26 vowels
        /// before such an /l/: at 0 heard wrong 15 times (the recognizer's
        /// margin for the vowel −0.30 nats), at 0.5 and 0.7 9 times (+0.50 and
        /// +0.49); gliding towards /ɔ/ instead, 12 times; towards the /l/'s own
        /// body without its tip, which lowers the jaw, 10 times (+0.09).
        dark_l_share = 0.6, 0.0..=1.0, B;

        // ----- following a speaker's schedule (the transfer's limits) ----------------

        /// Least breath a phone of a followed schedule may be given, a share of
        /// what the rules give it (physical engine; see
        /// `speech::FollowOptions::breath_range`).
        follow_breath_low = 0.5, 0.2..=1.0, A;
        /// Most breath a phone of a followed schedule may be given, a share of
        /// what the rules give it (the lungs push with at most 1.5 times the
        /// voice's pressure).
        follow_breath_high = 1.5, 1.0..=1.5, A;
        /// Least tenseness a voiced phone of a followed schedule may be given,
        /// a share of the rules' (physical engine; see
        /// `speech::FollowOptions::tenseness_range`). Round 8 measured that
        /// 12 % less hands the child voice's folds to the pulse source with a
        /// click and 20 % less loses the voice bar of a /g/.
        follow_tenseness_low = 0.9, 0.7..=1.0, A;
        /// Most tenseness a voiced phone of a followed schedule may be given,
        /// a share of the rules'.
        follow_tenseness_high = 1.15, 1.0..=1.4, A;
        /// Fastest the tongue body and its back may be driven through a phone
        /// a followed schedule makes shorter than the rules do, a multiple of
        /// their speed (see `Trombone::set_articulation_boost`).
        follow_speed_max = 1.6, 1.0..=2.5, B;
        /// Shortest a followed schedule may make a phone, a share of its rule
        /// duration (physical engine); `0` follows the schedule however short.
        /// The rule durations are about what this tract needs to reach its
        /// shapes. On the 16 LibriSpeech anchors with the reader's own timing
        /// (14 % faster than the rules, a third of his phones under 70 % of
        /// theirs) the phoneme error was 25.0 % at `0`, 21.7 % at 0.7 and
        /// 19.4 % at 0.8 (the rules: 18.3 %), Whisper's words 74.6, 83.3 and
        /// 85.6 % (the rules: 81.3 %); the naturalness distance to him 0.925,
        /// 0.884 and 0.934, with the warped spectral distance 15.3, 14.9 and
        /// 15.2 dB. A /d/ he runs into the next sound ("and the", "heard
        /// there") was the one most often lost.
        follow_shortest = 0.7, 0.0..=1.0, B;
        /// Shortest a followed schedule may make a phone the floor of
        /// `follow_shortest` does not hold (physical engine; see
        /// `speech::rhythm::held_in_transfer`), a share of its own rule
        /// duration: the vowels of unstressed syllables and function words,
        /// the consonants of function words but their stops and sibilants, the
        /// single nasals, liquids, glides and weak fricatives of content words.
        /// `0.7`, the same floor as the rest: left off one class at a time on
        /// the 16 LibriSpeech anchors with the reader's own timing, the floor
        /// mattered most in the function words (their vowels: phoneme error
        /// 22.5 to 24.7 %; their consonants: 25.2 %, Whisper 81.8 to 74.6 %)
        /// and not on the stressed vowels or clusters (22.2 and 22.1 %); held
        /// only on the stressed vowels, stops, sibilants and clusters (this at
        /// `0`) the phoneme error was 27.6 % and Whisper 74.6 %, at 0.6 23.5
        /// and 80.9 %, at 0.5 25.1 and 75.6 % (round 10's rules).
        follow_shortest_light = 0.7, 0.0..=1.0, B;
        /// How far a followed schedule may move a vowel's F1 and F2 from the
        /// voice's own, a share either way (see `speech::PhoneSpan::formants`).
        follow_formant_range = 0.15, 0.0..=0.3, C;

        // ----- articulator speeds (and `aliases` below: the tiers' springs) ----------

        /// Natural frequency of the back of the tongue making a closure or a
        /// gap (/k g ŋ/), rad/s at speed 1: between the two.
        dorsum_rate = 20.0, 10.0..=40.0, B;
        /// Natural frequency of the tongue as it drops behind a nasal's
        /// closure to [`nasal_cavity`](Self::nasal_cavity), rad/s at speed 1,
        /// as `articulation_rate` is. At the back of the tongue's own speed (20)
        /// the mouth behind an /n/ after /ʌ ə ɛ ɑ aʊ/ was still the vowel's for
        /// the first third of the murmur, its antiresonance 1.2 to 1.35 kHz
        /// rising to 1.5 kHz only as the murmur ended, and 5 of 94 /n/ before
        /// consonants were heard as /ŋ/. Lowered this quickly (with the tip
        /// rising, the jaw and the blade under it drop), it stands at 1.45 to
        /// 1.55 kHz from the start of the murmur.
        nasal_cavity_rate = 60.0, 10.0..=70.0, B;
        /// How much faster articulators close than open.
        closing_ratio = 2.0, 1.0..=3.0, B;
        /// Speed at which a closure lets go, as a share of its full swing. At
        /// 0.37, set while the tract's areas were drawn across 10.7 ms blocks,
        /// the tongue tip of a /t/ now crosses the width that hisses slowly
        /// enough for single bursts to stand up to the vowel after them ("two",
        /// "to": -1 dB in the deep voice, +1.3 dB in the elder one); at 0.5
        /// the loudest are 3 to 7 dB under it. 0.5 is the owner's choice: at
        /// 0.6 Whisper heard "background" as "back drawing".
        release_speed = 0.5, 0.1..=0.8, B;
        /// How many times faster than usual the velum shuts out of an /n/ or
        /// /ŋ/ into a vowel, from the moment the tongue lets go (physical
        /// engine; usually some 45 ms from open to shut, here 15). Even a
        /// quarter open, the port (0.2 cm²) couples the nose to a close vowel
        /// so strongly that the coupled tract's loudest peak moves to 700 Hz,
        /// above the vowel's own first formant (transfer function of /u/: 322
        /// and 704 Hz, 10 and 16 dB), and the nose's resonances at 2.5 to 3.4
        /// kHz come up 17 dB. Shutting at the usual speed, the velum left the
        /// first 35 ms of the /u/ of "canoe" so: its 550 to 950 Hz band 7 to 9
        /// dB under its 200 to 450 Hz band, where the owner's "canoe" and
        /// "smooth" and the LibriSpeech reader's "the new" stand 16 to 26 dB
        /// under from 5 ms after the release, with a 3.5 dB step in the level
        /// as it shut; the recognizer heard /ʌ/, /oʊ/ or /l/ there. At 3 the
        /// band is 18 dB under from 20 ms on and the step is gone; the
        /// recognizer hears N UW in sentence 1 and in 6 of 10 carrier phrases
        /// (1 before). At 4 the mouth, still opening, chokes the flow as the
        /// nose shuts and a cycle of the folds fails to close (heard as /h/).
        /// Not after /m/, whose lips open slowly onto a rounded vowel's narrow
        /// ones: with the nose shut that soon, the voice of "smooth" dipped 6
        /// to 8 dB 20 to 40 ms into the /u/.
        velum_release = 3.0, 1.0..=5.0, B;

        // ----- the gesture score (Wave 1, w1-gesture) ------------------------

        /// Whether the physical engine's speech is played as a gesture score
        /// (`1`; see `crate::gesture`): each segment's articulation becomes
        /// gestures on the tiers that make it (the tongue body, the tip, the
        /// lips and jaw, the velum), the larynx's settings and the pitch
        /// gestures on the glottal and laryngeal tiers, and the tract follows
        /// the tiers' blend (the tract's articulators move towards it as
        /// they always have, at the speed of each tier's stiffness). The
        /// default since Wave 1: with `gesture_overlap` 0 it renders bit for
        /// bit as the segment path; with 1 the 30 held-out and 30 tuning
        /// sentences, the vowels and the consonants in the LibriSpeech texts
        /// score as on it (26.1 / 74.3 %, 22.3 / 72.8 %, 24.8 % wrong, 88.2
        /// % right). `0` sends each segment's shape to the tract as before,
        /// kept to compare by ear. `2` is the gesture path with the series
        /// plant, a measurement: the tiers' own task-space state, integrated
        /// at their stiffness, is what the tract is sent, so the tract's
        /// articulators lag behind a second spring; the stops of the
        /// LibriSpeech texts then shut 21 ms into their phone (2 ms on the
        /// single spring) and 43 of 193 never do, and the phoneme error of
        /// the ten `sentence_heldout` sentences is 62 % (21.6 on the single
        /// spring).
        gesture_score = 1.0, 0.0..=2.0, B;
        /// On the gesture path, which of the rules about neighbouring sounds
        /// are left to the overlap of gestures instead: the sum of the
        /// families' bits (`0` none). `1`: the tongue body and the lips under
        /// a consonant made by the tip or the lips (and /h/) are the gestures
        /// of the vowel the consonant leans on, spanning it, and the
        /// consonant adds only its own (its constriction, a weak pull of the
        /// body towards the tip's, lips of its own or, under a gap meant to
        /// hiss, lips held as they are). `2`: a stop released into an /l r/
        /// of its word has the liquid's tongue tip on its way
        /// (`speech::HELD_TIP`) through its closure, the lips' or the back of
        /// the tongue's closure and the tip overlapping on their two tiers
        /// ("dark blue", "glue"); measured and left off (in sentence 2 alone
        /// "glue" lost its /g l/; `sentence_heldout` words 81.2 to 80.0 %).
        /// `4`: an unstressed /ə/ is a weak tongue body gesture
        /// ([`schwa_dominance`](Self::schwa_dominance)) under the gesture of
        /// the vowel after it, which spans it, in place of the rule of
        /// [`schwa_colour`](Self::schwa_colour).
        gesture_overlap = 1.0, 0.0..=7.0, B;

        // ----- "canoe": an /n/ into a fronted vowel (wave 1, w1-canoe) -------

        /// How many times faster the tongue tip lets go of a closure while the
        /// velum is open (physical engine; out of /n/; `1` is off): the
        /// closure and the tip's shoulder around it, until each cell is back
        /// with the tongue body or closing again. At the tip's own speed
        /// (`articulation_rate`, starting at `release_speed`) the gap out of
        /// an /n/ was at 20 % of its opening after 10 ms and 37 % after 20 (2
        /// to 8 cm²/s, where Stevens gives 50 to 100 for an alveolar release);
        /// over a fronted body the second formant overshot to 1900 to 2134 Hz
        /// and fell back by 250 over 25 ms, F3 and F4 11 dB over F2: a /j/,
        /// "canoe" heard as "can use". At 3 it still overshot (1968 Hz); at 6
        /// the narrowest cell is at 0.22 cm² 9 ms after the release (some 25
        /// cm²/s) and the second formant rises onto the vowel. A stop's
        /// release is not hastened: its burst and breath cover it, and its
        /// transitions are already quicker than a speaker's.
        nasal_tip_release = 6.0, 1.0..=8.0, B;
        /// Whether the /u/ after /n/ is the fronted one (`1`) or the plain back
        /// one (`0`, as until wave 1; physical engine; see `speech::fronts_u`):
        /// the owner's "canoe" holds F2 1860 to 1910 Hz from the release on,
        /// the LibriSpeech reader's "the new" starts near 1790 and rises.
        /// Fronted with the slow tip release, it was heard as /ju/. On by the
        /// owner's ear; the phoneme recognizer hears the fronted /u/ after /n/
        /// less often than the back one (see the README).
        fronted_after_n = 1.0, 0.0..=1.0, C;
        /// Share of the tongue's drop behind an /n/'s closure
        /// ([`nasal_cavity`](Self::nasal_cavity)) held back when the /n/ leads
        /// into the fronted /u/ (physical engine; `0` is off; see
        /// `speech::keeps_tongue_up`): dropped, the tongue sprang back after
        /// the release at the cavity's speed into the vowel's first 15 ms. Kept
        /// up, the mouth behind the /n/ is the fronted vowel's: its
        /// antiresonance stands at 2.1 to 2.3 kHz (1.45 to 1.55 dropped). Not
        /// before /i ɪ j/: kept up there it rose from 1.4 to 1.8 kHz, where
        /// round 6's recognizer heard /ŋ/ in a fifth of the /n/. Off since
        /// wave 2 (on in wave 1): by the owner's ear the /n/ leaned toward /ŋ/.
        nasal_cavity_held = 0.0, 0.0..=1.0, B;
        /// Whether an unstressed /ə/ takes the colour of a high front or
        /// fronted vowel after it (`1`) or stays the one central schwa (`0`,
        /// as until wave 1; physical engine; see `speech::schwa_colour`): its
        /// shape is `VowelKey::Fronted` of /ə/ (F1 400, F2 1750 Hz), where the
        /// LibriSpeech reader's /ə/ before a front vowel stands at 372 and
        /// 1604 and the owner's in "canoe" near 300 and 1800.
        schwa_colour = 1.0, 0.0..=1.0, C;
        /// Dominance of an unstressed /ə/'s tongue body and lips against the
        /// next vowel's, which span it on the gesture path with
        /// `gesture_overlap`'s bit 4 (see `Trombone::articulate_as_gestures`):
        /// the schwa's body is the blend, `1 / (1 + this)` of the way to the
        /// next vowel's. Vowels and consonants have dominance 1.
        schwa_dominance = 2.0, 0.25..=8.0, C;

        // ----- who lowers the larynx (wave 2A; each `0` off) ------------------------

        /// How far the larynx is lowered through the closure of a voiced stop
        /// or affricate, cm (physical engine; read by the tract only with
        /// `larynx_travel` on): a gesture on the larynx tier on the gesture
        /// path, `Trombone::set_larynx_height` on the rule path; it comes back
        /// up after the release at the larynx's own speed (`larynx_rate`).
        voiced_stop_larynx = 0.0, 0.0..=0.8, A;
        /// How far the larynx is lowered in a rounded vowel (/u ʊ oʊ ɝ w/ and
        /// the rounded glides), cm (physical engine; with `larynx_travel` on):
        /// the measured speaker's /u/ tract is 2.7 cm longer than his /i/, lips
        /// and larynx together (Story 2008).
        rounded_larynx = 0.0, 0.0..=0.8, C;
    } tiers {
        /// The folds coming together and parting (adduction and abduction).
        /// Not read yet: today the glottis follows first-order lags
        /// ([`adduction_time`](Self::adduction_time),
        /// [`abduction_time`](Self::abduction_time),
        /// [`spread_closing`](Self::spread_closing)), not a spring, so its
        /// stiffness is `0` (none).
        glottal_adduction = TierDynamics::new(0.0, 1.0, 1.0);
        /// The larynx's height and the pitch it carries. The pitch follows
        /// the speech compiler's contour directly; the height is a spring of
        /// the tract (wave 2A): its `stiffness` is the field `larynx_rate`,
        /// its `damping` `larynx_damping` (read only with `larynx_travel`
        /// on).
        larynx_and_pitch = TierDynamics::new(18.0, 1.0, 1.0);
        /// The velum. Not read yet: today it moves at a fixed slew rate
        /// (`tract::Tract::move_damped`: open at 0.25, shut at
        /// `VELUM_SHUTTING` times the articulation speed, faster out of a
        /// nasal by [`velum_release`](Self::velum_release)), not a spring,
        /// so its stiffness is `0` (none).
        velum = TierDynamics::new(0.0, 1.0, 1.0);
        /// The tongue body and the jaw under the consonants: the vowel
        /// shape. Its `stiffness` is the field `body_rate` and its `damping`
        /// `body_damping` ([`body_rate`](Self::body_rate),
        /// [`body_damping`](Self::body_damping)).
        tongue_body = TierDynamics::new(11.4, 1.0, 1.0);
        /// The tongue tip and blade, and with them today the lips (the
        /// tract's cells from the tip's on share one spring). Its
        /// `stiffness` is the field `articulation_rate`
        /// ([`articulation_rate`](Self::articulation_rate)); its `damping`
        /// is not read yet (the tract's tip, lips, back of the tongue and
        /// nasal cavity are critically damped, `1`, in `tract.rs`).
        tongue_tip = TierDynamics::new(40.0, 1.0, 1.0);
        /// The lips and the jaw. Not read yet: today the lips ride the
        /// tongue tip's spring (40 rad/s, critically damped), filled in
        /// here, and the jaw follows a first-order lag of 20 ms
        /// (`JAW_TIME` in `tract.rs`).
        lips_and_jaw = TierDynamics::new(40.0, 1.0, 1.0);
    } aliases {
        /// Natural frequency of the articulators, rad/s, at speed 1 (physical
        /// engine). Higher is nimbler.
        articulation_rate => tongue_tip.stiffness, 25.0..=70.0, B;
        /// Natural frequency of the lips, rad/s at speed 1 (physical engine,
        /// gesture path): the stiffness of the gestures on
        /// [`Tier::LipsAndJaw`](crate::Tier::LipsAndJaw); the tract's lip
        /// cells move at `articulation_rate` scaled by this over it (alike
        /// at 40: on the segment path the lips ride the tip's spring).
        lips_rate => lips_and_jaw.stiffness, 25.0..=70.0, B;
        /// Natural frequency of the tongue body and the jaw, rad/s at speed 1
        /// (physical engine). They are heavy and slow (a time constant of
        /// about 40 ms in speech), so the vowel shape glides from vowel to
        /// vowel underneath the consonants; `articulation_rate` is that of the
        /// light, quick tongue tip, blade and lips that make the consonants on
        /// top of it.
        body_rate => tongue_body.stiffness, 6.0..=25.0, B;
        /// Damping ratio of the tongue body and the jaw (physical engine): `1`
        /// is the critically damped spring of `body_rate`, which from rest
        /// covers half its way in 67 ms and nine tenths in 156 ms at the speech
        /// speed, its speed peaking a quarter of the way through that time and
        /// then dying away over a long tail. Below `1` it overshoots a little
        /// and settles sooner: at 0.9 nine tenths in 136 ms and 95 % in 161
        /// (190 critically damped) with an overshoot of 0.15 % of the move; at
        /// 0.8 in 119 and 136 ms with 1.5 %. The tip, the lips and the back of
        /// the tongue stay critically damped. Off (`1`) by default: measured
        /// in the LibriSpeech texts the engine's vowels already hold their
        /// formants for more of their length than the reader's (58 % against
        /// 46 % within 40 Hz on F1 and 80 Hz on F2) and come out of a stop
        /// sooner (20 ms against 35); at 0.8 the vowels there were heard wrong
        /// less often (25.5 to 21.7 %) but the 30 held-out and 30 tuning
        /// sentences no better (25.7 and 21.9 % phoneme error against 25.9 and
        /// 22.0), a /z/ hissed 9.2 dB under /s/ (8.1 at 1, the gate's limit 9)
        /// and the tongue's way out of an /n/ came to depend on the control
        /// block by 1.1 % of its travel (the gate's limit 1 %; 0.98 % at 1, and
        /// 1.07 % at 0.9); at 0.7 the sentences were heard worse.
        body_damping => tongue_body.damping, 0.6..=1.0, B;
        /// Natural frequency of the larynx's height, rad/s at speed 1
        /// (physical engine, with `larynx_travel` on; wave 2A): at the speech
        /// speed (2.2) 40 rad/s, the larynx tier's own, so a critically damped
        /// larynx covers seven tenths of its way in 60 ms, nine tenths in 97.
        larynx_rate => larynx_and_pitch.stiffness, 8.0..=40.0, B;
        /// Damping ratio of the larynx's height (`1` critically damped).
        larynx_damping => larynx_and_pitch.damping, 0.6..=1.0, B;
    }

    /// The rhythm and melody of a phrase (see [`Tuning`]): the
    /// rules' lengths of sounds and their lengths by stress and position,
    /// rhythm, pauses, intonation, the breath over a phrase, and tenseness
    /// and level by syllable.
    prosody: ProsodyConfig, Prosody {
        // ----- the rules' lengths of sounds ------------------------------------------

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
        /// Length of /z ʒ/, s: longer than the weak /v ð/
        /// (`voiced_fricative_length`), as in English; a short /z/ is heard as
        /// /ð/ or /d/.
        voiced_sibilant_length = 0.095, 0.06..=0.14, B;
        /// Length of an approximant, s.
        approximant_length = 0.055, 0.04..=0.1, B;
        /// Length of an `h`, s.
        aspirate_length = 0.06, 0.04..=0.1, B;

        // ----- lengths by stress and position ----------------------------------------

        /// How much longer a stressed vowel is than an unstressed one.
        stress_stretch = 1.5, 1.2..=1.9, B;
        /// How much longer every vowel and diphthong is held on the physical
        /// engine than `vowel_length` and `diphthong_length` say (those are the
        /// classic engine's). Its tongue body is heavy and slow
        /// (`body_rate`), and a vowel held for the classic engine's time ends
        /// before the body has reached it.
        vowel_hold = 1.18, 1.0..=1.6, C;
        /// Length of a stressed lax vowel /ɪ ɛ ʌ ʊ/ against a tense vowel's
        /// (/i u/) in the same place. A speaker reading LibriSpeech holds his
        /// stressed lax vowels for some 60 % of his tense ones. At 0.62 the
        /// recognizer heard the vowels better (vowel lab held-out −2.7 points)
        /// but the sentences of the tuning material worse (6 points of phoneme
        /// accuracy, Whisper 8.5 of words) than at the round 4 value, 0.905.
        /// The owner chose 0.62 by ear in round 5 and a milder 0.82 for the
        /// stressed lax vowels in round 6 ("sells", "punch"; the unstressed
        /// ones keep `unstressed_lax_length`). At 0.82 on the round 6 engine
        /// the recognizer heard more stressed /ɪ/ as /eɪ/ (vowel error in the
        /// LibriSpeech texts 28.3 % against 26.2 % at 0.62), the 30 held-out
        /// sentences' phoneme error was 23.5 % against 24.4 % and Whisper's
        /// words 65.7 % against 63.1 %, all within the sets' noise.
        lax_length = 0.82, 0.5..=1.0, C;
        /// The same as `lax_length` for an unstressed lax vowel ("it", "-ing"),
        /// which the stress already makes shorter.
        unstressed_lax_length = 0.62, 0.5..=1.0, C;
        /// Length of a stressed /ɪ/ ("slid", "big"), as a share of a tense
        /// vowel's: it keeps the short length while the other stressed lax
        /// vowels take `lax_length`. At 0.82 the recognizer heard the stressed
        /// /ɪ/ of the LibriSpeech texts as /eɪ/ ("slid" as "slayed"): a long
        /// /ɪ/ that glides towards schwa is what /eɪ/ sounds like backwards.
        ih_length = 0.65, 0.5..=1.0, C;
        /// Shortest the held part of a stressed lax vowel may be, seconds (its
        /// nucleus: `lax_nucleus` of its length, before its glide towards
        /// schwa), however fast the speech; `0` for none. The owner asked for
        /// 60 ms with the milder stressed length: of the 136 stressed lax
        /// vowels of the judge's sentences it lengthens none at the default
        /// rate, 39 at 1.2 times the rate and 93 at 1.4 times (at `lax_length`
        /// 0.62 it would lengthen 39 already at the default rate).
        lax_floor = 0.06, 0.0..=0.09, C;
        /// Length of a vowel before a voiceless consonant that ends its
        /// syllable, relative to before a sonorant (pre-fortis clipping).
        prefortis_clip = 0.85, 0.6..=1.0, B;
        /// Length of a vowel before a voiced obstruent that ends its syllable,
        /// relative to before a sonorant.
        prelenis_stretch = 1.2, 1.0..=1.5, B;
        /// How much longer the consonants after the last vowel of a phrase are
        /// (physical engine), on top of that vowel's own lengthening: the
        /// LibriSpeech reader draws out his whole last syllable, its closing
        /// consonants most (1.2 to 1.6 times the rules' at his pace, the vowel
        /// 1.2 times), where the rules lengthened the vowel alone.
        final_coda_stretch = 1.25, 1.0..=1.8, B;

        // ----- rhythm (round 11, off) ------------------------------------------------

        /// How much longer a consonant that opens a stressed syllable of a
        /// content word is than its own length (physical engine; see
        /// `speech::rhythm`, which has the measurements): the LibriSpeech
        /// reader's, the two references' and the owner's are 1.30 to 1.37
        /// times the rules' at their pace with the class of each consonant
        /// fitted alongside, 1.11 to 1.27 without. Not the weak fricatives
        /// /f v θ ð h/, which are long in the rules already. With this and
        /// `cluster_length` alone the 30 held-out sentences' phoneme error went
        /// from 25.9 to 23.4 %, the 30 tuning sentences' from 22.0 to 18.6 %,
        /// the vowels heard wrong in the LibriSpeech texts from 25.5 to 22.4 %
        /// for 1.4 % more time; every stop, every fricative, every nasal or
        /// every liquid 1.3 times as long, wherever it stood, gained 0.4 to 1.4
        /// points over the two sets for 2 to 5 % more time. Whisper's words
        /// fell 2 to 5 points with any of them. `1` makes them as long as
        /// anywhere else.
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `1.3` is the measured setting.)
        stressed_onset_length = 1.0, 1.0..=1.6, B;
        /// Length of each consonant of a cluster against a single one's
        /// (physical engine): the reader's are 0.81 of his single ones (the
        /// references' 0.94), where the rules made them alike. Not a sibilant
        /// that closes one, which the reader holds 1.11 times the rules'.
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `0.85` is the measured setting.)
        cluster_length = 1.0, 0.6..=1.2, B;
        /// Length of the consonants of a function word, its stops excepted
        /// (they keep their closure and release), against their own (physical
        /// engine): the reader's /ð v/ of "the", "of" are 0.78 of the rules'.
        /// At 0.85 (with `schwa_length` 0.75) Whisper's words over the 30
        /// held-out sentences fell 7.8 points and rose 3.7 over the tuning
        /// ones; at 0.9 (and 0.85) -2.6 and +5.5.
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `0.9` is the measured setting.)
        function_consonant_length = 1.0, 0.5..=1.0, B;
        /// Length of /ə/ against an unstressed vowel's (physical engine): the
        /// speakers' are 0.69 to 0.76 of the rules', in function and content
        /// words alike (the /ə/ of "canoe" was 100 ms, the reader's 65 and the
        /// references' 50 to 52; now 77).
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `0.75` is the measured setting.)
        schwa_length = 1.0, 0.5..=1.0, C;
        /// Length of a diphthong against what `diphthong_length`, `vowel_hold`
        /// and the stress give it (physical engine): the reader's are 0.67 of
        /// the rules' (a stressed /oʊ eɪ/ was 226 to 266 ms, his 70 to 250 at
        /// the rules' pace).
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `0.75` is the measured setting.)
        diphthong_hold = 1.0, 0.5..=1.0, C;
        /// How much longer every phone of the physical engine's rules is held,
        /// the first and last syllable of a phrase excepted, to give back what
        /// the fields above take from the rate of speech (the four listening
        /// sentences last 11.86 s, 11.84 before). At 1.032 a vowel held 3 %
        /// longer before /ʃ/ darkened its hiss past the bound of
        /// `hisses_are_broad_and_s_is_brighter_than_sh` (centroid 4350 to 3840
        /// Hz, flatness 0.42 to 0.29; at 1.02 0.32).
        /// (Off, `1`, since the end of round 11: the owner's ears chose round
        /// 10's lengths; `1.02` is the measured setting.)
        rhythm_pace = 1.0, 0.8..=1.25, B;

        // ----- intonation ------------------------------------------------------------

        /// Rise of the pitch on a stressed vowel, semitones at expressiveness
        /// 1 (a speaker's accents are 2 to 5). 6 until round 10: the rules'
        /// pitch spanned 7.2 semitones over the LibriSpeech texts where the
        /// reader's spans 10.4, and stepped 1.8 semitones from vowel to vowel
        /// where his steps 3.1; at 8 (4 semitones in the default voice) they
        /// rise into an accent by 2.9 semitones, he by 2.7.
        accent_size = 8.0, 0.0..=16.0, B;
        /// Seconds over which an accent sinks back after its vowel.
        accent_decay = 0.2, 0.05..=0.5, B;
        /// Where in an accented vowel its pitch accent peaks, as a share of the
        /// vowel (physical engine): the accent rises from the start of its
        /// syllable to there and sinks back over `accent_decay` (a statement's
        /// last accent peaks as its vowel begins, whatever this says). At `1`,
        /// as until round 9, it rose from the vowel's start to its end, and the
        /// accented vowels of the LibriSpeech texts rose by 1.2 semitones
        /// (median) where the reader's go either way (median 0.0).
        accent_peak = 0.55, 0.0..=1.0, B;
        /// Fall of the pitch over a phrase, semitones at expressiveness 1.
        declination = 6.0, 0.0..=12.0, B;
        /// Fall across the last vowel of a statement, semitones at
        /// expressiveness 1.
        final_fall = 8.0, 0.0..=16.0, B;

        // ----- the breath over a phrase, and pauses ----------------------------------

        /// Seconds over which the lungs build up their pressure at the start of
        /// a phrase, from nothing: the first sound swells in rather than
        /// switching on.
        phrase_onset = 0.15, 0.04..=0.3, B;
        /// Share of the lung pressure a phrase that opens on a vowel or a
        /// sonorant starts from, rising to all of it over `phrase_onset`: about
        /// what the folds need to start, so the voice grows in from its first
        /// cycles without the first sound being lost.
        onset_pressure = 0.5, 0.0..=1.0, A;
        /// Seconds by which a phrase that opens on a nasal or a liquid starts
        /// that sound earlier, out of the pause before it (physical engine;
        /// see `speech::breathe`): the lungs need some 40 ms to bring the
        /// pressure up to what the folds need, so the first 40 to 45 ms of an
        /// opening /n m l/ were 33 dB under the vowel and silent, where the
        /// LibriSpeech reader's opening /n m/ stand 8 to 20 dB under it from
        /// their first 10 ms. At 0 the recognizer heard no /n/ in 4 short
        /// phrases opening on one and 4 of 12 opening /l/ (the rest as /t f v/,
        /// only the release onto the vowel being heard); at 0.05 4 of 4 and 6
        /// of 12.
        sonorant_onset = 0.05, 0.0..=0.08, B;
        /// Share of the lung pressure lost between the start and the end of a
        /// phrase: the speaker's breath running down, which lowers the level
        /// and the pitch together (all of it over a phrase of two seconds or
        /// more, a shorter one in proportion to its time). 0.15 until round 10,
        /// when the rules' vowels grew 1 dB louder over a phrase where the
        /// LibriSpeech reader's lose some 3 (fitted over stress, function word,
        /// place in the phrase and vowel height; at 0.3 they lose 1). At 0.35
        /// with `unaccented_level` 3, nearer him, the 30 held-out sentences'
        /// phoneme error rose from 25 to 28 %.
        phrase_decline = 0.3, 0.0..=0.4, A;
        /// Share of the lung pressure let go over the last vowel of a
        /// statement: the voice dies away instead of being cut off (half of it
        /// before a comma).
        final_drop = 0.5, 0.0..=0.9, A;
        /// How far the breath lets the level of each vowel sink from its start
        /// to its end, dB (physical engine; the lungs take it back in the
        /// sounds after it). `0` holds it.
        vowel_decline = 2.5, 0.0..=6.0, A;
        /// How much quieter an unaccented vowel is than an accented one beyond
        /// what its pitch and length give, dB of level (physical engine): the
        /// lungs push less through it, and the consonants between two vowels
        /// take the breath of the line from one to the other (see
        /// `speech::weigh_syllables`). `0` breathes every syllable alike. The
        /// LibriSpeech reader's accented vowels stand 2.7 dB over his others,
        /// the rules' 1.2 without this and 1.9 with it; at 3 dB the 30
        /// held-out sentences' phoneme error went from 25.5 to 26.6 %.
        unaccented_level = 1.5, 0.0..=6.0, A;
        /// Pause after a sentence, s (physical engine; the classic engine keeps
        /// 0.38, a comma 0.18 in both): the two reference recordings pause 0.54
        /// and 0.62 s between their two sentences, the owner 0.74.
        sentence_pause = 0.55, 0.3..=1.0, B;

        // ----- tenseness and softness by syllable ------------------------------------

        /// Share by which the folds are held more loosely through the vowel of
        /// a function word (physical engine; see `speech::shade_syllables`).
        light_laxness = 0.08, 0.0..=0.3, A;
        /// How much more loosely the folds are held through nasals, liquids,
        /// glides and unaccented vowels than on an accented vowel, as a share
        /// of the voice's tenseness (physical engine); `0` (the default since
        /// round 8, which softens the voice there instead: `sonorant_softness`)
        /// holds them alike throughout. Round 7 had 0.12: it opens the folds,
        /// and at 0.2 the voice gave out a quarter of the way through the
        /// closure of a /g/ after an unaccented vowel, at 0.15 the child
        /// voice's folds (290 Hz) fell out of the range they can sing and were
        /// handed to the pulse source with a click 14 times the vowel's peak;
        /// at 0.12 it made the nasals and liquids of sentences 1 to 2 dB
        /// brighter above 2 kHz against the accented vowels, not darker.
        sonorant_slack = 0.0, 0.0..=0.4, A;
        /// How much the folds slacken over the last 150 ms before a pause, as
        /// a share of their tenseness there (physical engine). Round 7 had
        /// 0.2; `0` since round 8: the LibriSpeech reader's last vowel is as
        /// bright as his accented ones (+1.0 dB in 2-4 kHz against 0-1 kHz)
        /// and has the same H1-H2 (+0.0 dB), where the engine's, with the
        /// breath dying away under it, is already 5 dB darker without it, and
        /// the slackening raised its H1-H2 from +1.0 to +1.9 dB.
        final_slack = 0.0, 0.0..=0.4, A;
        /// How soft the voice is in nasals and liquids, `0` (as on an accented
        /// vowel) to `1` (physical engine). The glides are left firm: under
        /// their narrow lips they are already darker than the reader's (11.6
        /// dB under the accented vowels in 2-4 kHz, his 4.4). Round 8 had 1;
        /// `0` since round 9: the owner heard the accented vowels jump out
        /// sharp where the filter let go of them.
        sonorant_softness = 0.0, 0.0..=1.0, A;
        /// The same in unaccented vowels (round 8: 0.5).
        unstressed_softness = 0.0, 0.0..=1.0, A;
        /// The same at the very end of a phrase, reached over its last 150
        /// ms; `0`, since the engine's last vowels are already darker than the
        /// reader's (see `final_slack`).
        final_softness = 0.0, 0.0..=1.0, A;
    } tiers {
    } aliases {
    }
}

impl Default for Tuning {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The JSON (or any serde) form of a [`Tuning`]: written nested, one object
/// per part (`{"anatomy": {"lung_pressure": 8, ..}, "articulation": {..},
/// "prosody": {..}}`, every field of [`Tuning::FIELDS`] under its name);
/// read in that form, flat (`{"lung_pressure": 8}`) or both mixed, onto
/// [`Tuning::DEFAULT`]. A field given twice (flat and in its part) is an
/// error, so the result never depends on the order of the keys. An unknown
/// name is skipped, as before the split, so that a voice saved with a field
/// since removed (a take of the terminal app) still loads; a profile
/// ([`VoiceSettings::with_profile`](crate::VoiceSettings::with_profile))
/// refuses it.
#[cfg(feature = "serde")]
mod form {
    use super::{Part, Tuning};
    use core::fmt;
    use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
    use serde::ser::{SerializeMap, Serializer};

    const COUNT: usize = Tuning::FIELDS.len();

    /// One part of a tuning, as a map of its fields.
    struct PartRef<'a>(&'a Tuning, Part);

    impl serde::Serialize for PartRef<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let fields = || Tuning::FIELDS.iter().filter(|f| f.part == self.1);
            let mut map = s.serialize_map(Some(fields().count()))?;
            for f in fields() {
                map.serialize_entry(f.name, &self.0.get(f.name).unwrap_or(f.default))?;
            }
            map.end()
        }
    }

    impl serde::Serialize for Tuning {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut map = s.serialize_map(Some(Part::ALL.len()))?;
            for part in Part::ALL {
                map.serialize_entry(part.name(), &PartRef(self, part))?;
            }
            map.end()
        }
    }

    /// A key of a tuning object: a part, or a field (its index in
    /// [`Tuning::FIELDS`]).
    enum Key {
        Part(Part),
        Field(usize),
        Unknown,
    }

    /// Reads a key at the top (`None`: a part or any field) or inside a
    /// part (only that part's fields).
    struct KeySeed(Option<Part>);

    impl<'de> DeserializeSeed<'de> for KeySeed {
        type Value = Key;
        fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Key, D::Error> {
            d.deserialize_str(self)
        }
    }

    impl Visitor<'_> for KeySeed {
        type Value = Key;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("the name of a tuning field or part")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Key, E> {
            if self.0.is_none()
                && let Some(part) = Part::from_name(v)
            {
                return Ok(Key::Part(part));
            }
            let found = Tuning::FIELDS
                .iter()
                .position(|f| f.name == v && self.0.is_none_or(|p| f.part == p));
            Ok(found.map_or(Key::Unknown, Key::Field))
        }
    }

    /// Reads a tuning object (or, with a part, one part's object) onto a
    /// tuning, noting each field it sets.
    struct Fill<'a> {
        tuning: &'a mut Tuning,
        seen: &'a mut [bool; COUNT],
        part: Option<Part>,
    }

    impl<'de> DeserializeSeed<'de> for Fill<'_> {
        type Value = ();
        fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_map(self)
        }
    }

    impl<'de> Visitor<'de> for Fill<'_> {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an object of tuning fields")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            while let Some(key) = map.next_key_seed(KeySeed(self.part))? {
                match key {
                    Key::Part(part) => map.next_value_seed(Fill {
                        tuning: &mut *self.tuning,
                        seen: &mut *self.seen,
                        part: Some(part),
                    })?,
                    Key::Field(i) => {
                        let name = Tuning::FIELDS[i].name;
                        if core::mem::replace(&mut self.seen[i], true) {
                            return Err(de::Error::custom(format_args!(
                                "tuning field `{name}` given twice"
                            )));
                        }
                        let value: f32 = map.next_value()?;
                        self.tuning.set(name, value);
                    }
                    Key::Unknown => {
                        map.next_value::<de::IgnoredAny>()?;
                    }
                }
            }
            Ok(())
        }
    }

    impl<'de> serde::Deserialize<'de> for Tuning {
        fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let mut tuning = Tuning::DEFAULT;
            let mut seen = [false; COUNT];
            Fill {
                tuning: &mut tuning,
                seen: &mut seen,
                part: None,
            }
            .deserialize(d)?;
            Ok(tuning)
        }
    }
}

impl ArticulationConfig {
    /// Whether the gesture path leaves the family of rules with bit `bit`
    /// to the overlap of gestures (see
    /// [`gesture_overlap`](Self::gesture_overlap)).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn overlaps(&self, bit: u32) -> bool {
        (self.gesture_overlap.clamp(0.0, 7.0) as u32) & bit != 0
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
    fn three_parts_of_thirty_two_seventy_one_and_forty_three() {
        let count = |part| Tuning::FIELDS.iter().filter(|f| f.part == part).count();
        assert_eq!(
            (
                count(Part::Anatomy),
                count(Part::Articulation),
                count(Part::Prosody)
            ),
            (33, 71, 43)
        );
        assert_eq!(Tuning::FIELDS.len(), 147);
        // In order: the anatomy's, the articulation's, the prosody's.
        assert!(Tuning::FIELDS.is_sorted_by_key(|f| f.part as u8));
        // One name per field, across the parts.
        for (i, f) in Tuning::FIELDS.iter().enumerate() {
            assert!(
                Tuning::FIELDS[..i].iter().all(|g| g.name != f.name),
                "{}",
                f.name
            );
            assert!(Part::from_name(f.name).is_none(), "{}", f.name);
        }
    }

    #[test]
    fn two_passes_of_sixty_seven_and_sixty_five() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (67, 65));
    }

    #[test]
    fn vowels_are_held_longer_on_the_physical_engine() {
        let hold = Tuning::FIELDS.iter().find(|f| f.name == "vowel_hold");
        assert!(hold.is_some_and(|f| f.default > 1.0 && f.lo >= 1.0));
        let c = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::C).count();
        assert_eq!(c, 15);
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
    fn the_springs_live_in_their_tiers() {
        let mut t = Tuning::DEFAULT;
        let a = &t.articulation;
        assert_eq!(a.body_rate(), a.tongue_body.stiffness);
        assert_eq!(a.body_damping(), a.tongue_body.damping);
        assert_eq!(a.articulation_rate(), a.tongue_tip.stiffness);
        assert!(t.set("body_rate", 20.0) && t.set("articulation_rate", 50.0));
        assert!(t.set("body_damping", 0.8));
        let a = &t.articulation;
        assert_eq!(
            (
                a.tongue_body.stiffness,
                a.tongue_body.damping,
                a.tongue_tip.stiffness
            ),
            (20.0, 0.8, 50.0)
        );
        // Clamped as any other field.
        t.articulation.tongue_body.damping = 3.0;
        assert_eq!(t.clamped().articulation.tongue_body.damping, 1.0);
    }

    #[test]
    fn clamping_tames_garbage() {
        let mut t = Tuning::DEFAULT;
        t.anatomy.lung_pressure = f32::NAN;
        t.articulation.lead = 99.0;
        t.anatomy.modal_gap = -1.0;
        let c = t.clamped();
        assert!((c.anatomy.lung_pressure - 8.0).abs() < f32::EPSILON);
        assert!((c.articulation.lead - 0.06).abs() < f32::EPSILON);
        assert!((c.anatomy.modal_gap - 0.004).abs() < f32::EPSILON);
    }
}
