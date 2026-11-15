//! [`Tuning`]: the one hundred and twenty-two numbers that decide how a voice turns sounds into
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
        /// How a voice turns sounds into speech: one hundred and twenty-two numbers in three
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
    /// Speed at which a closure lets go, as a share of its full swing. At
    /// 0.37, set while the tract's areas were drawn across 10.7 ms blocks,
    /// the tongue tip of a /t/ now crosses the width that hisses slowly
    /// enough for single bursts to stand up to the vowel after them ("two",
    /// "to": -1 dB in the deep voice, +1.3 dB in the elder one); at 0.5
    /// the loudest are 3 to 7 dB under it. 0.5 is the owner's choice: at
    /// 0.6 Whisper heard "background" as "back drawing".
    release_speed = 0.5, 0.1..=0.8, B;
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
    /// 1 (a speaker's accents are 2 to 5). 6 until round 10: the rules'
    /// pitch spanned 7.2 semitones over the LibriSpeech texts where the
    /// reader's spans 10.4, and stepped 1.8 semitones from vowel to vowel
    /// where his steps 3.1; at 8 (4 semitones in the default voice) they
    /// rise into an accent by 2.9 semitones, he by 2.7.
    accent_size = 8.0, 0.0..=16.0, B;
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
    /// The owner chose 0.62 by ear in round 5 and a milder 0.82 for the
    /// stressed lax vowels in round 6 ("sells", "punch"; the unstressed
    /// ones keep `unstressed_lax_length`). At 0.82 on the round 6 engine
    /// the recognizer heard more stressed /ɪ/ as /eɪ/ (vowel error in the
    /// LibriSpeech texts 28.3 % against 26.2 % at 0.62), the 30 held-out
    /// sentences' phoneme error was 23.5 % against 24.4 % and Whisper's
    /// words 65.7 % against 63.1 %, all within the sets' noise.
    lax_length = 0.82, 0.5..=1.0, C;
    /// Share of a lax vowel held on its own shape before it glides towards
    /// schwa (to its end shape in `vowel_targets`) for the rest.
    lax_nucleus = 0.67, 0.4..=1.0, C;
    /// The same as `lax_length` for an unstressed lax vowel ("it", "-ing"),
    /// which the stress already makes shorter.
    unstressed_lax_length = 0.62, 0.5..=1.0, C;

    // ----- stops into liquids, and nasal murmurs (physical engine) -------

    /// How long after a /b p g k/ in front of an /l r/ (as in "blue",
    /// "play", "glue", "cream") opens the mouth the liquid's tongue tip is
    /// held back, s: the release goes into the liquid's tongue body without
    /// its narrowing, so a burst and a stretch of the stop's own formant
    /// transition are heard before the liquid takes over. `0` sends the
    /// liquid with the release, as before.
    liquid_transition = 0.018, 0.0..=0.04, B;
    /// How far the velum opens for /n/, against /m ŋ/ (1: as far). Lowered
    /// less, the murmur of /n/ grew weaker and was lost to other sounds
    /// (at 0.7, 86 of 94 /n/ before consonants heard right against 85,
    /// with fewer heard as /ŋ/ but more as nothing or as /l/), so it opens
    /// as far by default.
    alveolar_velum = 1.0, 0.5..=1.0, A;
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

    // ----- A and B: the weak fricatives and the end of a sibilant -------

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

    // ----- C: vowel lengths and glides by stress (round 6) ---------------

    /// Shortest the held part of a stressed lax vowel may be, seconds (its
    /// nucleus: `lax_nucleus` of its length, before its glide towards
    /// schwa), however fast the speech; `0` for none. The owner asked for
    /// 60 ms with the milder stressed length: of the 136 stressed lax
    /// vowels of the judge's sentences it lengthens none at the default
    /// rate, 39 at 1.2 times the rate and 93 at 1.4 times (at `lax_length`
    /// 0.62 it would lengthen 39 already at the default rate).
    lax_floor = 0.06, 0.0..=0.09, C;
    /// Share of a stressed /æ/ held on its own shape before it glides
    /// towards schwa for the rest ("planks", "background"); `1.0` (the
    /// default) holds it to the end. At 0.55 it ends about a fifth of the
    /// way to schwa (`tests/vowel_glide_gate.rs`; at 0.67, the lax vowels'
    /// share, 13 %), and the recognizer heard more /æ/ as /ɛ/ (with it
    /// the 30 held-out sentences' phoneme error was 24.9 % against 23.5 %
    /// and vowel error in the LibriSpeech texts 30.1 % against 28.3 %). A
    /// stressed /æ/ reaches its formants without it ("planks": 229 ms).
    ae_nucleus = 1.0, 0.4..=1.0, C;
    /// Length of a stressed /ɪ/ ("slid", "big"), as a share of a tense
    /// vowel's: it keeps the short length while the other stressed lax
    /// vowels take `lax_length`. At 0.82 the recognizer heard the stressed
    /// /ɪ/ of the LibriSpeech texts as /eɪ/ ("slid" as "slayed"): a long
    /// /ɪ/ that glides towards schwa is what /eɪ/ sounds like backwards.
    ih_length = 0.65, 0.5..=1.0, C;
    /// Added to the opening the tongue leaves beside its tip for /l/,
    /// diameter units (physical engine). At the classic engine's opening
    /// the first formant of /l/ stood near 500 Hz, where a speaker's has
    /// 350 to 400, and an /l/ running into /ɪ/ or /u/ was heard as the
    /// start of a diphthong: "slid" as "slayed", "blue" as "b-uh-ue".
    lateral_gap = -0.12, -0.3..=0.1, A;
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

    // ----- A: a softer voice off the accent (round 8) -------------------

    /// Corner of the one-pole low-pass on the voice at full softness, Hz
    /// (physical engine; see `Trombone::set_fold_softness`). At softness
    /// `s` the corner is this over `s`. With the default softnesses the
    /// nasals, liquids and unaccented vowels of the LibriSpeech texts come
    /// out 6.1, 4.0 and 0.1 dB darker in 2-4 kHz against 0-1 kHz than the
    /// accented vowels (the reader: 5.6, 5.1 and 0.6; at 2500 Hz 6.7, 4.6
    /// and 0.4).
    soft_corner = 3000.0, 1000.0..=8000.0, A;
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

    // ----- B: from one shape to the next (round 8) -----------------------

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

    // ----- B and C: the front of the tongue (round 9) -------------------

    /// Share of an /u/ after /t d s z/ held on its fronted shape (F2 1600
    /// Hz against the plain /u/'s 1000) before it glides back towards the
    /// plain /u/ for the rest (physical engine; see `speech::fronts_u`):
    /// the LibriSpeech reader's /u/ after /t d/ starts at 1700 to 2200 Hz
    /// and ends some 400 Hz lower. `1.0` holds it to the end (held, the
    /// recognizer heard 3 to 6 of 20 such /u/ as /i/), `0` makes every /u/
    /// the plain back one, as until round 9.
    fronted_nucleus = 0.5, 0.0..=1.0, C;
    /// Length of an alveolar tap, s (physical engine): a /t d/ between a
    /// vowel and an unstressed one ("slid on", "get it", "water", "city")
    /// is one voiced segment, the tongue tip thrown against the ridge and
    /// let go, with no closure held, no pressure built behind it, no burst
    /// and no breath after it (see `speech::taps`). At 0.02 the mouth is
    /// shut for 3 to 7 ms and the level dips 6 to 15 dB under the vowels
    /// beside it in 25 ms frames (the LibriSpeech reader's taps: 5 to 13
    /// dB, voiced throughout). `0` makes them stops again.
    tap_length = 0.02, 0.0..=0.05, B;
    /// How long before an alveolar tap begins its tip sets off, s
    /// (physical engine), in place of `lead`: the tap is a quick throw of
    /// the tip, which reaches the ridge from a vowel in some 10 to 18 ms.
    /// Sent with the usual head start (`lead`, 0.03), it shut the mouth 10
    /// to 15 ms before the tap was due, for 15 to 25 ms, and the level
    /// dipped 12 to 29 dB.
    tap_lead = 0.015, 0.0..=0.04, B;

    // ----- A: the voice's source (round 9) -------------------------------

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
    /// Seconds for the arytenoids to bring the glottis back together
    /// behind the folds once a voiceless sound is over (physical engine;
    /// opening it takes `abduction_time`, and the shorter of the two
    /// applies). With both at 20 ms the spread glottis was still closing
    /// as the vowel after a voiceless consonant began, and the vowel took
    /// 19 ms from its voicing onset to come within 3 dB of its peak (the
    /// LibriSpeech reader's: 7.5 ms), growing for 40 ms.
    spread_closing = 0.012, 0.004..=0.05, A;
    /// How far the breath lets the level of each vowel sink from its start
    /// to its end, dB (physical engine; the lungs take it back in the
    /// sounds after it). `0` holds it.
    vowel_decline = 2.5, 0.0..=6.0, A;
    /// Where in an accented vowel its pitch accent peaks, as a share of the
    /// vowel (physical engine): the accent rises from the start of its
    /// syllable to there and sinks back over `accent_decay` (a statement's
    /// last accent peaks as its vowel begins, whatever this says). At `1`,
    /// as until round 9, it rose from the vowel's start to its end, and the
    /// accented vowels of the LibriSpeech texts rose by 1.2 semitones
    /// (median) where the reader's go either way (median 0.0).
    accent_peak = 0.55, 0.0..=1.0, B;

    // ----- following a speaker's schedule (round 9) ----------------------

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
    /// How far a followed schedule may move a vowel's F1 and F2 from the
    /// voice's own, a share either way (see `speech::PhoneSpan::formants`).
    follow_formant_range = 0.15, 0.0..=0.3, C;

    // ----- the rules' prosody against speakers' (round 10) ----------------

    /// How much quieter an unaccented vowel is than an accented one beyond
    /// what its pitch and length give, dB of level (physical engine): the
    /// lungs push less through it, and the consonants between two vowels
    /// take the breath of the line from one to the other (see
    /// `speech::weigh_syllables`). `0` breathes every syllable alike. The
    /// LibriSpeech reader's accented vowels stand 2.7 dB over his others,
    /// the rules' 1.2 without this and 1.9 with it; at 3 dB the 30
    /// held-out sentences' phoneme error went from 25.5 to 26.6 %.
    unaccented_level = 1.5, 0.0..=6.0, A;
    /// How much longer the consonants after the last vowel of a phrase are
    /// (physical engine), on top of that vowel's own lengthening: the
    /// LibriSpeech reader draws out his whole last syllable, its closing
    /// consonants most (1.2 to 1.6 times the rules' at his pace, the vowel
    /// 1.2 times), where the rules lengthened the vowel alone.
    final_coda_stretch = 1.25, 1.0..=1.8, B;
    /// Pause after a sentence, s (physical engine; the classic engine keeps
    /// 0.38, a comma 0.18 in both): the two reference recordings pause 0.54
    /// and 0.62 s between their two sentences, the owner 0.74.
    sentence_pause = 0.55, 0.3..=1.0, B;
    /// Share by which the folds are held more loosely through the vowel of
    /// a function word (physical engine; see `speech::shade_syllables`).
    light_laxness = 0.08, 0.0..=0.3, A;

    // ----- the shape of the source tilt (round 10) -----------------------

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

    // ----- the lateral, taps and close vowels (round 10) -----------------

    /// Cells in front of its place over which the side channels of an /l/
    /// run (physical engine; see `tract::carve_lateral`): the air of a
    /// lateral leaves round the sides of a tongue whose tip touches the
    /// ridge, between them and the teeth. At 3 the channel ends at the
    /// front of the ridge; at 1 (a pinch at the place) the /l/ after a
    /// consonant was heard as /r/ in 8 of 27 short phrases, at 3 and 4 in
    /// none (at 4, into the teeth's cell, the lips of "a blue" were 0.36
    /// apart 10 ms after the release, where they open to 0.63).
    lateral_channel = 3.0, 1.0..=5.0, A;

    // ----- the rhythm of the rules and the transfer's floor (round 11) ----

    /// How much longer a consonant that opens a stressed syllable of a
    /// content word is than its own length (physical engine; see
    /// `speech::rhythm`, which has the measurements): the LibriSpeech
    /// reader's, the two references' and the owner's are 1.30 to 1.37
    /// times the rules' at their pace with the class of each consonant
    /// fitted alongside, 1.11 to 1.27 without. Not the weak fricatives
    /// /f v θ ð h/, which are long in the rules already. At 1.3 (with
    /// `cluster_length` alone) the 30 held-out sentences' phoneme error
    /// went from 25.9 to 23.4 %, the 30 tuning sentences' from 22.0 to
    /// 18.6 %, the vowels heard wrong in the LibriSpeech texts from 25.5 to
    /// 22.4 %, but the android's /b/ released 4.0 dB under the vowel after
    /// it on average, at the bound of `stop_releases_are_bursts_not_clicks`
    /// (at 1.2 -4.3 dB, before -4.6): the longer closure builds more
    /// pressure. `1` makes them as long as anywhere else.
    stressed_onset_length = 1.2, 1.0..=1.6, B;
    /// Length of each consonant of a cluster against a single one's
    /// (physical engine): the reader's are 0.81 of his single ones (the
    /// references' 0.94), where the rules made them alike.
    cluster_length = 0.85, 0.6..=1.0, B;
    /// Length of the consonants of a function word, its stops excepted
    /// (they keep their closure and release), against their own (physical
    /// engine): the reader's /ð v/ of "the", "of" are 0.78 of the rules'.
    /// The reader's 0.85 with `schwa_length` at his 0.75 lost Whisper 7.8
    /// points of the held-out sentences' words (it gained 3.7 on the
    /// tuning sentences); at 0.9 and 0.85 the two sets moved -2.6 and
    /// +5.5.
    function_consonant_length = 0.9, 0.5..=1.0, B;
    /// Length of /ə/ against an unstressed vowel's (physical engine): the
    /// speakers' are 0.69 to 0.76 of the rules', in function and content
    /// words alike; milder than theirs, see `function_consonant_length`.
    schwa_length = 0.85, 0.5..=1.0, C;
    /// Length of a diphthong against what `diphthong_length`, `vowel_hold`
    /// and the stress give it (physical engine): the reader's are 0.67 of
    /// the rules' (a stressed /oʊ eɪ/ was 226 to 266 ms, his 70 to 250 at
    /// the rules' pace).
    diphthong_hold = 0.75, 0.5..=1.0, C;
    /// Shortest a followed schedule may make a phone the floor of
    /// `follow_shortest` does not hold (physical engine; see
    /// `speech::rhythm::held_in_transfer`), a share of its rule duration:
    /// the unstressed vowels, and the nasals, liquids, glides and weak
    /// fricatives that are not in a cluster of a content word. Left off
    /// one class at a time on the 16 LibriSpeech anchors with the reader's
    /// own timing, the floor mattered most in the function words (their
    /// vowels: phoneme error 22.5 to 24.7 %; their consonants: 25.2 %,
    /// Whisper 81.8 to 74.6 %) and not on the stressed vowels or clusters
    /// (22.2 and 22.1 %); held only on the stressed vowels, stops,
    /// sibilants and clusters (this at `0`) the phoneme error was 27.6 %
    /// and Whisper 74.6 %, at 0.6 23.5 and 80.9 %, at 0.5 25.1 and 75.6 %.
    follow_shortest_light = 0.7, 0.0..=1.0, B;
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
    fn two_passes_of_fifty_eight_and_fifty_three() {
        let a = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::A).count();
        let b = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::B).count();
        assert_eq!((a, b), (58, 53));
    }

    #[test]
    fn vowels_are_held_longer_on_the_physical_engine() {
        let hold = Tuning::FIELDS.iter().find(|f| f.name == "vowel_hold");
        assert!(hold.is_some_and(|f| f.default > 1.0 && f.lo >= 1.0));
        let c = Tuning::FIELDS.iter().filter(|f| f.pass == Pass::C).count();
        assert_eq!(c, 11);
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
