//! [`Trombone`]: the complete voice — glottis, tract, noise and resampling —
//! behind a small, real-time safe API.

use crate::INTERNAL_RATE;
use crate::glottis::{DEFAULT_ATTACK, DEFAULT_RELEASE, Glottis};
use crate::math::midi_to_hz;
use crate::noise::{BandPass, Rng, Simplex};
use crate::phoneme::{Articulation, Phoneme};
use crate::phys::source::{Source, SourceKind, WHISPER_TENSENESS};
use crate::phys::{CM_H2O, Drive, Phys};
use crate::tract::{Constriction, Dynamics, HISS_WIDEST, MOVEMENT_SPEED, N, NOSE_LEN, Tract};
use crate::tuning::Tuning;
use core::f32::consts::PI;

/// Internal samples per parameter block (classic engine): 10.7 ms.
const BLOCK: u32 = 512;
/// Internal samples per parameter block in the physical engine: 2.7 ms.
/// What the articulators are told takes effect at the start of a block,
/// so at 512 samples a gesture of 20 to 30 ms was made in two or three
/// steps, and where they fell depended on what had been said before it.
const PHYSICAL_BLOCK: u32 = 128;
/// Constriction slot used by [`Trombone::articulate`] for the main closure.
pub const SLOT_MAIN: usize = 0;
/// Constriction slot used by [`Trombone::articulate`] for lip rounding.
pub const SLOT_LIPS: usize = crate::tract::ROUNDING_SLOT;

/// Output level of the physical engine: radiated flow derivative
/// (cm³/s²) to sample units, set so both engines are about equally loud.
const PHYSICAL_GAIN: f32 = 2.5e-7;

/// Widest opening (touch-diameter units) that still counts as a fricative;
/// wider consonant openings are approximants.
pub(crate) const FRICATIVE_WIDEST: f32 = HISS_WIDEST;

/// Places of the tongue (tract index) behind the alveolar ridge whose
/// rounded consonants are /ʃ ʒ tʃ dʒ/ (see [`Tuning::postalveolar_lips`]);
/// /r/, further back, keeps its rounding.
const POSTALVEOLAR_PLACE: core::ops::RangeInclusive<f32> = 32.5..=34.5;
/// Where the tongue tip closes for /n/ (the alveolar ridge, and the teeth
/// for an /n/ made there), as a constriction index.
const ALVEOLAR_NASAL_PLACE: core::ops::RangeInclusive<f32> = 34.5..=40.0;

/// Which model turns the articulation into sound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Engine {
    /// Pink Trombone as Neil Thapen wrote it: a scripted glottal pulse into
    /// a lossless tube, with noise and bursts added by rule.
    #[default]
    Classic,
    /// The same tract in physical units: lung pressure through a glottis,
    /// frequency-dependent losses, yielding walls and lip radiation.
    Physical,
}

/// What moves the vocal folds in the physical engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GlottalSource {
    /// The opening between the folds follows a prescribed pulse at the
    /// requested pitch; the air flow through it is real. Always in tune.
    Pulse,
    /// Self-oscillating folds: masses on springs driven by the air
    /// pressure around them. Pitch comes from their tension, and voicing
    /// starts, weakens and stops by itself as pressures change. Notes the
    /// folds cannot reach (below or above [`Trombone::fold_range`]) are
    /// sung with the pulse instead, unless [`Fallback::Never`] is set;
    /// [`Gauges::sounding`] tells which is singing.
    #[default]
    Folds,
}

/// When the self-oscillating folds hand the voice to the pulse source (see
/// [`GlottalSource`]). To sing with the pulse alone, choose
/// [`GlottalSource::Pulse`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Fallback {
    /// Notes outside the folds' range ([`Trombone::fold_range`]) are sung
    /// by the pulse: below it as vocal fry, above it as a falsetto the
    /// folds have no register for. Inside it the folds sing every note,
    /// even one they keep stalling on. The range is about 56 to 288 Hz for
    /// the native-size voice at modal tenseness and scales with the
    /// inverse of the size, so a small voice speaking high (the critter
    /// preset, up to about 520 Hz) is sung by the pulse at its peaks.
    #[default]
    Auto,
    /// The folds sing everything; a note they cannot hold is not sung.
    Never,
}

/// Readings from inside the physical engine, for measurements and
/// displays. All zero while the classic engine is in use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gauges {
    /// The mouth is shut somewhere in front of the pharynx, as the sound
    /// sees it (it follows the articulators by up to one parameter block).
    pub mouth_shut: bool,
    /// Air flow through the glottis, cm³/s.
    pub glottal_flow: f32,
    /// Pressure just below the vocal folds, cmH2O.
    pub subglottal_pressure: f32,
    /// Pressure just above the vocal folds, cmH2O.
    pub supraglottal_pressure: f32,
    /// Air flow out of the lips, cm³/s.
    pub lip_flow: f32,
    /// Air flow out of the nostrils, cm³/s.
    pub nose_flow: f32,
    /// Flow radiated through the walls of the throat and cheeks, cm³/s.
    pub skin_flow: f32,
    /// Flow the jet of /f v θ ð/ radiates as where it strikes the edges of
    /// the upper teeth and the lip, cm³/s (a dipole's sound, as the flow
    /// that would make it; it does not pass through the tract).
    pub edge_flow: f32,
    /// Area between the vocal folds, cm².
    pub glottal_area: f32,
    /// The part of [`glottal_area`](Self::glottal_area) between the
    /// vibrating edges, cm²: zero while they touch. The rest is the
    /// standing gap behind them (a breathy voice's leak).
    pub edge_area: f32,
    /// Which source is making the voice: the folds or the pulse (see
    /// [`GlottalSource`]); `None` while no voice is wanted.
    pub sounding: Option<GlottalSource>,
    /// Pitch the self-oscillating folds are actually producing, Hz (`0`
    /// while they are not vibrating or the pulse source is in use).
    pub fold_pitch: f32,
    /// Noise pressure of the strongest jet in the mouth, cmH2O (`0` while
    /// no stream there is fast enough to be turbulent).
    pub jet_noise: f32,
    /// Speed of that jet, cm/s.
    pub jet_speed: f32,
    /// The tract cell that jet leaves.
    pub jet_cell: usize,
    /// Furthest the hiss regulator has widened a fricative gap that would
    /// not hiss, in [`Constriction`] diameter units.
    pub gap_nudge: f32,
    /// How far it is opening a vowel shape that hisses, same units.
    pub vowel_relief: f32,
}

/// One Pink Trombone voice.
///
/// The model runs at a fixed [`INTERNAL_RATE`] (tract at twice that) and is
/// resampled to the output rate, so formants and timing do not depend on the
/// host sample rate. Parameters update every 512 internal samples and are
/// interpolated in between, independent of the host's buffer size.
///
/// No method allocates; [`process`](Self::process) is safe to call from an
/// audio callback.
#[derive(Clone, Debug)]
pub struct Trombone {
    glottis: Glottis,
    tract: Tract,
    engine: Engine,
    phys: Phys,
    source: Source,
    /// Radiated flow at the previous internal sample (physical engine).
    phys_flow: f32,
    /// The physical engine's output after the low-pass of soft folds.
    soft_out: f32,
    glottal_area: f32,
    regulated: bool,
    tuning: Tuning,
    /// Vowel shapes held in place of the built-in ones (physical engine).
    own_vowels: crate::vowel_size::OwnShapes,
    /// One vowel shape held over those for the sound being spoken (see
    /// [`Trombone::hold_speech_vowel`]).
    speech_vowel: Option<(crate::vowel_targets::VowelKey, crate::inverse::Shape)>,
    simplex: Simplex,
    rng: Rng,
    frication: BandPass,
    /// Internal samples per parameter block (see [`Trombone::set_block`]).
    block: u32,
    block_pos: u32,
    /// Internal samples since the hiss regulator last looked (physical
    /// engine): it reads the streams' averages over a long block (10.7 ms,
    /// a pitch period or more) whatever the control block's length.
    since_regulated: u32,

    sample_rate: f32,
    internal_rate: f32,
    tract_length: f32,
    step: f64,
    frac: f64,
    prev: f32,
    cur: f32,

    dc_x: f32,
    dc_y: f32,
    dc_r: f32,
    gain: f32,
}

impl Trombone {
    /// A voice for the given output sample rate with the default seed.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is not positive and finite.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::with_seed(sample_rate, 0x5EED_7B0E)
    }

    /// A voice with an explicit noise seed. Equal seeds render equal audio.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is not positive and finite.
    #[must_use]
    pub fn with_seed(sample_rate: f32, seed: u32) -> Self {
        assert!(
            sample_rate.is_finite() && sample_rate > 0.0,
            "sample_rate must be positive"
        );
        let mut rng = Rng::new(seed);
        let simplex = Simplex::new(&mut rng);
        let mut phys = Phys::new();
        phys.seed(seed ^ 0x9E37_79B9);
        Self {
            glottis: Glottis::new(),
            tract: Tract::new(),
            engine: Engine::Classic,
            phys,
            source: {
                let mut source = Source::new();
                source.kind = SourceKind::Folds;
                source
            },
            phys_flow: 0.0,
            soft_out: 0.0,
            glottal_area: 0.0,
            regulated: true,
            tuning: Tuning::DEFAULT,
            own_vowels: crate::vowel_size::OwnShapes::NONE,
            speech_vowel: None,
            simplex,
            rng,
            frication: BandPass::new(1000.0, 0.5, INTERNAL_RATE),
            block: BLOCK,
            block_pos: 0,
            since_regulated: BLOCK,
            sample_rate,
            internal_rate: INTERNAL_RATE,
            tract_length: 1.0,
            step: f64::from(INTERNAL_RATE) / f64::from(sample_rate),
            frac: 1.0,
            prev: 0.0,
            cur: 0.0,
            dc_x: 0.0,
            dc_y: 0.0,
            dc_r: 1.0 - 2.0 * PI * 10.0 / sample_rate,
            gain: 1.0,
        }
    }

    /// Output sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Silence all sound immediately; articulation and settings are kept.
    pub fn reset(&mut self) {
        self.glottis.reset();
        self.tract.reset_waves();
        self.phys.reset();
        self.source.reset();
        self.phys_flow = 0.0;
        self.soft_out = 0.0;
        self.frication.reset();
        self.block_pos = 0;
        self.since_regulated = BLOCK;
        self.frac = 1.0;
        self.prev = 0.0;
        self.cur = 0.0;
        self.dc_x = 0.0;
        self.dc_y = 0.0;
    }

    // ----- engine ------------------------------------------------------

    /// Choose the sound model. Switching silences the voice; articulation
    /// and settings are kept.
    pub fn set_engine(&mut self, engine: Engine) {
        if engine == self.engine {
            return;
        }
        self.engine = engine;
        self.tract.dynamics = match engine {
            Engine::Classic => Dynamics::Linear,
            Engine::Physical => Dynamics::Damped,
        };
        self.tract.jaw = engine == Engine::Physical;
        self.sync_wander();
        self.set_block(match engine {
            Engine::Classic => BLOCK,
            Engine::Physical => PHYSICAL_BLOCK,
        });
        self.reset();
    }

    /// How often the articulators, the folds' controls and the tract's
    /// areas are brought up to date, in internal samples (48 kHz at tract
    /// length 1): 512 is 10.7 ms. For measuring how far the sound depends
    /// on it; the classic engine is only itself at 512.
    #[doc(hidden)]
    pub fn set_block(&mut self, samples: u32) {
        self.block = samples.clamp(16, 4096);
        self.block_pos = 0;
        self.since_regulated = BLOCK;
        #[allow(clippy::cast_precision_loss)]
        let share = self.block as f32 / BLOCK as f32;
        self.tract.block_share = share;
        self.phys.block_share = share;
    }

    /// The sound model in use.
    #[must_use]
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Choose what moves the vocal folds in the physical engine.
    pub fn set_glottal_source(&mut self, source: GlottalSource) {
        let kind = match source {
            GlottalSource::Pulse => SourceKind::Kinematic,
            GlottalSource::Folds => SourceKind::Folds,
        };
        if kind != self.source.kind {
            self.source.kind = kind;
            self.source.reset();
        }
    }

    /// How this voice turns sounds into speech: the breath and folds, the
    /// noise of jets, consonant gaps and articulator speeds (see
    /// [`Tuning`]). Values outside their allowed range are pulled into it.
    /// This also sets the lung pressure. The timing fields act where speech
    /// is compiled ([`Prosody`](crate::speech::Prosody) carries them).
    pub fn set_tuning(&mut self, tuning: &Tuning) {
        let tuning = tuning.clamped();
        self.tuning = tuning;
        self.source.set_tuning(&tuning);
        self.phys.set_tuning(&tuning);
        self.tract.motion = tuning;
        self.sync_wander();
    }

    /// The physical engine's pitch keeps a clock of its own and flutters
    /// as `Tuning::pitch_flutter` says; the classic engine's moves as it
    /// always has.
    fn sync_wander(&mut self) {
        self.glottis.flutter = match self.engine {
            Engine::Classic => 0.0,
            Engine::Physical => self.tuning.pitch_flutter,
        };
        self.glottis.own_clock = self.engine == Engine::Physical;
    }

    /// The tuning in use.
    #[must_use]
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// Switch the hiss regulator of the physical engine on or off (it is on
    /// by default). While on, the voice corrects itself from what its own
    /// air is doing: a gap that is meant to hiss but does not is widened a
    /// little, and a vowel that hisses is opened a little; neither by more
    /// than 0.15 in [`Constriction`] diameter. What it learns about a
    /// place of articulation it keeps until the voice changes size.
    pub fn set_regulation(&mut self, on: bool) {
        self.regulated = on;
        if !on {
            self.tract.forget();
        }
    }

    /// Lung pressure of the physical engine while sounding, in cmH2O
    /// (default 8, clamped to 1‥40). More pressure is louder and brighter;
    /// below a few cmH2O the folds stop vibrating.
    pub fn set_lung_pressure(&mut self, cm_h2o: f32) {
        if cm_h2o.is_finite() {
            self.source.set_lung_pressure(cm_h2o.clamp(1.0, 40.0));
        }
    }

    /// Lung pressure setting, cmH2O.
    #[must_use]
    pub fn lung_pressure(&self) -> f32 {
        self.source.lung_pressure()
    }

    /// Hold the self-oscillating folds at a fixed tension (relative to
    /// their standard stiffness) and rest gap (half-width, cm) instead of
    /// deriving them from pitch and tenseness. For measuring the folds;
    /// `None` returns control to the pitch and tenseness settings.
    #[doc(hidden)]
    pub fn override_folds(&mut self, tension: Option<f32>, gap: Option<f32>) {
        self.source.tension_override = tension;
        self.source.gap_override = gap;
    }

    /// Replace the folds by body–cover folds held at fixed muscle
    /// activations (cricothyroid, thyroarytenoid, lateral cricoarytenoid,
    /// each `0‥1`), after Titze & Story (2002). For measuring them; `None`
    /// returns to the standard folds. Use with
    /// [`override_folds`](Self::override_folds)`(Some(1.0), ..)`.
    #[doc(hidden)]
    pub fn override_muscles(&mut self, muscles: Option<(f32, f32, f32)>) {
        self.source.muscle_override = muscles;
        self.source.set_size(self.tract_length);
    }

    /// Switch parts of the fold control off, to find out which of them a
    /// behaviour comes from: the pitch trim, the stall watch, and the
    /// pressure the tract above and the trachea below send back to the
    /// folds (without it they feel only the lungs and the open air). For
    /// measuring; everything is on by default.
    #[doc(hidden)]
    #[allow(clippy::fn_params_excessive_bools)] // four independent switches
    pub fn probe_folds(&mut self, trim: bool, monitor: bool, supraglottal: bool, subglottal: bool) {
        self.source.probe = crate::phys::source::Probe {
            trim,
            monitor,
            supraglottal,
            subglottal,
        };
    }

    /// When the folds hand the voice to the pulse source (default
    /// [`Fallback::Auto`]). [`Gauges::sounding`] shows which is singing.
    pub fn set_fallback(&mut self, fallback: Fallback) {
        self.source.fallback = fallback;
    }

    /// When the folds hand the voice to the pulse source.
    #[must_use]
    pub fn fallback(&self) -> Fallback {
        self.source.fallback
    }

    /// Lowest and highest pitch, Hz, the self-oscillating folds can sing at
    /// the present size and tenseness; outside it the pulse takes over
    /// (see [`Fallback`]).
    #[must_use]
    pub fn fold_range(&self) -> (f32, f32) {
        self.source.range()
    }

    /// What moves the vocal folds in the physical engine.
    #[must_use]
    pub fn glottal_source(&self) -> GlottalSource {
        match self.source.kind {
            SourceKind::Kinematic => GlottalSource::Pulse,
            SourceKind::Folds => GlottalSource::Folds,
        }
    }

    /// Cross-section scale of the physical engine's tract (`1.0` reads the
    /// articulation model's diameters as centimetres). The classic engine
    /// only depends on area ratios and ignores it.
    pub fn set_area_scale(&mut self, scale: f32) {
        if scale.is_finite() {
            self.phys.set_area_scale(scale.clamp(0.1, 4.0));
        }
    }

    /// Cross-section scale (see [`set_area_scale`](Self::set_area_scale)).
    #[must_use]
    pub fn area_scale(&self) -> f32 {
        self.phys.area_scale()
    }

    // ----- voicing and pitch -------------------------------------------

    /// Start phonating (the original's "touching the voicebox").
    pub fn note_on(&mut self) {
        self.glottis.voiced = true;
    }

    /// Stop phonating; the voice fades out.
    pub fn note_off(&mut self) {
        self.glottis.voiced = false;
    }

    /// Start phonating at a MIDI note.
    pub fn note_on_midi(&mut self, note: f32) {
        self.set_note(note);
        self.note_on();
    }

    /// Set the voicing gate directly.
    pub fn set_voiced(&mut self, voiced: bool) {
        self.glottis.voiced = voiced;
    }

    /// Whether the voicing gate is on.
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.glottis.voiced
    }

    /// Hold the glottis open behind the folds by `cm2` (cm² at native size,
    /// `0‥1`), as a speaker spreads it for a voiceless consonant: on top of
    /// whatever the voicing gate does, and moving at the arytenoids' pace
    /// ([`Tuning::abduction_time`]). With the gate on, a small opening
    /// leaves the folds vibrating beside a stream of breath (a voiced
    /// fricative); a large one with the gate off is the wide glottis of an
    /// aspirated stop. Physical engine only; `0` (the default) leaves the
    /// glottis to the voicing gate.
    pub fn set_glottal_spread(&mut self, cm2: f32) {
        if cm2.is_finite() {
            self.source.spread_target = cm2.clamp(0.0, 1.0);
        }
    }

    /// Push with `share` (`0‥1.5`) of the lung pressure
    /// ([`set_lung_pressure`](Self::set_lung_pressure)): a speaker's
    /// breath building up at the start of a phrase and dying away at its
    /// end, while the lung pressure setting stays the voice's. The pressure
    /// follows within a few tens of milliseconds. Physical engine only; `1`
    /// by default.
    pub fn set_breath(&mut self, share: f32) {
        if share.is_finite() {
            self.source.breath = share.clamp(0.0, 1.5);
        }
    }

    /// Slacken the cover of the folds by `share` (`0‥0.5`) of its tension,
    /// as a speaker does in a voiced stop: they then keep vibrating with
    /// less pressure across them, and their pitch dips a little. Physical
    /// engine only; `0` by default.
    pub fn set_fold_slack(&mut self, share: f32) {
        if share.is_finite() {
            self.source.slack_target = share.clamp(0.0, 0.5);
        }
    }

    /// Soften the voice, `0` (firm, the default) to `1`, as a speaker's
    /// voice is softer on nasals, liquids and unaccented syllables than on
    /// an accented vowel: the voice passes a one-pole low-pass whose corner
    /// is [`Tuning::soft_corner`] at `1` and moves up out of hearing
    /// towards `0`, so the spectrum above 2 kHz falls faster. The folds
    /// themselves are left alone: how far apart they are held, the
    /// pressure they need to start and every note they can sing stay as
    /// the tenseness sets them. The change is spread over some 10 ms.
    /// Physical engine only.
    ///
    /// Why not in the folds: a softer contact made them close for longer
    /// and lowered H1-H2 (by 3.4 dB at a third of the stiffness), edges
    /// that meet progressively along their length closed faster under the
    /// suction of the narrowing slit and at 0.012 cm sent the 210 Hz voice
    /// into a regime where its edges never met, a low-pass on the glottal
    /// area put the flow behind the folds' motion and pumped them louder
    /// and brighter, and one on the flow the tract takes up was given back
    /// by the tract's pressure on the folds (−0.3 to −2 dB at 2 to 4 kHz).
    /// The high frequencies of this voice come from the flow's ripple under
    /// the tract's own pressure, not from the instant of closure.
    pub fn set_fold_softness(&mut self, share: f32) {
        if share.is_finite() {
            self.source.soft_target = share.clamp(0.0, 1.0);
        }
    }

    /// Widen the pharynx actively at `cm3_per_s` (cm³/s at native size,
    /// `0‥500`), as a speaker does to keep the voice going through a voiced
    /// closure: the air the folds send into the shut tract is taken up
    /// there, so the pressure across the folds holds. Stops by itself after
    /// 15 cm³ until it is set to `0` again. Physical engine only; `0` by
    /// default.
    pub fn set_cavity_expansion(&mut self, cm3_per_s: f32) {
        if cm3_per_s.is_finite() {
            self.phys.expansion_target = cm3_per_s.clamp(0.0, 500.0);
        }
    }

    /// Phonate continuously, ignoring the gate, like the original web toy.
    pub fn set_always_voice(&mut self, on: bool) {
        self.glottis.always_voice = on;
    }

    /// Target pitch in Hz (clamped to 20‥2000).
    pub fn set_frequency(&mut self, hz: f32) {
        if hz.is_finite() {
            self.glottis.target_frequency = hz.clamp(20.0, 2000.0);
        }
    }

    /// Target pitch as a (fractional) MIDI note.
    pub fn set_note(&mut self, note: f32) {
        self.set_frequency(midi_to_hz(note));
    }

    /// Target pitch in Hz.
    #[must_use]
    pub fn target_frequency(&self) -> f32 {
        self.glottis.target_frequency
    }

    /// Current pitch in Hz, after glide, vibrato and jitter.
    #[must_use]
    pub fn frequency(&self) -> f32 {
        self.glottis.frequency()
    }

    /// Glottal tenseness, `0` (breathy whisper) to `1` (pressed).
    pub fn set_tenseness(&mut self, tenseness: f32) {
        if tenseness.is_finite() {
            self.glottis.target_tenseness = tenseness.clamp(0.0, 1.0);
        }
    }

    /// Glottal tenseness.
    #[must_use]
    pub fn tenseness(&self) -> f32 {
        self.glottis.target_tenseness
    }

    /// Vibrato depth (relative pitch, default `0.005`) and rate in Hz.
    pub fn set_vibrato(&mut self, amount: f32, hz: f32) {
        if amount.is_finite() && hz.is_finite() {
            self.glottis.vibrato_amount = amount.clamp(0.0, 0.5);
            self.glottis.vibrato_frequency = hz.clamp(0.0, 30.0);
        }
    }

    /// The original's large, slow pitch wobble.
    pub fn set_wobble(&mut self, on: bool) {
        self.glottis.wobble = on;
    }

    /// Glide speed: relative pitch change per 10.7 ms block (default `0.1`).
    /// Larger is faster; `f32::INFINITY` jumps.
    pub fn set_glide(&mut self, speed: f32) {
        if !speed.is_nan() {
            self.glottis.glide = speed.max(0.0);
        }
    }

    /// Voicing onset and offset rates, in intensity per second. Defaults are
    /// the original's slow, singing-like rates; speech wants about `30`.
    pub fn set_voicing_rates(&mut self, attack: f32, release: f32) {
        if attack.is_finite() && release.is_finite() {
            self.glottis.attack = attack.max(0.1);
            self.glottis.release = release.max(0.1);
        }
    }

    /// Restore the original's voicing rates.
    pub fn reset_voicing_rates(&mut self) {
        self.glottis.attack = DEFAULT_ATTACK;
        self.glottis.release = DEFAULT_RELEASE;
    }

    /// Vocal tract length relative to the model's native tract (`1.0`,
    /// about 17 cm); `0.7` is a small speaker such as a child. Clamped to
    /// `0.5‥1.6`. Shorter tracts raise all formants. CPU cost scales with
    /// `1 / length`.
    pub fn set_tract_length(&mut self, length: f32) {
        if !length.is_finite() {
            return;
        }
        let length = length.clamp(0.5, 1.6);
        if length.to_bits() == self.tract_length.to_bits() {
            return;
        }
        self.tract_length = length;
        self.internal_rate = INTERNAL_RATE / length;
        self.step = f64::from(self.internal_rate) / f64::from(self.sample_rate);
        self.glottis.set_rate(self.internal_rate);
        self.tract.set_rate(self.internal_rate);
        // A different-sized speaker is a different tube: the air in the old
        // one means nothing in the new, so the physical engine starts over
        // (a few milliseconds of silence rather than a bang).
        self.phys.set_rate(0.5 / self.internal_rate, length);
        self.phys.reset();
        self.tract.forget();
        self.source.set_size(length);
        self.source.reset();
        self.phys_flow = 0.0;
        self.soft_out = 0.0;
        self.frication = BandPass::new(1000.0, 0.5, self.internal_rate);
    }

    /// Vocal tract length (see [`set_tract_length`](Self::set_tract_length)).
    #[must_use]
    pub fn tract_length(&self) -> f32 {
        self.tract_length
    }

    /// Small random pitch jitter, `1.0` natural (default), `0.0` perfectly
    /// steady (robotic).
    pub fn set_jitter(&mut self, amount: f32) {
        if amount.is_finite() {
            self.glottis.jitter = amount.clamp(0.0, 5.0);
        }
    }

    /// Glottal intensity, `0‥1` (rises while voiced, falls when not).
    #[must_use]
    pub fn intensity(&self) -> f32 {
        self.glottis.intensity
    }

    // ----- articulation ------------------------------------------------

    /// Tongue body position (clamped to [`TONGUE_INDEX`](crate::TONGUE_INDEX))
    /// and height (clamped to [`TONGUE_DIAMETER`](crate::TONGUE_DIAMETER)).
    pub fn set_tongue(&mut self, index: f32, diameter: f32) {
        if index.is_finite() && diameter.is_finite() {
            self.tract.set_tongue(index, diameter);
        }
    }

    /// Tongue body `(index, diameter)`.
    #[must_use]
    pub fn tongue(&self) -> (f32, f32) {
        (self.tract.tongue_index, self.tract.tongue_diameter)
    }

    /// Place, move, or release (`None`) a constriction. Slots
    /// `0..MAX_CONSTRICTIONS`; out-of-range slots are ignored.
    pub fn set_constriction(&mut self, slot: usize, c: Option<Constriction>) {
        self.tract.set_constriction(slot, c);
    }

    /// The live constriction in `slot`.
    #[must_use]
    pub fn constriction(&self, slot: usize) -> Option<Constriction> {
        self.tract.constriction(slot)
    }

    /// Velum target, `0` (closed, oral) to `1` (open, nasal).
    pub fn set_velum(&mut self, openness: f32) {
        if openness.is_finite() {
            self.tract.set_velum(openness);
        }
    }

    /// Current velum openness, `0‥1`.
    #[must_use]
    pub fn velum(&self) -> f32 {
        self.tract.velum_openness()
    }

    /// Articulator speed multiplier (`1.0` is the original's).
    pub fn set_articulation_speed(&mut self, factor: f32) {
        if factor.is_finite() {
            self.tract.movement_speed = MOVEMENT_SPEED * factor.clamp(0.05, 20.0);
        }
    }

    /// Move the tongue body and the back of the tongue `factor` times as
    /// fast as the [articulation speed](Self::set_articulation_speed) sets
    /// them (physical engine; clamped to `1‥3`, `1` by default): a speaker
    /// in a hurry gets the tongue to its shapes in the time a fast phone
    /// leaves. The tip, the lips and the velum keep their speed.
    pub fn set_articulation_boost(&mut self, factor: f32) {
        if factor.is_finite() {
            self.tract.boost = factor.clamp(1.0, 3.0);
        }
    }

    /// Hold `shape` for the vowel `key` in the articulations sent from now
    /// on, over the voice's own and those of
    /// [`set_vowel_shape`](Self::set_vowel_shape), until it is set again;
    /// `None` lets go of it (physical engine). The speech of a followed
    /// schedule gives each vowel, and the consonants made on its body, a
    /// shape of its own this way (see
    /// [`Utterance::follow_with`](crate::speech::Utterance::follow_with)).
    pub fn hold_speech_vowel(
        &mut self,
        shape: Option<(crate::vowel_targets::VowelKey, &crate::inverse::Shape)>,
    ) {
        self.speech_vowel = shape.map(|(k, s)| (k, *s));
    }

    /// Move to an articulation. Uses [`SLOT_MAIN`] and [`SLOT_LIPS`]; leaves
    /// voicing and pitch alone (see [`Articulation::voiced`]). On the
    /// physical engine a vowel takes the shape that suits the current
    /// [tract length](Self::set_tract_length), so its formants scale with
    /// the speaker's size; after changing the size, articulate again.
    pub fn articulate(&mut self, a: &Articulation) {
        // A consonant with a body of its own moves it at the speed of the
        // back of the tongue (physical engine).
        self.tract.quick_body = a.own_body();
        self.tract.nasal = a.velum > 0.5;
        self.tract.own_lips = a.lips.is_some() && a.constriction.is_some();
        self.tract.lateral = self.engine == Engine::Physical && a.lateral();
        // Vowel shapes are tuned per engine.
        let (a, narrowing) = match self.engine {
            Engine::Classic => (*a, None),
            // Physical vowel shapes also follow the speaker's size, and the
            // tongue root goes with the tongue body.
            Engine::Physical => {
                let mut held = self.own_vowels;
                if let Some((key, shape)) = &self.speech_vowel {
                    held.set(*key, Some(shape));
                }
                if let Some(root) = a.physical_root(self.tract_length, &held) {
                    self.tract.set_pharynx(root);
                }
                a.physical_in(self.tract_length, &held)
            }
        };
        self.tract.set_narrowing(narrowing);
        if let Some((index, diameter)) = a.tongue {
            self.set_tongue(index, diameter);
        }
        // The voice's tuning widens or narrows consonant gaps: fricatives
        // (a narrow opening) and approximants (a wider one). Closures and
        // vowel shapes are left alone.
        let gap = |c: Constriction| {
            let d = c.diameter;
            if d > 0.3 && d <= FRICATIVE_WIDEST {
                Constriction::new(c.index, (d + self.tuning.fricative_gap).max(0.32))
            } else if d > FRICATIVE_WIDEST && d < 1.3 {
                Constriction::new(
                    c.index,
                    (d + self.tuning.approximant_gap).max(FRICATIVE_WIDEST + 0.01),
                )
            } else {
                c
            }
        };
        let mut a = Articulation {
            constriction: a.constriction.map(gap),
            ..a
        };
        // The physical /ʃ ʒ tʃ dʒ/ are rounded less than the classic
        // engine's: lips nearly shut over the long front cavity behind them
        // make it ring as one narrow, low resonance (a whistle near 1.5
        // kHz), where a speaker's hiss is broad from about 2 kHz up.
        if self.engine == Engine::Physical
            && let (Some(c), Some(lips)) = (a.constriction, &mut a.lips)
            && POSTALVEOLAR_PLACE.contains(&c.index)
            && c.diameter <= FRICATIVE_WIDEST
        {
            lips.diameter = lips.diameter.max(self.tuning.postalveolar_lips);
        }
        // The physical /n/ lowers the velum less than /m ŋ/.
        let velum = match a.constriction {
            Some(c)
                if self.engine == Engine::Physical
                    && a.velum > 0.0
                    && c.diameter <= 0.3
                    && ALVEOLAR_NASAL_PLACE.contains(&c.index) =>
            {
                a.velum * self.tuning.alveolar_velum
            }
            _ => a.velum,
        };
        self.tract.set_constriction(SLOT_MAIN, a.constriction);
        self.tract.set_constriction(SLOT_LIPS, a.lips);
        self.tract.set_velum(velum);
    }

    /// Move to a phoneme's articulation (see [`articulate`](Self::articulate)).
    pub fn articulate_phoneme(&mut self, p: Phoneme) {
        self.articulate(&p.articulation());
    }

    /// Shape the vowel `key` as `shape` from now on (physical engine), in
    /// place of the built-in shape solved for its
    /// [target](crate::vowel_targets); `None` returns to the built-in one.
    /// For trying out other vowel targets: solve the shape for the voice's
    /// size with [`inverse::solve_for`](crate::inverse::solve_for), and
    /// set it again after [`set_tract_length`](Self::set_tract_length).
    /// Returns `false` (and changes nothing) for a key with no target.
    pub fn set_vowel_shape(
        &mut self,
        key: crate::vowel_targets::VowelKey,
        shape: Option<&crate::inverse::Shape>,
    ) -> bool {
        self.own_vowels.set(key, shape)
    }

    /// Hold a vowel shape of the physical engine, such as one found by
    /// [`inverse::solve`](crate::inverse::solve): tongue body, lip rounding
    /// (in [`SLOT_LIPS`]) and the narrowing along the tongue. The other
    /// constriction slots, the velum and the voice are left alone.
    pub fn set_shape(&mut self, shape: &crate::inverse::Shape) {
        self.set_tongue(shape.tongue_index, shape.tongue_diameter);
        let lips = shape
            .lips
            .map(|d| Constriction::new(crate::inverse::LIP_CELL, d));
        self.tract.set_constriction(SLOT_LIPS, lips);
        self.tract
            .set_narrowing(shape.narrowing.map(|(at, d)| Constriction::new(at, d)));
        self.tract.set_pharynx(shape.pharynx);
    }

    // ----- inspection --------------------------------------------------

    /// Current oral tract diameters, glottis to lips.
    #[must_use]
    pub fn diameters(&self) -> &[f32; N] {
        &self.tract.diameter
    }

    /// Diameters the tract is moving towards.
    #[must_use]
    pub fn target_diameters(&self) -> &[f32; N] {
        &self.tract.target
    }

    /// Nasal tract diameters, velum to nostrils.
    #[must_use]
    pub fn nose_diameters(&self) -> &[f32; NOSE_LEN] {
        &self.tract.nose_diameter
    }

    /// Recent peak pressure per oral cell (slowly decaying), for display.
    #[must_use]
    pub fn amplitudes(&self) -> &[f32; N] {
        self.tract.amplitudes()
    }

    /// Recent peak pressure per nasal cell.
    #[must_use]
    pub fn nose_amplitudes(&self) -> &[f32; NOSE_LEN] {
        self.tract.nose_amplitudes()
    }

    // ----- measurement -------------------------------------------------

    /// Rate of the tract model's own time steps, Hz (twice the glottal
    /// rate; it rises as the tract gets shorter).
    #[must_use]
    pub fn tick_rate(&self) -> f32 {
        2.0 * self.internal_rate
    }

    /// Readings from inside the physical engine at the latest tick.
    #[must_use]
    pub fn gauges(&self) -> Gauges {
        if self.engine != Engine::Physical {
            return Gauges::default();
        }
        let g = self.phys.gauge;
        Gauges {
            glottal_flow: g.u_glottis,
            subglottal_pressure: g.p_sub / CM_H2O,
            supraglottal_pressure: g.p_supra / CM_H2O,
            lip_flow: g.u_lips,
            nose_flow: g.u_nose,
            skin_flow: g.u_skin,
            edge_flow: g.u_edge,
            glottal_area: self.glottal_area,
            edge_area: self.source.edge_area,
            sounding: self.source.sounding().map(|kind| match kind {
                SourceKind::Kinematic => GlottalSource::Pulse,
                SourceKind::Folds => GlottalSource::Folds,
            }),
            fold_pitch: self.source.measured,
            jet_noise: g.jet / CM_H2O,
            jet_speed: g.jet_speed,
            jet_cell: g.jet_cell,
            mouth_shut: g.shut,
            gap_nudge: self.tract.nudges().0,
            vowel_relief: self.tract.nudges().1,
        }
    }

    /// Whether the mouth is shut somewhere in front of the pharynx right now
    /// (physical engine; see [`Gauges::mouth_shut`]). The speech front end
    /// times the larynx from it: the voice stops as a voiceless stop shuts
    /// and comes back a voice onset time after it opens.
    #[must_use]
    pub fn mouth_shut(&self) -> bool {
        self.engine == Engine::Physical && self.phys.gauge.shut
    }

    /// Air pressure in tract cell `cell` (`0` at the glottis), cmH2O, in
    /// the physical engine.
    #[must_use]
    pub fn pressure(&self, cell: usize) -> f32 {
        if self.engine == Engine::Physical && cell < N {
            self.phys.pressure(cell) / CM_H2O
        } else {
            0.0
        }
    }

    /// Let the articulators finish moving, without rendering sound.
    #[allow(clippy::cast_precision_loss)]
    pub fn settle(&mut self) {
        let dt = BLOCK as f32 / self.internal_rate;
        for _ in 0..400 {
            self.tract.update_block(dt);
        }
        self.phys.set_teeth(self.tract.teeth());
        self.phys.set_edge(self.tract.edges());
        self.phys
            .snap(&self.tract.diameter, self.tract.velum_openness());
    }

    /// The physical tract's response with the glottis held open at `area`
    /// (cm²) and the lungs at `lung` (cmH2O): the change in the flow at
    /// the lips, nostrils and skin, one value per tick, after a pressure
    /// impulse at the lungs (per dyn/cm²), once the steady flow has settled.
    /// It is the transfer function as the open glottis loads it, around the
    /// flow it passes. For measuring; the classic engine gives zeros.
    #[doc(hidden)]
    pub fn impulse_response_open(&self, area: f32, lung: f32, out: &mut [f32]) {
        out.fill(0.0);
        if self.engine != Engine::Physical {
            return;
        }
        let mut quiet = self.phys.clone();
        quiet.reset();
        quiet.freeze();
        let lung = lung * CM_H2O;
        // Long enough for the flow to settle: walls, trachea and all.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let settle = (0.8 * self.internal_rate) as usize;
        for _ in 0..settle {
            quiet.tick(1.0, Drive::Glottis { area, lung });
        }
        let mut kicked = quiet.clone();
        for (n, o) in out.iter_mut().enumerate() {
            let a = quiet.tick(1.0, Drive::Glottis { area, lung });
            // A kick of 0.1 cmH2O: small next to the lungs, large next to
            // the rounding of the flows it is told from.
            let kick = if n == 0 { 0.1 * CM_H2O } else { 0.0 };
            let b = kicked.tick(
                1.0,
                Drive::Glottis {
                    area,
                    lung: lung + kick,
                },
            );
            *o = (b - a) / (0.1 * CM_H2O);
        }
    }

    /// The tract's response to a unit flow impulse at a closed glottis, as
    /// flow at the lips and nostrils, one value per tick (see
    /// [`tick_rate`](Self::tick_rate)). Its spectrum is the vocal tract
    /// transfer function for the current shape. The voice itself is not
    /// disturbed.
    pub fn impulse_response(&self, out: &mut [f32]) {
        match self.engine {
            Engine::Classic => {
                let mut tract = self.tract.clone();
                tract.reset_waves();
                for (n, o) in out.iter_mut().enumerate() {
                    *o = tract.step(if n == 0 { 1.0 } else { 0.0 }, 0.0, 1.0);
                }
            }
            Engine::Physical => {
                let mut phys = self.phys.clone();
                phys.reset();
                phys.freeze();
                for (n, o) in out.iter_mut().enumerate() {
                    *o = phys.tick(1.0, Drive::Flow(if n == 0 { 1.0 } else { 0.0 }));
                }
            }
        }
    }

    /// The physical tract's response, as flow at the lips, nostrils and
    /// skin per tick, to a pressure impulse (per dyn/cm²) in series at
    /// joint `joint` (between cells `joint − 1` and `joint`): how a noise
    /// source there is heard. With `lung` (cmH2O) above zero the glottis is
    /// held open at `area` (cm²) and the steady stream through the tract
    /// settles first, so a narrow gap resists as its jet does; otherwise
    /// the glottis is closed and nothing flows. The noise sources are off.
    /// Its spectrum times frequency is the radiated sound. For measuring;
    /// the classic engine gives zeros.
    #[doc(hidden)]
    pub fn source_response(&self, joint: usize, area: f32, lung: f32, out: &mut [f32]) {
        out.fill(0.0);
        if self.engine != Engine::Physical {
            return;
        }
        let mut quiet = self.phys.clone();
        quiet.reset();
        quiet.freeze();
        quiet.set_noisy(false);
        let drive = if lung > 0.0 {
            Drive::Glottis {
                area,
                lung: lung * CM_H2O,
            }
        } else {
            Drive::Flow(0.0)
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let settle = if lung > 0.0 {
            (0.8 * self.internal_rate) as usize
        } else {
            0
        };
        for _ in 0..settle {
            quiet.tick(1.0, drive);
        }
        let mut kicked = quiet.clone();
        // A kick of 0.1 cmH2O: small next to the stream's own pressures.
        let kick = 0.1 * CM_H2O;
        kicked.kick(joint, kick);
        for o in out.iter_mut() {
            let a = if lung > 0.0 {
                quiet.tick(1.0, drive)
            } else {
                0.0
            };
            *o = (kicked.tick(1.0, drive) - a) / kick;
        }
    }

    // ----- audio -------------------------------------------------------

    /// Output gain (default `1.0`).
    pub fn set_gain(&mut self, gain: f32) {
        if gain.is_finite() {
            self.gain = gain.max(0.0);
        }
    }

    /// One output sample.
    #[allow(clippy::cast_possible_truncation)]
    pub fn process(&mut self) -> f32 {
        while self.frac >= 1.0 {
            self.prev = self.cur;
            self.cur = self.internal_sample();
            self.frac -= 1.0;
        }
        let x = self.prev + (self.cur - self.prev) * self.frac as f32;
        self.frac += self.step;

        // DC blocker: the glottal flow derivative is not quite zero-mean.
        let y = x - self.dc_x + self.dc_r * self.dc_y;
        self.dc_x = x;
        self.dc_y = y;
        y * self.gain
    }

    /// Fill `out` with mono samples.
    pub fn render(&mut self, out: &mut [f32]) {
        for s in out {
            *s = self.process();
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn internal_sample(&mut self) -> f32 {
        // Two clocks in the physical engine. The folds' controls (pitch,
        // tenseness, voicing) and the hiss regulator are brought up to date
        // once a long block, 10.7 ms: they are interpolated across it or
        // read averages over it, and a pitch period is that long. The
        // articulators and the tract's areas go by the control block, which
        // may be shorter.
        let block = self.block;
        let slow = self.since_regulated >= BLOCK;
        if self.block_pos == 0 {
            let dt = block as f32 / self.internal_rate;
            let long = self.since_regulated as f32 / self.internal_rate;
            if slow {
                self.since_regulated = 0;
                if self.engine == Engine::Physical {
                    self.glottis.advance(long);
                }
                self.glottis.update_block(long, &self.simplex);
                if self.engine == Engine::Physical {
                    self.phys.end_block();
                    if self.regulated {
                        let breath = self.source.breath_is_up();
                        let voiced = (self.glottis.voiced || self.glottis.always_voice)
                            && self.glottis.block_tenseness().1 >= WHISPER_TENSENESS;
                        let phys = &self.phys;
                        self.tract
                            .regulate(long, breath, voiced, |i| phys.stream(i));
                    }
                }
            }
            self.tract.update_block(dt);
            if self.engine == Engine::Physical {
                self.phys.set_motion(self.tract.velocity());
                self.phys.set_teeth(self.tract.teeth());
                self.phys.set_edge(self.tract.edges());
                self.phys.begin_block(
                    &self.tract.diameter,
                    self.tract.velum_openness(),
                    dt,
                    self.glottis.voiced || self.glottis.always_voice,
                );
                if slow {
                    self.source
                        .begin_block(&self.glottis, self.tract.any_constriction());
                    self.phys.voiced_target = if self.source.sounding().is_some() {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
        }
        let lambda1 = self.block_pos as f32 / block as f32;
        let lambda2 = (self.block_pos as f32 + 0.5) / block as f32;
        let slow1 = self.since_regulated as f32 / BLOCK as f32;
        let slow2 = (self.since_regulated as f32 + 0.5) / BLOCK as f32;
        self.block_pos = (self.block_pos + 1) % block;
        self.since_regulated = self.since_regulated.saturating_add(1);

        if self.engine == Engine::Physical {
            return self.physical_sample([lambda1, lambda2], [slow1, slow2]);
        }

        let glottal = self
            .glottis
            .step(lambda1, self.rng.bipolar(), &self.simplex);
        let frication = self.frication.process(self.rng.bipolar()) * self.glottis.noise_modulator();
        let out = self.tract.step(glottal, frication, lambda1)
            + self.tract.step(glottal, frication, lambda2);
        let out = out * 0.125;
        if out.is_finite() {
            out
        } else {
            // Never let a NaN circulate in the waveguide.
            self.tract.reset_waves();
            self.glottis.reset();
            0.0
        }
    }

    /// One internal sample of the physical engine: two tract ticks. The
    /// sound that reaches a listener is the rate of change of the flow
    /// leaving the mouth and nose.
    fn physical_sample(&mut self, tract: [f32; 2], folds: [f32; 2]) -> f32 {
        let dt = 0.5 / self.internal_rate;
        let (area, lung) = self.source.tick(folds[0], dt);
        let flow = self.phys.tick(tract[0], Drive::Glottis { area, lung });
        self.phys.radiate(flow);
        self.source.react(&self.phys.gauge, folds[0], dt);
        let (area, lung) = self.source.tick(folds[1], dt);
        let flow = self.phys.tick(tract[1], Drive::Glottis { area, lung });
        let flow = self.phys.radiate(flow);
        self.source.react(&self.phys.gauge, folds[1], dt);
        self.glottal_area = area;
        let out = (flow - self.phys_flow) * (0.5 / dt) * PHYSICAL_GAIN;
        self.phys_flow = flow;
        // Soft folds: the voice through a one-pole low-pass. The tract is
        // linear, so for the voice this is the low-pass on the derivative
        // of the glottal flow, but outside the loop of folds and tract: put
        // on the flow the tract takes up, the tract's pressure fed back on
        // the folds gave most of the high frequencies back (see
        // `Trombone::set_fold_softness`).
        let pass = self.source.soft_pass(2.0 * dt);
        self.soft_out += (out - self.soft_out) * pass;
        let out = self.soft_out;
        if out.is_finite() {
            out
        } else {
            self.phys.reset();
            self.source.reset();
            self.phys_flow = 0.0;
            self.soft_out = 0.0;
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::lpc;

    use super::*;
    use crate::tract::Constriction;
    use std::vec;
    use std::vec::Vec;

    fn render(t: &mut Trombone, seconds: f32) -> Vec<f32> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mut buf = vec![0.0; (seconds * t.sample_rate()) as usize];
        t.render(&mut buf);
        buf
    }

    fn rms(x: &[f32]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let n = x.len() as f32;
        (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt()
    }

    #[test]
    fn silent_until_voiced() {
        let mut t = Trombone::new(48_000.0);
        assert!(render(&mut t, 0.5).iter().all(|s| s.abs() < 1e-3));
    }

    #[test]
    fn voiced_output_is_audible_and_bounded() {
        for sr in [8_000.0, 22_050.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let mut t = Trombone::new(sr);
            t.note_on();
            let out = render(&mut t, 1.0);
            assert!(out.iter().all(|s| s.is_finite() && s.abs() < 1.5), "{sr}");
            assert!(rms(&out[out.len() / 2..]) > 0.02, "{sr}: too quiet");
        }
    }

    #[test]
    fn fades_out_after_note_off() {
        let mut t = Trombone::new(48_000.0);
        t.note_on();
        render(&mut t, 0.5);
        t.note_off();
        render(&mut t, 0.5);
        assert!(rms(&render(&mut t, 0.1)) < 1e-3);
    }

    #[test]
    fn same_seed_same_audio() {
        let mut a = Trombone::with_seed(44_100.0, 42);
        let mut b = Trombone::with_seed(44_100.0, 42);
        a.note_on();
        b.note_on();
        assert_eq!(render(&mut a, 0.3), render(&mut b, 0.3));
    }

    #[test]
    fn unvoiced_fricative_hisses() {
        let mut t = Trombone::new(48_000.0);
        t.articulate_phoneme(Phoneme::S);
        let out = render(&mut t, 0.5);
        assert!(rms(&out[out.len() / 2..]) > 0.003, "{}", rms(&out));
    }

    #[test]
    fn releasing_a_closure_makes_a_burst() {
        let mut t = Trombone::new(48_000.0);
        t.set_constriction(SLOT_MAIN, Some(Constriction::new(41.0, 0.0)));
        render(&mut t, 0.5);
        let before = rms(&render(&mut t, 0.05));
        t.set_constriction(SLOT_MAIN, None);
        let after = render(&mut t, 0.2);
        let peak = after.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            peak > 10.0 * before.max(1e-4),
            "peak {peak}, before {before}"
        );
    }

    #[test]
    fn velum_opens_and_closes() {
        let mut t = Trombone::new(48_000.0);
        t.set_velum(1.0);
        render(&mut t, 0.5);
        assert!(t.velum() > 0.95);
        t.set_velum(0.0);
        render(&mut t, 0.5);
        assert!(t.velum() < 0.05);
    }

    #[test]
    fn rejects_garbage_parameters() {
        let mut t = Trombone::new(48_000.0);
        t.set_frequency(f32::NAN);
        t.set_tongue(f32::INFINITY, 2.0);
        t.set_constriction(0, Some(Constriction::new(f32::NAN, f32::NAN)));
        t.set_constriction(99, Some(Constriction::new(20.0, 0.0)));
        t.set_tenseness(f32::NAN);
        t.note_on();
        assert!(render(&mut t, 0.3).iter().all(|s| s.is_finite()));
    }

    /// F1 and F2 of a sustained phoneme, by LPC.
    fn formants(p: Phoneme, tract_length: f32) -> (f32, f32) {
        let sr = 16_000.0;
        let mut t = Trombone::new(sr);
        t.set_tract_length(tract_length);
        t.set_frequency(100.0);
        t.set_vibrato(0.0, 0.0);
        t.articulate_phoneme(p);
        t.note_on();
        render(&mut t, 0.6);
        let f = lpc::formants(&render(&mut t, 0.4), sr, 5000.0);
        (f[0], f[1])
    }

    fn vowel(p: Phoneme) -> (f32, f32) {
        formants(p, 1.0)
    }

    #[test]
    fn oh_is_a_proper_o() {
        // Hillenbrand et al. (1995), men: /o/ at 497/910 Hz.
        let (f1, f2) = formants(Phoneme::Ow, 1.0);
        assert!((440.0..560.0).contains(&f1), "F1 {f1}");
        assert!((820.0..1000.0).contains(&f2), "F2 {f2}");
        // Closer and more rounded than "aw".
        let (a1, _) = formants(Phoneme::Ao, 1.0);
        assert!(f1 < a1 - 40.0, "oh {f1} vs aw {a1}");
    }

    #[test]
    fn shorter_tract_raises_formants() {
        let (m1, m2) = formants(Phoneme::Aa, 1.0);
        let (c1, c2) = formants(Phoneme::Aa, 0.7);
        // Formants scale with 1 / length: about +43 % for 0.7.
        assert!(
            c1 > m1 * 1.3 && c2 > m2 * 1.3,
            "native {m1}/{m2}, short {c1}/{c2}"
        );
    }

    #[test]
    fn tract_length_one_is_the_default() {
        let mut a = Trombone::with_seed(44_100.0, 3);
        let mut b = Trombone::with_seed(44_100.0, 3);
        b.set_tract_length(0.8);
        b.set_tract_length(1.0);
        a.note_on();
        b.note_on();
        assert_eq!(render(&mut a, 0.2), render(&mut b, 0.2));
    }

    #[test]
    fn vowels_have_the_right_formant_layout() {
        let (i1, i2) = vowel(Phoneme::Iy);
        let (a1, a2) = vowel(Phoneme::Aa);
        let (u1, u2) = vowel(Phoneme::Uw);
        std::println!("IY {i1}/{i2}  AA {a1}/{a2}  UW {u1}/{u2}");
        // The vowel triangle: /a/ is open (high F1), /i/ front (high F2),
        // /u/ close and back (low F1, low F2).
        assert!(a1 > i1 && a1 > u1, "F1: AA {a1} IY {i1} UW {u1}");
        assert!(i2 > a2 && i2 > u2, "F2: IY {i2} AA {a2} UW {u2}");
    }
}
