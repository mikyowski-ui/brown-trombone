//! Listening: turn microphone samples into a small stream of measurements
//! (pitch, formants, level and hiss) that a voice can copy.
//!
//! [`Listener`] takes mono samples one at a time at any rate from 12 kHz to
//! 96 kHz (it is tuned for 16–48 kHz), converts them internally to the
//! fixed [`ANALYSIS_RATE`], and once per [`HOP`] analysis samples (5 ms)
//! returns a [`Heard`]:
//!
//! - **Pitch** by a cumulative normalised difference function (YIN) over
//!   the last 37 ms, 60–500 Hz; the period is then placed at the peak of
//!   the normalised cross-correlation, refined between samples.
//! - **Formants** by linear prediction of order 12: pre-emphasis in two
//!   stages (+6 dB per octave from 60 Hz and again from 1 kHz, for the
//!   dark spectrum of a speaking voice), a 20 ms
//!   Hann window, autocorrelation with a Gaussian lag window a third of
//!   the pitch wide (so the model follows the envelope, not single
//!   harmonics), the Levinson–Durbin recursion, and the roots of the
//!   predictor polynomial (Aberth–Ehrlich, warm-started from the previous
//!   frame). The three lowest resonances narrower than 700 Hz are F1–F3; a
//!   broader one counts only where it continues a formant just measured.
//!   A frame whose F2 is smeared into a broad resonance (F1 and F3 still
//!   in place) holds the formants instead, and a median over the last
//!   three frames removes other single-frame slips.
//!   Nothing assumes a speaker size: men, women and children are measured
//!   the same way.
//! - **Level, hiss and brightness** from smoothed energies at the input
//!   rate: the share of energy above 1.5 kHz, discounted by how periodic
//!   that band is while the voice is on, and its spectral centre from a
//!   bank of 1 kHz wide band-pass filters between 1.5 and 7.5 kHz.
//!
//! Nothing allocates and all state is fixed-size (about 23 KB), so
//! [`Listener::push`] can run inside an audio callback: on a desktop
//! processor it costs about 0.1 µs per input sample at 48 kHz, plus
//! about 15 µs in the one push per hop that analyses.
//!
//! How well it does, measured on this crate's own voices (see
//! `tests/listen_gate.rs`): pitch within 2 % in every frame from 80 to
//! 400 Hz (0.2 % on the classic engine). With the classic engine, F1 and
//! F2 come within 6 % of the tract's resonances at 110 and 220 Hz, except
//! where F1 falls between the first two harmonics (/u/ at 220 Hz: F1 20 %
//! high), the hard case of any envelope method. With the physical engine
//! (whose voice is as dark as a speaker's) they come within 8 %, except
//! the F1 of /i/ and /u/ at 110 Hz and of /ɔ/ at 220 Hz, which lean 9 to
//! 13 % towards the bump its closing folds put in the source near 440 Hz.
//! Formants reach a new vowel (within 10 %) 20–30 ms after an abrupt
//! change; [`Listener::latency`] gives the bound.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::math::{cos, exp, ln, sin, sqrt};

/// Rate of the internal analysis stream, Hz. Input at any rate is
/// converted to this before pitch and formants are measured.
pub const ANALYSIS_RATE: f32 = 12_000.0;

/// Analysis samples between two measurements (5 ms).
pub const HOP: usize = 60;

const PI: f32 = core::f32::consts::PI;

// ----- tuning ------------------------------------------------------------

/// Linear prediction order (one pole pair per kHz of bandwidth).
const ORDER: usize = 12;
/// Linear prediction window, analysis samples (20 ms).
const LPC_WIN: usize = 240;
/// A dip of YIN's normalised difference this deep at a whole fraction of
/// the chosen period is taken for the period itself (a voice's own period
/// sits under 0.15; half the period of a clear voice, above 0.29).
const SUBPERIOD_DIP: f32 = 0.2;
/// Pre-emphasis coefficient: +6 dB per octave from about 60 Hz, for the
/// lip radiation's tilt.
const PRE: f32 = 0.97;
/// Coefficient of a second stage of pre-emphasis, +6 dB per octave more
/// from about 1 kHz (a zero at 975 Hz): for the glottal source's own fall
/// above its return-phase corner. A speaking voice is that dark (the
/// LibriSpeech reader's vowels have 2-4 kHz 29 dB under 0-1 kHz, and so has
/// the physical engine's since its source tilt), and with only the first
/// stage linear prediction spends its poles on the strong band under 1
/// kHz: in about one frame in ten of the physical /æ/ the pole of F2 came
/// out 1.9-3.6 kHz broad and F3 (2.2-2.3 kHz) was read as F2, so the step
/// from /u/ to /æ/ took 220 ms to settle (22 ms with this stage), and the
/// close vowels' F1 leaned 3 to 9 points further towards the source's
/// bump. 0.55 to 0.65 pass every gate; 0.5 lets an F2 slip through on the
/// copied /æ/, 0.7 reads the classic /i/ at 220 Hz 8.2 % high. On the
/// reader's 16 sentences the vowels' F2 jumps less from frame to frame
/// (90th percentile 4.6 → 3.8 %), and his vowels are told apart as well.
const PRE_SOURCE: f32 = 0.6;
/// Width of the Gaussian lag window, as a fraction of the pitch.
const LAG_WIDTH: f32 = 0.3;
/// Most iterations of the root finder per attempt (a warm start needs
/// three to five).
const ROOT_ITERATIONS: usize = 50;
/// Lowest frequency that can be a formant, Hz.
const F_LOW: f32 = 180.0;
/// Highest frequency that can be a formant, Hz.
const F_HIGH: f32 = 5_200.0;
/// Widest bandwidth that counts as a formant, Hz.
const MAX_BW: f32 = 700.0;
/// Widest bandwidth that counts as a formant if it continues one just
/// measured, Hz.
const MAX_BW_TRACKED: f32 = 1_600.0;
/// How close (relative) a broad resonance must be to a formant just
/// measured to continue it.
const TRACK_REACH: f32 = 0.15;
/// Widest bandwidth of a resonance that shows a formant smeared rather
/// than gone, Hz.
const MAX_BW_SMEARED: f32 = 2_000.0;
/// How far (as a ratio) a smeared resonance must lie from its sharp
/// neighbours to count as one.
const GAP_EDGE: f32 = 1.15;
/// How far (as a ratio less one) the lowest sharp resonance may lie from
/// the F1 just measured for a smeared frame to be held.
const F1_REACH: f32 = 0.5;
/// How far (as a ratio less one) the next sharp resonance may lie from
/// the F3 just measured for a smeared frame to be held.
const F3_REACH: f32 = 0.2;
/// Most measurements in a row that smeared frames hold the formants
/// (40 ms); after that the frame is used as it is.
const MAX_DISTURBED: usize = 8;

/// YIN integration window, analysis samples (20 ms).
const YIN_W: usize = 240;
/// Shortest period searched (545 Hz).
const TAU_MIN: usize = 22;
/// Longest period searched (58.5 Hz).
const TAU_MAX: usize = 205;
/// Periods either side of the YIN dip searched for the correlation peak.
const REFINE: usize = 3;
/// Aperiodicity below which a frame starts to count as voiced.
const VOICE_ON: f32 = 0.25;
/// Aperiodicity above which a voiced stretch ends.
const VOICE_OFF: f32 = 0.35;
/// Share of energy above the hiss edge below which an aperiodic sound
/// still counts as a vowel for the formants.
const VOWEL_SHARE: f32 = 0.3;
/// Measurements whose window can reach past the end of a vowel; the held
/// formants are taken from before them.
const ROLLBACK: usize = LPC_WIN / HOP;
/// Pitch assumed for the lag window before any voice was heard, Hz.
const DEFAULT_PITCH: f32 = 150.0;
/// Level (RMS) below which nothing counts as voiced.
const LEVEL_FLOOR: f32 = 3e-4;

/// Time constant of the energy smoothers, seconds.
const ENERGY_TAU: f32 = 0.008;
/// Lower edge of the hiss band, Hz.
const HISS_EDGE: f32 = 1_500.0;
/// Centres of the brightness bands, Hz (each 1 kHz wide).
const BANDS: [f32; 7] = [
    1_500.0, 2_500.0, 3_500.0, 4_500.0, 5_500.0, 6_500.0, 7_500.0,
];

// ----- sizes -------------------------------------------------------------

/// Analysis stream history (power of two, at least the longest frame).
const RING: usize = 512;
/// Longest frame analysed (the pitch frame).
const FRAME: usize = YIN_W + TAU_MAX;

/// Resampler: zero crossings of the kernel on each side.
const RS_ZC: usize = 10;
/// Resampler: kernel phases tabulated per input sample (interpolated in
/// between).
const RS_PHASES: usize = 16;
/// Resampler: most taps (at the highest input rate).
const RS_TAPS: usize = 180;
/// Resampler: input history (power of two, at least `RS_TAPS`), stored
/// twice so that every window is contiguous.
const RS_RING: usize = 256;
/// Resampler: pass band edge, Hz.
const RS_CUTOFF: f32 = 5_400.0;

/// Lowest and highest accepted input rates, Hz.
const RATE_RANGE: (f32, f32) = (12_000.0, 96_000.0);

/// One measurement of what the listener heard.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Heard {
    /// Whether the input is a voiced sound (a vowel, a hum, a voiced
    /// consonant): loud enough and periodic.
    pub voiced: bool,
    /// Fundamental frequency, Hz; `0` when not voiced.
    pub pitch: f32,
    /// First three formants F1, F2, F3, Hz. They are measured while the
    /// input is a voice: voiced, or at least not dominated by hiss (so they
    /// keep following a vowel through the moment the pitch tracker loses
    /// it at an abrupt change). Through unvoiced stretches (/s/, /f/, a
    /// stop's closure) and silence they **hold** the last good values of
    /// the vowel before, taken from before the analysis window reached
    /// past its end. `0` until a voice has been heard.
    pub formants: [f32; 3],
    /// Level, RMS of the input over the last few milliseconds (`1.0` is a
    /// full-scale square wave).
    pub level: f32,
    /// Share of the energy that is noise-like sound above 1.5 kHz, `0‥1`:
    /// near `1` for /s/, /sh/ or /f/, near `0` for a clean vowel, in
    /// between for /z/ or a breathy voice, depending on how much of it is
    /// voice. It is a ratio, so it says nothing about loudness; read
    /// [`level`](Self::level) to tell a quiet room from a fricative.
    pub hiss: f32,
    /// Spectral centre of the sound between 1.5 and 7.5 kHz, Hz. Read it
    /// when [`hiss`](Self::hiss) is high: /s/ (about 5 kHz and up) is
    /// brighter than /sh/ (2–3 kHz); /f/ and /th/ are told apart from
    /// /sh/ by their much lower [`level`](Self::level) rather than by this.
    pub brightness: f32,
}

/// Real-time analyser: microphone samples in, a [`Heard`] every 5 ms out.
///
/// ```
/// use synthie_trombone::listen::Listener;
///
/// let mut ear = Listener::new(48_000.0);
/// for n in 0..48_000 {
///     let x = (2.0 * core::f32::consts::PI * 220.0 * n as f32 / 48_000.0).sin();
///     if let Some(heard) = ear.push(0.3 * x) {
///         // hand `heard` to whatever copies the voice
///         let _ = heard.pitch;
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Listener {
    sample_rate: f32,
    resampler: Resampler,

    // Input-rate energies.
    hiss_hp: [Biquad; 2],
    bands: [Biquad; BANDS.len()],
    energy_coef: f32,
    e_total: f32,
    e_hiss: f32,
    e_bands: [f32; BANDS.len()],

    // Analysis stream.
    dc_x: f32,
    dc_y: f32,
    stream: [f32; RING],
    hf_filter: Biquad,
    hf_stream: [f32; RING],
    /// Analysis samples written (wrapping).
    written: usize,
    /// Analysis samples written, up to a full frame.
    filled: usize,
    since_hop: usize,

    // Scratch and tables.
    frame: [f32; FRAME],
    window: [f32; LPC_WIN],
    cmndf: [f32; TAU_MAX + 2],
    roots: [Complex; ORDER],

    // Held results.
    voiced: bool,
    formants: [f32; 3],
    /// The latest raw formant estimates, newest last, and how many of
    /// them are valid.
    recent: [[f32; 3]; 3],
    recent_len: usize,
    /// Formants reported over the last `ROLLBACK + 1` measurements of
    /// this vowel, newest last, and how many are valid.
    history: [[f32; 3]; ROLLBACK + 1],
    history_len: usize,
    /// Measurements in a row held because a formant seemed to vanish.
    disturbed: usize,
    /// Latest voiced pitch, Hz (sets the lag window).
    last_pitch: f32,
}

impl Listener {
    /// A listener for input sampled at `sample_rate` Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is not between 12 kHz and 96 kHz.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        assert!(
            (RATE_RANGE.0..=RATE_RANGE.1).contains(&sample_rate),
            "Listener: sample rate {sample_rate} Hz is outside 12-96 kHz"
        );
        let mut window = [0.0; LPC_WIN];
        for (n, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * cos(2.0 * PI * (n as f32 + 0.5) / LPC_WIN as f32);
        }
        let nyquist = 0.49 * sample_rate;
        let mut s = Self {
            sample_rate,
            resampler: Resampler::new(sample_rate),
            hiss_hp: [
                Biquad::highpass(HISS_EDGE, 0.541, sample_rate),
                Biquad::highpass(HISS_EDGE, 1.307, sample_rate),
            ],
            bands: BANDS.map(|hz| Biquad::bandpass(hz.min(nyquist), 1_000.0, sample_rate)),
            energy_coef: 1.0 - exp(-1.0 / (ENERGY_TAU * sample_rate)),
            e_total: 0.0,
            e_hiss: 0.0,
            e_bands: [0.0; BANDS.len()],
            dc_x: 0.0,
            dc_y: 0.0,
            stream: [0.0; RING],
            hf_filter: Biquad::highpass(HISS_EDGE, core::f32::consts::FRAC_1_SQRT_2, ANALYSIS_RATE),
            hf_stream: [0.0; RING],
            written: 0,
            filled: 0,
            since_hop: 0,
            frame: [0.0; FRAME],
            window,
            cmndf: [0.0; TAU_MAX + 2],
            roots: initial_roots(),
            voiced: false,
            formants: [0.0; 3],
            recent: [[0.0; 3]; 3],
            recent_len: 0,
            history: [[0.0; 3]; ROLLBACK + 1],
            history_len: 0,
            disturbed: 0,
            last_pitch: DEFAULT_PITCH,
        };
        s.reset();
        s
    }

    /// The input sample rate this listener was made for, Hz.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Forget everything heard so far (as if just created).
    pub fn reset(&mut self) {
        self.resampler.reset();
        for f in &mut self.hiss_hp {
            f.reset();
        }
        for f in &mut self.bands {
            f.reset();
        }
        self.hf_filter.reset();
        self.e_total = 0.0;
        self.e_hiss = 0.0;
        self.e_bands = [0.0; BANDS.len()];
        self.dc_x = 0.0;
        self.dc_y = 0.0;
        self.stream = [0.0; RING];
        self.hf_stream = [0.0; RING];
        self.written = 0;
        self.filled = 0;
        self.since_hop = 0;
        self.roots = initial_roots();
        self.voiced = false;
        self.formants = [0.0; 3];
        self.recent = [[0.0; 3]; 3];
        self.recent_len = 0;
        self.disturbed = 0;
        self.history = [[0.0; 3]; ROLLBACK + 1];
        self.history_len = 0;
        self.last_pitch = DEFAULT_PITCH;
    }

    /// Seconds from an abrupt change in the input until the measurements
    /// reflect it (about 31 ms): an upper bound on the time until the
    /// formants are within 10 % of a new vowel's and the pitch within 2 %
    /// of a new note's, wherever the change falls within a hop. Measured
    /// on the synthesizer's vowels the formants take 20–28 ms and the
    /// pitch 21 ms; level, hiss and brightness follow within about 10 ms.
    #[must_use]
    pub fn latency(&self) -> f32 {
        // The window has to fill with the new sound, the change can fall
        // just after a measurement, and the median of three waits one hop.
        self.resampler.delay() + (LPC_WIN + 2 * HOP) as f32 / ANALYSIS_RATE
    }

    /// Feed one input sample. Returns a measurement once every [`HOP`]
    /// analysis samples (every 5 ms), `None` otherwise.
    ///
    /// Non-finite samples are treated as silence.
    pub fn push(&mut self, sample: f32) -> Option<Heard> {
        let x = if sample.is_finite() {
            sample.clamp(-8.0, 8.0)
        } else {
            0.0
        };

        // Energies at the input rate.
        let k = self.energy_coef;
        smooth(&mut self.e_total, x * x, k);
        let h = self.hiss_hp[0].process(x);
        let h = self.hiss_hp[1].process(h);
        smooth(&mut self.e_hiss, h * h, k);
        for (f, e) in self.bands.iter_mut().zip(&mut self.e_bands) {
            let b = f.process(x);
            smooth(e, b * b, k);
        }

        let y = self.resampler.push(x)?;
        // DC blocker (about 40 Hz) on the analysis stream.
        let y_hp = flush(y - self.dc_x + 0.98 * self.dc_y);
        self.dc_x = y;
        self.dc_y = y_hp;
        let at = self.written % RING;
        self.stream[at] = y_hp;
        self.hf_stream[at] = self.hf_filter.process(y_hp);
        self.written = self.written.wrapping_add(1);
        self.filled = (self.filled + 1).min(FRAME);
        self.since_hop += 1;
        if self.since_hop < HOP {
            return None;
        }
        self.since_hop = 0;
        Some(self.analyse())
    }

    /// Copy the newest `len` samples of `ring` into the front of `frame`.
    fn latest(&mut self, len: usize, hf: bool) {
        let ring = if hf { &self.hf_stream } else { &self.stream };
        let start = self.written.wrapping_sub(len);
        for (n, v) in self.frame[..len].iter_mut().enumerate() {
            *v = ring[start.wrapping_add(n) % RING];
        }
    }

    fn analyse(&mut self) -> Heard {
        let level = sqrt(self.e_total);
        let ready = self.filled >= FRAME;

        // Pitch.
        let mut pitch = 0.0;
        let mut aperiodicity = 1.0;
        if ready {
            self.latest(FRAME, false);
            if let Some((tau, ap)) = self.yin() {
                pitch = ANALYSIS_RATE / tau;
                aperiodicity = ap;
            }
        }
        let threshold = if self.voiced { VOICE_OFF } else { VOICE_ON };
        self.voiced = ready && level > LEVEL_FLOOR && pitch > 0.0 && aperiodicity < threshold;
        if !self.voiced {
            pitch = 0.0;
        }

        // Formants, while the sound is a voice rather than hiss or silence
        // (also through a moment when the pitch tracker loses the voicing,
        // as at an abrupt change of vowel).
        let share = (self.e_hiss / self.e_total.max(1e-20)).clamp(0.0, 1.0);
        if self.voiced {
            self.last_pitch = pitch;
        }
        if ready && level > LEVEL_FLOOR && (self.voiced || share < VOWEL_SHARE) {
            self.latest(LPC_WIN + 2, false);
            if let Some(f) = self.formants_now(self.last_pitch) {
                self.recent = [self.recent[1], self.recent[2], f];
                self.recent_len = (self.recent_len + 1).min(3);
                self.formants = if self.recent_len == 3 {
                    // Median of three: drops a frame that lost a weak
                    // formant, at the cost of one hop.
                    let [a, b, c] = self.recent;
                    [0, 1, 2].map(|k| median3(a[k], b[k], c[k]))
                } else {
                    f
                };
                self.history.copy_within(1.., 0);
                self.history[ROLLBACK] = self.formants;
                self.history_len = (self.history_len + 1).min(ROLLBACK + 1);
            }
        } else {
            // The vowel has ended. The last few windows already reached
            // into what came after it: hold the values from before them.
            if self.history_len > 0 {
                self.formants = self.history[ROLLBACK + 1 - self.history_len];
            }
            self.recent_len = 0;
            self.history_len = 0;
            self.disturbed = 0;
        }

        // Hiss: the share above the hiss edge, less its periodic part.
        let hiss = if self.voiced {
            share * (1.0 - self.hf_periodicity(ANALYSIS_RATE / pitch))
        } else {
            share
        };
        let (mut num, mut den) = (0.0, 0.0);
        for (hz, e) in BANDS.iter().zip(&self.e_bands) {
            num += hz * e;
            den += e;
        }
        let brightness = if den > 1e-20 { num / den } else { 0.0 };

        Heard {
            voiced: self.voiced,
            pitch,
            formants: self.formants,
            level,
            hiss: if hiss.is_finite() { hiss } else { 0.0 },
            brightness,
        }
    }

    /// YIN on `frame[..FRAME]`: the period in analysis samples (refined
    /// between samples) and the aperiodicity at that period.
    fn yin(&mut self) -> Option<(f32, f32)> {
        let cur = &self.frame[TAU_MAX..FRAME];
        let energy = dot(cur, cur);
        if energy < 1e-12 {
            return None;
        }
        self.cmndf[0] = 1.0;
        let mut running = 0.0;
        for tau in 1..=TAU_MAX {
            let lagged = &self.frame[TAU_MAX - tau..TAU_MAX - tau + YIN_W];
            let d = distance(cur, lagged);
            running += d;
            self.cmndf[tau] = if running > 0.0 {
                d * tau as f32 / running
            } else {
                1.0
            };
        }
        self.cmndf[TAU_MAX + 1] = self.cmndf[TAU_MAX];
        let threshold = 0.15;
        let mut best = None;
        let mut tau = TAU_MIN;
        while tau <= TAU_MAX {
            if self.cmndf[tau] < threshold {
                while tau < TAU_MAX && self.cmndf[tau + 1] < self.cmndf[tau] {
                    tau += 1;
                }
                best = Some(tau);
                break;
            }
            tau += 1;
        }
        let mut tau = best.unwrap_or_else(|| {
            let mut m = TAU_MIN;
            for t in TAU_MIN..=TAU_MAX {
                if self.cmndf[t] < self.cmndf[m] {
                    m = t;
                }
            }
            m
        });
        // A dip at a whole number of periods can be deeper than the one at
        // the period itself: a voice whose second harmonic stands on F1
        // can carry a weak subharmonic (the physical /i/ at 170 Hz: 0.04
        // at two periods, 0.10-0.15 at one), and a listener still hears
        // the period. Take the shortest whole fraction of the chosen period
        // whose dip is still clearly a period's.
        for k in [3, 2] {
            let near = tau / k;
            if near < TAU_MIN + 2 {
                continue;
            }
            let mut d = near - 2;
            for t in near - 2..=near + 2 {
                if self.cmndf[t] < self.cmndf[d] {
                    d = t;
                }
            }
            if self.cmndf[d] < SUBPERIOD_DIP {
                tau = d;
                break;
            }
        }
        let aperiodicity = self.cmndf[tau];
        // The normalised difference leans towards short periods where its
        // dip is broad; place the period by the normalised
        // cross-correlation instead, near the chosen dip.
        let nccf = |t: usize| {
            let lagged = &self.frame[TAU_MAX - t..TAU_MAX - t + YIN_W];
            let e = dot(lagged, lagged);
            if e > 0.0 {
                dot(cur, lagged) / sqrt(energy * e)
            } else {
                0.0
            }
        };
        let lo = tau.saturating_sub(REFINE).max(TAU_MIN);
        let hi = (tau + REFINE).min(TAU_MAX - 1);
        let mut peak = (tau, nccf(tau));
        for t in lo..=hi {
            let v = nccf(t);
            if v > peak.1 {
                peak = (t, v);
            }
        }
        let (t, b) = peak;
        let shift = if t > TAU_MIN && t < TAU_MAX {
            let (a, c) = (nccf(t - 1), nccf(t + 1));
            let curve = a - 2.0 * b + c;
            if curve < -1e-9 {
                (0.5 * (a - c) / curve).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };
        Some((t as f32 + shift, aperiodicity))
    }

    /// Normalised autocorrelation of the band above the hiss edge at the
    /// pitch period (best of the neighbouring whole periods), `0‥1`.
    fn hf_periodicity(&mut self, period: f32) -> f32 {
        let t = (period + 0.5) as usize;
        if !(TAU_MIN..TAU_MAX).contains(&t) {
            return 0.0;
        }
        self.latest(FRAME, true);
        let cur = &self.frame[TAU_MAX..FRAME];
        let mut best = 0.0f32;
        for lag in t - 1..=t + 1 {
            let old = &self.frame[TAU_MAX - lag..FRAME - lag];
            let (xy, xx, yy) = (dot(cur, old), dot(cur, cur), dot(old, old));
            if xx > 1e-20 && yy > 1e-20 {
                best = best.max(xy / sqrt(xx * yy));
            }
        }
        best.clamp(0.0, 1.0)
    }

    /// F1–F3 by linear prediction on `frame[..LPC_WIN + 2]`, with the lag
    /// window set for `pitch`.
    #[allow(clippy::many_single_char_names)]
    fn formants_now(&mut self, pitch: f32) -> Option<[f32; 3]> {
        let mut u = [0.0f32; LPC_WIN];
        for (n, v) in u.iter_mut().enumerate() {
            let e = |m: usize| self.frame[m + 1] - PRE * self.frame[m];
            *v = (e(n + 1) - PRE_SOURCE * e(n)) * self.window[n];
        }
        let mut r = [0.0f64; ORDER + 1];
        for (k, rk) in r.iter_mut().enumerate() {
            *rk = f64::from(dot(&u[..LPC_WIN - k], &u[k..]));
        }
        if r[0] <= 1e-18 {
            return None;
        }
        r[0] *= 1.0 + 1e-6;
        // Gaussian lag window: smooths the spectrum over about a third of
        // the harmonic spacing.
        let w = 2.0 * PI * LAG_WIDTH * pitch / ANALYSIS_RATE;
        for (k, rk) in r.iter_mut().enumerate().skip(1) {
            let t = w * k as f32;
            *rk *= f64::from(exp(-0.5 * t * t));
        }
        let a = levinson(&r)?;
        // Start from the previous frame's roots (a few iterations), or
        // from scratch if that fails.
        if !find_roots(&a, &mut self.roots) {
            self.roots = initial_roots();
            if !find_roots(&a, &mut self.roots) {
                return None;
            }
        }
        // Sharp resonances are formants; a broad one only where it carries
        // on a formant that was there a moment ago (a weak formant whose
        // bandwidth the analysis overestimates in some frames).
        let mut cand = [0.0f32; ORDER];
        let mut n = 0;
        let mut broad = [0.0f32; ORDER];
        let mut nb = 0;
        let mut gap = [0.0f32; ORDER];
        let mut ng = 0;
        for z in &self.roots {
            if z.im <= 0.0 {
                continue;
            }
            let hz = atan2(z.im, z.re) as f32 * ANALYSIS_RATE / (2.0 * PI);
            let r2 = (z.re * z.re + z.im * z.im) as f32;
            let bw = -ln(r2.max(1e-12)) * ANALYSIS_RATE / (2.0 * PI);
            if !(F_LOW..F_HIGH).contains(&hz) {
                continue;
            }
            if bw < MAX_BW {
                cand[n] = hz;
                n += 1;
            } else if bw < MAX_BW_SMEARED {
                gap[ng] = hz;
                ng += 1;
                if bw < MAX_BW_TRACKED {
                    broad[nb] = hz;
                    nb += 1;
                }
            }
        }
        // A frame with a broad resonance between its two lowest sharp
        // ones, F1 about where it was and the next sharp one where F3 was,
        // shows F2 smeared, not a new vowel: a voice disturbed for a moment
        // (a few cycles in which the folds do not close lift F1 and smear
        // F2, as in the physical engine's /u/ every 50 ms or so). Ranking
        // what is left would make F3 into F2, and a run of such frames gets
        // past the median: hold the formants instead, for a while at most.
        if self.recent_len > 0 && self.disturbed < MAX_DISTURBED {
            let mut sharp = cand;
            let sharp = &mut sharp[..n];
            sharp.sort_unstable_by(f32::total_cmp);
            let [f1, _, f3] = self.formants;
            let within = |a: f32, b: f32, reach: f32| b > 0.0 && (a / b).max(b / a) < 1.0 + reach;
            let smeared = n >= 2
                && within(sharp[0], f1, F1_REACH)
                && within(sharp[1], f3, F3_REACH)
                && gap[..ng]
                    .iter()
                    .any(|&b| b > GAP_EDGE * sharp[0] && GAP_EDGE * b < sharp[1]);
            if smeared {
                self.disturbed += 1;
                return None;
            }
        }
        if self.recent_len > 0 {
            let near = |a: f32, b: f32| (a / b - 1.0).abs() < TRACK_REACH;
            for prev in self.formants {
                if prev > 0.0
                    && n < ORDER
                    && !cand[..n].iter().any(|&c| near(c, prev))
                    && let Some(&b) = broad[..nb].iter().find(|&&b| near(b, prev))
                {
                    cand[n] = b;
                    n += 1;
                }
            }
        }
        if n < 2 {
            return None;
        }
        let cand = &mut cand[..n];
        cand.sort_unstable_by(f32::total_cmp);
        self.disturbed = 0;
        Some([
            cand[0],
            cand[1],
            if n > 2 { cand[2] } else { self.formants[2] },
        ])
    }
}

/// One-pole smoothing of `value` into `state`, flushing values too small
/// to matter to zero (denormal numbers are slow on many processors).
#[inline]
fn smooth(state: &mut f32, value: f32, k: f32) {
    *state = flush(*state + k * (value - *state));
}

/// `x`, or zero if it is too small to matter.
#[inline]
fn flush(x: f32) -> f32 {
    if x.abs() < 1e-30 { 0.0 } else { x }
}

/// Middle one of three values.
fn median3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).min(a.min(b).max(c))
}

// ----- sums --------------------------------------------------------------

/// Lanes for the sums below: independent accumulators that the compiler
/// can keep in vector registers.
const LANES: usize = 8;

/// `Σ a·b` over the common length.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (ca, ta) = a[..n].as_chunks::<LANES>();
    let (cb, tb) = b[..n].as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ta.iter().zip(tb).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// `Σ (a−b)²` over the common length.
#[inline]
fn distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (ca, ta) = a[..n].as_chunks::<LANES>();
    let (cb, tb) = b[..n].as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let tail: f32 = ta.iter().zip(tb).map(|(x, y)| (x - y) * (x - y)).sum();
    acc.iter().sum::<f32>() + tail
}

// ----- linear prediction -------------------------------------------------

/// Levinson–Durbin: predictor `a[0..=ORDER]` (`a[0] = 1`) from the
/// autocorrelation `r`.
fn levinson(r: &[f64; ORDER + 1]) -> Option<[f64; ORDER + 1]> {
    let mut a = [0.0f64; ORDER + 1];
    let mut prev = [0.0f64; ORDER + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        prev[..i].copy_from_slice(&a[..i]);
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err.is_nan() || err <= 0.0 {
            return None;
        }
    }
    Some(a)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn norm1(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
}

/// Starting points for the root finder: a circle inside the unit circle,
/// turned off the real axis.
fn initial_roots() -> [Complex; ORDER] {
    let mut z = [Complex::new(0.0, 0.0); ORDER];
    for (k, v) in z.iter_mut().enumerate() {
        let ang = 2.0 * PI * (k as f32 + 0.35) / ORDER as f32;
        *v = Complex::new(f64::from(0.9 * cos(ang)), f64::from(0.9 * sin(ang)));
    }
    z
}

/// Roots of `z^P + a[1] z^(P-1) + … + a[P]` by the Aberth–Ehrlich method,
/// starting from (and written back to) `z`. `false` if it did not converge.
#[allow(clippy::many_single_char_names)]
fn find_roots(a: &[f64; ORDER + 1], z: &mut [Complex; ORDER]) -> bool {
    for _ in 0..ROOT_ITERATIONS {
        let mut moved = 0.0f64;
        for i in 0..ORDER {
            let zi = z[i];
            let mut p = Complex::new(1.0, 0.0);
            let mut dp = Complex::new(0.0, 0.0);
            for &c in &a[1..] {
                dp = dp.mul(zi).add(p);
                p = p.mul(zi).add(Complex::new(c, 0.0));
            }
            if dp.norm1() < 1e-300 {
                z[i] = zi.add(Complex::new(1e-6, 1e-6));
                moved = 1.0;
                continue;
            }
            let ratio = p.div(dp);
            let mut s = Complex::new(0.0, 0.0);
            for (j, &zj) in z.iter().enumerate() {
                if j != i {
                    let d = zi.sub(zj);
                    if d.norm1() > 1e-300 {
                        s = s.add(Complex::new(1.0, 0.0).div(d));
                    }
                }
            }
            let w = ratio.div(Complex::new(1.0, 0.0).sub(ratio.mul(s)));
            if !(w.re.is_finite() && w.im.is_finite()) {
                return false;
            }
            z[i] = zi.sub(w);
            moved = moved.max(w.norm1());
        }
        if moved < 1e-9 {
            return true;
        }
    }
    false
}

/// `atan2` in `f64` without `std` (error below 1e-5 rad).
fn atan2(y: f64, x: f64) -> f64 {
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
    const PI64: f64 = core::f64::consts::PI;
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let (t, swap) = if ay > ax {
        (ax / ay, true)
    } else {
        (ay / ax, false)
    };
    let t2 = t * t;
    let mut r = t
        * (0.999_866_0
            + t2 * (-0.330_299_5 + t2 * (0.180_141_0 + t2 * (-0.085_133_0 + t2 * 0.020_835_1))));
    if swap {
        r = HALF_PI - r;
    }
    if x < 0.0 {
        r = PI64 - r;
    }
    if y < 0.0 { -r } else { r }
}

// ----- filters -----------------------------------------------------------

/// Second-order section (transposed direct form II).
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn from_rbj(b: [f32; 3], a: [f32; 3]) -> Self {
        Self {
            b0: b[0] / a[0],
            b1: b[1] / a[0],
            b2: b[2] / a[0],
            a1: a[1] / a[0],
            a2: a[2] / a[0],
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn highpass(hz: f32, q: f32, rate: f32) -> Self {
        let w = 2.0 * PI * hz / rate;
        let (s, c) = (sin(w), cos(w));
        let alpha = s / (2.0 * q);
        Self::from_rbj(
            [f32::midpoint(1.0, c), -(1.0 + c), f32::midpoint(1.0, c)],
            [1.0 + alpha, -2.0 * c, 1.0 - alpha],
        )
    }

    /// Band-pass with 0 dB peak and the given bandwidth in Hz.
    fn bandpass(hz: f32, width: f32, rate: f32) -> Self {
        let w = 2.0 * PI * hz / rate;
        let (s, c) = (sin(w), cos(w));
        let alpha = s / (2.0 * (hz / width));
        Self::from_rbj([alpha, 0.0, -alpha], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = flush(self.b1 * x - self.a1 * y + self.z2);
        self.z2 = flush(self.b2 * x - self.a2 * y);
        y
    }
}

// ----- sample-rate conversion --------------------------------------------

/// Windowed-sinc converter from the input rate to [`ANALYSIS_RATE`],
/// with the kernel tabulated at [`RS_PHASES`] fractional positions.
#[derive(Clone, Debug)]
struct Resampler {
    /// Kernel rows: row `p` holds the taps for an output `p / RS_PHASES`
    /// of an input sample past the newest-but-`taps/2` sample.
    table: [f32; (RS_PHASES + 1) * RS_TAPS],
    /// Taps in use (even).
    taps: usize,
    /// Input history, each sample stored at `n` and `n + RS_RING`.
    ring: [f32; 2 * RS_RING],
    /// Input samples received.
    count: usize,
    /// Input position (in input samples) of the next output sample.
    next: f64,
    /// Input samples per output sample.
    step: f64,
    /// Pass band edge, Hz.
    cutoff: f32,
}

impl Resampler {
    fn new(rate: f32) -> Self {
        let cutoff = RS_CUTOFF.min(0.45 * rate);
        // Kernel argument per input sample, in zero crossings.
        let scale = 2.0 * cutoff / rate;
        let half = RS_ZC as f32 / scale;
        let m = (half as usize + 1).min(RS_TAPS / 2);
        let taps = 2 * m;
        let mut table = [0.0; (RS_PHASES + 1) * RS_TAPS];
        for p in 0..=RS_PHASES {
            let phase = p as f32 / RS_PHASES as f32;
            for i in 0..taps {
                // Distance from the output instant to tap `i`.
                let d = i as f32 + 1.0 - m as f32 - phase;
                let u = d * scale;
                let sinc = if u.abs() < 1e-6 {
                    1.0
                } else {
                    sin(PI * u) / (PI * u)
                };
                let w = if d.abs() < half {
                    0.5 + 0.5 * cos(PI * d / half)
                } else {
                    0.0
                };
                table[p * RS_TAPS + i] = sinc * w * scale;
            }
        }
        Self {
            table,
            taps,
            ring: [0.0; 2 * RS_RING],
            count: 0,
            next: 0.0,
            step: f64::from(rate) / f64::from(ANALYSIS_RATE),
            cutoff,
        }
    }

    fn reset(&mut self) {
        self.ring = [0.0; 2 * RS_RING];
        self.count = 0;
        self.next = 0.0;
    }

    /// Delay from input to output, seconds.
    fn delay(&self) -> f32 {
        RS_ZC as f32 / (2.0 * self.cutoff)
    }

    /// Take one input sample; return an output sample when one is due (at
    /// most one per input, as the input rate is at least the analysis
    /// rate).
    #[allow(clippy::many_single_char_names)]
    fn push(&mut self, x: f32) -> Option<f32> {
        let at = self.count % RS_RING;
        self.ring[at] = x;
        self.ring[at + RS_RING] = x;
        self.count += 1;
        let m = self.taps / 2;
        // Output at `next` needs inputs up to floor(next) + m.
        let base = self.next as usize;
        if base + m >= self.count {
            return None;
        }
        let phase = (self.next - base as f64) as f32 * RS_PHASES as f32;
        let p = (phase as usize).min(RS_PHASES - 1);
        let frac = phase - p as f32;
        self.next += self.step;
        // Taps cover inputs base + 1 - m ..= base + m; earlier than the
        // first input counts as silence.
        let first = (base + 1).wrapping_sub(m);
        let y = if base + 1 >= m {
            let start = first % RS_RING;
            let window = &self.ring[start..start + self.taps];
            let lo = &self.table[p * RS_TAPS..p * RS_TAPS + self.taps];
            let hi = &self.table[(p + 1) * RS_TAPS..(p + 1) * RS_TAPS + self.taps];
            let a = dot(window, lo);
            let b = dot(window, hi);
            a + (b - a) * frac
        } else {
            0.0
        };
        // Keep the clock small so it never loses precision.
        if self.next > 1e8 {
            let shift = 1e8 - (1e8 % RS_RING as f64);
            self.next -= shift;
            self.count -= shift as usize;
        }
        Some(y)
    }
}
