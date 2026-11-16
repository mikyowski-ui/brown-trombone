//! What the instrument panel shows besides the controls: the glottis's
//! pressures and closure, gathered sample by sample in the audio thread,
//! and the tract's own formants, measured off it.
//!
//! - [`GlottisWatch`] runs in the audio callback: a few additions per
//!   sample, no allocation. [`GlottisWatch::read`] gives a [`GlottisGauge`]
//!   for each snapshot.
//! - The formants are the peaks of the transfer function of the tract's
//!   shape (its resonances, as `examples/replay.rs --trace` and
//!   `tune_physical` measure them), not an analysis of the sound. One
//!   measurement takes some 5 ms, too long for the audio callback: the
//!   audio thread sends a copy of the voice ([`Probe`], allocation-free,
//!   at most [`PROBE_RATE`] times a second and only when the shape has
//!   changed), the UI passes it on to the [`FormantMeter`]'s own thread,
//!   and the reading comes back stamped with the sample the shape was
//!   taken at, so the panel can say how late it is.

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use std::time::{Duration, Instant};
use synthie_trombone::{GlottalSource, N, NOSE_LEN, Trombone};

/// Area under which the vibrating edges count as shut, cm².
const SHUT_AREA: f32 = 1e-4;
/// Snapshots a gauge reading averages over (about 50 ms at 60 a second:
/// several cycles of the lowest voice).
const WINDOWS: usize = 3;

/// Measured shapes sent per second at most.
pub const PROBE_RATE: f32 = 30.0;
/// Seconds of impulse response the meter analyses, as `replay --trace`.
const FORMANT_SECONDS: f32 = 0.04;

/// The glottis as the instrument panel shows it, over the last few
/// snapshots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlottisGauge {
    /// The physical engine is in use (the classic one has no air).
    pub physical: bool,
    /// Pressure across the folds, cmH2O: below them less above them,
    /// averaged.
    pub transglottal: f32,
    /// Pressure below the folds, cmH2O, averaged.
    pub subglottal: f32,
    /// The lung pressure the voice asks for, cmH2O.
    pub lung: f32,
    /// Share of the time the vibrating edges were shut while the voice
    /// sounded (the closed quotient); `None` while it does not.
    pub closed: Option<f32>,
    /// Smallest and largest glottal area, cm².
    pub area: (f32, f32),
    /// What is making the voice: the folds, the pulse, or nothing.
    pub sounding: Option<GlottalSource>,
    /// Pitch the folds are producing, Hz (`0` when they are not).
    pub fold_pitch: f32,
}

#[derive(Clone, Copy, Debug)]
struct Window {
    samples: u32,
    sounding: u32,
    shut: u32,
    transglottal: f32,
    subglottal: f32,
    area: (f32, f32),
}

impl Default for Window {
    fn default() -> Self {
        Self {
            samples: 0,
            sounding: 0,
            shut: 0,
            transglottal: 0.0,
            subglottal: 0.0,
            area: (f32::INFINITY, 0.0),
        }
    }
}

/// Gathers the glottis's readings in the audio thread (see the module).
#[derive(Clone, Debug, Default)]
pub struct GlottisWatch {
    windows: [Window; WINDOWS],
    /// The window being filled.
    at: usize,
}

impl GlottisWatch {
    /// Look at the voice after one sample.
    #[inline]
    pub fn sample(&mut self, voice: &Trombone) {
        let w = &mut self.windows[self.at];
        w.samples += 1;
        let g = voice.gauges();
        if g.sounding.is_some() {
            w.sounding += 1;
            w.shut += u32::from(g.edge_area <= SHUT_AREA);
        }
        w.transglottal += g.subglottal_pressure - g.supraglottal_pressure;
        w.subglottal += g.subglottal_pressure;
        w.area = (w.area.0.min(g.glottal_area), w.area.1.max(g.glottal_area));
    }

    /// The reading over the last few windows; a new window starts.
    #[allow(clippy::cast_precision_loss)]
    pub fn read(&mut self, voice: &Trombone) -> GlottisGauge {
        let mut all = Window::default();
        for w in &self.windows {
            all.samples += w.samples;
            all.sounding += w.sounding;
            all.shut += w.shut;
            all.transglottal += w.transglottal;
            all.subglottal += w.subglottal;
            all.area = (all.area.0.min(w.area.0), all.area.1.max(w.area.1));
        }
        self.at = (self.at + 1) % WINDOWS;
        self.windows[self.at] = Window::default();
        let n = all.samples.max(1) as f32;
        let g = voice.gauges();
        let physical = voice.engine() == synthie_trombone::Engine::Physical;
        GlottisGauge {
            physical,
            transglottal: all.transglottal / n,
            subglottal: all.subglottal / n,
            lung: voice.lung_pressure(),
            // Most of the window sounding, or the share means little.
            closed: (physical && all.sounding * 2 > all.samples)
                .then(|| all.shut as f32 / all.sounding as f32),
            area: if all.area.0.is_finite() {
                all.area
            } else {
                (0.0, 0.0)
            },
            sounding: g.sounding,
            fold_pitch: g.fold_pitch,
        }
    }
}

/// A copy of the voice for the formant meter, and the sample it was taken
/// at.
#[derive(Clone)]
pub struct Probe {
    pub voice: Trombone,
    pub at: u64,
}

/// The shape a probe measures, to tell whether it has changed.
#[derive(Clone, Copy, PartialEq)]
pub struct ShapeKey([u32; N + NOSE_LEN + 2]);

impl ShapeKey {
    pub const NONE: Self = Self([u32::MAX; N + NOSE_LEN + 2]);

    pub fn of(voice: &Trombone) -> Self {
        let mut k = [0u32; N + NOSE_LEN + 2];
        for (o, d) in k
            .iter_mut()
            .zip(voice.diameters().iter().chain(voice.nose_diameters()))
        {
            *o = d.to_bits();
        }
        k[N + NOSE_LEN] = voice.tract_length().to_bits();
        k[N + NOSE_LEN + 1] = voice.engine() as u32;
        Self(k)
    }
}

/// F1, F2 and F3 of the tract at a moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Formants {
    /// Hz; `None` when the transfer function shows fewer than three peaks
    /// (a closed mouth).
    pub hz: Option<[f32; 3]>,
    /// The sample the shape was taken at.
    pub at: u64,
    /// How long the measurement took.
    pub took: Duration,
}

/// Measures the formants of probes on a thread of its own (see the
/// module); the thread ends with the meter.
pub struct FormantMeter {
    to: Sender<Probe>,
    from: Receiver<Formants>,
}

impl Default for FormantMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormantMeter {
    pub fn new() -> Self {
        let (to, probes) = bounded::<Probe>(1);
        let (results, from) = bounded(4);
        std::thread::Builder::new()
            .name("formant meter".into())
            .spawn(move || {
                let mut meter = crate::survey::Meter::with_seconds(FORMANT_SECONDS);
                while let Ok(p) = probes.recv() {
                    let t0 = Instant::now();
                    let hz = meter.formants(&p.voice);
                    let reading = Formants {
                        hz,
                        at: p.at,
                        took: t0.elapsed(),
                    };
                    if results.send(reading).is_err() {
                        break;
                    }
                }
            })
            .expect("the formant meter's thread starts");
        Self { to, from }
    }

    /// Measure `probe` once the meter is free; a probe arriving while it
    /// is busy is dropped (the next one is newer anyway).
    pub fn submit(&self, probe: Probe) {
        match self.to.try_send(probe) {
            Ok(()) | Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {}
        }
    }

    /// The newest reading since the last call.
    pub fn poll(&self) -> Option<Formants> {
        self.from.try_iter().last()
    }
}

/// F1, F2, F3 of `voice` as the meter measures them, at once (for tests
/// and for comparing with `replay --trace`).
#[cfg(test)]
pub fn measure(voice: &Trombone) -> Option<[f32; 3]> {
    crate::survey::Meter::with_seconds(FORMANT_SECONDS).formants(voice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synthie_trombone::{Engine, Phoneme};

    fn render(v: &mut Trombone, seconds: f32) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = (seconds * v.sample_rate()) as usize;
        for _ in 0..n {
            v.process();
        }
    }

    fn watch(v: &mut Trombone, seconds: f32) -> GlottisGauge {
        let mut w = GlottisWatch::default();
        let mut g = GlottisGauge::default();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let per = (v.sample_rate() / 60.0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = (seconds * 60.0) as usize;
        for _ in 0..n {
            for _ in 0..per {
                v.process();
                w.sample(v);
            }
            g = w.read(v);
        }
        g
    }

    #[test]
    fn the_gauge_reads_the_folds() {
        let mut v = Trombone::new(48_000.0);
        v.set_engine(Engine::Physical);
        v.articulate_phoneme(Phoneme::Aa);
        let quiet = watch(&mut v, 0.3);
        assert_eq!(quiet.closed, None);
        assert_eq!(quiet.sounding, None);
        assert!(quiet.transglottal.abs() < 0.5, "{quiet:?}");
        v.set_frequency(110.0);
        v.note_on();
        render(&mut v, 0.4);
        let g = watch(&mut v, 0.3);
        println!("folds at 110 Hz: {g:?}");
        assert!(g.physical);
        assert_eq!(g.sounding, Some(GlottalSource::Folds));
        let cq = g.closed.expect("a closed quotient while sounding");
        assert!((0.1..0.9).contains(&cq), "{g:?}");
        // Most of the lungs' pressure drops across the folds of a vowel.
        assert!(
            g.transglottal > 0.5 * g.lung && g.transglottal <= g.subglottal + 0.1,
            "{g:?}"
        );
        assert!(g.area.1 > g.area.0 + 0.02, "{g:?}");
        // Pressed folds stay shut longer than breathy ones.
        v.set_tenseness(0.9);
        render(&mut v, 0.3);
        let pressed = watch(&mut v, 0.3).closed.unwrap();
        v.set_tenseness(0.35);
        render(&mut v, 0.3);
        let breathy = watch(&mut v, 0.3).closed.unwrap_or(0.0);
        println!("closed quotient pressed {pressed:.2}, breathy {breathy:.2}");
        assert!(
            pressed > breathy + 0.03,
            "pressed {pressed}, breathy {breathy}"
        );
        // A closed mouth: the pressure above the folds rises to meet the
        // lungs', and the drop across them falls.
        let vowel = watch(&mut v, 0.2).transglottal;
        v.articulate_phoneme(Phoneme::B);
        render(&mut v, 0.06);
        let closed = watch(&mut v, 0.05).transglottal;
        println!("across the folds: vowel {vowel:.1}, /b/ closure {closed:.1} cmH2O");
        assert!(closed < vowel, "vowel {vowel}, closure {closed}");
    }

    /// How long a measurement takes, and how far a shorter one is off.
    #[test]
    #[ignore = "a measurement"]
    fn meter_cost() {
        let mut long = crate::survey::Meter::with_seconds(0.04);
        let mut short = crate::survey::Meter::with_seconds(0.025);
        let mut worst = 0.0f32;
        let (mut tl, mut ts) = (Duration::ZERO, Duration::ZERO);
        for engine in [Engine::Classic, Engine::Physical] {
            for p in Phoneme::ALL {
                let mut v = Trombone::new(48_000.0);
                v.set_engine(engine);
                v.articulate_phoneme(*p);
                v.settle();
                let t0 = Instant::now();
                let a = long.formants(&v);
                tl += t0.elapsed();
                let t0 = Instant::now();
                let b = short.formants(&v);
                ts += t0.elapsed();
                if let (Some(a), Some(b)) = (a, b) {
                    for k in 0..3 {
                        worst = worst.max((a[k] / b[k]).ln().abs());
                    }
                }
            }
        }
        let n = f64::from(u32::try_from(2 * Phoneme::ALL.len()).unwrap());
        println!(
            "0.04 s: {:.2} ms each; 0.025 s: {:.2} ms each; worst difference {:.1} %",
            tl.as_secs_f64() * 1e3 / n,
            ts.as_secs_f64() * 1e3 / n,
            worst * 100.0
        );
    }

    #[test]
    fn the_classic_engine_has_no_air() {
        let mut v = Trombone::new(48_000.0);
        v.note_on();
        let g = watch(&mut v, 0.3);
        assert!(!g.physical);
        assert_eq!(g.closed, None);
        assert_eq!(g.transglottal, 0.0);
    }

    #[test]
    fn the_meter_measures_off_the_audio_thread() {
        let mut v = Trombone::new(48_000.0);
        v.set_engine(Engine::Physical);
        v.articulate_phoneme(Phoneme::Iy);
        v.settle();
        let want = measure(&v).expect("an /i/ has three formants");
        let meter = FormantMeter::new();
        meter.submit(Probe {
            voice: v.clone(),
            at: 1234,
        });
        let t0 = Instant::now();
        let got = loop {
            if let Some(f) = meter.poll() {
                break f;
            }
            assert!(t0.elapsed() < Duration::from_secs(5), "no reading");
            std::thread::sleep(Duration::from_millis(1));
        };
        println!(
            "/i/: {:?} Hz, measured in {:.1} ms; a probe is {} bytes",
            got.hz,
            got.took.as_secs_f64() * 1e3,
            std::mem::size_of::<Probe>()
        );
        assert_eq!(got.hz, Some(want));
        assert_eq!(got.at, 1234);
        // /i/: F1 low, F2 high.
        assert!(want[0] < 400.0 && want[1] > 1900.0, "{want:?}");
    }
}
