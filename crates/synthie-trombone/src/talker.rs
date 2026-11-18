//! [`Talker`]: a voice you tune once and then hand sentences to.

use crate::preset::{Preset, VoiceSettings};
use crate::speech::{Speaker, Utterance};
use crate::{ArticulatorFrame, Listener, Stereo, Trombone};
use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Seconds rendered after the last segment so the voice can fade out.
const TAIL: f32 = 0.25;
/// Samples at 48 kHz between two frames of the articulators
/// ([`ArticulatorFrame::PERIOD`]).
const FRAME_SAMPLES: f64 = 512.0;

/// A tuned voice that speaks queued sentences.
///
/// Use it two ways:
///
/// - **Bake:** [`bake`](Self::bake) renders a sentence to a mono buffer, to
///   play with any audio engine (kira, rodio, a game toolkit's
///   `play_samples`) or to save. This is the easy path for games.
/// - **Stream:** [`say`](Self::say) queues text and [`render`](Self::render)
///   produces samples as they are needed, for audio callbacks and live
///   dialogue.
///
/// ```
/// use synthie_trombone::{Preset, Talker};
///
/// let mut goblin = Talker::new(44_100.0, Preset::Critter);
/// let line: Vec<f32> = goblin.bake("Shiny! Mine, all mine!");
/// assert!(!line.is_empty());
/// ```
///
/// Compiling text allocates. When streaming from a real-time callback,
/// compile with [`compile`](Self::compile) elsewhere and pass the result to
/// [`say_utterance`](Self::say_utterance). Finished utterances are freed
/// where `render` runs.
#[derive(Clone, Debug)]
pub struct Talker {
    voice: Trombone,
    speaker: Speaker,
    settings: VoiceSettings,
    queue: VecDeque<Utterance>,
    /// The articulators' frames kept while they are recorded (see
    /// [`Talker::record_articulators`]), and the samples at 48 kHz since
    /// the last was taken.
    frames: Option<Vec<ArticulatorFrame>>,
    frame_phase: f64,
}

impl Talker {
    /// A talker at `sample_rate` with the given voice.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is not positive and finite.
    #[must_use]
    pub fn new(sample_rate: f32, settings: impl Into<VoiceSettings>) -> Self {
        let mut voice = Trombone::new(sample_rate);
        Speaker::prepare(&mut voice);
        let settings = settings.into();
        settings.apply(&mut voice);
        Self {
            voice,
            speaker: Speaker::new(),
            settings,
            queue: VecDeque::new(),
            frames: None,
            frame_phase: FRAME_SAMPLES,
        }
    }

    /// A talker with one of the built-in voices.
    #[must_use]
    pub fn with_preset(sample_rate: f32, preset: Preset) -> Self {
        Self::new(sample_rate, preset)
    }

    /// Output sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.voice.sample_rate()
    }

    /// The voice settings.
    #[must_use]
    pub fn settings(&self) -> &VoiceSettings {
        &self.settings
    }

    /// Change the voice. Affects sentences compiled from now on.
    pub fn set_settings(&mut self, settings: impl Into<VoiceSettings>) {
        self.settings = settings.into();
        self.settings.apply(&mut self.voice);
    }

    /// The underlying voice, for direct articulation control.
    pub fn voice_mut(&mut self) -> &mut Trombone {
        &mut self.voice
    }

    /// Compile text with this talker's voice, without queueing it.
    #[must_use]
    pub fn compile(&self, text: &str) -> Utterance {
        Utterance::from_text(text, &self.settings.prosody())
    }

    /// Queue text to be spoken after anything already queued.
    pub fn say(&mut self, text: &str) {
        let u = self.compile(text);
        self.say_utterance(u);
    }

    /// Queue a compiled utterance.
    pub fn say_utterance(&mut self, utterance: Utterance) {
        if !utterance.is_empty() {
            self.queue.push_back(utterance);
        }
    }

    /// Make a vocalization now (wave 3; see [`Trombone::vocalize`]): a
    /// sentence being said waits for it and carries on after it (see
    /// [`Speaker::interject`]). Returns its length in seconds (`None` on
    /// the classic engine). Tags in text (`"[laugh] That was close."`) do
    /// the same within a sentence.
    pub fn vocalize(&mut self, v: impl Into<crate::Vocal>) -> Option<f32> {
        self.speaker.interject(&mut self.voice, v)
    }

    /// Break off what is being said, mid-word, with a vocalization (a
    /// cough, a sigh, a laugh, a throat clearing) and forget the queue (see
    /// [`Speaker::interrupt_with`]): the voice is in it within 10 ms, with
    /// no click, and whatever is said next ([`say`](Self::say)) follows it.
    /// Returns its length in seconds (`None` on the classic engine, where
    /// the speech simply stops).
    pub fn interrupt_with(&mut self, v: impl Into<crate::Vocal>) -> Option<f32> {
        self.queue.clear();
        self.speaker.interrupt_with(&mut self.voice, v).1
    }

    /// Knock the speaker's body while it speaks (see
    /// [`Trombone::perturb`]): a push on the lungs (cmH2O; negative a gasp)
    /// and a clamp of the jaw (`0‥1`), each dying away by itself.
    pub fn perturb(&mut self, delta_lung_pressure: f32, jaw_clamp: f32) {
        self.voice.perturb(delta_lung_pressure, jaw_clamp);
    }

    /// Stop now and forget the queue; the voice fades out.
    pub fn interrupt(&mut self) {
        self.queue.clear();
        self.speaker.stop(&mut self.voice);
    }

    /// Whether anything is being said or waiting to be said.
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.speaker.is_speaking() || !self.queue.is_empty()
    }

    /// One output sample.
    pub fn process(&mut self) -> f32 {
        // (What is said after a vocalization made on its own, as after an
        // interruption, waits for it to be over.)
        if !self.speaker.is_speaking()
            && !self.queue.is_empty()
            && !self.voice.is_vocalizing()
            && let Some(next) = self.queue.pop_front()
        {
            self.speaker.speak(next);
        }
        self.step()
    }

    /// One sample of the speaker's voice; while the articulators are
    /// recorded, their frame before it whenever one is due.
    fn step(&mut self) -> f32 {
        if let Some(frames) = &mut self.frames {
            if self.frame_phase >= FRAME_SAMPLES {
                self.frame_phase -= FRAME_SAMPLES;
                frames.push(self.voice.articulators());
            }
            self.frame_phase += 48_000.0 / f64::from(self.voice.sample_rate());
        }
        self.speaker.process(&mut self.voice)
    }

    /// Keep a frame of the articulators every
    /// [`ArticulatorFrame::PERIOD`] (512 samples at 48 kHz, 10.67 ms) of
    /// what is rendered from now on, the first at the next sample, for
    /// lip-sync baked with the sound: frame `k` is the mouth at second
    /// `k · PERIOD` of the samples. [`take_articulators`](Self::take_articulators)
    /// hands them over. Keeping them allocates as they accumulate: for a
    /// voice rendered in an audio callback read
    /// [`Trombone::articulators`] or a stream's frames instead.
    pub fn record_articulators(&mut self, on: bool) {
        self.frames = on.then(Vec::new);
        self.frame_phase = FRAME_SAMPLES;
    }

    /// The frames recorded since [`record_articulators`](Self::record_articulators)
    /// or the last call; recording goes on.
    pub fn take_articulators(&mut self) -> Vec<ArticulatorFrame> {
        self.frames
            .as_mut()
            .map(core::mem::take)
            .unwrap_or_default()
    }

    /// [`bake`](Self::bake) with the mouth that speaks it: the samples, and
    /// a frame of the articulators for every 10.67 ms of them (frame `k` at
    /// sample `k · 512` at 48 kHz).
    #[must_use]
    pub fn bake_with_articulators(&mut self, text: &str) -> (Vec<f32>, Vec<ArticulatorFrame>) {
        let recording = self.frames.is_some();
        self.record_articulators(true);
        let out = self.bake(text);
        let frames = self.take_articulators();
        if !recording {
            self.record_articulators(false);
        }
        (out, frames)
    }

    /// A bake starts the recorded frames over, the first at its first
    /// sample.
    fn restart_frames(&mut self) {
        if let Some(frames) = &mut self.frames {
            frames.clear();
            self.frame_phase = FRAME_SAMPLES;
        }
    }

    /// Fill `out` with mono samples.
    pub fn render(&mut self, out: &mut [f32]) {
        for s in out {
            *s = self.process();
        }
    }

    /// Hear the talker in two channels (see [`Trombone::set_stereo`]).
    pub fn set_stereo(&mut self, stereo: Stereo) {
        self.voice.set_stereo(stereo);
    }

    /// Move the listener (see [`Trombone::set_listener`]): turn the
    /// talker's head away with `set_listener(Listener { azimuth: 180.0,
    /// ..Listener::FRONT })`.
    pub fn set_listener(&mut self, listener: Listener) {
        self.voice.set_listener(listener);
    }

    /// Put the voice into a vocal effort state while it speaks (see
    /// [`Trombone::set_effort`]); [`VoiceSettings::with_effort`] gives a
    /// talker the state from the start, and a tag in a text (`[whisper]`)
    /// switches from there on.
    pub fn set_effort(&mut self, effort: crate::Effort) {
        self.voice.set_effort(effort);
    }

    /// Morph the voice towards another body while it speaks (see
    /// [`Trombone::set_morph`]); [`VoiceSettings::with_morph`] gives a
    /// talker the body from the start.
    pub fn set_morph(&mut self, target: crate::MorphTarget) {
        self.voice.set_morph(target);
    }

    /// One output frame, `[left, right]`.
    pub fn process_stereo(&mut self) -> [f32; 2] {
        self.process();
        self.voice.frame()
    }

    /// Fill `out` with interleaved frames, left then right.
    pub fn render_stereo(&mut self, out: &mut [f32]) {
        for frame in out.as_chunks_mut::<2>().0 {
            *frame = self.process_stereo();
        }
    }

    /// Render `text` to a mono buffer at [`sample_rate`](Self::sample_rate),
    /// from silence, including a short fade-out tail. Drops anything queued.
    ///
    /// The level follows the voice (a whisper is quiet); see
    /// [`normalize`] for a uniform peak.
    #[must_use]
    pub fn bake(&mut self, text: &str) -> Vec<f32> {
        let u = self.compile(text);
        self.bake_utterance(u)
    }

    /// [`bake`](Self::bake) for a compiled utterance.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn bake_utterance(&mut self, utterance: Utterance) -> Vec<f32> {
        self.queue.clear();
        self.speaker.stop(&mut self.voice);
        self.voice.reset();
        let len = ((utterance.duration() + TAIL) * self.sample_rate()) as usize;
        let mut out = alloc::vec![0.0; len];
        self.speaker.speak(utterance);
        self.restart_frames();
        for s in &mut out {
            *s = self.step();
        }
        out
    }

    /// [`bake`](Self::bake) in two channels, interleaved left then right
    /// (see [`set_stereo`](Self::set_stereo)).
    #[must_use]
    pub fn bake_stereo(&mut self, text: &str) -> Vec<f32> {
        let u = self.compile(text);
        self.bake_utterance_stereo(u)
    }

    /// [`bake_stereo`](Self::bake_stereo) for a compiled utterance.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn bake_utterance_stereo(&mut self, utterance: Utterance) -> Vec<f32> {
        self.queue.clear();
        self.speaker.stop(&mut self.voice);
        self.voice.reset();
        let len = ((utterance.duration() + TAIL) * self.sample_rate()) as usize;
        let mut out = alloc::vec![0.0; 2 * len];
        self.speaker.speak(utterance);
        self.restart_frames();
        for frame in out.as_chunks_mut::<2>().0 {
            self.step();
            *frame = self.voice.frame();
        }
        out
    }
}

/// Scale `samples` so the loudest one reaches `peak` (e.g. `0.9`). Silence
/// is left alone.
pub fn normalize(samples: &mut [f32], peak: f32) {
    let max = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if max > 1e-6 {
        let gain = peak / max;
        for s in samples {
            *s *= gain;
        }
    }
}

/// Render `text` in one go: `bake(text, Preset::Bright, 44_100.0)`.
#[must_use]
pub fn bake(text: &str, voice: impl Into<VoiceSettings>, sample_rate: f32) -> Vec<f32> {
    Talker::new(sample_rate, voice).bake(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let n = x.len().max(1) as f32;
        (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt()
    }

    #[test]
    fn every_preset_speaks() {
        for p in Preset::ALL {
            let out = bake("Hello there, how are you?", p, 22_050.0);
            assert!(out.iter().all(|s| s.is_finite()), "{p}");
            assert!(rms(&out) > 0.002, "{p} too quiet: {}", rms(&out));
            assert!(out.iter().all(|s| s.abs() < 2.0), "{p} too loud");
        }
    }

    #[test]
    fn faster_presets_are_shorter() {
        let giant = bake("The quick brown fox.", Preset::Giant, 16_000.0);
        let critter = bake("The quick brown fox.", Preset::Critter, 16_000.0);
        assert!(critter.len() * 3 < giant.len() * 2);
    }

    #[test]
    fn bake_is_repeatable() {
        let mut t = Talker::new(16_000.0, Preset::Bright);
        let a = t.bake("Again.");
        let b = t.bake("Again.");
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn queued_sentences_play_in_order_then_stop() {
        let mut t = Talker::new(16_000.0, Preset::Robot);
        t.say("One.");
        t.say("Two.");
        assert!(t.is_speaking());
        let mut buf = alloc::vec![0.0; 16_000 * 4];
        t.render(&mut buf);
        assert!(!t.is_speaking());
        t.say("Three.");
        t.interrupt();
        assert!(!t.is_speaking());
    }

    #[test]
    fn presets_round_trip_by_name() {
        for p in Preset::ALL {
            assert_eq!(Preset::from_name(p.name()), Some(p));
        }
        assert_eq!(Preset::from_name("CRITTER"), Some(Preset::Critter));
    }

    #[test]
    fn normalize_hits_the_peak() {
        let mut x = [0.1, -0.4, 0.2];
        normalize(&mut x, 0.8);
        assert!((x[1] + 0.8).abs() < 1e-6);
        let mut silent = [0.0; 4];
        normalize(&mut silent, 0.8);
        assert_eq!(silent, [0.0; 4]);
    }
}
