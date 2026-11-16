//! Audio thread: one [`Trombone`], a [`Speaker`] for typed text, and
//! synthie's reverb; it also records takes (see [`record`]). Talks to the
//! UI over bounded channels; the callback never allocates or blocks.

use crate::record::{self, Clock, Recorder, Tap};
use crate::take::{Logged, Performer};
use crate::telemetry::{self, GlottisGauge, GlottisWatch, PROBE_RATE, Probe, ShapeKey};
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use synthie::audio::fx::Reverb;
use synthie_trombone::listen::{Heard, Listener};
use synthie_trombone::mimic::{Gesture, Mimic};
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{
    Constriction, Engine as SoundModel, MAX_CONSTRICTIONS, N, NOSE_LEN, Phoneme, Trombone,
    VoiceSettings,
};

/// Snapshots sent to the UI per second.
const SNAPSHOT_RATE: f32 = 60.0;

/// Samples kept for the oscilloscope.
pub const SCOPE_LEN: usize = 1024;

/// UI → audio commands.
#[derive(Debug)]
pub enum Cmd {
    NoteOn(f32),
    NoteOff,
    Note(f32),
    Tongue(f32, f32),
    Constriction(usize, Option<Constriction>),
    Velum(f32),
    Tenseness(f32),
    Vibrato(f32),
    Wobble(bool),
    Drone(bool),
    Reverb(f32),
    Voice(VoiceSettings),
    /// Lung pressure of the physical engine, cmH2O.
    Lung(f32),
    /// Switch the sound model.
    Model(SoundModel),
    /// Move to a vowel's shape, as the current sound model tunes it.
    Vowel(Phoneme),
    /// Speak an utterance; the number is the one the recorder gave it.
    Speak(Utterance, u32),
    StopSpeaking,
    /// Copy what the microphone hears (physical engine), or stop.
    Mimic(bool),
    /// Hold a vowel shape of the physical engine (Formant Lock).
    Shape(synthie_trombone::inverse::Shape),
    /// Start (`true`) or stop recording a take.
    Record(bool),
    /// Play a gesture on the voice's gesture score, kept in a slot.
    Push(u8, synthie_trombone::Gesture),
    /// End the gesture in a slot over this many seconds.
    EndGesture(u8, f32),
}

/// Audio → UI state, sent about 60 times a second. `Copy`, so sending it
/// through a bounded channel never allocates.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub diameters: [f32; N],
    pub targets: [f32; N],
    pub amplitudes: [f32; N],
    pub nose: [f32; NOSE_LEN],
    pub nose_amplitudes: [f32; NOSE_LEN],
    pub constrictions: [Option<Constriction>; MAX_CONSTRICTIONS],
    pub tongue: (f32, f32),
    pub velum: f32,
    pub intensity: f32,
    pub frequency: f32,
    pub tenseness: f32,
    /// Segment index while speaking.
    pub speaking: Option<usize>,
    pub scope: [f32; SCOPE_LEN],
    pub level: f32,
    /// What the microphone last heard, while mimicking.
    pub heard: Option<Heard>,
    /// How far the take being recorded has got.
    pub take: Option<Clock>,
    /// Samples the audio thread had rendered when it took the snapshot.
    pub at: u64,
    /// The glottis's pressures and closure over the last snapshots.
    pub glottis: GlottisGauge,
    /// What each tier of the gesture score is doing, in `Tier::ALL` order,
    /// and whether the score's gestures act on it (`false`: read off the
    /// controls; see [`telemetry::tiers`]).
    pub tiers: [synthie_trombone::TierActivity; synthie_trombone::Tier::COUNT],
    pub scored: [bool; synthie_trombone::Tier::COUNT],
}

impl Snapshot {
    pub(crate) fn capture(voice: &Trombone, speaker: &Speaker, scope: &Scope) -> Self {
        let mut constrictions = [None; MAX_CONSTRICTIONS];
        for (k, c) in constrictions.iter_mut().enumerate() {
            *c = voice.constriction(k);
        }
        let (tiers, scored) = telemetry::tiers(voice);
        Self {
            diameters: *voice.diameters(),
            targets: *voice.target_diameters(),
            amplitudes: *voice.amplitudes(),
            nose: *voice.nose_diameters(),
            nose_amplitudes: *voice.nose_amplitudes(),
            constrictions,
            tongue: voice.tongue(),
            velum: voice.velum(),
            intensity: voice.intensity(),
            frequency: voice.frequency(),
            tenseness: voice.tenseness(),
            speaking: speaker.current().map(|(i, _)| i),
            scope: scope.ordered(),
            level: scope.level,
            heard: None,
            take: None,
            at: 0,
            glottis: GlottisGauge::default(),
            tiers,
            scored,
        }
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::capture(&Trombone::new(48_000.0), &Speaker::new(), &Scope::default())
    }
}

pub(crate) struct Scope {
    buf: [f32; SCOPE_LEN],
    pos: usize,
    level: f32,
}

impl Default for Scope {
    fn default() -> Self {
        Self {
            buf: [0.0; SCOPE_LEN],
            pos: 0,
            level: 0.0,
        }
    }
}

impl Scope {
    pub(crate) fn push(&mut self, s: f32) {
        self.buf[self.pos] = s;
        self.pos = (self.pos + 1) % SCOPE_LEN;
        self.level = (self.level * 0.9995).max(s.abs());
    }

    fn ordered(&self) -> [f32; SCOPE_LEN] {
        let mut out = [0.0; SCOPE_LEN];
        let (a, b) = self.buf.split_at(self.pos);
        out[..b.len()].copy_from_slice(b);
        out[b.len()..].copy_from_slice(a);
        out
    }
}

pub(crate) struct Engine {
    /// The voice and the speaker, played by commands.
    core: Performer,
    reverb: Reverb,
    scope: Scope,
    snapshot_every: usize,
    since_snapshot: usize,
    cmd_rx: Receiver<Cmd>,
    snap_tx: Sender<Snapshot>,
    trash_tx: Sender<Utterance>,
    /// Gestures from the microphone thread, with what it heard.
    ear_rx: Option<Receiver<(Gesture, Heard)>>,
    /// The voice's tract length, for the microphone thread (`f32` bits).
    size: Arc<AtomicU32>,
    heard: Option<Heard>,
    /// Where takes are recorded.
    tap: Tap,
    /// Samples rendered since the engine started.
    clock: u64,
    /// The glottis's readings between snapshots.
    watch: GlottisWatch,
    /// Copies of the voice for the formant meter, and the shape last sent.
    probe_tx: Sender<Probe>,
    probe_shape: ShapeKey,
    probe_every: u64,
    probe_last: u64,
}

impl Engine {
    fn apply(&mut self, cmd: Cmd) {
        let logged = match cmd {
            Cmd::Record(on) => {
                self.record(on);
                return;
            }
            Cmd::Speak(u, id) => {
                let old = self.core.speak(u);
                self.trash(old);
                Logged::Speak(id)
            }
            Cmd::Mimic(on) => {
                self.heard = None;
                Logged::Mimic(on && self.ear_rx.is_some())
            }
            Cmd::NoteOn(note) => Logged::NoteOn(note),
            Cmd::NoteOff => Logged::NoteOff,
            Cmd::Note(note) => Logged::Note(note),
            Cmd::Tongue(i, d) => Logged::Tongue(i, d),
            Cmd::Constriction(slot, c) => Logged::Constriction(slot, c),
            Cmd::Velum(x) => Logged::Velum(x),
            Cmd::Tenseness(t) => Logged::Tenseness(t),
            Cmd::Vibrato(a) => Logged::Vibrato(a),
            Cmd::Wobble(w) => Logged::Wobble(w),
            Cmd::Drone(on) => Logged::Drone(on),
            Cmd::Reverb(mix) => Logged::Reverb(mix),
            Cmd::Voice(settings) => Logged::Voice(settings),
            Cmd::Lung(p) => Logged::Lung(p),
            Cmd::Model(model) => Logged::Model(model),
            Cmd::Vowel(p) => Logged::Vowel(p),
            Cmd::StopSpeaking => Logged::StopSpeaking,
            Cmd::Shape(shape) => Logged::Shape(shape),
            Cmd::Push(slot, g) => Logged::Push(slot, g),
            Cmd::EndGesture(slot, ramp) => Logged::EndGesture(slot, ramp),
        };
        if let Some(old) = self.core.apply(&logged) {
            self.trash(old);
        }
        self.tap.command(logged);
        self.size.store(
            self.core.voice().tract_length().to_bits(),
            Ordering::Relaxed,
        );
    }

    /// Start or stop a take. A take starts from silence: speech stops and
    /// the voice is rebuilt from its controls, as a replay builds it, so
    /// that it can be replayed sample for sample.
    fn record(&mut self, on: bool) {
        if !on {
            self.tap.end();
        } else if !self.tap.is_recording() {
            let old = self.core.restart();
            self.trash(old);
            self.tap.begin(&self.core.controls());
        }
    }

    /// Hand an utterance back so it is freed off this thread.
    fn trash(&self, old: Utterance) {
        let _ = self.trash_tx.try_send(old);
    }

    /// Carry out what the microphone thread asks for.
    fn follow(&mut self) {
        let Some(rx) = &self.ear_rx else { return };
        while let Ok((gesture, heard)) = rx.try_recv() {
            if self.core.follow(&gesture) {
                self.heard = Some(heard);
                self.tap.command(Logged::Gesture(gesture));
            }
        }
    }

    /// Fill one block of the output: what the audio callback does.
    /// Commands are taken off the queue at the start of the block, so a
    /// take stamps each with the sample it is applied before.
    fn render(&mut self, data: &mut [f32], channels: usize) {
        self.tap.service();
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            self.apply(cmd);
        }
        self.follow();
        for frame in data.chunks_mut(channels.max(1)) {
            let dry = self.core.sample();
            self.watch.sample(self.core.voice());
            self.clock += 1;
            self.tap.sample(dry);
            let s = (self.reverb.process(dry, self.core.reverb()) * OUTPUT_GAIN).clamp(-1.0, 1.0);
            frame.fill(s);
            self.scope.push(s);

            self.since_snapshot += 1;
            if self.since_snapshot >= self.snapshot_every {
                self.since_snapshot = 0;
                let mut snap =
                    Snapshot::capture(self.core.voice(), self.core.speaker(), &self.scope);
                snap.heard = self.heard;
                snap.take = self.tap.clock();
                snap.at = self.clock;
                snap.glottis = self.watch.read(self.core.voice());
                let _ = self.snap_tx.try_send(snap);
                self.probe();
            }
        }
    }
}

impl Engine {
    /// Send the formant meter a copy of the voice, if the shape has
    /// changed, the meter has taken the last one, and it is time. The
    /// copy goes into a channel slot made up front: nothing is allocated.
    fn probe(&mut self) {
        if self.clock < self.probe_last + self.probe_every || !self.probe_tx.is_empty() {
            return;
        }
        let shape = ShapeKey::of(self.core.voice());
        if shape == self.probe_shape {
            return;
        }
        let probe = Probe {
            voice: self.core.voice().clone(),
            at: self.clock,
        };
        if self.probe_tx.try_send(probe).is_ok() {
            self.probe_shape = shape;
            self.probe_last = self.clock;
        }
    }
}

/// Gain of the output after the reverb.
const OUTPUT_GAIN: f32 = 0.9;

/// Channels the UI keeps.
pub struct Link {
    pub cmd: Sender<Cmd>,
    pub snapshots: Receiver<Snapshot>,
    pub trash: Receiver<Utterance>,
    /// Copies of the voice for the formant meter.
    pub probes: Receiver<Probe>,
    pub sample_rate: f32,
    /// Whether a microphone is open for mimic mode, and which.
    pub microphone: bool,
    pub microphone_name: Option<String>,
    /// Records takes (`None`: recording is not available).
    pub recorder: Option<Recorder>,
}

/// Open the default input device and listen to it: every measurement is
/// turned into a gesture for the voice and sent to the audio thread. The
/// search for a vowel shape runs here, off the output callback.
fn listen(
    size: &Arc<AtomicU32>,
    ear_tx: &Sender<(Gesture, Heard)>,
    wanted: Option<&str>,
) -> Result<(cpal::Stream, String)> {
    let host = cpal::default_host();
    let name = |d: &cpal::Device| {
        d.description()
            .map_or_else(|_| "unnamed".to_string(), |x| x.name().to_string())
    };
    // The system's default first, then sound servers, then the rest: on
    // some systems the default capture device cannot be opened while a
    // sound server holds the card.
    let mut devices: Vec<cpal::Device> = host.default_input_device().into_iter().collect();
    let mut others: Vec<cpal::Device> = host
        .input_devices()
        .context("failed to list audio inputs")?
        .collect();
    others.sort_by_key(|d| {
        let n = name(d).to_lowercase();
        !(n.contains("pipewire") || n.contains("pulse"))
    });
    devices.extend(others);
    if let Some(wanted) = wanted {
        let wanted = wanted.to_lowercase();
        devices.retain(|d| name(d).to_lowercase().contains(&wanted));
    }
    let mut tried = Vec::new();
    for device in devices {
        let label = name(&device);
        match open(&device, Arc::clone(size), ear_tx.clone()) {
            Ok(stream) => return Ok((stream, label)),
            Err(e) => tried.push(format!("{label}: {e:#}")),
        }
    }
    anyhow::bail!(
        "no microphone could be opened for --mimic{}",
        if tried.is_empty() {
            ": no audio input found".to_string()
        } else {
            format!(":\n  {}", tried.join("\n  "))
        }
    )
}

/// Open one input device and start listening to it.
fn open(
    device: &cpal::Device,
    size: Arc<AtomicU32>,
    ear_tx: Sender<(Gesture, Heard)>,
) -> Result<cpal::Stream> {
    let config = device
        .default_input_config()
        .context("failed to query its config")?;
    #[allow(clippy::cast_precision_loss)]
    let sample_rate = config.sample_rate() as f32;
    let channels = usize::from(config.channels()).max(1);
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };
    anyhow::ensure!(
        (12_000.0..=96_000.0).contains(&sample_rate),
        "unsupported sample rate {sample_rate}"
    );
    let mut listener = Listener::new(sample_rate);
    let mut mimic = Mimic::new(1.0);
    let stream = device
        .build_input_stream(
            stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                mimic.set_tract_length(f32::from_bits(size.load(Ordering::Relaxed)));
                for frame in data.chunks(channels) {
                    #[allow(clippy::cast_precision_loss)]
                    let mono = frame.iter().sum::<f32>() / channels as f32;
                    if let Some(heard) = listener.push(mono) {
                        let _ = ear_tx.try_send((mimic.follow(&heard), heard));
                    }
                }
            },
            |_err| {},
            None,
        )
        .context("failed to open it")?;
    stream.play().context("failed to start it")?;
    Ok(stream)
}

/// The body of the output callback: the engine filling interleaved blocks
/// of `channels` channels. The device runs exactly this (see [`start`]),
/// and so do the tests, with plain buffers.
pub(crate) fn output(mut engine: Engine, channels: usize) -> impl FnMut(&mut [f32]) + Send {
    move |data: &mut [f32]| engine.render(data, channels)
}

/// The engine and the UI's end of its channels, without a sound device:
/// [`output`] turns the engine into the device's callback. With `takes`,
/// takes are recorded into that directory.
pub(crate) fn wire(
    sample_rate: f32,
    ear_rx: Option<Receiver<(Gesture, Heard)>>,
    size: Arc<AtomicU32>,
    takes: Option<PathBuf>,
) -> (Engine, Link) {
    let (cmd_tx, cmd_rx) = bounded(256);
    let (snap_tx, snap_rx) = bounded(4);
    let (trash_tx, trash_rx) = bounded(16);
    let (probe_tx, probe_rx) = bounded(1);
    let (tap, recorder) = match takes {
        Some(dir) => {
            let (events, saved) = record::spawn();
            (
                Tap::new(Some(events.clone())),
                Some(Recorder::new(dir, events, saved)),
            )
        }
        None => (Tap::new(None), None),
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let snapshot_every = (sample_rate / SNAPSHOT_RATE) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let probe_every = (sample_rate / PROBE_RATE) as u64;
    let mut reverb = Reverb::new();
    reverb.set_params(0.55, 0.45);
    let engine = Engine {
        core: Performer::new(sample_rate),
        reverb,
        scope: Scope::default(),
        snapshot_every,
        since_snapshot: 0,
        cmd_rx,
        snap_tx,
        trash_tx,
        ear_rx,
        size,
        heard: None,
        tap,
        clock: 0,
        watch: GlottisWatch::default(),
        probe_tx,
        probe_shape: ShapeKey::NONE,
        probe_every,
        probe_last: 0,
    };
    let link = Link {
        cmd: cmd_tx,
        snapshots: snap_rx,
        trash: trash_rx,
        probes: probe_rx,
        sample_rate,
        microphone: false,
        microphone_name: None,
        recorder,
    };
    (engine, link)
}

/// Open the default output device and start the engine; with `mimic`, the
/// default microphone too. Takes are recorded into `takes`. The streams
/// stop when dropped.
pub fn start(
    mimic: bool,
    microphone: Option<&str>,
    takes: PathBuf,
) -> Result<(Vec<cpal::Stream>, Link)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no default audio output device")?;
    let config = device
        .default_output_config()
        .context("failed to query default output config")?;
    #[allow(clippy::cast_precision_loss)]
    let sample_rate = config.sample_rate() as f32;
    let channels = usize::from(config.channels());
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let size = Arc::new(AtomicU32::new(1.0f32.to_bits()));
    let mut streams = Vec::new();
    let mut heard_on = None;
    let ear_rx = if mimic {
        let (ear_tx, ear_rx) = bounded(64);
        let (stream, name) = listen(&size, &ear_tx, microphone)?;
        streams.push(stream);
        heard_on = Some(name);
        Some(ear_rx)
    } else {
        None
    };
    let (engine, mut link) = wire(sample_rate, ear_rx, size, Some(takes));
    let mut callback = output(engine, channels);
    link.microphone = heard_on.is_some();
    link.microphone_name = heard_on;

    let stream = device
        .build_output_stream(
            stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| callback(data),
            |_err| {},
            None,
        )
        .context("failed to build output stream")?;
    stream.play().context("failed to start audio stream")?;
    streams.push(stream);
    Ok((streams, link))
}
