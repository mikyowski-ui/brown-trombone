//! Recording takes (see [`take`](crate::take) for what a take is).
//!
//! Three parts, on three threads:
//!
//! - [`Tap`], in the audio callback: stamps each command with the sample it
//!   is applied before and collects the sound in fixed blocks. It never
//!   allocates, locks or touches a file; it hands everything to a bounded
//!   queue whose slots are allocated up front, and counts what does not fit.
//! - the writer ([`spawn`]): takes the queue and writes the two files.
//! - [`Recorder`], in the app: numbers the takes, creates their files, and
//!   passes on what only the app knows (the text of speech, the voice's
//!   preset and persona).

use crate::take::{self, Controls, Logged, Speech};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use synthie_trombone::speech::Prosody;

/// Samples of sound in one message from the audio thread.
pub const CHUNK: usize = 256;
/// Messages the queue to the writer holds: about 11 s of sound at 48 kHz.
pub const QUEUE: usize = 2048;
/// How often, in samples written, the writer brings the WAV header up to
/// date and flushes both files, so that a take survives a crash.
const REFRESH: u64 = 48_000;
/// How long the app waits for room in the queue.
const PATIENCE: Duration = Duration::from_millis(200);
/// How often the writer looks at the queue. It looks rather than waits: a
/// waiting receiver would make every send from the audio thread take a
/// lock to wake it.
const POLL: Duration = Duration::from_millis(5);

/// A message to the writer.
// The sound blocks make this large; boxing them would allocate in the
// audio callback. The queue's slots are allocated once, up front.
#[allow(clippy::large_enum_variant)]
pub enum Event {
    /// From the app: the files of the next take.
    Open(Box<Opening>),
    /// From the app: what speech number `id` says.
    Said(u32, Box<Speech>),
    /// From the app: the take did not start after all; remove its files.
    Cancel,
    /// From the audio thread: the take starts, with these controls.
    Start(Controls),
    /// From the audio thread: a command, applied before sample `at`.
    Command { at: u64, cmd: Logged },
    /// From the audio thread: `len` samples of sound from sample `at`.
    Audio {
        at: u64,
        len: usize,
        samples: [f32; CHUNK],
    },
    /// From the audio thread: the take ends before sample `at`.
    Stop {
        at: u64,
        dropped_commands: u64,
        dropped_samples: u64,
    },
}

/// The files of a take and what the app knows about its voice.
pub struct Opening {
    pub name: String,
    pub log: File,
    pub wav: File,
    pub log_path: PathBuf,
    pub wav_path: PathBuf,
    pub sample_rate: f32,
    /// Informational part of the header (preset, persona, the app's view).
    pub app: Value,
}

/// A take as saved, for the app to report.
#[derive(Clone, Debug)]
pub struct Saved {
    pub log_path: PathBuf,
    pub samples: u64,
    pub sample_rate: f32,
    pub commands: u64,
    pub dropped_commands: u64,
    pub dropped_samples: u64,
    /// The first thing that went wrong writing it.
    pub error: Option<String>,
}

impl Saved {
    #[must_use]
    pub fn seconds(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let n = self.samples as f32;
        n / self.sample_rate
    }
}

/// How far the take being recorded has got, for the display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clock {
    pub samples: u64,
    /// Commands and blocks of sound that did not fit in the queue.
    pub dropped: u64,
}

// ----- the audio thread's side ----------------------------------------------

/// A take in progress, on the audio thread.
struct Running {
    /// Samples since the take began.
    at: u64,
    /// Whether the writer has been told the take started.
    started: bool,
    controls: Controls,
    block: [f32; CHUNK],
    fill: usize,
    dropped_commands: u64,
    dropped_samples: u64,
}

/// The audio callback's end of the recording queue. Never allocates,
/// locks or blocks.
pub struct Tap {
    tx: Option<Sender<Event>>,
    take: Option<Running>,
    /// A stop the queue had no room for yet.
    stop: Option<Event>,
}

impl Tap {
    /// A tap into `tx`; with `None` it records nothing.
    #[must_use]
    pub fn new(tx: Option<Sender<Event>>) -> Self {
        Self {
            tx,
            take: None,
            stop: None,
        }
    }

    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.take.is_some()
    }

    /// Start a take at the next sample, from these controls.
    pub fn begin(&mut self, controls: &Controls) {
        if self.take.is_some() || self.tx.is_none() {
            return;
        }
        self.take = Some(Running {
            at: 0,
            started: false,
            controls: *controls,
            block: [0.0; CHUNK],
            fill: 0,
            dropped_commands: 0,
            dropped_samples: 0,
        });
        self.service();
    }

    /// End the take after the last sample recorded.
    pub fn end(&mut self) {
        let Some(mut r) = self.take.take() else {
            return;
        };
        if let Some(tx) = &self.tx {
            r.flush(tx);
        }
        self.stop = Some(Event::Stop {
            at: r.at,
            dropped_commands: r.dropped_commands,
            dropped_samples: r.dropped_samples,
        });
        self.service();
    }

    /// Send what is still waiting (a stop, a start): call at the start of
    /// every audio block.
    pub fn service(&mut self) {
        let Some(tx) = &self.tx else { return };
        if let Some(stop) = self.stop.take()
            && let Err(e) = tx.try_send(stop)
        {
            self.stop = Some(e.into_inner());
            return;
        }
        if let Some(r) = &mut self.take
            && !r.started
        {
            r.started = tx.try_send(Event::Start(r.controls)).is_ok();
        }
    }

    /// Log a command, applied before the next sample.
    #[allow(clippy::large_types_passed_by_value)] // moved on into the channel
    pub fn command(&mut self, cmd: Logged) {
        let (Some(r), Some(tx)) = (&mut self.take, &self.tx) else {
            return;
        };
        if !r.started || tx.try_send(Event::Command { at: r.at, cmd }).is_err() {
            r.dropped_commands += 1;
        }
    }

    /// Record one sample of sound.
    pub fn sample(&mut self, s: f32) {
        let (Some(r), Some(tx)) = (&mut self.take, &self.tx) else {
            return;
        };
        r.block[r.fill] = s;
        r.fill += 1;
        r.at += 1;
        if r.fill == CHUNK {
            r.flush(tx);
        }
    }

    /// How far the take has got, while recording.
    #[must_use]
    pub fn clock(&self) -> Option<Clock> {
        self.take.as_ref().map(|r| Clock {
            samples: r.at,
            dropped: r.dropped_commands + r.dropped_samples.div_ceil(CHUNK as u64),
        })
    }
}

impl Running {
    /// Send the samples collected so far.
    fn flush(&mut self, tx: &Sender<Event>) {
        if self.fill == 0 {
            return;
        }
        let event = Event::Audio {
            at: self.at - self.fill as u64,
            len: self.fill,
            samples: self.block,
        };
        if !self.started || tx.try_send(event).is_err() {
            self.dropped_samples += self.fill as u64;
        }
        self.fill = 0;
    }
}

// ----- the writer -----------------------------------------------------------

/// Start the writer thread. The audio thread and the app send it
/// [`Event`]s; it reports each take it saves. It ends when every sender
/// is gone.
///
/// # Panics
///
/// When the thread cannot be started.
#[must_use]
pub fn spawn() -> (Sender<Event>, Receiver<Saved>) {
    let (tx, rx) = bounded(QUEUE);
    let (saved_tx, saved_rx) = bounded(16);
    std::thread::Builder::new()
        .name("take writer".into())
        .spawn(move || write_takes(&rx, &saved_tx))
        .expect("failed to start the take writer");
    (tx, saved_rx)
}

/// A take being written.
struct Writing {
    name: String,
    log_path: PathBuf,
    wav_path: PathBuf,
    sample_rate: f32,
    app: Value,
    log: BufWriter<File>,
    wav: BufWriter<File>,
    started: bool,
    /// Samples written to the WAV file.
    written: u64,
    since_refresh: u64,
    commands: u64,
    said: HashMap<u32, Speech>,
    error: Option<String>,
}

impl Writing {
    fn new(open: Opening) -> Self {
        Self {
            name: open.name,
            log_path: open.log_path,
            wav_path: open.wav_path,
            sample_rate: open.sample_rate,
            app: open.app,
            log: BufWriter::new(open.log),
            wav: BufWriter::new(open.wav),
            started: false,
            written: 0,
            since_refresh: 0,
            commands: 0,
            said: HashMap::new(),
            error: None,
        }
    }

    /// Keep the first error; once there is one, stop writing.
    fn check(&mut self, r: io::Result<()>) {
        if let Err(e) = r
            && self.error.is_none()
        {
            self.error = Some(e.to_string());
        }
    }

    fn line(&mut self, text: &str) {
        if self.error.is_some() {
            return;
        }
        let r = self
            .log
            .write_all(text.as_bytes())
            .and_then(|()| self.log.write_all(b"\n"));
        self.check(r);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn wav_header(&mut self) {
        if self.error.is_some() {
            return;
        }
        let rate = self.sample_rate.round() as u32;
        let header = take::wav_header(rate, u32::try_from(self.written).unwrap_or(u32::MAX));
        let r = (|| {
            self.wav.seek(SeekFrom::Start(0))?;
            self.wav.write_all(&header)?;
            self.wav.seek(SeekFrom::End(0))?;
            self.wav.flush()?;
            self.log.flush()
        })();
        self.check(r);
    }

    fn start(&mut self, controls: &Controls) {
        if self.started {
            return;
        }
        self.started = true;
        let header = take::header_line(&self.name, self.sample_rate, controls, self.app.take());
        self.line(&header);
        self.wav_header();
    }

    fn command(&mut self, at: u64, cmd: &Logged) {
        let speech = match cmd {
            Logged::Speak(id) => self.said.remove(id),
            _ => None,
        };
        let line = take::command_line(at, self.sample_rate, cmd, speech.as_ref());
        self.line(&line);
        self.commands += 1;
    }

    /// Write sound up to sample `at` as silence (sound that was lost).
    fn pad(&mut self, at: u64) {
        while self.written < at && self.error.is_none() {
            let r = self.wav.write_all(&0f32.to_le_bytes());
            self.check(r);
            self.written += 1;
        }
    }

    fn audio(&mut self, at: u64, samples: &[f32]) {
        self.pad(at);
        let skip = usize::try_from(self.written - at).unwrap_or(usize::MAX);
        for s in samples.iter().skip(skip) {
            if self.error.is_some() {
                break;
            }
            let r = self.wav.write_all(&s.to_le_bytes());
            self.check(r);
            self.written += 1;
            self.since_refresh += 1;
        }
        if self.since_refresh >= REFRESH {
            self.since_refresh = 0;
            self.wav_header();
        }
    }

    fn finish(mut self, at: u64, dropped_commands: u64, dropped_samples: u64) -> Saved {
        self.pad(at);
        let end = take::end_line(at, self.sample_rate, dropped_commands, dropped_samples);
        self.line(&end);
        self.wav_header();
        Saved {
            log_path: self.log_path,
            samples: self.written,
            sample_rate: self.sample_rate,
            commands: self.commands,
            dropped_commands,
            dropped_samples,
            error: self.error,
        }
    }

    /// Remove the files of a take that never started.
    fn discard(self) {
        drop((self.log, self.wav));
        let _ = fs::remove_file(self.log_path);
        let _ = fs::remove_file(self.wav_path);
    }
}

fn write_takes(rx: &Receiver<Event>, saved: &Sender<Saved>) {
    let mut current: Option<Writing> = None;
    let report = |w: Writing, at, dc, ds| {
        let _ = saved.send(w.finish(at, dc, ds));
    };
    loop {
        let event = match rx.try_recv() {
            Ok(event) => event,
            Err(TryRecvError::Empty) => {
                std::thread::sleep(POLL);
                continue;
            }
            Err(TryRecvError::Disconnected) => break,
        };
        match event {
            Event::Open(open) => {
                if let Some(w) = current.take() {
                    if w.started {
                        let at = w.written;
                        report(w, at, 0, 0);
                    } else {
                        w.discard();
                    }
                }
                current = Some(Writing::new(*open));
            }
            Event::Said(id, speech) => {
                if let Some(w) = &mut current {
                    w.said.insert(id, *speech);
                }
            }
            Event::Cancel => {
                if let Some(w) = current.take() {
                    w.discard();
                }
            }
            Event::Start(controls) => {
                if let Some(w) = &mut current {
                    w.start(&controls);
                }
            }
            Event::Command { at, cmd } => {
                if let Some(w) = &mut current {
                    w.command(at, &cmd);
                }
            }
            Event::Audio { at, len, samples } => {
                if let Some(w) = &mut current {
                    w.audio(at, &samples[..len]);
                }
            }
            Event::Stop {
                at,
                dropped_commands,
                dropped_samples,
            } => {
                if let Some(w) = current.take() {
                    report(w, at, dropped_commands, dropped_samples);
                }
            }
        }
    }
    // Every sender is gone (the app has ended): keep what was written.
    if let Some(w) = current.take()
        && w.started
    {
        let at = w.written;
        report(w, at, 0, 0);
    }
}

// ----- the app's side -------------------------------------------------------

/// The app's end of recording: numbers takes and creates their files.
pub struct Recorder {
    dir: PathBuf,
    tx: Sender<Event>,
    saved: Receiver<Saved>,
    /// Name of the take being recorded.
    current: Option<String>,
    speeches: u32,
}

/// The number of a take file's name (`take-007.wav` → 7).
fn take_number(name: &str) -> Option<u32> {
    let stem = name
        .strip_suffix(".jsonl")
        .or_else(|| name.strip_suffix(".wav"))?;
    stem.strip_prefix("take-")?.parse().ok()
}

impl Recorder {
    /// Takes go to `dir` (created on the first take), through the writer's
    /// queue `tx`; `saved` is where the writer reports them.
    #[must_use]
    pub fn new(dir: PathBuf, tx: Sender<Event>, saved: Receiver<Saved>) -> Self {
        Self {
            dir,
            tx,
            saved,
            current: None,
            speeches: 0,
        }
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The take being recorded.
    #[must_use]
    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Create the next take's files, numbered past every take already in
    /// the directory, and hand them to the writer. Its name.
    ///
    /// # Errors
    ///
    /// When the directory or the files cannot be created.
    pub fn open(&mut self, sample_rate: f32, app: Value) -> Result<String, String> {
        let dir = &self.dir;
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let last = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| take_number(&e.ok()?.file_name().to_string_lossy()))
            .max()
            .unwrap_or(0);
        let create = |path: &Path| {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)
        };
        let mut n = last + 1;
        loop {
            let name = format!("take-{n:03}");
            let log_path = dir.join(format!("{name}.jsonl"));
            let wav_path = dir.join(format!("{name}.wav"));
            let log = match create(&log_path) {
                Ok(f) => f,
                // Taken in the meantime: never overwrite, try the next.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    n += 1;
                    continue;
                }
                Err(e) => return Err(format!("{}: {e}", log_path.display())),
            };
            let wav = match create(&wav_path) {
                Ok(f) => f,
                Err(e) => {
                    drop(log);
                    let _ = fs::remove_file(&log_path);
                    if e.kind() == io::ErrorKind::AlreadyExists {
                        n += 1;
                        continue;
                    }
                    return Err(format!("{}: {e}", wav_path.display()));
                }
            };
            let open = Opening {
                name: name.clone(),
                log,
                wav,
                log_path: log_path.clone(),
                wav_path: wav_path.clone(),
                sample_rate,
                app,
            };
            if self
                .tx
                .send_timeout(Event::Open(Box::new(open)), PATIENCE)
                .is_err()
            {
                let _ = fs::remove_file(&log_path);
                let _ = fs::remove_file(&wav_path);
                return Err("the take writer is not keeping up".into());
            }
            self.current = Some(name.clone());
            return Ok(name);
        }
    }

    /// The audio thread never got the word to start: forget the take.
    pub fn cancel(&mut self) {
        if self.current.take().is_some() {
            let _ = self.tx.send_timeout(Event::Cancel, PATIENCE);
        }
    }

    /// The take is over (the audio thread has been told); its name.
    pub fn close(&mut self) -> Option<String> {
        self.current.take()
    }

    /// Number a new piece of speech and, while recording, tell the writer
    /// what it says. Call before the speech is sent to the audio thread.
    pub fn said(&mut self, text: &str, prosody: &Prosody) -> u32 {
        self.speeches = self.speeches.wrapping_add(1);
        if self.current.is_some() {
            let speech = Speech {
                text: text.to_string(),
                prosody: *prosody,
            };
            let _ = self
                .tx
                .send_timeout(Event::Said(self.speeches, Box::new(speech)), PATIENCE);
        }
        self.speeches
    }

    /// A take the writer has saved since the last call.
    #[must_use]
    pub fn poll(&self) -> Option<Saved> {
        self.saved.try_recv().ok()
    }

    /// Wait up to `timeout` for the writer to save a take.
    #[must_use]
    pub fn wait(&self, timeout: Duration) -> Option<Saved> {
        self.saved.recv_timeout(timeout).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_numbers_are_read_from_file_names() {
        assert_eq!(take_number("take-007.wav"), Some(7));
        assert_eq!(take_number("take-1234.jsonl"), Some(1234));
        assert_eq!(take_number("take-007.txt"), None);
        assert_eq!(take_number("notes.jsonl"), None);
    }
}
