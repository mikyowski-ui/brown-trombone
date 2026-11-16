//! Takes end to end: the app and the audio thread's own processing (no
//! sound device) record a take into a directory, and the take is read back
//! and replayed.

use crate::app::{App, Mode, RECORD_KEY, X_BOUNDS, Y_BOUNDS};
use crate::audio::{Cmd, output, wire};
use crate::record::{self, Clock, Event, Tap};
use crate::take::{
    self, Controls, Logged, Replay, Shaping, Speech, Take, command_line, header_line,
};
use crossbeam_channel::{Sender, bounded};
use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::{Terminal, backend::TestBackend};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::{Duration, Instant};
use synthie_trombone::inverse::Shape;
use synthie_trombone::listen::Heard;
use synthie_trombone::mimic::Gesture;
use synthie_trombone::speech::Prosody;
use synthie_trombone::{Constriction, Engine as SoundModel, Phoneme, Preset};
use synthie_ui::{Key, KeyCode, KeyKind};

const SR: f32 = 48_000.0;
/// Frames per audio callback in these tests; a device's varies, and so do
/// these (see `play`).
const BLOCK: usize = 512;
/// Output channels, interleaved, as a sound card has them.
const CHANNELS: usize = 2;

/// Set to a directory to keep the tests' takes there (to try the replay
/// example on them).
const KEEP: &str = "SYNTHIE_KEEP_TAKES";

/// An empty directory for one test's takes (not created).
fn scratch(name: &str) -> PathBuf {
    let dir = match std::env::var_os(KEEP) {
        Some(keep) => PathBuf::from(keep).join(name),
        None => std::env::temp_dir().join(format!("synthie-takes-{}-{name}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Remove a test's takes, unless they are to be kept.
fn clean(dir: &Path) {
    if std::env::var_os(KEEP).is_none() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The output callback, as the sound device runs it.
type Callback = Box<dyn FnMut(&mut [f32]) + Send>;

/// The app and its audio engine, wired as `audio::start` wires them but
/// without a device, recording into `dir`; with `ear`, a microphone that
/// the test speaks into.
struct Rig {
    app: App,
    /// The output callback, as the sound device runs it.
    callback: Callback,
    ear: Option<Sender<(Gesture, Heard)>>,
    /// Frames rendered so far, to vary the callback's block size.
    frames: usize,
}

impl Rig {
    fn new(dir: &Path, ear: bool) -> Self {
        let (ear_tx, ear_rx) = if ear {
            let (tx, rx) = bounded(64);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let size = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let (engine, mut link) = wire(SR, ear_rx, size, Some(dir.to_path_buf()));
        link.microphone = ear;
        let app = App::new(link, true, SoundModel::Classic);
        Self {
            app,
            callback: Box::new(output(engine, CHANNELS)),
            ear: ear_tx,
            frames: 0,
        }
    }

    /// Run the audio for `seconds`, in callbacks of changing size (as a
    /// device may), with the app ticking in between.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn play(&mut self, seconds: f32) {
        let mut left = (seconds * SR) as usize;
        let mut buf = vec![0.0f32; 2 * BLOCK * CHANNELS];
        while left > 0 {
            let n = (BLOCK / 2 + (self.frames * 7) % BLOCK).min(left);
            (self.callback)(&mut buf[..n * CHANNELS]);
            self.frames += n;
            left -= n;
            self.app.tick();
        }
    }

    fn press(&mut self, code: KeyCode) {
        self.app.key(Key::press(code));
    }

    fn type_text(&mut self, text: &str) {
        self.press(KeyCode::Tab);
        for c in text.chars() {
            self.press(KeyCode::Char(c));
        }
        self.press(KeyCode::Enter);
        // Tab, not Esc: Esc would stop speech still under way.
        self.press(KeyCode::Tab);
    }

    /// Stop the take and keep the audio running until the app reports it
    /// saved.
    fn stop_and_save(&mut self) -> String {
        assert!(self.app.recording().is_some(), "not recording");
        self.press(RECORD_KEY);
        let t0 = Instant::now();
        while !self.app.status.starts_with("saved") {
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "{}",
                self.app.status
            );
            self.play(0.01);
        }
        self.app.status.clone()
    }
}

fn read_take(dir: &Path, name: &str) -> (Take, Vec<f32>) {
    let log = std::fs::read_to_string(dir.join(format!("{name}.jsonl"))).unwrap();
    let take = Take::parse(&log).unwrap();
    let bytes = std::fs::read(dir.join(format!("{name}.wav"))).unwrap();
    let (rate, sound) = take::read_wav(&bytes).unwrap();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let want = SR as u32;
    assert_eq!(rate, want);
    (take, sound)
}

/// Replay `take` from its header; the samples.
fn replay(take: &Take) -> Vec<f32> {
    let mut r = Replay::new(take);
    let n = take.end.expect("the take was saved");
    let out = (0..n).map(|_| r.sample()).collect();
    assert_eq!(r.unspoken, 0);
    out
}

fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn recording_toggles_and_numbers_takes() {
    let dir = scratch("numbers");
    let mut rig = Rig::new(&dir, false);
    // Playing without recording writes nothing, not even the directory.
    rig.press(KeyCode::Char('a'));
    rig.play(0.2);
    assert!(!dir.exists(), "{}", dir.display());
    assert_eq!(rig.app.recording(), None);

    rig.press(RECORD_KEY);
    assert_eq!(rig.app.recording(), Some("take-001"));
    assert!(
        rig.app.status.contains("recording take-001"),
        "{}",
        rig.app.status
    );
    rig.play(0.3);
    // The display learns how far the take has got from the audio thread.
    let clock = rig
        .app
        .snap
        .take
        .expect("the audio thread reports the take");
    assert!(clock.samples > 12_000 && clock.dropped == 0, "{clock:?}");
    let status = rig.stop_and_save();
    assert!(status.contains("take-001.jsonl"), "{status}");
    assert_eq!(rig.app.recording(), None);
    rig.play(0.05);
    assert_eq!(rig.app.snap.take, None);
    let first = std::fs::read(dir.join("take-001.jsonl")).unwrap();

    // The next take gets the next number; nothing is overwritten.
    rig.press(RECORD_KEY);
    assert_eq!(rig.app.recording(), Some("take-002"));
    rig.play(0.05);
    rig.stop_and_save();
    // Numbers continue past the highest in the directory, whatever made it.
    std::fs::write(dir.join("take-010.wav"), b"").unwrap();
    rig.press(RECORD_KEY);
    assert_eq!(rig.app.recording(), Some("take-011"));
    rig.play(0.05);
    rig.stop_and_save();
    assert_eq!(std::fs::read(dir.join("take-001.jsonl")).unwrap(), first);
    assert_eq!(
        files(&dir),
        [
            "take-001.jsonl",
            "take-001.wav",
            "take-002.jsonl",
            "take-002.wav",
            "take-010.wav",
            "take-011.jsonl",
            "take-011.wav"
        ]
    );
    // Between takes, still nothing is written.
    let size = |p: &str| std::fs::metadata(dir.join(p)).unwrap().len();
    let before = size("take-011.wav");
    rig.play(0.2);
    assert_eq!(size("take-011.wav"), before);

    // A take still recording when the app leaves is saved.
    rig.press(RECORD_KEY);
    rig.play(0.1);
    assert!(rig.app.key(Key::press(KeyCode::Char('q'))));
    // `finish` needs the audio running to take the stop: run it alongside.
    let mut callback = rig.callback;
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let keep = Arc::clone(&running);
    let audio = std::thread::spawn(move || {
        let mut buf = [0.0f32; BLOCK * CHANNELS];
        while keep.load(std::sync::atomic::Ordering::Relaxed) {
            callback(&mut buf);
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    rig.app.finish();
    running.store(false, std::sync::atomic::Ordering::Relaxed);
    audio.join().unwrap();
    assert!(rig.app.status.contains("saved"), "{}", rig.app.status);
    let (take, sound) = read_take(&dir, "take-012");
    assert_eq!(take.end, Some(sound.len() as u64));
    clean(&dir);
}

/// Larger difference between two signals, and the larger of their peaks.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let peak = a.iter().map(|x| x.abs()).fold(0.0, f32::max);
    (diff, peak)
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn a_take_replays_to_the_same_samples() {
    let dir = scratch("replay");
    let mut rig = Rig::new(&dir, false);
    // Something already going when the take starts: a drone on a vowel,
    // the tongue moved, a voice of another size, speech under way.
    rig.press(KeyCode::Char('v'));
    rig.press(KeyCode::Char('3'));
    rig.press(KeyCode::Left);
    rig.press(KeyCode::Char('i'));
    rig.type_text("hello there");
    rig.play(0.4);
    rig.press(RECORD_KEY);
    rig.play(0.3);
    // The take: notes, vowels, the tongue, tenseness, a lip pop, the
    // velum, a persona dial, speech, both engines, Formant Lock.
    rig.press(KeyCode::Char('i'));
    for (c, wait) in [('g', 0.2), ('1', 0.15), ('5', 0.15), ('9', 0.1)] {
        rig.press(KeyCode::Char(c));
        rig.play(wait);
    }
    rig.app.key(Key {
        code: KeyCode::Char('g'),
        kind: KeyKind::Release,
        ctrl: false,
        shift: false,
    });
    rig.press(KeyCode::Char('h'));
    for code in [
        KeyCode::Right,
        KeyCode::Right,
        KeyCode::Up,
        KeyCode::Char(']'),
    ] {
        rig.press(code);
        rig.play(0.03);
    }
    rig.press(KeyCode::Char('b'));
    rig.play(0.05);
    rig.app.key(Key {
        code: KeyCode::Char('b'),
        kind: KeyKind::Release,
        ctrl: false,
        shift: false,
    });
    rig.press(KeyCode::Char('n'));
    rig.play(0.1);
    rig.press(KeyCode::PageUp);
    rig.play(0.2);
    rig.press(KeyCode::Char(' '));
    rig.type_text("a big red cat");
    rig.play(1.2);
    rig.press(KeyCode::Char('\\'));
    rig.press(KeyCode::Char('/'));
    rig.press(KeyCode::Char('d'));
    for code in [KeyCode::Left, KeyCode::Down, KeyCode::Down] {
        rig.play(0.08);
        rig.press(code);
    }
    rig.type_text("[S IY1]");
    rig.play(0.6);
    rig.press(KeyCode::Char('r'));
    rig.press(KeyCode::Char(' '));
    rig.play(0.2);
    rig.stop_and_save();

    let (take, sound) = read_take(&dir, "take-001");
    assert_eq!(take.end, Some(sound.len() as u64));
    assert_eq!((take.dropped_commands, take.dropped_samples), (0, 0));
    let speak = take
        .commands
        .iter()
        .filter(|c| matches!(c.cmd, Logged::Speak(_)))
        .count();
    assert_eq!(speak, 2, "{:?}", take.commands);
    assert!(take.commands.len() > 25, "{}", take.commands.len());
    // Speech under way when the take began was stopped; the drone held on.
    assert!(take.controls.drone && take.controls.voice.is_some());
    let replayed = replay(&take);
    let (diff, peak) = compare(&replayed, &sound);
    println!(
        "{} commands, {:.2} s, peak {peak:.3}, largest difference {diff:e}",
        take.commands.len(),
        sound.len() as f32 / SR
    );
    assert!(peak > 0.05, "the take is not silent");
    assert!(diff == 0.0, "replay differs by up to {diff}");
    clean(&dir);
}

#[test]
fn mimic_gestures_replay_too() {
    let dir = scratch("mimic");
    let mut rig = Rig::new(&dir, true);
    // With a microphone the app starts copying it on the physical engine.
    assert_eq!(rig.app.mimic, Some(true));
    rig.play(0.1);
    rig.press(RECORD_KEY);
    let ear = rig.ear.clone().unwrap();
    let shapes = [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw];
    for k in 0..60 {
        let shape = crate::app::CHART_VOWELS
            .iter()
            .find(|v| v.1 == shapes[k / 20])
            .map(|v| v.2);
        #[allow(clippy::cast_precision_loss)]
        let gesture = Gesture {
            voiced: k % 20 < 16,
            pitch: 120.0 + k as f32,
            shape,
            hiss: (k % 20 >= 17).then(|| Constriction::new(36.0, 0.5)),
        };
        ear.send((gesture, Heard::default())).unwrap();
        rig.play(0.01);
    }
    // Mimic off and on again, and speech, which mimic gives way to.
    rig.press(KeyCode::Char('`'));
    rig.play(0.1);
    rig.press(KeyCode::Char('`'));
    rig.type_text("ah");
    ear.send((Gesture::REST, Heard::default())).unwrap();
    rig.play(0.5);
    rig.stop_and_save();
    let (take, sound) = read_take(&dir, "take-001");
    let gestures = take
        .commands
        .iter()
        .filter(|c| matches!(c.cmd, Logged::Gesture(_)))
        .count();
    // The last gesture came while speaking: not carried out, not logged.
    assert_eq!(gestures, 60, "{:?}", take.commands);
    assert!(take.controls.mimic);
    let (diff, peak) = compare(&replay(&take), &sound);
    println!("{gestures} gestures, peak {peak:.3}, largest difference {diff:e}");
    assert!(peak > 0.01 && diff == 0.0, "peak {peak}, difference {diff}");
    clean(&dir);
}

#[test]
#[allow(clippy::too_many_lines)]
fn header_and_commands_round_trip() {
    let shape = Shape {
        tongue_index: 21.123_457,
        tongue_diameter: 2.718_281_7,
        lips: Some(0.733),
        narrowing: Some((14.0, 1.203)),
        pharynx: -0.25,
    };
    let mut voice = Preset::Critter.settings();
    voice.tuning.lung_pressure = 9.37;
    let controls = Controls {
        model: SoundModel::Physical,
        voice: Some(voice),
        shaping: Some(Shaping::Shape(shape)),
        tongue: (17.3, 2.9),
        constrictions: [
            None,
            Some(Constriction::new(41.0, 0.0)),
            Some(Constriction::new(30.2, 0.5)),
            None,
        ],
        velum: 0.4,
        tenseness: 0.71,
        lung: 12.5,
        frequency: 233.081_88,
        voiced: true,
        vibrato: Some((0.01, 5.5)),
        wobble: true,
        drone: false,
        reverb: 0.4,
        mimic: true,
    };
    let header = header_line("take-042", SR, &controls, json!({"preset": "critter"}));
    let mut log = header;
    // Every command, and numbers of every size and precision.
    let prosody = Prosody {
        pitch: 187.3,
        ..Prosody::default()
    };
    let speech = Speech {
        text: "hello, \"world\"".into(),
        prosody,
    };
    let mut lcg = 0x1234_5678_u32;
    let mut odd = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f32::from_bits(0x3c00_0000 + (lcg >> 5)) // 0.008 to 512
    };
    let mut cmds = vec![
        Logged::NoteOn(57.0),
        Logged::NoteOff,
        Logged::Note(45.5),
        Logged::Tongue(odd(), odd()),
        Logged::Constriction(2, Some(Constriction::new(odd(), odd()))),
        Logged::Constriction(3, None),
        Logged::Velum(0.25),
        Logged::Tenseness(odd()),
        Logged::Vibrato(0.015),
        Logged::Wobble(true),
        Logged::Drone(false),
        Logged::Reverb(0.18),
        Logged::Voice(voice),
        Logged::Lung(odd()),
        Logged::Model(SoundModel::Classic),
        Logged::Vowel(Phoneme::Er),
        Logged::Speak(3),
        Logged::StopSpeaking,
        Logged::Mimic(false),
        Logged::Shape(shape),
        Logged::Gesture(Gesture {
            voiced: true,
            pitch: odd(),
            shape: Some(shape),
            hiss: Some(Constriction::new(36.0, odd())),
        }),
        Logged::Gesture(Gesture::REST),
    ];
    for _ in 0..500 {
        cmds.push(Logged::Tongue(odd(), odd()));
    }
    for (k, cmd) in cmds.iter().enumerate() {
        let said = matches!(cmd, Logged::Speak(_)).then_some(&speech);
        let line = command_line(k as u64 * 97, SR, cmd, said);
        log += "\n";
        log += &line;
    }
    log += "\n";
    log += &take::end_line(60_000, SR, 2, 512);
    for line in log.lines().take(4) {
        println!("{line}");
    }

    let take = Take::parse(&log).unwrap();
    assert_eq!(take.name, "take-042");
    assert_eq!(take.controls, controls);
    assert_eq!(take.sample_rate.to_bits(), SR.to_bits());
    assert_eq!(take.commands.len(), cmds.len());
    for (k, (got, want)) in take.commands.iter().zip(&cmds).enumerate() {
        assert_eq!(got.at, k as u64 * 97);
        // Bit for bit: `PartialEq` on the floats would let -0 pass for 0.
        assert_eq!(format!("{:?}", got.cmd), format!("{want:?}"));
        assert_eq!(got.cmd, *want);
    }
    assert_eq!(take.commands[16].speech.as_ref(), Some(&speech));
    assert_eq!(
        (take.end, take.dropped_commands, take.dropped_samples),
        (Some(60_000), 2, 512)
    );
    // The numbers are written the short way.
    assert!(log.contains("\"velum\":0.4,"), "{log}");
}

#[test]
fn the_tap_writes_nothing_until_a_take_begins_and_counts_what_it_drops() {
    let (tx, rx) = bounded(4);
    let mut tap = Tap::new(Some(tx));
    for k in 0..10_000 {
        tap.command(Logged::NoteOff);
        #[allow(clippy::cast_precision_loss)]
        tap.sample(k as f32);
        tap.service();
    }
    assert!(rx.is_empty());
    assert_eq!(tap.clock(), None);
    // A queue that is not read: the start and three blocks fit, the rest
    // is counted, and the stop waits for room.
    tap.begin(&Controls::initial());
    for _ in 0..(5 * record::CHUNK) {
        tap.sample(0.5);
    }
    tap.command(Logged::NoteOff);
    assert_eq!(
        tap.clock(),
        Some(Clock {
            samples: 5 * record::CHUNK as u64,
            dropped: 3
        })
    );
    tap.end();
    assert_eq!(tap.clock(), None);
    let mut got: Vec<Event> = rx.try_iter().collect();
    assert!(matches!(got[0], Event::Start(_)));
    assert!(got[1..].iter().all(|e| matches!(e, Event::Audio { .. })));
    assert_eq!(got.len(), 4);
    tap.service();
    got = rx.try_iter().collect();
    let [
        Event::Stop {
            at,
            dropped_commands,
            dropped_samples,
        },
    ] = got[..]
    else {
        panic!("no stop");
    };
    assert_eq!(at, 5 * record::CHUNK as u64);
    assert_eq!(
        (dropped_commands, dropped_samples),
        (1, 2 * record::CHUNK as u64)
    );
}

#[test]
fn the_record_key_works_in_both_modes_and_says_when_it_cannot() {
    // Without a recorder the key only says so.
    let (cmd, rx) = bounded::<Cmd>(64);
    let (_s, snapshots) = bounded(1);
    let (_t, trash) = bounded(1);
    let link = crate::audio::Link {
        cmd,
        snapshots,
        trash,
        probes: crossbeam_channel::bounded(1).1,
        sample_rate: SR,
        microphone: false,
        microphone_name: None,
        recorder: None,
    };
    let mut app = App::new(link, true, SoundModel::Classic);
    while rx.try_recv().is_ok() {}
    app.key(Key::press(RECORD_KEY));
    assert!(app.status.contains("not available"), "{}", app.status);
    assert!(rx.is_empty());
    // In type mode it records rather than typing.
    let dir = scratch("modes");
    let mut rig = Rig::new(&dir, false);
    rig.press(KeyCode::Tab);
    rig.press(RECORD_KEY);
    assert_eq!(rig.app.recording(), Some("take-001"));
    assert_eq!(rig.app.line.text(), "");
    // …and goes to play mode, where the keys sound.
    assert_eq!(rig.app.mode, Mode::Play);
    rig.play(0.05);
    rig.stop_and_save();
    clean(&dir);
}

/// Three sentences pasted while a take is recorded (the paste switches to
/// type mode and says nothing), then Enter: they are spoken in order, with
/// a pause after each, until Esc stops all of it in the third. The take
/// holds the text as it was said and replays exactly.
#[test]
#[allow(clippy::cast_precision_loss)]
fn pasted_sentences_are_spoken_in_order_stopped_and_replayed() {
    let dir = scratch("pasted");
    let mut rig = Rig::new(&dir, false);
    rig.press(RECORD_KEY);
    rig.play(0.1);
    let pasted = "The cat sat on the mat.\r\nThe dog ran to the park!\n\nWhere did the bird go?\n";
    rig.app.paste(pasted);
    assert_eq!(rig.app.mode, Mode::Type);
    rig.play(0.2);
    assert_eq!(rig.app.snap.speaking, None, "the paste was spoken");
    rig.press(KeyCode::Enter);

    // Follow the speech as the display sees it, and stop it in the third
    // sentence.
    let said = "The cat sat on the mat. The dog ran to the park! Where did the bird go?";
    let speech_of = |take: &Take| take.commands.iter().find_map(|c| c.speech.clone());
    let mut seen: Vec<usize> = Vec::new();
    let pauses: Vec<usize> = rig
        .app
        .spoken
        .iter()
        .filter(|(sym, r)| *sym == "SIL" && r.start > 0)
        .map(|(_, r)| r.start)
        .collect();
    assert!(pauses.len() >= 2, "{:?}", rig.app.spoken);
    let third = pauses[1] + 1;
    let t0 = Instant::now();
    loop {
        rig.play(0.01);
        if let Some(k) = rig.app.snap.speaking {
            seen.push(k);
            if k > third + 4 {
                break;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "{seen:?}");
        assert!(seen.len() < 2_000, "never reached the third sentence");
    }
    assert!(
        seen.windows(2).all(|w| w[0] <= w[1]),
        "out of order: {seen:?}"
    );
    for p in &pauses[..2] {
        assert!(seen.contains(p), "pause at {p} not heard: {seen:?}");
    }
    rig.press(KeyCode::Esc);
    rig.play(0.05);
    assert_eq!(rig.app.snap.speaking, None, "Esc did not stop the speech");
    assert_eq!(rig.app.mode, Mode::Type);
    rig.play(0.3);
    rig.stop_and_save();

    let (take, sound) = read_take(&dir, "take-001");
    let kinds: Vec<&Logged> = take
        .commands
        .iter()
        .map(|c| &c.cmd)
        .filter(|c| matches!(c, Logged::Speak(_) | Logged::StopSpeaking))
        .collect();
    assert!(
        matches!(kinds[..], [Logged::Speak(_), Logged::StopSpeaking]),
        "{kinds:?}"
    );
    let speech = speech_of(&take).expect("the take says what was said");
    assert_eq!(speech.text, said);
    // One utterance, a pause after each sentence.
    let u = speech.utterance();
    let long_pauses = u
        .segments()
        .iter()
        .filter(|s| s.phoneme == Phoneme::Sil && s.duration > 0.25)
        .count();
    assert_eq!(long_pauses, 2);
    let (diff, peak) = compare(&replay(&take), &sound);
    println!(
        "{} commands, {:.2} s, peak {peak:.3}, largest difference {diff:e}",
        take.commands.len(),
        sound.len() as f32 / SR
    );
    assert!(peak > 0.05, "the take is silent");
    assert!(diff == 0.0, "replay differs by up to {diff}");
    clean(&dir);
}

/// A mouse event at a terminal cell.
fn mouse(kind: MouseEventKind, (column, row): (u16, u16)) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

/// Playing by hand, as on a terminal: speech first (the app is left in
/// type mode), then F5, a held piano key, a finger dragged along the mouth
/// and let go, the tongue handle dragged, F5. Everything goes from the
/// app's key and mouse handlers through the command queue and the output
/// callback the device runs into the take, which sounds and replays
/// exactly.
#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn playing_by_hand_goes_into_the_take() {
    let dir = scratch("hand");
    let mut rig = Rig::new(&dir, false);
    let mut term = Terminal::new(TestBackend::new(120, 34)).unwrap();
    let mut show = |rig: &mut Rig| {
        term.draw(|f| crate::ui::draw(f, &mut rig.app)).unwrap();
    };
    rig.press(KeyCode::Tab);
    for c in "good dog".chars() {
        rig.press(KeyCode::Char(c));
    }
    rig.press(KeyCode::Enter);
    rig.play(1.2);
    show(&mut rig);
    assert_eq!(rig.app.mode, Mode::Type);

    rig.press(RECORD_KEY);
    assert_eq!(rig.app.mode, Mode::Play);
    rig.play(0.1);
    show(&mut rig);
    let key = |c, kind| Key {
        code: KeyCode::Char(c),
        kind,
        ctrl: false,
        shift: false,
    };
    rig.app.key(key('a', KeyKind::Press));
    rig.play(0.3);
    // A finger on the midline row of the mouth, dragged towards the lips.
    let a = rig.app.tract_area;
    let col = |x: f64| {
        a.x + ((x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0]) * f64::from(a.width)) as u16
    };
    let midline = a.y + (Y_BOUNDS[1] / (Y_BOUNDS[1] - Y_BOUNDS[0]) * f64::from(a.height)) as u16;
    let left = MouseEventKind::Down(MouseButton::Left);
    let drag = MouseEventKind::Drag(MouseButton::Left);
    let up = MouseEventKind::Up(MouseButton::Left);
    rig.app.mouse(mouse(left, (col(28.0), midline)));
    for k in 1..=6 {
        rig.play(0.03);
        show(&mut rig);
        rig.app.mouse(mouse(drag, (col(28.0) + k, midline)));
    }
    rig.app.mouse(mouse(up, (col(28.0) + 6, midline)));
    rig.play(0.2);
    // The tongue handle, taken and dragged back.
    show(&mut rig);
    let handle = rig.app.tongue_cell().expect("the tract is drawn");
    rig.app.mouse(mouse(left, handle));
    for k in 1..=5 {
        rig.play(0.03);
        rig.app.mouse(mouse(drag, (handle.0 - 2 * k, handle.1)));
    }
    rig.app.mouse(mouse(up, (handle.0 - 10, handle.1)));
    rig.play(0.2);
    rig.app.key(key('a', KeyKind::Release));
    rig.play(0.2);
    rig.stop_and_save();

    let (take, sound) = read_take(&dir, "take-001");
    let has = |f: &dyn Fn(&Logged) -> bool| take.commands.iter().any(|c| f(&c.cmd));
    assert!(
        has(&|c| matches!(c, Logged::NoteOn(_))),
        "{:?}",
        take.commands
    );
    assert!(
        has(&|c| matches!(c, Logged::NoteOff)),
        "{:?}",
        take.commands
    );
    assert!(
        has(&|c| matches!(c, Logged::Constriction(2, Some(k)) if k.diameter < 0.1)),
        "{:?}",
        take.commands
    );
    assert!(has(&|c| matches!(c, Logged::Constriction(2, None))));
    let tongues = take
        .commands
        .iter()
        .filter(|c| matches!(c.cmd, Logged::Tongue(..)))
        .count();
    assert!(tongues >= 5, "{:?}", take.commands);
    let (diff, peak) = compare(&replay(&take), &sound);
    println!(
        "{} commands, peak {peak:.3}, largest difference {diff:e}",
        take.commands.len()
    );
    assert!(peak > 0.05, "the take is silent");
    assert!(diff == 0.0, "replay differs by up to {diff}");
    clean(&dir);
}
