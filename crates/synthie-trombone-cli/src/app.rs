//! UI state and input handling.

use crate::audio::{Cmd, Link, Snapshot};
use crate::line::{self, MAX_CHARS, TextLine};
use crate::record::Recorder;
use crossbeam_channel::Sender;
use crossterm::event::{
    KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;
use serde_json::json;
use std::fmt::Write as _;
use std::time::{Duration, Instant};
use synthie_trombone::inverse::{self, Shape};
use synthie_trombone::speech::{Prosody, Utterance};
use synthie_trombone::{
    Constriction, Engine, N, Persona, Phoneme, Preset, TONGUE_DIAMETER, TONGUE_INDEX, VoiceSettings,
};
use synthie_ui::{Key, KeyCode, KeyKind};

/// Constriction slot for mouse dragging (0 and 1 belong to the speaker).
const SLOT_MOUSE: usize = 2;
/// Constriction slot for the lip-pop key.
const SLOT_POP: usize = 3;
const NO_RELEASE_HINT: &str =
    "terminal sends no key releases: notes hold, Space stops · Tab to type or paste text";
/// Piano keys, one semitone apart, starting at C.
const PIANO: &str = "awsedftgyhujkolp;";
/// Canvas y range of the tract view (shared with `ui`).
pub const Y_BOUNDS: [f64; 2] = [-2.6, 5.2];
/// Canvas x range of the tract view.
pub const X_BOUNDS: [f64; 2] = [-1.0, 45.0];
/// Vertical centre of the nasal tract in canvas units.
pub const NOSE_Y: f64 = 3.6;
/// Half-height of the band the mouth and throat can occupy. Mouse drags
/// outside it never become constrictions.
pub const LUMEN: f32 = 1.9;
/// Lower edge of the nasal cavity; between it and `LUMEN` lies the velum.
const NOSE_FLOOR: f32 = 2.65;
/// Width of a meter in the glottis panel, and the column it starts at.
pub const BAR_WIDTH: u16 = 22;
const BAR_START: u16 = 9;
/// The persona dials, in panel order: name and range.
pub const DIALS: [(&str, f32, f32); 4] = [
    ("size", 0.6, 1.4),
    ("effort", 0.0, 1.0),
    ("breath", 0.0, 1.0),
    ("clarity", 0.0, 1.0),
];
/// Openings (touch-diameter units) a finger leaves on the row of the
/// midline, on the row beside it, and at least on the rows beyond: a
/// closure, a gap that hisses, and shapings that do not. Terminal rows are
/// too coarse to find a fricative by its exact width, so the row itself
/// says which is meant; the physical engine's hiss regulator sees to the
/// rest.
const FINGER_SHUT: f32 = 0.0;
const FINGER_HISS: f32 = 0.5;
const FINGER_OPEN: f32 = 0.9;
/// Range of the `loud` control, cmH2O of lung pressure.
pub const LOUD_RANGE: (f32, f32) = (2.0, 20.0);
/// How long the voice keeps sounding after the last turn of a persona dial
/// or glottis meter made while nothing else was sounding, so that the
/// change can be heard.
const PREVIEW_HOLD: Duration = Duration::from_millis(400);

/// What the status line says on entering type mode.
const TYPE_HINT: &str = "type or paste, Enter to speak · ↑↓ history · Tab back";
/// How much of a long text the status line quotes.
const QUOTE_CHARS: usize = 40;

/// The key that starts and stops recording a take, in either mode.
pub const RECORD_KEY: KeyCode = KeyCode::F(5);
/// How long a command that must arrive (to start or stop a take) may wait
/// for room in the queue to the audio thread.
const MUST_ARRIVE: Duration = Duration::from_millis(200);
/// How long leaving the app waits for the take being recorded to be saved.
const SAVE_WAIT: Duration = Duration::from_secs(3);

/// Vowels marked on the tract view; a click sets them.
pub const LANDMARKS: [(&str, Phoneme); 5] = [
    ("i", Phoneme::Iy),
    ("e", Phoneme::Eh),
    ("a", Phoneme::Aa),
    ("o", Phoneme::Ow),
    ("u", Phoneme::Uw),
];

/// Where the tongue handle is drawn for a tongue `(index, diameter)`.
#[must_use]
pub fn tongue_handle(tongue: (f32, f32)) -> (f32, f32) {
    let span = TONGUE_DIAMETER.end() - TONGUE_DIAMETER.start();
    let depth = (tongue.1 - TONGUE_DIAMETER.start()) / span;
    (tongue.0, -0.6 - 1.5 * depth)
}

/// `cells` read at a fractional cell `x`, straight between neighbours.
#[must_use]
pub fn lerp_cell(cells: &[f32], x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = (x.floor().max(0.0) as usize).min(cells.len() - 1);
    let hi = (lo + 1).min(cells.len() - 1);
    #[allow(clippy::cast_precision_loss)]
    let frac = (x - lo as f64).clamp(0.0, 1.0);
    f64::from(cells[lo]) * (1.0 - frac) + f64::from(cells[hi]) * frac
}

/// Something in the glottis panel or header that the mouse can work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Tense,
    Tongue,
    Velum,
    Vibrato,
    Reverb,
    Loud,
    /// A persona dial (index into [`DIALS`]).
    Dial(usize),
    OctaveDown,
    OctaveUp,
    Mode,
    Engine,
    Wobble,
    Drone,
    /// The Formant Lock button: vowel chart on or off.
    FormantLock,
}

impl Control {
    /// A meter to drag, as opposed to a button to click.
    fn is_bar(self) -> bool {
        matches!(
            self,
            Control::Tense
                | Control::Tongue
                | Control::Velum
                | Control::Vibrato
                | Control::Reverb
                | Control::Loud
                | Control::Dial(_)
        )
    }
}

/// What a mouse press has taken hold of.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grab {
    /// A finger on the tract wall.
    Constriction,
    /// The tongue handle (or Shift-drag), with how far the handle's own
    /// place `(index, height)` lies from where it was pressed: the handle
    /// is drawn inside the mouth, which is not always where the tongue's
    /// height would put it, and taking hold of it should not move it.
    Tongue((f32, f32)),
    /// Right-drag: the tongue, as the original maps it.
    TongueOld,
    Velum,
    Control(Control),
    /// The vowel chart of Formant Lock; `true` if the press landed on a
    /// vowel mark (the press itself is then that vowel).
    Chart(bool),
}

/// Number keys → vowels.
pub const VOWEL_KEYS: [(char, Phoneme); 10] = [
    ('1', Phoneme::Iy),
    ('2', Phoneme::Ih),
    ('3', Phoneme::Eh),
    ('4', Phoneme::Ae),
    ('5', Phoneme::Aa),
    ('6', Phoneme::Ah),
    ('7', Phoneme::Ao),
    ('8', Phoneme::Uh),
    ('9', Phoneme::Uw),
    ('0', Phoneme::Er),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Play,
    Type,
}

#[allow(clippy::struct_excessive_bools)] // independent toggles
pub struct App {
    cmd: Sender<Cmd>,
    link: Link,
    pub snap: Snapshot,
    pub mode: Mode,
    pub keyboard_release: bool,

    pub octave: i32,
    pub note: i32,
    held: Option<char>,
    pub voiced: bool,

    pub tongue: (f32, f32),
    pub tenseness: f32,
    pub vibrato: f32,
    pub wobble: bool,
    pub nasal: bool,
    pub drone: bool,
    pub reverb: f32,
    pub rate: f32,
    pub preset: Preset,
    /// The voice before the persona dials: a preset, or a profile laid
    /// over one.
    base: VoiceSettings,
    /// The persona dials, and the one the keys work.
    pub persona: Persona,
    pub dial: usize,
    /// The voice in use: `base` as the persona reshapes it.
    pub settings: VoiceSettings,
    pub engine: Engine,

    /// The text line of type mode.
    pub line: TextLine,
    history: Vec<String>,
    history_pos: usize,
    /// Phoneme symbols of the last utterance with their segment ranges.
    pub spoken: Vec<(&'static str, std::ops::Range<usize>)>,
    pub last_vowel: Option<Phoneme>,

    /// Inner area of the tract canvas, for mouse mapping.
    pub tract_area: Rect,
    /// Where each mouse-operated control was last drawn.
    pub controls: Vec<(Rect, Control)>,
    /// Lung pressure of the physical engine, cmH2O.
    pub loud: f32,
    grab: Option<Grab>,
    pop_until: Option<Instant>,
    /// The time of the latest [`tick`](Self::tick): the clock the timers
    /// set from key and mouse handlers run on.
    now: Instant,
    /// While a dial or meter is being heard (see [`PREVIEW_HOLD`]): when
    /// the voice it started stops.
    preview_until: Option<Instant>,
    pub status: String,
    /// Mimic mode: copying what the microphone hears (`None` without a
    /// microphone).
    pub mimic: Option<bool>,
    /// Formant Lock: the vowel shape the voice holds while the vowel chart
    /// steers it, `None` when the lock is off.
    pub formant_lock: Option<Shape>,
    /// Inner area of the vowel chart, for mouse mapping (empty when it is
    /// not drawn).
    pub chart_area: Rect,
}

/// Lowercase name of a sound model.
pub fn engine_name(engine: Engine) -> &'static str {
    match engine {
        Engine::Classic => "classic",
        Engine::Physical => "physical",
    }
}

impl App {
    pub fn new(link: Link, keyboard_release: bool, engine: Engine) -> Self {
        let mut app = Self {
            cmd: link.cmd.clone(),
            link,
            snap: Snapshot::default(),
            mode: Mode::Play,
            keyboard_release,
            octave: 3,
            note: 45,
            held: None,
            voiced: false,
            tongue: (12.9, 2.43),
            tenseness: 0.6,
            vibrato: 0.005,
            wobble: false,
            nasal: false,
            drone: false,
            reverb: 0.18,
            rate: 1.0,
            preset: Preset::Deep,
            base: Preset::Deep.settings().with_engine(engine),
            persona: Persona::NEUTRAL,
            dial: 0,
            settings: Preset::Deep.settings().with_engine(engine),
            engine,
            line: TextLine::default(),
            history: Vec::new(),
            history_pos: 0,
            spoken: Vec::new(),
            last_vowel: None,
            tract_area: Rect::default(),
            controls: Vec::new(),
            loud: 8.0,
            grab: None,
            pop_until: None,
            now: Instant::now(),
            preview_until: None,
            status: String::new(),
            mimic: None,
            formant_lock: None,
            chart_area: Rect::default(),
        };
        app.status = if keyboard_release {
            "hold a piano key to sing · Space stops · Tab to type or paste text".into()
        } else {
            NO_RELEASE_HINT.into()
        };
        app.send(Cmd::Model(engine));
        app.send(Cmd::Note(app.note_f32()));
        if app.link.microphone {
            app.set_mimic(true);
        }
        app
    }

    pub fn sample_rate(&self) -> f32 {
        self.link.sample_rate
    }

    /// Start from a voice other than the first preset's, with the persona
    /// dials set (as given on the command line).
    pub fn set_voice(&mut self, preset: Preset, base: VoiceSettings, persona: Persona) {
        self.preset = preset;
        self.base = base.with_engine(self.engine);
        self.persona = persona;
        self.voice_changed();
    }

    /// The persona dials as numbers, in [`DIALS`] order.
    #[must_use]
    pub fn dials(&self) -> [f32; 4] {
        let p = &self.persona;
        [p.size, p.effort, p.breathiness, p.clarity]
    }

    /// Start or stop copying the microphone.
    fn set_mimic(&mut self, on: bool) {
        if self.mimic.is_none() && !self.link.microphone {
            self.status = "no microphone: start with --mimic".into();
            return;
        }
        if on && self.engine != Engine::Physical {
            self.toggle_engine();
        }
        if on {
            // The microphone takes over the voice.
            self.end_preview();
        }
        self.mimic = Some(on);
        self.send(Cmd::Mimic(on));
        self.status = if on {
            format!(
                "mimic: copying {} (wear headphones) · ` stops",
                self.link
                    .microphone_name
                    .as_deref()
                    .unwrap_or("the microphone")
            )
        } else {
            "mimic off · ` starts it again".into()
        };
    }

    /// Hand the keys the next persona dial.
    fn next_dial(&mut self) {
        self.dial = (self.dial + 1) % DIALS.len();
        self.status = format!(
            "persona dial: {} (PgUp/PgDn to turn, Del for neutral)",
            DIALS[self.dial].0
        );
    }

    /// Set persona dial `k` and reshape the voice.
    fn set_dial(&mut self, k: usize, value: f32) {
        let Some(&(name, lo, hi)) = DIALS.get(k) else {
            return;
        };
        let value = value.clamp(lo, hi);
        let p = &mut self.persona;
        match k {
            0 => p.size = value,
            1 => p.effort = value,
            2 => p.breathiness = value,
            _ => p.clarity = value,
        }
        self.dial = k;
        self.voice_changed();
        self.preview();
        self.status = format!("{name}: {value:.2}");
    }

    /// The base voice or the persona changed: rebuild the voice in use,
    /// bring the meters that show it up to date, and move to the note
    /// nearest its speaking pitch.
    fn voice_changed(&mut self) {
        self.settings = self
            .base
            .with_engine(self.engine)
            .with_persona(&self.persona);
        self.tenseness = self.settings.tenseness;
        self.vibrato = self.settings.vibrato;
        self.loud = self.settings.tuning.lung_pressure;
        #[allow(clippy::cast_possible_truncation)]
        let note = (69.0 + 12.0 * (self.settings.pitch / 440.0).log2()).round() as i32;
        self.note = note;
        self.octave = (note / 12 - 1).clamp(1, 6);
        self.send(Cmd::Voice(self.settings));
        self.send(Cmd::Note(self.note_f32()));
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.cmd.try_send(cmd);
    }

    #[allow(clippy::cast_precision_loss)]
    fn note_f32(&self) -> f32 {
        self.note as f32
    }

    /// Per-frame housekeeping: newest snapshot, timers, freeing old
    /// utterances.
    pub fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    /// [`tick`](Self::tick) at time `now`.
    pub fn tick_at(&mut self, now: Instant) {
        self.now = now;
        while let Ok(s) = self.link.snapshots.try_recv() {
            self.snap = s;
        }
        while self.link.trash.try_recv().is_ok() {}
        if let Some(saved) = self.link.recorder.as_ref().and_then(Recorder::poll) {
            self.status = saved_status(&saved);
        }
        if self.mimic == Some(true)
            && let Some(h) = self.snap.heard
        {
            self.status = if h.voiced {
                format!(
                    "mimic: hearing {:.0} Hz, formants {:.0} / {:.0} Hz · ` stops",
                    h.pitch, h.formants[0], h.formants[1]
                )
            } else if h.hiss > 0.5 {
                format!("mimic: hearing hiss at {:.0} Hz · ` stops", h.brightness)
            } else {
                "mimic: listening (wear headphones) · ` stops".into()
            };
        }
        if self.pop_until.is_some_and(|t| now >= t) {
            self.pop_until = None;
            self.send(Cmd::Constriction(SLOT_POP, None));
        }
        if self.preview_until.is_some_and(|t| now >= t) {
            self.end_preview();
        }
    }

    /// A persona dial or glottis meter changed: let it be heard. With
    /// nothing sounding (no note held, no drone, no speech, no mimicking)
    /// the voice sounds at the current note until [`PREVIEW_HOLD`] after
    /// the last change.
    fn preview(&mut self) {
        if self.preview_until.is_none() {
            let busy = self.held.is_some()
                || self.drone
                || self.snap.speaking.is_some()
                || self.mimic == Some(true);
            if busy {
                return;
            }
            self.send(Cmd::NoteOn(self.note_f32()));
        }
        self.preview_until = Some(self.now + PREVIEW_HOLD);
    }

    /// Stop the voice a dial or meter started, if it is still sounding.
    fn end_preview(&mut self) {
        if self.preview_until.take().is_some() {
            self.send(Cmd::NoteOff);
        }
    }

    /// Handle a key; returns `true` to quit.
    pub fn key(&mut self, key: Key) -> bool {
        if key.code == RECORD_KEY {
            if key.kind == KeyKind::Press {
                self.toggle_recording();
            }
            return false;
        }
        let quit = if key.ctrl && key.code == KeyCode::Char('c') {
            true
        } else {
            match self.mode {
                Mode::Play => self.play_key(key),
                Mode::Type => {
                    if key.kind != KeyKind::Release {
                        self.type_key(key);
                    }
                    false
                }
            }
        };
        if quit {
            // Leave nothing sounding behind.
            self.end_preview();
        }
        quit
    }

    fn play_key(&mut self, key: Key) -> bool {
        let KeyCode::Char(c) = key.code else {
            if key.kind != KeyKind::Release {
                self.play_special(key.code);
            }
            return false;
        };
        let upper = c.is_ascii_uppercase();
        let c = c.to_ascii_lowercase();

        if let Some(semi) = PIANO.find(c) {
            match key.kind {
                KeyKind::Press if self.keyboard_release && self.held == Some(c) => {
                    // A second press of a held key without a release in
                    // between: this terminal does not report releases after
                    // all. Fall back to Space-to-stop.
                    self.keyboard_release = false;
                    self.status = NO_RELEASE_HINT.into();
                }
                KeyKind::Press => {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let semi = semi as i32;
                    self.note = 12 * (self.octave + 1) + semi;
                    self.held = Some(c);
                    self.voiced = true;
                    // The note takes over from a dial being heard; that
                    // ending must not stop it.
                    self.preview_until = None;
                    self.send(Cmd::NoteOn(self.note_f32()));
                }
                KeyKind::Release if self.held == Some(c) => {
                    self.held = None;
                    self.voiced = false;
                    self.send(Cmd::NoteOff);
                }
                _ => {}
            }
            return false;
        }

        if key.kind == KeyKind::Release {
            if c == 'b' && self.pop_until.is_none() {
                self.send(Cmd::Constriction(SLOT_POP, None));
            }
            return false;
        }
        if key.kind == KeyKind::Repeat && c == 'b' {
            return false;
        }

        if let Some(&(_, v)) = VOWEL_KEYS.iter().find(|(k, _)| *k == c) {
            self.vowel(v);
            return false;
        }

        match c {
            'q' => return true,
            ' ' => self.silence(),
            'z' => self.octave = (self.octave - 1).max(1),
            'x' => self.octave = (self.octave + 1).min(6),
            'b' => {
                // Lips shut; open on release (or after 150 ms without
                // release events) for a "b"/"p" burst.
                self.send(Cmd::Constriction(
                    SLOT_POP,
                    Some(Constriction::new(41.0, 0.0)),
                ));
                if !self.keyboard_release {
                    self.pop_until = Some(Instant::now() + Duration::from_millis(150));
                }
            }
            'n' => {
                self.nasal = !self.nasal;
                self.send(Cmd::Velum(if self.nasal { 1.0 } else { 0.0 }));
            }
            'v' => self.cycle_preset(if upper || key.shift { -1 } else { 1 }),
            'm' => {
                self.wobble = !self.wobble;
                self.send(Cmd::Wobble(self.wobble));
            }
            'i' => self.toggle_drone(),
            'c' => {
                for slot in 0..synthie_trombone::MAX_CONSTRICTIONS {
                    self.send(Cmd::Constriction(slot, None));
                }
            }
            'r' => {
                self.reverb = match self.reverb {
                    r if r < 0.05 => 0.18,
                    r if r < 0.3 => 0.4,
                    _ => 0.0,
                };
                self.send(Cmd::Reverb(self.reverb));
            }
            '\\' => self.toggle_engine(),
            '\'' => self.next_dial(),
            '`' => self.set_mimic(self.mimic != Some(true)),
            '[' => self.set_tenseness(self.tenseness - 0.05),
            ']' => self.set_tenseness(self.tenseness + 0.05),
            '-' => self.set_vibrato(self.vibrato - 0.005),
            '=' => self.set_vibrato(self.vibrato + 0.005),
            ',' => self.rate = (self.rate - 0.1).max(0.4),
            '.' => self.rate = (self.rate + 0.1).min(2.5),
            '/' => self.toggle_lock(),
            _ => {}
        }
        false
    }

    fn play_special(&mut self, code: KeyCode) {
        if self.nudge(code) {
            return;
        }
        let (mut i, mut d) = self.tongue;
        match code {
            KeyCode::Tab => {
                self.mode = Mode::Type;
                self.status = TYPE_HINT.into();
                return;
            }
            KeyCode::Enter => {
                if let Some(t) = self.history.last().cloned() {
                    self.speak(&t);
                }
                return;
            }
            KeyCode::Esc => {
                self.silence();
                return;
            }
            KeyCode::PageUp | KeyCode::PageDown => {
                let (_, lo, hi) = DIALS[self.dial];
                let step = if code == KeyCode::PageUp { 0.05 } else { -0.05 };
                self.set_dial(self.dial, self.dials()[self.dial] + step * (hi - lo));
                return;
            }
            KeyCode::Delete => {
                self.persona = Persona::NEUTRAL;
                self.voice_changed();
                self.preview();
                self.status = "persona: neutral".into();
                return;
            }
            KeyCode::Left => i -= 0.5,
            KeyCode::Right => i += 0.5,
            KeyCode::Up => d -= 0.08,
            KeyCode::Down => d += 0.08,
            _ => return,
        }
        self.set_tongue(i, d);
    }

    fn type_key(&mut self, key: Key) {
        let line = &mut self.line;
        match key.code {
            KeyCode::Esc if self.snap.speaking.is_some() => {
                // Esc stops the speech first, all of it; again, it goes
                // back to play mode.
                self.send(Cmd::StopSpeaking);
                self.status = "stopped · Esc again to play".into();
            }
            KeyCode::Esc | KeyCode::Tab => {
                self.mode = Mode::Play;
                self.status.clear();
            }
            KeyCode::Enter => self.say_line(),
            KeyCode::Left | KeyCode::Right if key.ctrl => {
                if key.code == KeyCode::Left {
                    line.word_left();
                } else {
                    line.word_right();
                }
            }
            KeyCode::Left => line.left(),
            KeyCode::Right => line.right(),
            KeyCode::Backspace if key.ctrl => line.delete_word(),
            KeyCode::Backspace => line.backspace(),
            KeyCode::Delete => line.delete(),
            KeyCode::Up if self.history_pos > 0 => {
                self.history_pos -= 1;
                line.set(&self.history[self.history_pos]);
            }
            KeyCode::Down => {
                self.history_pos = (self.history_pos + 1).min(self.history.len());
                line.set(
                    self.history
                        .get(self.history_pos)
                        .map_or("", String::as_str),
                );
            }
            // The usual line-editing keys of a shell.
            KeyCode::Char(c) if key.ctrl => match c.to_ascii_lowercase() {
                'a' => line.home(),
                'e' => line.end(),
                'b' => line.left(),
                'f' => line.right(),
                'u' => line.clear(),
                'k' => line.delete_to_end(),
                'w' => line.delete_word(),
                'd' => line.delete(),
                // Backspace, in terminals that send it as Ctrl+H.
                'h' => line.backspace(),
                // A line feed: what a pasted line break arrives as when
                // the terminal pastes without bracketing it. A space, as
                // in a bracketed paste; only Enter speaks.
                'j' => {
                    line.insert_pasted(" ");
                }
                _ => {}
            },
            KeyCode::Char(_) if line.is_full() => {
                self.status = format!("the text line is full ({MAX_CHARS} characters)");
            }
            KeyCode::Char(c) => {
                line.insert(c.encode_utf8(&mut [0; 4]));
            }
            _ => {}
        }
    }

    /// Speak the text line and keep it in the history.
    fn say_line(&mut self) {
        let t = line::normalise(self.line.text()).trim().to_string();
        if t.is_empty() {
            return;
        }
        self.speak(&t);
        if self.history.last() != Some(&t) {
            self.history.push(t);
        }
        self.history_pos = self.history.len();
        self.line.clear();
    }

    /// Text pasted into the terminal (a bracketed paste): it goes into the
    /// text line at the cursor, tidied for the speech front end (see
    /// [`line::normalise`]); in play mode the app switches to type mode
    /// first. Pasting never speaks: a line break in the text is a space,
    /// and only Enter speaks.
    pub fn paste(&mut self, text: &str) {
        self.mode = Mode::Type;
        let tidy = line::normalise(text);
        let left_out = self.line.insert_pasted(&tidy);
        self.status = if left_out > 0 {
            format!(
                "the text line is full ({MAX_CHARS} characters): {left_out} pasted characters left out"
            )
        } else {
            "pasted · Enter to speak · Ctrl+U clears".into()
        };
    }

    /// A key as the terminal reports it. Keys [`Key`] has no room for
    /// (Home, End, Alt with a key) are handled here, the rest by
    /// [`key`](Self::key). With `burst` the key came in a burst of typed
    /// text faster than anyone types: the terminal pasting without
    /// bracketed paste. In type mode a line break or tab in it is then a
    /// space, as in a bracketed paste, so that half a paste is not spoken
    /// and the rest played on the piano.
    pub fn terminal_key(&mut self, ev: KeyEvent, burst: bool) -> bool {
        use crossterm::event::KeyCode as C;
        if self.mode == Mode::Type && ev.kind != KeyEventKind::Release {
            let alt = ev.modifiers.contains(KeyModifiers::ALT);
            let line = &mut self.line;
            let handled = match ev.code {
                C::Home => {
                    line.home();
                    true
                }
                C::End => {
                    line.end();
                    true
                }
                C::Enter | C::Tab if burst => {
                    line.insert_pasted(" ");
                    true
                }
                C::Backspace if alt => {
                    line.delete_word();
                    true
                }
                C::Left | C::Char('b') if alt => {
                    line.word_left();
                    true
                }
                C::Right | C::Char('f') if alt => {
                    line.word_right();
                    true
                }
                _ => false,
            };
            if handled {
                return false;
            }
        }
        self.key(ev.into())
    }

    /// Stop singing and speaking; the voice fades out. Also ends a drone.
    fn silence(&mut self) {
        self.held = None;
        self.voiced = false;
        self.preview_until = None;
        if self.drone {
            self.drone = false;
            self.send(Cmd::Drone(false));
        }
        self.send(Cmd::NoteOff);
        self.send(Cmd::StopSpeaking);
    }

    /// Start or stop the drone. A dial being heard ends at once: the drone
    /// keeps the voice going on its own, and a note left on would sound on
    /// after the drone stops.
    fn toggle_drone(&mut self) {
        self.drone = !self.drone;
        if self.drone {
            self.end_preview();
        }
        self.send(Cmd::Drone(self.drone));
    }

    /// Switch between the classic and the physical sound model.
    fn toggle_engine(&mut self) {
        self.engine = match self.engine {
            Engine::Classic => Engine::Physical,
            Engine::Physical => Engine::Classic,
        };
        self.settings.engine = self.engine;
        self.base.engine = self.engine;
        self.send(Cmd::Model(self.engine));
        if let Some(v) = self.last_vowel {
            self.send(Cmd::Vowel(v));
        }
        self.status = format!("engine: {}", engine_name(self.engine));
        self.lock_engine_changed();
    }

    /// Switch to the next (`+1`) or previous (`-1`) voice preset.
    fn cycle_preset(&mut self, dir: isize) {
        let all = Preset::ALL;
        let i = all.iter().position(|&p| p == self.preset).unwrap_or(0);
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let next = (i as isize + dir).rem_euclid(all.len() as isize) as usize;
        self.preset = all[next];
        self.base = self.preset.settings().with_engine(self.engine);
        self.voice_changed();
        self.status = format!("voice: {}", self.preset);
    }

    fn speak(&mut self, text: &str) {
        let prosody = Prosody {
            pitch: synthie_trombone::midi_to_hz(self.note_f32()),
            rate: self.settings.rate * self.rate,
            tenseness: self.tenseness,
            ..self.settings.prosody()
        };
        let u = Utterance::from_text(text, &prosody);
        self.spoken.clear();
        for (k, s) in u.segments().iter().enumerate() {
            let sym = s.phoneme.symbol();
            match self.spoken.last_mut() {
                Some((last, range)) if *last == sym => range.end = k + 1,
                _ => self.spoken.push((sym, k..k + 1)),
            }
        }
        self.status = format!("“{}” · {:.1} s", quote(text), u.duration());
        // Speech takes over the voice; a dial being heard stops first.
        self.end_preview();
        // A take logs what was said, so that it can be said again.
        let id = self
            .link
            .recorder
            .as_mut()
            .map_or(0, |r| r.said(text, &prosody));
        self.send(Cmd::Speak(u, id));
    }

    fn vowel(&mut self, v: Phoneme) {
        // The sound model picks the shape; remember where it puts the
        // tongue so the handle and the arrow keys start from there.
        if let Some((i, d)) = v.tongue_for(self.engine, self.settings.tract_length) {
            self.tongue = (
                i.clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
                d.clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
            );
        }
        self.send(Cmd::Vowel(v));
        self.last_vowel = Some(v);
        self.lock_follow_vowel(v);
    }

    fn set_tongue(&mut self, i: f32, d: f32) {
        self.tongue = (
            i.clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
            d.clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
        );
        self.send(Cmd::Tongue(self.tongue.0, self.tongue.1));
        self.lock_follow_tongue();
    }

    fn set_tenseness(&mut self, t: f32) {
        self.tenseness = t.clamp(0.0, 1.0);
        self.send(Cmd::Tenseness(self.tenseness));
        self.preview();
    }

    fn set_vibrato(&mut self, v: f32) {
        self.vibrato = v.clamp(0.0, 0.05);
        self.send(Cmd::Vibrato(self.vibrato));
        self.preview();
    }

    /// Map a terminal cell to tract coordinates `(index, y)`.
    fn to_tract(&self, col: u16, row: u16) -> Option<(f32, f32)> {
        let a = self.tract_area;
        if a.width == 0 || !a.contains((col, row).into()) {
            return None;
        }
        let fx = (f64::from(col - a.x) + 0.5) / f64::from(a.width);
        let fy = (f64::from(row - a.y) + 0.5) / f64::from(a.height);
        let x = X_BOUNDS[0] + fx * (X_BOUNDS[1] - X_BOUNDS[0]);
        let y = Y_BOUNDS[1] - fy * (Y_BOUNDS[1] - Y_BOUNDS[0]);
        #[allow(clippy::cast_possible_truncation)]
        Some((x as f32, y as f32))
    }

    /// How many terminal rows `row` lies from the row that holds the
    /// tract's midline.
    fn rows_from_midline(&self, row: u16) -> u16 {
        let a = self.tract_area;
        let fy = Y_BOUNDS[1] / (Y_BOUNDS[1] - Y_BOUNDS[0]);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let midline = a.y + (fy * f64::from(a.height)) as u16;
        row.abs_diff(midline)
    }

    /// The terminal cell the tract canvas paints a Braille dot for tract
    /// coordinates `(x, y)` in (the canvas rounds to the nearest of its
    /// 2×4 dots per cell). Text labels on a canvas are placed by a
    /// different rule, up to a row higher, so whatever must line up with
    /// the walls is placed with this instead.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn dot_cell(&self, x: f64, y: f64) -> (u16, u16) {
        let a = self.tract_area;
        let x = x.clamp(X_BOUNDS[0], X_BOUNDS[1]);
        let y = y.clamp(Y_BOUNDS[0], Y_BOUNDS[1]);
        let dots_x = f64::from(a.width) * 2.0 - 1.0;
        let dots_y = f64::from(a.height) * 4.0 - 1.0;
        let dx = ((x - X_BOUNDS[0]) * dots_x / (X_BOUNDS[1] - X_BOUNDS[0])).round() as u16;
        let dy = ((Y_BOUNDS[1] - y) * dots_y / (Y_BOUNDS[1] - Y_BOUNDS[0])).round() as u16;
        (a.x + dx / 2, a.y + dy / 4)
    }

    /// The cell the tongue handle `◆` is drawn in: at the tongue's place
    /// along the tract and the height [`tongue_handle`] gives it, but held
    /// between the walls there as they are drawn, so it never floats
    /// outside the mouth. A press on it takes hold of the tongue. `None`
    /// before the tract is drawn.
    #[must_use]
    pub fn tongue_cell(&self) -> Option<(u16, u16)> {
        let a = self.tract_area;
        if a.width == 0 || a.height == 0 {
            return None;
        }
        let (hx, hy) = tongue_handle(self.snap.tongue);
        let x = f64::from(hx);
        let half = lerp_cell(&self.snap.diameters, x) / 2.0;
        let (col, row) = self.dot_cell(x, f64::from(hy));
        let (_, upper) = self.dot_cell(x, half);
        let (_, lower) = self.dot_cell(x, -half);
        Some((col, row.clamp(upper, lower)))
    }

    /// Hit areas of the glottis panel, given its inner area. Rows follow
    /// `ui::draw_glottis`.
    #[must_use]
    pub fn glottis_controls(inner: Rect) -> Vec<(Rect, Control)> {
        let bar = |row: u16, c: Control| {
            (
                Rect::new(inner.x + BAR_START, inner.y + row, BAR_WIDTH, 1),
                c,
            )
        };
        let button = |row: u16, from: u16, width: u16, c: Control| {
            (Rect::new(inner.x + from, inner.y + row, width, 1), c)
        };
        vec![
            button(0, 25, 3, Control::OctaveDown),
            button(0, 35, 3, Control::OctaveUp),
            bar(2, Control::Tense),
            bar(3, Control::Tongue),
            bar(4, Control::Velum),
            bar(5, Control::Vibrato),
            bar(6, Control::Reverb),
            bar(7, Control::Loud),
            button(8, 0, 12, Control::Wobble),
            button(8, 12, 11, Control::Drone),
            button(8, 23, 21, Control::Engine),
        ]
    }

    /// Hit areas of the persona panel, given its inner area: one dial per
    /// row, as `ui::draw_persona` draws them.
    #[must_use]
    pub fn persona_controls(inner: Rect) -> Vec<(Rect, Control)> {
        (0..DIALS.len())
            .map(|k| {
                #[allow(clippy::cast_possible_truncation)]
                let row = inner.y + k as u16;
                (
                    Rect::new(inner.x + BAR_START, row, BAR_WIDTH, 1),
                    Control::Dial(k),
                )
            })
            .collect()
    }

    /// Where a vowel landmark sits on the tract view for the engine in use.
    #[must_use]
    pub fn landmark(&self, vowel: Phoneme) -> Option<(f32, f32)> {
        vowel
            .tongue_for(self.engine, self.settings.tract_length)
            .map(tongue_handle)
    }

    /// Whether tract coordinates fall in the nasal cavity or the velum gap
    /// below it.
    #[allow(clippy::cast_precision_loss)]
    fn in_nose(x: f32, y: f32) -> bool {
        y > LUMEN && x >= synthie_trombone::NOSE_START as f32 - 1.0
    }

    /// What a press at this cell takes hold of.
    fn grab_at(&mut self, button: MouseButton, shift: bool, col: u16, row: u16) -> Option<Grab> {
        if button == MouseButton::Left
            && let Some(&(_, c)) = self
                .controls
                .iter()
                .find(|(r, _)| r.contains((col, row).into()))
        {
            return Some(Grab::Control(c));
        }
        if let Some(g) = self.chart_press(button, col, row) {
            return Some(g);
        }
        let (x, y) = self.to_tract(col, row)?;
        if button == MouseButton::Right {
            return (y.abs() <= LUMEN + 0.4).then_some(Grab::TongueOld);
        }
        if button != MouseButton::Left {
            return None;
        }
        if Self::in_nose(x, y) {
            return Some(Grab::Velum);
        }
        if shift {
            return Some(Grab::Tongue((0.0, 0.0)));
        }
        // The handle is drawn from what the voice reports, in the cell
        // `tongue_cell` gives; a press on or near that cell takes hold of
        // it where it is.
        let on_handle = self
            .tongue_cell()
            .and_then(|(c, r)| self.to_tract(c, r))
            .is_some_and(|(hx, hy)| (x - hx).abs() <= 1.6 && (y - hy).abs() <= 0.45);
        if on_handle {
            let (ix, iy) = tongue_handle(self.snap.tongue);
            return Some(Grab::Tongue((ix - x, iy - y)));
        }
        for (_, vowel) in LANDMARKS {
            if let Some((lx, ly)) = self.landmark(vowel)
                && (x - lx).abs() <= 1.0
                && (y - ly).abs() <= 0.35
            {
                // A click on a landmark is that vowel; nothing to drag.
                self.vowel(vowel);
                return None;
            }
        }
        (y.abs() <= LUMEN).then_some(Grab::Constriction)
    }

    /// Left press and drag: a finger on the tract wall, the tongue handle,
    /// the velum, or a control in the panel. Shift-drag and right-drag move
    /// the tongue.
    pub fn mouse(&mut self, ev: MouseEvent) {
        let (col, row) = (ev.column, ev.row);
        match ev.kind {
            MouseEventKind::Down(b) => {
                let shift = ev.modifiers.contains(KeyModifiers::SHIFT);
                self.grab = self.grab_at(b, shift, col, row);
                if let Some(g) = self.grab {
                    self.pull(g, col, row, true);
                }
            }
            MouseEventKind::Drag(_) => {
                if let Some(g) = self.grab {
                    self.pull(g, col, row, false);
                }
            }
            MouseEventKind::Up(_) => {
                if self.grab == Some(Grab::Constriction) {
                    self.send(Cmd::Constriction(SLOT_MOUSE, None));
                }
                self.grab = None;
            }
            _ => {}
        }
    }

    /// Follow the mouse with whatever it holds. `first` is the press itself.
    #[allow(clippy::cast_precision_loss)]
    fn pull(&mut self, grab: Grab, col: u16, row: u16, first: bool) {
        if let Grab::Control(c) = grab {
            self.work(c, col, first);
            return;
        }
        if let Grab::Chart(on_vowel) = grab {
            if !(first && on_vowel) {
                self.steer_to_cell(col, row, !first);
            }
            return;
        }
        // Outside the canvas the drag holds its last position.
        let Some((x, y)) = self.to_tract(col, row) else {
            return;
        };
        let index = x.clamp(2.0, N as f32 - 0.5);
        let span = TONGUE_DIAMETER.end() - TONGUE_DIAMETER.start();
        match grab {
            Grab::Constriction => {
                // Distance from the midline is the opening left; the finger
                // cannot leave the tract. The row of the midline shuts the
                // tract, the row beside it leaves a gap that hisses, and
                // rows further out only shape it.
                let diameter = match self.rows_from_midline(row) {
                    0 => FINGER_SHUT,
                    1 => FINGER_HISS,
                    _ => (y.abs().min(LUMEN) * 2.0 - 0.1).max(FINGER_OPEN),
                };
                self.send(Cmd::Constriction(
                    SLOT_MOUSE,
                    Some(Constriction::new(index, diameter)),
                ));
            }
            Grab::Tongue((dx, dy)) => {
                // The handle hangs below the midline: move it as far as the
                // mouse has moved since the press.
                let (x, y) = (x + dx, y + dy);
                let depth = ((y.abs() - 0.6) / 1.5).clamp(0.0, 1.0);
                self.set_tongue(
                    x.clamp(2.0, N as f32 - 0.5),
                    TONGUE_DIAMETER.start() + depth * span,
                );
            }
            Grab::TongueOld => {
                let d = TONGUE_DIAMETER.start() + (y.abs() / 1.75) * span;
                self.set_tongue(index, d);
            }
            Grab::Velum => {
                let open = if y >= NOSE_FLOOR {
                    // In the nose itself: a press opens it, or shuts it if
                    // it was open; dragging around inside changes nothing.
                    if !first {
                        return;
                    }
                    if self.nasal { 0.0 } else { 1.0 }
                } else {
                    // In the gap: the height is how far the velum hangs open.
                    ((y - LUMEN) / (NOSE_FLOOR - LUMEN)).clamp(0.0, 1.0)
                };
                self.nasal = open > 0.5;
                self.send(Cmd::Velum(open));
            }
            Grab::Control(_) | Grab::Chart(_) => {}
        }
    }

    /// Operate a panel control from a press or drag at column `col`.
    fn work(&mut self, c: Control, col: u16, first: bool) {
        if !c.is_bar() {
            if first {
                self.press(c);
            }
            return;
        }
        let Some(&(r, _)) = self.controls.iter().find(|(_, k)| *k == c) else {
            return;
        };
        let at = (f32::from(col.saturating_sub(r.x)) + 0.5) / f32::from(r.width.max(1));
        let v = at.clamp(0.0, 1.0);
        match c {
            Control::Tense => self.set_tenseness(v),
            Control::Tongue => {
                let (lo, hi) = (*TONGUE_INDEX.start(), *TONGUE_INDEX.end());
                self.set_tongue(lo + v * (hi - lo), self.tongue.1);
            }
            Control::Velum => {
                self.nasal = v > 0.5;
                self.send(Cmd::Velum(v));
            }
            Control::Vibrato => self.set_vibrato(v * 0.05),
            Control::Reverb => {
                self.reverb = v * 0.4;
                self.send(Cmd::Reverb(self.reverb));
            }
            Control::Loud => {
                self.loud = LOUD_RANGE.0 + v * (LOUD_RANGE.1 - LOUD_RANGE.0);
                self.send(Cmd::Lung(self.loud));
            }
            Control::Dial(k) => {
                let (_, lo, hi) = DIALS[k.min(DIALS.len() - 1)];
                self.set_dial(k, lo + v * (hi - lo));
            }
            _ => {}
        }
        self.preview();
    }

    /// Click a button.
    fn press(&mut self, c: Control) {
        match c {
            Control::OctaveDown => self.octave = (self.octave - 1).max(1),
            Control::OctaveUp => self.octave = (self.octave + 1).min(6),
            Control::Mode => match self.mode {
                Mode::Play => {
                    self.mode = Mode::Type;
                    self.status = TYPE_HINT.into();
                }
                Mode::Type => {
                    self.mode = Mode::Play;
                    self.status.clear();
                }
            },
            Control::Engine => self.toggle_engine(),
            Control::Wobble => {
                self.wobble = !self.wobble;
                self.send(Cmd::Wobble(self.wobble));
            }
            Control::Drone => self.toggle_drone(),
            Control::FormantLock => self.toggle_lock(),
            _ => {}
        }
    }
}

/// The start of `text` for the status line: [`QUOTE_CHARS`] characters at
/// most.
fn quote(text: &str) -> String {
    match text.char_indices().nth(QUOTE_CHARS) {
        Some((b, _)) => format!("{}…", text[..b].trim_end()),
        None => text.to_string(),
    }
}

// ----- recording takes ----------------------------------------------------

/// What the status line says about a saved take.
fn saved_status(saved: &crate::record::Saved) -> String {
    let mut status = format!(
        "saved {} · {:.1} s · {} commands",
        saved.log_path.display(),
        saved.seconds(),
        saved.commands
    );
    if saved.dropped_commands > 0 || saved.dropped_samples > 0 {
        let _ = write!(
            status,
            " · LOST {} commands, {} samples (the disk did not keep up)",
            saved.dropped_commands, saved.dropped_samples
        );
    }
    if let Some(e) = &saved.error {
        let _ = write!(status, " · write failed: {e}");
    }
    status
}

/// Send a command that must arrive, waiting up to [`MUST_ARRIVE`] for
/// room in the queue. It tries again rather than waits on the queue: a
/// sender waiting there would make the audio thread take a lock to wake
/// it. Whether it was sent.
fn send_surely(tx: &Sender<Cmd>, cmd: Cmd) -> bool {
    let start = Instant::now();
    let mut cmd = cmd;
    loop {
        match tx.try_send(cmd) {
            Ok(()) => return true,
            Err(crossbeam_channel::TrySendError::Full(c)) if start.elapsed() < MUST_ARRIVE => {
                cmd = c;
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(_) => return false,
        }
    }
}

impl App {
    /// The take being recorded.
    #[must_use]
    pub fn recording(&self) -> Option<&str> {
        self.link.recorder.as_ref().and_then(Recorder::current)
    }

    /// Start recording a take, or stop the one being recorded.
    fn toggle_recording(&mut self) {
        let info = self.take_info();
        let rate = self.sample_rate();
        let Some(recorder) = self.link.recorder.as_mut() else {
            self.status = "recording is not available".into();
            return;
        };
        if let Some(name) = recorder.close() {
            send_surely(&self.cmd, Cmd::Record(false));
            self.status = format!("saving {name}…");
            return;
        }
        match recorder.open(rate, info) {
            Ok(name) => {
                if send_surely(&self.cmd, Cmd::Record(true)) {
                    self.status = format!(
                        "recording {name} into {} · F5 stops",
                        recorder.dir().display()
                    );
                    // A take is for playing: in type mode the keys would
                    // only fill the text line, and nothing would sound.
                    self.mode = Mode::Play;
                } else {
                    recorder.cancel();
                    self.status = "cannot record: the audio thread is not answering".into();
                }
            }
            Err(e) => self.status = format!("cannot record: {e}"),
        }
    }

    /// Before leaving: stop the take being recorded and wait (a little)
    /// for it to be saved.
    pub fn finish(&mut self) {
        if self.recording().is_none() {
            return;
        }
        self.toggle_recording();
        if let Some(saved) = self.link.recorder.as_ref().and_then(|r| r.wait(SAVE_WAIT)) {
            self.status = saved_status(&saved);
        }
    }

    /// What the app knows about the voice, for a take's header: the
    /// preset, profile and persona it was made from, and the app's own
    /// view of the controls. For reading; a replay needs only the state
    /// the audio thread reports.
    fn take_info(&self) -> serde_json::Value {
        let value = |x: Result<serde_json::Value, _>| x.unwrap_or_default();
        crate::take::tidied(json!({
            "preset": self.preset.name(),
            "base": value(serde_json::to_value(self.base)),
            "persona": value(serde_json::to_value(self.persona)),
            "settings": value(serde_json::to_value(self.settings)),
            "engine": engine_name(self.engine),
            "note": self.note,
            "octave": self.octave,
            "held": self.held.is_some(),
            "speech_rate": self.rate,
            "tongue": [self.tongue.0, self.tongue.1],
            "tenseness": self.tenseness,
            "vibrato": self.vibrato,
            "wobble": self.wobble,
            "drone": self.drone,
            "nasal": self.nasal,
            "reverb": self.reverb,
            "loud": self.loud,
            "mimic": self.mimic,
            "formant_lock": self.formant_lock.as_ref().map(crate::take::shape_json),
            "keyboard_release": self.keyboard_release,
        }))
    }
}

// ----- Formant Lock -------------------------------------------------------

/// A vowel shape of the physical engine.
const fn shape(index: f32, height: f32, lips: Option<f32>, narrowing: (f32, f32)) -> Shape {
    Shape {
        tongue_index: index,
        tongue_diameter: height,
        lips,
        narrowing: Some(narrowing),
    }
}

/// Vowels marked on the vowel chart: IPA letter, phoneme, and the shape the
/// physical engine gives it (the library's physical vowel shapes; a test
/// checks that they still agree). Where two marks share a cell the first
/// is shown, so the corner vowels come first.
#[rustfmt::skip]
pub const CHART_VOWELS: [(&str, Phoneme, Shape); 11] = [
    ("i", Phoneme::Iy, shape(29.0, 2.257, None, (21.648, 0.858))),
    ("u", Phoneme::Uw, shape(26.181, 2.05, Some(0.733), (14.0, 1.203))),
    ("ɑ", Phoneme::Aa, shape(16.514, 3.165, None, (10.0, 0.927))),
    ("æ", Phoneme::Ae, shape(26.564, 2.93, None, (14.0, 1.621))),
    ("ɪ", Phoneme::Ih, shape(26.372, 2.449, None, (24.0, 3.0))),
    ("ɛ", Phoneme::Eh, shape(24.422, 2.663, None, (14.0, 1.486))),
    ("ʌ", Phoneme::Ah, shape(15.41, 2.297, None, (29.283, 1.584))),
    ("ɔ", Phoneme::Ao, shape(12.0, 2.219, None, (23.829, 1.8))),
    ("ʊ", Phoneme::Uh, shape(19.849, 2.38, Some(0.998), (32.217, 1.027))),
    ("o", Phoneme::Ow, shape(15.493, 2.345, Some(0.989), (30.202, 1.519))),
    ("ɝ", Phoneme::Er, shape(15.234, 2.777, Some(1.242), (30.498, 0.7))),
];

/// The chart vowel `v`: its IPA letter and shape.
fn chart_vowel(v: Phoneme) -> Option<(&'static str, Shape)> {
    CHART_VOWELS
        .iter()
        .find(|(_, p, _)| *p == v)
        .map(|&(name, _, s)| (name, s))
}

/// The babble map's formants for `shape` at the native length; where the
/// map has no data for the shape itself (a few extreme ones, among them
/// the physical /ʊ/), its formants a hair away from it.
fn predict_near(shape: &Shape) -> Option<[f32; 3]> {
    if let Some(f) = inverse::predict(shape) {
        return Some(f);
    }
    let x = shape.coordinates();
    for step in [0.01, 0.02, 0.04, 0.08] {
        for k in 0..5 {
            for sign in [-1.0, 1.0] {
                let mut y = x;
                y[k] = (y[k] + sign * step).clamp(0.0, 1.0);
                if let Some(f) = inverse::predict(&Shape::from_coordinates(y)) {
                    return Some(f);
                }
            }
        }
    }
    None
}

/// How far the vowel chart reaches past the babble map, as a ratio.
const CHART_MARGIN: f32 = 1.04;
/// Ratio an arrow key moves F1 or F2 by in Formant Lock.
const NUDGE: f32 = 1.02;
/// How far (ratio) a gliding Formant Lock may miss F1 or F2 to keep F3
/// steady.
const GLIDE_SLACK: f32 = 0.015;

/// The F1 × F2 rectangle the vowel chart spans, Hz, on logarithmic axes:
/// F2 falls from left to right and F1 rises from top to bottom, as
/// phoneticians draw it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartRange {
    pub f1: (f32, f32),
    pub f2: (f32, f32),
}

impl ChartRange {
    /// The babble map's extent for a tract of `length`, with a margin.
    pub fn for_length(length: f32) -> Self {
        let lo =
            inverse::formants_at_length([inverse::F1_RANGE.0, inverse::F2_RANGE.0, 2500.0], length);
        let hi =
            inverse::formants_at_length([inverse::F1_RANGE.1, inverse::F2_RANGE.1, 2500.0], length);
        Self {
            f1: (lo[0] / CHART_MARGIN, hi[0] * CHART_MARGIN),
            f2: (lo[1] / CHART_MARGIN, hi[1] * CHART_MARGIN),
        }
    }

    /// Where F1 `f1` and F2 `f2` sit on the chart, `(x, y)` with `0‥1`
    /// across the chart from the left and down from the top (not clamped).
    pub fn unit(&self, f1: f32, f2: f32) -> (f32, f32) {
        let x = (self.f2.1.ln() - f2.ln()) / (self.f2.1.ln() - self.f2.0.ln());
        let y = (f1.ln() - self.f1.0.ln()) / (self.f1.1.ln() - self.f1.0.ln());
        (x, y)
    }

    /// `(F1, F2)`, Hz, at chart position `(x, y)` (see [`unit`](Self::unit)).
    pub fn hz(&self, x: f32, y: f32) -> (f32, f32) {
        let f2 = (self.f2.1.ln() - x * (self.f2.1.ln() - self.f2.0.ln())).exp();
        let f1 = (self.f1.0.ln() + y * (self.f1.1.ln() - self.f1.0.ln())).exp();
        (f1, f2)
    }
}

impl App {
    /// The vowel chart's extent for the voice in use.
    pub fn chart_range(&self) -> ChartRange {
        ChartRange::for_length(self.settings.tract_length)
    }

    /// F1 and F2 (Hz) of a shape for the voice in use, as the babble map
    /// predicts them.
    pub fn shape_formants(&self, shape: &Shape) -> Option<(f32, f32)> {
        let [f1, f2, _] = self.shape_formants3(shape)?;
        Some((f1, f2))
    }

    /// F1, F2 and F3 (Hz) of a shape for the voice in use (see
    /// [`predict_near`]).
    fn shape_formants3(&self, shape: &Shape) -> Option<[f32; 3]> {
        let native = predict_near(shape)?;
        Some(inverse::formants_at_length(
            native,
            self.settings.tract_length,
        ))
    }

    /// Where the voice is on the vowel chart, `(F1, F2)` in Hz, while
    /// Formant Lock is on.
    pub fn lock_formants(&self) -> Option<(f32, f32)> {
        self.formant_lock.and_then(|s| self.shape_formants(&s))
    }

    /// The chart cell that shows `(f1, f2)`; points off the chart are held
    /// at its edge. `None` while the chart is not drawn.
    pub fn chart_cell(&self, f1: f32, f2: f32) -> Option<(u16, u16)> {
        let a = self.chart_area;
        if a.width == 0 || a.height == 0 {
            return None;
        }
        let (x, y) = self.chart_range().unit(f1, f2);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let at = |t: f32, n: u16| ((t * f32::from(n)).floor().max(0.0) as u16).min(n - 1);
        Some((a.x + at(x, a.width), a.y + at(y, a.height)))
    }

    /// `(F1, F2)`, Hz, at the centre of a chart cell; cells off the chart
    /// are held at its edge. `None` while the chart is not drawn.
    pub fn chart_point(&self, col: u16, row: u16) -> Option<(f32, f32)> {
        let a = self.chart_area;
        if a.width == 0 || a.height == 0 {
            return None;
        }
        let col = col.clamp(a.x, a.right() - 1);
        let row = row.clamp(a.y, a.bottom() - 1);
        let x = (f32::from(col - a.x) + 0.5) / f32::from(a.width);
        let y = (f32::from(row - a.y) + 0.5) / f32::from(a.height);
        Some(self.chart_range().hz(x, y))
    }

    /// The vowel marks on the chart and their cells, one per cell.
    pub fn chart_marks(&self) -> Vec<(&'static str, Phoneme, (u16, u16))> {
        let mut marks: Vec<(&'static str, Phoneme, (u16, u16))> = Vec::new();
        for &(name, v, s) in &CHART_VOWELS {
            let Some(at) = self
                .shape_formants(&s)
                .and_then(|(f1, f2)| self.chart_cell(f1, f2))
            else {
                continue;
            };
            if marks.iter().all(|m| m.2 != at) {
                marks.push((name, v, at));
            }
        }
        marks
    }

    /// Turn Formant Lock on or off. It belongs to the physical engine, so
    /// turning it on there switches to it. Turning it off leaves the tract
    /// as it is.
    fn toggle_lock(&mut self) {
        if self.formant_lock.take().is_some() {
            self.status = "formant lock off · the tract stays as it is".into();
            return;
        }
        let switched = self.engine != Engine::Physical;
        if switched {
            self.toggle_engine();
        }
        // Start from where the voice is: the last vowel's lips and
        // narrowing with the tongue where it is now. After a change of
        // engine the voice has just taken up that vowel afresh.
        let tongue = |s: Shape, (i, d): (f32, f32)| Shape {
            tongue_index: i,
            tongue_diameter: d,
            ..s
        };
        let last = self.last_vowel.and_then(chart_vowel).map(|(_, s)| s);
        let start = match last {
            Some(s) if switched => s,
            Some(s) => tongue(s, self.tongue),
            None => Shape {
                tongue_index: self.tongue.0,
                tongue_diameter: self.tongue.1,
                lips: None,
                narrowing: None,
            },
        };
        if self.shape_formants(&start).is_some() {
            self.hold_shape(start);
        } else {
            // Somewhere the map does not know: begin in the middle.
            let r = self.chart_range();
            let (f1, f2) = r.hz(0.5, 0.5);
            self.formant_lock = Some(start);
            self.steer(f1, f2, false);
        }
        self.status = if switched {
            "formant lock · switched to the physical engine · drag the vowel chart".into()
        } else {
            "formant lock · drag the vowel chart, ←→↑↓ nudge F2/F1".into()
        };
    }

    /// Hold `shape`: send it, and keep the tongue state in step.
    fn hold_shape(&mut self, shape: Shape) {
        self.formant_lock = Some(shape);
        self.tongue = (
            shape
                .tongue_index
                .clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
            shape
                .tongue_diameter
                .clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
        );
        self.send(Cmd::Shape(shape));
    }

    /// Steer the voice to F1 `f1` and F2 `f2` (Hz, for the voice in use),
    /// or the nearest point it can reach, from the shape it holds.
    /// `glide` is a continuous move (a drag or an arrow key), as opposed
    /// to a fresh press.
    fn steer(&mut self, f1: f32, f2: f32, glide: bool) {
        let length = self.settings.tract_length;
        let (a, b) = inverse::nearest_reachable_for(f1, f2, length);
        let near = self.formant_lock;
        let plain = || inverse::solve_for(a, b, None, near.as_ref(), length);
        // F1 and F2 leave the shape a degree of freedom, and the solver's
        // pull towards the previous shape is weak: off a vowel's own
        // shape it would take a different one at once. While gliding,
        // holding F3 where it is pins that freedom, so the shape moves
        // smoothly; unless that costs F1 or F2 more than a little.
        let f3 = near.and_then(|s| self.shape_formants3(&s)).map(|f| f[2]);
        let shape = match f3 {
            Some(f3) if glide => {
                let held = inverse::solve_for(a, b, Some(f3), near.as_ref(), length);
                let close = self.shape_formants(&held).is_some_and(|(g1, g2)| {
                    (g1 / a - 1.0).abs().max((g2 / b - 1.0).abs()) < GLIDE_SLACK
                });
                if close { held } else { plain() }
            }
            _ => plain(),
        };
        self.hold_shape(shape);
        let off = |want: f32, got: f32| (got / want - 1.0).abs() > 0.005;
        self.status = if off(f1, a) || off(f2, b) {
            format!("out of reach · F1 {a:.0} Hz, F2 {b:.0} Hz is as close as this tract gets")
        } else {
            format!("F1 {a:.0} Hz · F2 {b:.0} Hz")
        };
    }

    /// Follow a press (`glide` false) or drag at a cell of the vowel chart.
    fn steer_to_cell(&mut self, col: u16, row: u16, glide: bool) {
        if self.formant_lock.is_some()
            && let Some((f1, f2)) = self.chart_point(col, row)
        {
            self.steer(f1, f2, glide);
        }
    }

    /// What a press on the vowel chart takes hold of. A press on a vowel
    /// mark is that vowel.
    fn chart_press(&mut self, button: MouseButton, col: u16, row: u16) -> Option<Grab> {
        if button != MouseButton::Left
            || self.formant_lock.is_none()
            || !self.chart_area.contains((col, row).into())
        {
            return None;
        }
        let mark = self.chart_marks().into_iter().find(|m| m.2 == (col, row));
        if let Some((_, v, _)) = mark {
            self.vowel(v);
            return Some(Grab::Chart(true));
        }
        Some(Grab::Chart(false))
    }

    /// Arrow keys under Formant Lock: move F2 (left is higher, as on the
    /// chart) and F1 (up is lower) by [`NUDGE`]. `false` if the key is not
    /// one of them or the lock is off.
    fn nudge(&mut self, code: KeyCode) -> bool {
        if self.formant_lock.is_none() {
            return false;
        }
        let Some((f1, f2)) = self.lock_formants() else {
            return false;
        };
        let (f1, f2) = match code {
            KeyCode::Left => (f1, f2 * NUDGE),
            KeyCode::Right => (f1, f2 / NUDGE),
            KeyCode::Up => (f1 / NUDGE, f2),
            KeyCode::Down => (f1 * NUDGE, f2),
            _ => return false,
        };
        self.steer(f1, f2, true);
        true
    }

    /// A vowel was chosen (number key, landmark or chart mark): under
    /// Formant Lock the chart follows it.
    fn lock_follow_vowel(&mut self, v: Phoneme) {
        if self.formant_lock.is_none() {
            return;
        }
        if let Some((name, s)) = chart_vowel(v) {
            self.formant_lock = Some(s);
            if let Some((f1, f2)) = self.shape_formants(&s) {
                self.status = format!("{name} ({}) · F1 {f1:.0} Hz · F2 {f2:.0} Hz", v.symbol());
            }
        }
    }

    /// The tongue was moved some other way (its handle or meter): under
    /// Formant Lock the chart follows it.
    fn lock_follow_tongue(&mut self) {
        if let Some(s) = &mut self.formant_lock {
            (s.tongue_index, s.tongue_diameter) = self.tongue;
        }
    }

    /// The engine changed: Formant Lock only works on the physical one.
    fn lock_engine_changed(&mut self) {
        if self.engine != Engine::Physical && self.formant_lock.take().is_some() {
            self.status = format!(
                "engine: {} · formant lock off (it needs the physical engine)",
                engine_name(self.engine)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::{Receiver, bounded};

    fn app(release: bool) -> (App, Receiver<Cmd>) {
        let (cmd, rx) = bounded(1024);
        let (_s, snapshots) = bounded(1);
        let (_t, trash) = bounded(1);
        let link = Link {
            cmd,
            snapshots,
            trash,
            sample_rate: 48_000.0,
            microphone: false,
            microphone_name: None,
            recorder: None,
        };
        let app = App::new(link, release, Engine::Classic);
        while rx.try_recv().is_ok() {}
        (app, rx)
    }

    fn key(c: char, kind: KeyKind) -> Key {
        Key {
            code: KeyCode::Char(c),
            kind,
            ctrl: false,
            shift: false,
        }
    }

    fn sent(rx: &Receiver<Cmd>) -> Vec<Cmd> {
        rx.try_iter().collect()
    }

    #[test]
    fn space_stops_the_voice() {
        let (mut app, rx) = app(false);
        app.key(key('a', KeyKind::Press));
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOn(_)]));
        app.key(key(' ', KeyKind::Press));
        let cmds = sent(&rx);
        assert!(cmds.iter().any(|c| matches!(c, Cmd::NoteOff)), "{cmds:?}");
        assert!(!app.voiced);
    }

    #[test]
    fn escape_stops_the_voice_in_play_mode() {
        let (mut app, rx) = app(true);
        app.key(key('a', KeyKind::Press));
        sent(&rx);
        app.key(Key::press(KeyCode::Esc));
        assert!(sent(&rx).iter().any(|c| matches!(c, Cmd::NoteOff)));
    }

    #[test]
    fn space_also_ends_a_drone() {
        let (mut app, rx) = app(true);
        app.key(key('i', KeyKind::Press));
        sent(&rx);
        app.key(key(' ', KeyKind::Press));
        let cmds = sent(&rx);
        assert!(
            cmds.iter().any(|c| matches!(c, Cmd::Drone(false))),
            "{cmds:?}"
        );
        assert!(cmds.iter().any(|c| matches!(c, Cmd::NoteOff)));
    }

    #[test]
    fn releasing_the_held_key_stops_the_note() {
        let (mut app, rx) = app(true);
        app.key(key('a', KeyKind::Press));
        app.key(key('a', KeyKind::Repeat));
        sent(&rx);
        app.key(key('a', KeyKind::Release));
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
    }

    #[test]
    fn repeated_presses_mean_no_release_events() {
        let (mut app, rx) = app(true);
        app.key(key('a', KeyKind::Press));
        app.key(key('a', KeyKind::Press));
        assert!(!app.keyboard_release);
        assert!(app.status.contains("Space stops"));
        // The note keeps sounding; Space ends it.
        sent(&rx);
        app.key(key(' ', KeyKind::Press));
        assert!(sent(&rx).iter().any(|c| matches!(c, Cmd::NoteOff)));
    }

    // ----- the text line -------------------------------------------------

    fn ctrl(c: char) -> Key {
        Key {
            ctrl: true,
            ..Key::press(KeyCode::Char(c))
        }
    }

    /// A key as the terminal reports it.
    fn terminal(code: crossterm::event::KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn type_in(app: &mut App, text: &str) {
        for c in text.chars() {
            app.key(Key::press(KeyCode::Char(c)));
        }
    }

    /// How many utterances were sent to be spoken.
    fn said(cmds: &[Cmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, Cmd::Speak(..))).count()
    }

    #[test]
    fn a_paste_goes_in_at_the_cursor() {
        let (mut app, rx) = app(true);
        app.key(Key::press(KeyCode::Tab));
        type_in(&mut app, "say  now");
        for _ in 0..4 {
            app.key(Key::press(KeyCode::Left));
        }
        app.paste("hello,\r\nworld ");
        assert_eq!(app.line.text(), "say hello, world now");
        assert_eq!(app.line.cursor(), "say hello, world".len());
        assert_eq!(app.mode, Mode::Type);
        assert_eq!(said(&sent(&rx)), 0);
    }

    #[test]
    fn pasting_in_play_mode_switches_to_type_mode() {
        let (mut app, rx) = app(true);
        assert_eq!(app.mode, Mode::Play);
        app.paste("Good dog.");
        assert_eq!(app.mode, Mode::Type);
        assert_eq!(app.line.text(), "Good dog.");
        assert!(app.status.contains("Enter to speak"), "{}", app.status);
        // Nothing was played on the piano, and nothing said.
        let cmds = sent(&rx);
        assert!(
            !cmds
                .iter()
                .any(|c| matches!(c, Cmd::NoteOn(_) | Cmd::Speak(..))),
            "{cmds:?}"
        );
    }

    #[test]
    fn a_pasted_line_break_never_speaks() {
        let (mut app, rx) = app(true);
        app.paste("First line.\nSecond line.\r\n\nThird line.\n");
        assert_eq!(app.line.text(), "First line. Second line. Third line. ");
        assert_eq!(said(&sent(&rx)), 0);
        // Without bracketed paste the terminal sends the keys: the line
        // breaks in the burst are spaces too, a line feed on its own as
        // well.
        app.line.clear();
        let plain = KeyModifiers::NONE;
        for c in "ab".chars() {
            app.terminal_key(terminal(crossterm::event::KeyCode::Char(c), plain), true);
        }
        app.terminal_key(terminal(crossterm::event::KeyCode::Enter, plain), true);
        app.terminal_key(terminal(crossterm::event::KeyCode::Tab, plain), true);
        app.key(ctrl('j'));
        app.terminal_key(terminal(crossterm::event::KeyCode::Char('c'), plain), true);
        assert_eq!(app.line.text(), "ab c");
        assert_eq!(app.mode, Mode::Type);
        assert_eq!(said(&sent(&rx)), 0);
        // Enter speaks, all of it, once.
        app.paste(" Done.");
        app.key(Key::press(KeyCode::Enter));
        assert_eq!(said(&sent(&rx)), 1);
        assert!(app.line.is_empty());
        assert_eq!(app.history.last().map(String::as_str), Some("ab c Done."));
    }

    #[test]
    fn editing_keys_work_the_text_line() {
        let (mut app, rx) = app(true);
        app.key(Key::press(KeyCode::Tab));
        type_in(&mut app, "the quick brown fox");
        app.key(ctrl('w'));
        assert_eq!(app.line.text(), "the quick brown ");
        app.key(ctrl('w'));
        assert_eq!(app.line.text(), "the quick ");
        let word_left = Key {
            ctrl: true,
            ..Key::press(KeyCode::Left)
        };
        app.key(word_left);
        assert_eq!(app.line.cursor(), 4);
        app.key(Key::press(KeyCode::Delete));
        assert_eq!(app.line.text(), "the uick ");
        let home = terminal(crossterm::event::KeyCode::Home, KeyModifiers::NONE);
        let end = terminal(crossterm::event::KeyCode::End, KeyModifiers::NONE);
        app.terminal_key(home, false);
        assert_eq!(app.line.cursor(), 0);
        app.key(Key::press(KeyCode::Char('A')));
        app.terminal_key(end, false);
        app.key(Key::press(KeyCode::Backspace));
        assert_eq!(app.line.text(), "Athe uick");
        let alt_backspace = terminal(crossterm::event::KeyCode::Backspace, KeyModifiers::ALT);
        app.terminal_key(alt_backspace, false);
        assert_eq!(app.line.text(), "Athe ");
        // Control keys edit; they do not type their letter.
        app.key(ctrl('a'));
        app.key(ctrl('k'));
        assert!(app.line.is_empty());
        type_in(&mut app, "again");
        app.key(ctrl('u'));
        assert!(app.line.is_empty());
        // In play mode Home and End are no editing keys.
        app.key(Key::press(KeyCode::Tab));
        app.terminal_key(home, false);
        assert_eq!(app.mode, Mode::Play);
        assert_eq!(said(&sent(&rx)), 0);
    }

    #[test]
    fn the_text_line_holds_a_long_text_and_says_when_it_is_full() {
        let (mut app, rx) = app(true);
        let sentence = "The pink trombone speaks one sentence after another. ";
        let long = sentence.repeat(MAX_CHARS / sentence.len() + 2);
        app.paste(&long);
        assert!(app.line.is_full());
        assert!(app.status.contains("full"), "{}", app.status);
        type_in(&mut app, "x");
        assert!(app.status.contains("full"), "{}", app.status);
        // All of it is compiled and sent as one utterance.
        let t0 = Instant::now();
        app.key(Key::press(KeyCode::Enter));
        let took = t0.elapsed();
        let cmds = sent(&rx);
        assert_eq!(said(&cmds), 1);
        println!("{MAX_CHARS} characters compiled in {took:?}");
        assert!(took < Duration::from_secs(1), "{took:?}");
        // The status quotes only the start.
        assert!(
            app.status.chars().count() < QUOTE_CHARS + 20,
            "{}",
            app.status
        );
    }

    #[test]
    fn escape_in_type_mode_stops_speech_first() {
        let (mut app, rx) = app(true);
        app.paste("One. Two. Three.");
        app.key(Key::press(KeyCode::Enter));
        sent(&rx);
        // Speech is under way, as the audio thread reports it.
        app.snap.speaking = Some(12);
        app.key(Key::press(KeyCode::Esc));
        let cmds = sent(&rx);
        assert!(
            matches!(cmds[..], [Cmd::StopSpeaking]),
            "only speech stops: {cmds:?}"
        );
        assert_eq!(app.mode, Mode::Type);
        app.snap.speaking = None;
        app.key(Key::press(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Play);
    }

    #[test]
    fn several_sentences_are_one_utterance_with_a_pause_and_a_new_contour_each() {
        let (mut app, rx) = app(true);
        app.paste("The cat sat down.\nThe dog ran far!\nWhere is the bird?\n");
        app.key(Key::press(KeyCode::Enter));
        let cmds = sent(&rx);
        let Some(Cmd::Speak(u, _)) = cmds.iter().find(|c| matches!(c, Cmd::Speak(..))) else {
            panic!("nothing said: {cmds:?}");
        };
        let segs = u.segments();
        // A sentence's pause, and the short release at the very end.
        let pauses: Vec<usize> = (0..segs.len())
            .filter(|&k| segs[k].phoneme == Phoneme::Sil && segs[k].duration > 0.25)
            .collect();
        assert_eq!(pauses.len(), 2, "{segs:?}");
        let vowel = |s: &&synthie_trombone::speech::Segment| {
            matches!(
                s.phoneme.manner(),
                synthie_trombone::Manner::Vowel | synthie_trombone::Manner::Diphthong
            )
        };
        let mut starts = Vec::new();
        for &p in &pauses {
            // Pitch falls into the pause and starts afresh after it.
            let before = segs[..p].iter().rev().find(vowel).unwrap().pitch.1;
            let after = segs[p..].iter().find(vowel).unwrap().pitch.1;
            println!("pause at {p}: {before:.1} Hz, then {after:.1} Hz");
            assert!(after > before * 1.05, "{before} Hz, then {after} Hz");
            starts.push(after);
        }
        // The question rises at the end, above where it started.
        let last = segs.iter().rev().find(vowel).unwrap().pitch.1;
        assert!(last > starts[1] * 1.1, "{last} Hz from {} Hz", starts[1]);
        // The strip of phonemes covers all three sentences.
        let n: usize = app.spoken.iter().map(|(_, r)| r.len()).sum();
        assert_eq!(n, segs.len());
    }

    // ----- mouse -------------------------------------------------------

    use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    /// An app with a tract canvas and a glottis panel laid out.
    fn laid_out() -> (App, Receiver<Cmd>) {
        let (mut app, rx) = app(true);
        app.tract_area = Rect::new(1, 2, 138, 26);
        app.controls = App::glottis_controls(Rect::new(1, 30, 44, 9));
        app.controls
            .extend(App::persona_controls(Rect::new(56, 30, 36, 9)));
        app.controls.push((Rect::new(40, 0, 6, 1), Control::Mode));
        // Park the tongue handle away from where the tests press.
        app.snap.tongue = (13.0, 3.5);
        (app, rx)
    }

    /// The terminal cell showing tract coordinates `(x, y)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn cell(app: &App, x: f64, y: f64) -> (u16, u16) {
        let a = app.tract_area;
        let fx = (x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0]);
        let fy = (Y_BOUNDS[1] - y) / (Y_BOUNDS[1] - Y_BOUNDS[0]);
        (
            a.x + (fx * f64::from(a.width)) as u16,
            a.y + (fy * f64::from(a.height)) as u16,
        )
    }

    fn mouse(app: &mut App, kind: MouseEventKind, at: (u16, u16), modifiers: KeyModifiers) {
        app.mouse(MouseEvent {
            kind,
            column: at.0,
            row: at.1,
            modifiers,
        });
    }

    const DOWN: MouseEventKind = MouseEventKind::Down(MouseButton::Left);
    const DRAG: MouseEventKind = MouseEventKind::Drag(MouseButton::Left);
    const UP: MouseEventKind = MouseEventKind::Up(MouseButton::Left);
    const PLAIN: KeyModifiers = KeyModifiers::NONE;

    #[test]
    fn a_finger_in_the_mouth_constricts_and_lets_go() {
        let (mut app, rx) = laid_out();
        let at = cell(&app, 30.0, 0.3);
        mouse(&mut app, DOWN, at, PLAIN);
        let cmds = sent(&rx);
        let [Cmd::Constriction(SLOT_MOUSE, Some(c))] = cmds[..] else {
            panic!("{cmds:?}");
        };
        assert!((c.index - 30.0).abs() < 1.0 && c.diameter < 1.0, "{c:?}");
        mouse(&mut app, UP, at, PLAIN);
        assert!(matches!(
            sent(&rx)[..],
            [Cmd::Constriction(SLOT_MOUSE, None)]
        ));
    }

    #[test]
    fn the_row_beside_the_midline_leaves_a_gap_that_hisses() {
        // Whatever the height of the terminal: on the midline's row the
        // finger shuts the tract, one row off it leaves a fricative's gap,
        // and further out the tract is only shaped.
        for height in [18, 26, 40, 61] {
            let (mut app, rx) = laid_out();
            app.tract_area = Rect::new(1, 2, 138, height);
            let (col, midline) = cell(&app, 36.0, 0.0);
            let mut openings = Vec::new();
            for rows in 0..4 {
                for row in [midline + rows, midline - rows] {
                    mouse(&mut app, DOWN, (col, row), PLAIN);
                    mouse(&mut app, UP, (col, row), PLAIN);
                    let cmds = sent(&rx);
                    let [Cmd::Constriction(SLOT_MOUSE, Some(c)), _] = cmds[..] else {
                        panic!("height {height}, {rows} rows off: {cmds:?}");
                    };
                    openings.push(c.diameter);
                }
            }
            let [shut, _, above, below, wide, _, wider, _] = openings[..] else {
                panic!("{openings:?}");
            };
            assert!(shut <= 0.01, "height {height}: {openings:?}");
            assert!(
                (above - 0.5).abs() < 0.01 && (below - 0.5).abs() < 0.01,
                "height {height}: {openings:?}"
            );
            assert!(
                wide >= 0.9 && wider >= wide,
                "height {height}: {openings:?}"
            );
        }
    }

    #[test]
    fn persona_dials_answer_the_mouse_and_the_keys() {
        let (mut app, rx) = laid_out();
        // Dragging the size dial to its left end makes a small, high voice.
        let r = app
            .controls
            .iter()
            .find(|(_, c)| *c == Control::Dial(0))
            .unwrap()
            .0;
        mouse(&mut app, DOWN, (r.x, r.y), PLAIN);
        mouse(&mut app, UP, (r.x, r.y), PLAIN);
        // With nothing else sounding, the new voice is heard.
        let cmds = sent(&rx);
        let [Cmd::Voice(small), Cmd::Note(note), Cmd::NoteOn(heard)] = cmds[..] else {
            panic!("{cmds:?}");
        };
        assert!((heard - note).abs() < 1e-3, "{cmds:?}");
        let deep = Preset::Deep.settings();
        assert!(app.persona.size < 0.65, "{:?}", app.persona);
        assert!(small.tract_length < 0.65 * deep.tract_length + 0.02);
        assert!(small.pitch > 2.5 * deep.pitch && note > 55.0, "{small:?}");
        // The apostrophe picks the next dial; PgUp and PgDn turn it.
        app.key(key('\'', KeyKind::Press));
        assert_eq!(app.dial, 1);
        let turn = |app: &mut App, code| {
            app.key(Key {
                code,
                kind: KeyKind::Press,
                ..key(' ', KeyKind::Press)
            });
        };
        turn(&mut app, KeyCode::PageUp);
        turn(&mut app, KeyCode::PageUp);
        assert!((app.persona.effort - 0.6).abs() < 1e-3, "{:?}", app.persona);
        let cmds = sent(&rx);
        let Some(Cmd::Voice(louder)) = cmds.iter().rev().find(|c| matches!(c, Cmd::Voice(_)))
        else {
            panic!("{cmds:?}");
        };
        assert!(louder.tuning.lung_pressure > small.tuning.lung_pressure);
        assert!((app.loud - louder.tuning.lung_pressure).abs() < 1e-3);
        // Del goes back to the voice as it was.
        turn(&mut app, KeyCode::Delete);
        assert_eq!(app.persona, Persona::NEUTRAL);
        assert_eq!(app.settings, deep);
        // A change of voice keeps the dials.
        turn(&mut app, KeyCode::PageUp);
        app.key(key('v', KeyKind::Press));
        assert!((app.persona.effort - 0.55).abs() < 1e-3);
        let next = Preset::ALL[1].settings();
        assert_eq!(
            app.settings,
            next.with_persona(&app.persona),
            "the next voice, with the same persona"
        );
    }

    #[test]
    fn mimic_needs_a_microphone_and_the_physical_engine() {
        // Without a microphone the key only says how to get one.
        let (mut app, rx) = app(true);
        app.key(key('`', KeyKind::Press));
        assert!(sent(&rx).is_empty());
        assert!(app.status.contains("--mimic"), "{}", app.status);
        // With one, it switches to the physical engine and starts copying;
        // the same key stops it.
        app.link.microphone = true;
        app.key(key('`', KeyKind::Press));
        let cmds = sent(&rx);
        assert!(
            matches!(cmds[..], [Cmd::Model(Engine::Physical), Cmd::Mimic(true)]),
            "{cmds:?}"
        );
        assert_eq!((app.mimic, app.engine), (Some(true), Engine::Physical));
        app.key(key('`', KeyKind::Press));
        assert!(matches!(sent(&rx)[..], [Cmd::Mimic(false)]));
        assert_eq!(app.mimic, Some(false));
    }

    #[test]
    fn a_finger_cannot_leave_the_tract() {
        let (mut app, rx) = laid_out();
        // Above the throat, where there is no nose: nothing to press on.
        let nowhere = cell(&app, 8.0, 3.4);
        mouse(&mut app, DOWN, nowhere, PLAIN);
        mouse(&mut app, UP, nowhere, PLAIN);
        assert!(sent(&rx).is_empty());
        // A drag that wanders off the mouth stays a finger at its edge.
        let inside = cell(&app, 30.0, 0.5);
        let outside = cell(&app, 30.0, 4.4);
        mouse(&mut app, DOWN, inside, PLAIN);
        mouse(&mut app, DRAG, outside, PLAIN);
        let cmds = sent(&rx);
        for c in &cmds {
            let Cmd::Constriction(SLOT_MOUSE, Some(c)) = c else {
                panic!("{cmds:?}");
            };
            assert!(c.diameter <= 2.0 * LUMEN, "{c:?}");
        }
        assert!(!app.nasal);
    }

    #[test]
    fn the_nose_and_the_velum_gap_work_the_velum() {
        let (mut app, rx) = laid_out();
        let nose = cell(&app, 30.0, 3.6);
        mouse(&mut app, DOWN, nose, PLAIN);
        mouse(&mut app, UP, nose, PLAIN);
        assert!(matches!(sent(&rx)[..], [Cmd::Velum(v)] if v > 0.99));
        assert!(app.nasal);
        mouse(&mut app, DOWN, nose, PLAIN);
        mouse(&mut app, UP, nose, PLAIN);
        assert!(matches!(sent(&rx)[..], [Cmd::Velum(v)] if v < 0.01));
        // In the gap under it, the height sets how far it hangs open.
        let gap = cell(&app, 19.0, 2.3);
        mouse(&mut app, DOWN, gap, PLAIN);
        let cmds = sent(&rx);
        assert!(
            matches!(cmds[..], [Cmd::Velum(v)] if v > 0.2 && v < 0.9),
            "{cmds:?}"
        );
    }

    #[test]
    fn the_tongue_handle_drags_with_the_left_button() {
        let (mut app, rx) = laid_out();
        app.snap.tongue = (20.0, 2.5);
        let (hx, hy) = tongue_handle(app.snap.tongue);
        let handle = app.tongue_cell().unwrap();
        mouse(&mut app, DOWN, handle, PLAIN);
        // Taking hold of it does not move it.
        let cmds = sent(&rx);
        let [Cmd::Tongue(i, d)] = cmds[..] else {
            panic!("{cmds:?}");
        };
        assert!((i - hx).abs() < 1e-4 && (d - 2.5).abs() < 1e-4, "{cmds:?}");
        let to = cell(&app, 25.0, -1.5);
        mouse(&mut app, DRAG, to, PLAIN);
        mouse(&mut app, UP, to, PLAIN);
        let cmds = sent(&rx);
        assert!(
            cmds.iter().all(|c| matches!(c, Cmd::Tongue(..))),
            "{cmds:?}"
        );
        let Some(Cmd::Tongue(i, d)) = cmds.last() else {
            panic!("{cmds:?}");
        };
        assert!((i - 25.0).abs() < 0.6, "{i}");
        // The handle moved as far as the mouse did.
        let (from, to) = (
            app.to_tract(handle.0, handle.1).unwrap(),
            app.to_tract(to.0, to.1).unwrap(),
        );
        let (_, y) = tongue_handle((*i, *d));
        assert!((y - (hy + to.1 - from.1)).abs() < 1e-3, "{y}");
    }

    #[test]
    fn shift_drag_moves_the_tongue_from_anywhere() {
        let (mut app, rx) = laid_out();
        let at = cell(&app, 24.0, 0.8);
        mouse(&mut app, DOWN, at, KeyModifiers::SHIFT);
        mouse(&mut app, UP, at, KeyModifiers::SHIFT);
        let cmds = sent(&rx);
        assert!(matches!(cmds[..], [Cmd::Tongue(..)]), "{cmds:?}");
    }

    #[test]
    fn clicking_a_vowel_landmark_sets_the_vowel() {
        let (mut app, rx) = laid_out();
        let (x, y) = app.landmark(Phoneme::Uw).unwrap();
        let at = cell(&app, f64::from(x), f64::from(y));
        mouse(&mut app, DOWN, at, PLAIN);
        mouse(&mut app, UP, at, PLAIN);
        let cmds = sent(&rx);
        assert!(matches!(cmds[..], [Cmd::Vowel(Phoneme::Uw)]), "{cmds:?}");
        assert_eq!(app.last_vowel, Some(Phoneme::Uw));
    }

    #[test]
    fn panel_meters_follow_the_mouse() {
        let (mut app, rx) = laid_out();
        // With the drone on, a meter sends only what it sets.
        app.drone = true;
        let bar = |app: &App, c: Control, share: f32| {
            let r = app.controls.iter().find(|(_, k)| *k == c).unwrap().0;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let col = r.x + (share * f32::from(r.width)) as u16;
            (col, r.y)
        };
        let at = bar(&app, Control::Tense, 0.5);
        mouse(&mut app, DOWN, at, PLAIN);
        let further = bar(&app, Control::Tense, 0.9);
        mouse(&mut app, DRAG, further, PLAIN);
        mouse(&mut app, UP, further, PLAIN);
        let cmds = sent(&rx);
        assert!(
            matches!(cmds[..], [Cmd::Tenseness(a), Cmd::Tenseness(b)] if (a - 0.5).abs() < 0.05 && (b - 0.9).abs() < 0.05),
            "{cmds:?}"
        );
        for (c, expect) in [
            (Control::Velum, "Velum"),
            (Control::Vibrato, "Vibrato"),
            (Control::Reverb, "Reverb"),
            (Control::Loud, "Lung"),
            (Control::Tongue, "Tongue"),
        ] {
            let at = bar(&app, c, 0.5);
            mouse(&mut app, DOWN, at, PLAIN);
            mouse(&mut app, UP, at, PLAIN);
            let cmds = sent(&rx);
            assert!(
                cmds.len() == 1 && format!("{:?}", cmds[0]).starts_with(expect),
                "{c:?}: {cmds:?}"
            );
        }
        assert!((app.loud - 11.0).abs() < 0.6, "{}", app.loud);
    }

    #[test]
    fn panel_buttons_click() {
        let (mut app, rx) = laid_out();
        let click = |app: &mut App, c: Control| {
            let r = app.controls.iter().find(|(_, k)| *k == c).unwrap().0;
            mouse(app, DOWN, (r.x + 1, r.y), PLAIN);
            mouse(app, UP, (r.x + 1, r.y), PLAIN);
        };
        let octave = app.octave;
        click(&mut app, Control::OctaveDown);
        assert_eq!(app.octave, octave - 1);
        click(&mut app, Control::OctaveUp);
        click(&mut app, Control::OctaveUp);
        assert_eq!(app.octave, octave + 1);
        assert!(sent(&rx).is_empty());

        click(&mut app, Control::Engine);
        assert_eq!(app.engine, Engine::Physical);
        assert!(matches!(sent(&rx)[..], [Cmd::Model(Engine::Physical)]));
        click(&mut app, Control::Wobble);
        assert!(matches!(sent(&rx)[..], [Cmd::Wobble(true)]));
        click(&mut app, Control::Drone);
        assert!(matches!(sent(&rx)[..], [Cmd::Drone(true)]));
        click(&mut app, Control::Mode);
        assert_eq!(app.mode, Mode::Type);
        click(&mut app, Control::Mode);
        assert_eq!(app.mode, Mode::Play);
    }

    // ----- Formant Lock -------------------------------------------------

    use synthie_trombone::Trombone;

    /// `laid_out()` with Formant Lock on and a vowel chart the size a
    /// 120×34 terminal draws it (26×9 cells), clear of the other areas.
    fn locked() -> (App, Receiver<Cmd>) {
        let (mut app, rx) = laid_out();
        app.key(key('/', KeyKind::Press));
        app.chart_area = Rect::new(93, 40, 26, 9);
        sent(&rx);
        (app, rx)
    }

    fn arrow(code: KeyCode) -> Key {
        Key {
            code,
            ..key(' ', KeyKind::Press)
        }
    }

    /// The shapes among `cmds`.
    fn shapes(cmds: &[Cmd]) -> Vec<Shape> {
        cmds.iter()
            .filter_map(|c| match c {
                Cmd::Shape(s) => Some(*s),
                _ => None,
            })
            .collect()
    }

    /// F1 and F2 of a shape for the app's voice, as the babble map has it.
    fn heard(app: &App, s: &Shape) -> (f32, f32) {
        app.shape_formants(s).expect("the map knows the shape")
    }

    /// Larger relative error of two `(F1, F2)` pairs.
    fn error(a: (f32, f32), b: (f32, f32)) -> f32 {
        (a.0 / b.0 - 1.0).abs().max((a.1 / b.1 - 1.0).abs())
    }

    /// Press and let go at a cell; what was sent.
    fn click_at(app: &mut App, rx: &Receiver<Cmd>, at: (u16, u16)) -> Vec<Cmd> {
        mouse(app, DOWN, at, PLAIN);
        mouse(app, UP, at, PLAIN);
        sent(rx)
    }

    /// Every cell of the chart.
    fn chart_cells(app: &App) -> Vec<(u16, u16)> {
        let a = app.chart_area;
        (a.y..a.bottom())
            .flat_map(|r| (a.x..a.right()).map(move |c| (c, r)))
            .collect()
    }

    #[test]
    fn formant_lock_toggles_on_the_physical_engine() {
        let (mut app, rx) = laid_out();
        assert_eq!(app.engine, Engine::Classic);
        // On the classic engine it switches to the physical one and says so.
        app.key(key('/', KeyKind::Press));
        assert_eq!(app.engine, Engine::Physical);
        assert!(app.formant_lock.is_some());
        assert!(app.status.contains("physical engine"), "{}", app.status);
        let cmds = sent(&rx);
        assert!(
            matches!(cmds[..], [Cmd::Model(Engine::Physical), Cmd::Shape(_)]),
            "{cmds:?}"
        );
        // Off again: the tract stays where it is.
        app.key(key('/', KeyKind::Press));
        assert!(app.formant_lock.is_none());
        assert_eq!(app.engine, Engine::Physical);
        assert!(sent(&rx).is_empty());
        // The button does the same.
        app.controls
            .push((Rect::new(108, 39, 9, 1), Control::FormantLock));
        click_at(&mut app, &rx, (110, 39));
        assert!(app.formant_lock.is_some());
        assert!(matches!(sent(&rx)[..], []));
        // Leaving the physical engine leaves the lock.
        app.key(key('\\', KeyKind::Press));
        assert_eq!(app.engine, Engine::Classic);
        assert!(app.formant_lock.is_none());
        assert!(app.status.contains("formant lock off"), "{}", app.status);
        // Arrow keys move the tongue again.
        sent(&rx);
        app.key(arrow(KeyCode::Left));
        assert!(matches!(sent(&rx)[..], [Cmd::Tongue(..)]));
    }

    #[test]
    fn a_press_on_the_chart_steers_the_voice_there() {
        let (mut app, rx) = locked();
        let length = app.settings.tract_length;
        let (mut worst, mut pressed) = (0.0f32, 0);
        for at in chart_cells(&app) {
            let target = app.chart_point(at.0, at.1).unwrap();
            if !inverse::reachable_for(target.0, target.1, length) {
                continue;
            }
            let cmds = click_at(&mut app, &rx, at);
            if matches!(cmds[..], [Cmd::Vowel(_)]) {
                continue; // a vowel mark
            }
            let [Cmd::Shape(s)] = cmds[..] else {
                panic!("{at:?}: {cmds:?}");
            };
            let e = error(heard(&app, &s), target);
            assert!(e < 0.03, "{at:?}: {target:?} → {:?}", heard(&app, &s));
            worst = worst.max(e);
            pressed += 1;
            // The tongue handle and the arrow keys start from the solved shape.
            assert_eq!(app.tongue, (s.tongue_index, s.tongue_diameter));
            assert_eq!(app.formant_lock, Some(s));
        }
        println!("{pressed} reachable cells, largest formant error {worst:.4}");
        assert!(pressed > 40, "{pressed}");
    }

    /// The largest change of any shape coordinate between consecutive
    /// shapes.
    fn largest_step(path: &[Shape]) -> f32 {
        path.windows(2)
            .map(|w| {
                let (a, b) = (w[0].coordinates(), w[1].coordinates());
                (0..5).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max)
            })
            .fold(0.0, f32::max)
    }

    /// Drag on the chart from the /i/ mark to the /ɑ/ mark, cell by cell;
    /// the shapes the voice went through.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn drag_i_to_a(app: &mut App, rx: &Receiver<Cmd>) -> Vec<Shape> {
        let mark = |app: &App, v| {
            app.chart_marks()
                .into_iter()
                .find(|m| m.1 == v)
                .map(|m| m.2)
                .unwrap()
        };
        let (from, to) = (mark(app, Phoneme::Iy), mark(app, Phoneme::Aa));
        mouse(app, DOWN, from, PLAIN);
        assert!(matches!(sent(rx)[..], [Cmd::Vowel(Phoneme::Iy)]));
        let mut path = vec![app.formant_lock.unwrap()];
        let steps = from.0.abs_diff(to.0).max(from.1.abs_diff(to.1));
        for k in 1..=steps {
            let t = f32::from(k) / f32::from(steps);
            let lerp =
                |a: u16, b: u16| (f32::from(a) + t * (f32::from(b) - f32::from(a))).round() as u16;
            mouse(app, DRAG, (lerp(from.0, to.0), lerp(from.1, to.1)), PLAIN);
        }
        mouse(app, UP, to, PLAIN);
        path.extend(shapes(&sent(rx)));
        assert_eq!(path.len(), usize::from(steps) + 1);
        let end = app.chart_point(to.0, to.1).unwrap();
        assert!(error(heard(app, path.last().unwrap()), end) < 0.03);
        path
    }

    #[test]
    fn dragging_from_i_to_a_moves_the_shape_smoothly() {
        // As large as a 120×34 terminal draws the chart.
        let (mut app, rx) = locked();
        let path = drag_i_to_a(&mut app, &rx);
        let coarse = largest_step(&path);
        // A chart three times finer: the steps shrink with it, so there is
        // no jump in the path, only the size of a terminal cell.
        app.chart_area = Rect::new(0, 40, 78, 27);
        let path_fine = drag_i_to_a(&mut app, &rx);
        let fine = largest_step(&path_fine);
        println!(
            "i → ɑ: {} steps, largest coordinate step {coarse:.3}; {} steps on a 3× finer chart, {fine:.3}",
            path.len() - 1,
            path_fine.len() - 1
        );
        // Near /i/ the narrowing opens fast as F1 rises (about 0.15 of its
        // range per step, over several steps), but no single step jumps:
        // in particular the first step off the vowel's own shape is small.
        // On the coarse chart a step is 10 % of F1, and the last one, onto
        // /ɑ/ (F1 above 760 Hz, which only a pharynx narrowed low reaches),
        // moves the shape by up to 0.35; the fine chart shows that it is a
        // big step, not a jump.
        assert!(coarse < 0.35, "{coarse}: {path:?}");
        assert!(fine < 0.2, "{fine}: {path_fine:?}");
        let first = largest_step(&path_fine[..2]);
        println!("first step off the /i/ mark: {first:.3}");
        assert!(first < 0.1, "{first}: {path_fine:?}");
    }

    #[test]
    fn a_press_out_of_reach_is_held_at_the_edge() {
        let (mut app, rx) = locked();
        let length = app.settings.tract_length;
        // Low F2 with a wide-open F1 (bottom right) is out of reach.
        let a = app.chart_area;
        let at = (a.right() - 1, a.bottom() - 1);
        let target = app.chart_point(at.0, at.1).unwrap();
        assert!(!inverse::reachable_for(target.0, target.1, length));
        let edge = inverse::nearest_reachable_for(target.0, target.1, length);
        assert!(error(edge, target) > 0.05, "{edge:?} {target:?}");
        let cmds = click_at(&mut app, &rx, at);
        let [Cmd::Shape(s)] = cmds[..] else {
            panic!("{cmds:?}");
        };
        let got = heard(&app, &s);
        println!("out of reach: {target:?} → edge {edge:?}, voice {got:?}");
        assert!(error(got, edge) < 0.03, "{got:?} vs {edge:?}");
        assert!(app.status.contains("out of reach"), "{}", app.status);
        // A drag that leaves the chart is held at its edge as well.
        let inside = (a.x + 3, a.y + 2);
        mouse(&mut app, DOWN, inside, PLAIN);
        mouse(&mut app, DRAG, (a.x, a.y + 30), PLAIN);
        mouse(&mut app, UP, (a.x, a.y + 30), PLAIN);
        let cmds = sent(&rx);
        assert_eq!(shapes(&cmds).len(), 2, "{cmds:?}");
    }

    #[test]
    fn the_chart_is_in_the_hz_of_the_voice_in_use() {
        let (mut app, rx) = locked();
        let native = app.chart_range();
        app.set_dial(0, 0.8);
        let length = app.settings.tract_length;
        assert!((length - 0.8).abs() < 1e-3, "{length}");
        assert!(app.formant_lock.is_some());
        sent(&rx);
        let small = app.chart_range();
        assert_eq!(small, ChartRange::for_length(length));
        assert!(small.f2.1 > native.f2.1 * 1.15, "{small:?} {native:?}");
        assert!(small.f1.0 > native.f1.0 * 1.1, "{small:?} {native:?}");
        // A press lands where it was made, in this voice's Hz…
        let a = app.chart_area;
        let at = (a.x + 12, a.y + 4);
        let target = app.chart_point(at.0, at.1).unwrap();
        assert!(inverse::reachable_for(target.0, target.1, length));
        let cmds = click_at(&mut app, &rx, at);
        let [Cmd::Shape(s)] = cmds[..] else {
            panic!("{cmds:?}");
        };
        let got = heard(&app, &s);
        assert!(error(got, target) < 0.03, "{got:?} vs {target:?}");
        // …which a tract of native length would not make.
        let n = inverse::predict(&s).unwrap();
        assert!(error((n[0], n[1]), target) > 0.12, "{n:?} vs {target:?}");
        // The vowels move with the voice: /i/ sits further left.
        let iy = CHART_VOWELS[0].2;
        let small_i = heard(&app, &iy);
        app.set_dial(0, 1.0);
        let native_i = heard(&app, &iy);
        assert!(small_i.1 > native_i.1 * 1.15, "{small_i:?} {native_i:?}");
    }

    #[test]
    fn arrow_keys_nudge_f1_and_f2() {
        let (mut app, rx) = locked();
        let a = app.chart_area;
        click_at(&mut app, &rx, (a.x + 12, a.y + 4));
        for (code, df1, df2) in [
            (KeyCode::Left, 1.0, NUDGE),
            (KeyCode::Right, 1.0, 1.0 / NUDGE),
            (KeyCode::Up, 1.0 / NUDGE, 1.0),
            (KeyCode::Down, NUDGE, 1.0),
        ] {
            let (f1, f2) = app.lock_formants().unwrap();
            app.key(arrow(code));
            let cmds = sent(&rx);
            let [Cmd::Shape(s)] = cmds[..] else {
                panic!("{code:?}: {cmds:?}");
            };
            let (g1, g2) = heard(&app, &s);
            println!("{code:?}: F1 ×{:.4}, F2 ×{:.4}", g1 / f1, g2 / f2);
            assert!(
                (g1 / f1 / df1 - 1.0).abs() < 0.006 && (g2 / f2 / df2 - 1.0).abs() < 0.006,
                "{code:?}: ({f1}, {f2}) → ({g1}, {g2})"
            );
        }
    }

    #[test]
    fn vowel_marks_on_the_chart_and_vowel_keys_move_the_cursor() {
        let (mut app, rx) = locked();
        let marks = app.chart_marks();
        assert!(marks.len() >= 8, "{marks:?}");
        let &(_, v, at) = marks.iter().find(|m| m.1 == Phoneme::Uw).unwrap();
        let cmds = click_at(&mut app, &rx, at);
        assert!(matches!(cmds[..], [Cmd::Vowel(Phoneme::Uw)]), "{cmds:?}");
        assert_eq!(app.formant_lock, chart_vowel(v).map(|(_, s)| s));
        let (f1, f2) = app.lock_formants().unwrap();
        assert_eq!(app.chart_cell(f1, f2), Some(at));
        // A number key is a vowel too: the chart follows.
        app.key(key('5', KeyKind::Press));
        assert_eq!(app.formant_lock, chart_vowel(Phoneme::Aa).map(|(_, s)| s));
        // So does the tongue, moved by its meter.
        sent(&rx);
        let r = app
            .controls
            .iter()
            .find(|(_, c)| *c == Control::Tongue)
            .unwrap()
            .0;
        click_at(&mut app, &rx, (r.x + 2, r.y));
        assert_eq!(
            app.formant_lock
                .map(|s| (s.tongue_index, s.tongue_diameter)),
            Some(app.tongue)
        );
    }

    #[test]
    fn chart_vowels_are_the_physical_engines_vowels() {
        for (name, v, s) in CHART_VOWELS {
            let mut said = Trombone::new(48_000.0);
            let mut held = Trombone::new(48_000.0);
            for voice in [&mut said, &mut held] {
                voice.set_engine(Engine::Physical);
            }
            said.articulate_phoneme(v);
            held.set_shape(&s);
            let mut buf = [0.0; 256];
            said.render(&mut buf);
            held.render(&mut buf);
            assert_eq!(said.tongue(), held.tongue(), "{name}");
            for (k, (a, b)) in said
                .target_diameters()
                .iter()
                .zip(held.target_diameters())
                .enumerate()
            {
                assert!((a - b).abs() < 1e-4, "{name} cell {k}: {a} vs {b}");
            }
            assert!(predict_near(&s).is_some(), "{name}");
        }
    }

    // ----- hearing a dial ---------------------------------------------

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// How many of `cmds` start the voice, and stop it.
    fn note_ons(cmds: &[Cmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, Cmd::NoteOn(_))).count()
    }
    fn note_offs(cmds: &[Cmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, Cmd::NoteOff)).count()
    }

    /// Press a panel meter at a share of its width, drag it to another and
    /// let go `release_at`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn drag_meter(app: &mut App, c: Control, from: f32, to: f32, release_at: (u16, u16)) {
        let r = app.controls.iter().find(|(_, k)| *k == c).unwrap().0;
        let at = |share: f32| (r.x + (share * f32::from(r.width)) as u16, r.y);
        mouse(app, DOWN, at(from), PLAIN);
        mouse(app, DRAG, at(to), PLAIN);
        mouse(app, UP, release_at, PLAIN);
    }

    #[test]
    fn turning_a_dial_in_silence_is_heard_until_it_rests() {
        let (mut app, rx) = laid_out();
        let t0 = Instant::now();
        app.tick_at(t0);
        app.key(arrow(KeyCode::PageUp));
        let cmds = sent(&rx);
        let Some(&Cmd::NoteOn(note)) = cmds.last() else {
            panic!("{cmds:?}");
        };
        assert!((note - app.note_f32()).abs() < 1e-3, "at the current note");
        // A further turn keeps the same voice going, for longer.
        app.tick_at(t0 + ms(300));
        app.key(arrow(KeyCode::PageUp));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (0, 0), "{cmds:?}");
        app.tick_at(t0 + ms(699));
        assert!(sent(&rx).is_empty());
        app.tick_at(t0 + ms(300) + PREVIEW_HOLD);
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
        app.tick_at(t0 + ms(2000));
        assert!(sent(&rx).is_empty());

        // A glottis meter likewise, even let go outside the panel; and the
        // Del key.
        let t1 = t0 + ms(3000);
        app.tick_at(t1);
        drag_meter(&mut app, Control::Loud, 0.3, 0.6, (120, 5));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (1, 0), "{cmds:?}");
        app.tick_at(t1 + PREVIEW_HOLD);
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
        app.key(arrow(KeyCode::Delete));
        assert_eq!(note_ons(&sent(&rx)), 1);
        // Leaving with it sounding stops it.
        assert!(app.key(key('q', KeyKind::Press)));
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
    }

    #[test]
    fn a_dial_is_not_heard_over_a_note_the_drone_or_the_microphone() {
        let t0 = Instant::now();
        // A held note: the dial changes it, and its end is the key's.
        let (mut app, rx) = laid_out();
        app.tick_at(t0);
        app.key(key('a', KeyKind::Press));
        sent(&rx);
        app.key(arrow(KeyCode::PageDown));
        drag_meter(&mut app, Control::Tense, 0.2, 0.8, (5, 5));
        app.tick_at(t0 + ms(1000));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (0, 0), "{cmds:?}");
        app.key(key('a', KeyKind::Release));
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
        // The drone.
        let (mut app, rx) = laid_out();
        app.tick_at(t0);
        app.key(key('i', KeyKind::Press));
        app.key(arrow(KeyCode::PageUp));
        drag_meter(&mut app, Control::Tense, 0.2, 0.8, (5, 5));
        app.key(key(']', KeyKind::Press));
        app.tick_at(t0 + ms(1000));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (0, 0), "{cmds:?}");
        // Mimicking.
        let (mut app, rx) = laid_out();
        app.link.microphone = true;
        app.key(key('`', KeyKind::Press));
        app.key(arrow(KeyCode::PageUp));
        app.tick_at(t0 + ms(1000));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (0, 0), "{cmds:?}");
    }

    #[test]
    fn a_note_played_while_a_dial_is_heard_is_left_alone() {
        let (mut app, rx) = laid_out();
        let t0 = Instant::now();
        app.tick_at(t0);
        app.key(arrow(KeyCode::PageUp));
        assert_eq!(note_ons(&sent(&rx)), 1);
        app.tick_at(t0 + ms(100));
        app.key(key('g', KeyKind::Press));
        let cmds = sent(&rx);
        assert!(matches!(cmds[..], [Cmd::NoteOn(_)]), "{cmds:?}");
        // The dial turned while the note is held starts nothing new.
        app.key(arrow(KeyCode::PageUp));
        app.tick_at(t0 + ms(1000));
        let cmds = sent(&rx);
        assert_eq!((note_ons(&cmds), note_offs(&cmds)), (0, 0), "{cmds:?}");
        app.key(key('g', KeyKind::Release));
        assert!(matches!(sent(&rx)[..], [Cmd::NoteOff]));
    }

    /// Something that takes over the voice.
    type TakeOver = fn(&mut App);

    #[test]
    fn nothing_a_dial_starts_is_left_sounding() {
        let t0 = Instant::now();
        // Whatever takes over the voice while a dial is heard, the voice
        // the dial started is stopped (and only once).
        let takeovers: [(&str, TakeOver); 4] = [
            ("drone", |app| {
                app.key(key('i', KeyKind::Press));
                app.key(key('i', KeyKind::Press));
            }),
            ("speech", |app| {
                app.speak("hi");
            }),
            ("mimic", |app| {
                app.link.microphone = true;
                app.key(key('`', KeyKind::Press));
            }),
            ("engine", |app| {
                app.key(key('\\', KeyKind::Press));
            }),
        ];
        for (name, take_over) in takeovers {
            let (mut app, rx) = laid_out();
            app.tick_at(t0);
            app.key(arrow(KeyCode::PageUp));
            assert_eq!(note_ons(&sent(&rx)), 1, "{name}");
            take_over(&mut app);
            app.tick_at(t0 + ms(1000));
            let cmds = sent(&rx);
            assert_eq!(
                (note_ons(&cmds), note_offs(&cmds)),
                (0, 1),
                "{name}: {cmds:?}"
            );
        }
    }
}
