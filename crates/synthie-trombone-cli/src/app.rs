//! UI state and input handling.

use crate::audio::{Cmd, Link, Snapshot};
use crossbeam_channel::Sender;
use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use std::time::{Duration, Instant};
use synthie_trombone::speech::{Prosody, Utterance};
use synthie_trombone::{
    Constriction, Engine, N, Phoneme, Preset, TONGUE_DIAMETER, TONGUE_INDEX, VoiceSettings,
};
use synthie_ui::{Key, KeyCode, KeyKind};

/// Constriction slot for mouse dragging (0 and 1 belong to the speaker).
const SLOT_MOUSE: usize = 2;
/// Constriction slot for the lip-pop key.
const SLOT_POP: usize = 3;
const NO_RELEASE_HINT: &str =
    "terminal sends no key releases: notes hold, Space stops · Tab to type text";
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
/// Range of the `loud` control, cmH2O of lung pressure.
pub const LOUD_RANGE: (f32, f32) = (2.0, 20.0);

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

/// Something in the glottis panel or header that the mouse can work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Tense,
    Tongue,
    Velum,
    Vibrato,
    Reverb,
    Loud,
    OctaveDown,
    OctaveUp,
    Mode,
    Engine,
    Wobble,
    Drone,
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
        )
    }
}

/// What a mouse press has taken hold of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grab {
    /// A finger on the tract wall.
    Constriction,
    /// The tongue handle (or Shift-drag).
    Tongue,
    /// Right-drag: the tongue, as the original maps it.
    TongueOld,
    Velum,
    Control(Control),
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
    pub settings: VoiceSettings,
    pub engine: Engine,

    pub text: String,
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
    pub status: String,
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
            settings: Preset::Deep.settings().with_engine(engine),
            engine,
            text: String::new(),
            history: Vec::new(),
            history_pos: 0,
            spoken: Vec::new(),
            last_vowel: None,
            tract_area: Rect::default(),
            controls: Vec::new(),
            loud: 8.0,
            grab: None,
            pop_until: None,
            status: String::new(),
        };
        app.status = if keyboard_release {
            "hold a piano key to sing · Space stops · Tab to type text".into()
        } else {
            NO_RELEASE_HINT.into()
        };
        app.send(Cmd::Model(engine));
        app.send(Cmd::Note(app.note_f32()));
        app
    }

    pub fn sample_rate(&self) -> f32 {
        self.link.sample_rate
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
        while let Ok(s) = self.link.snapshots.try_recv() {
            self.snap = s;
        }
        while self.link.trash.try_recv().is_ok() {}
        if self.pop_until.is_some_and(|t| Instant::now() >= t) {
            self.pop_until = None;
            self.send(Cmd::Constriction(SLOT_POP, None));
        }
    }

    /// Handle a key; returns `true` to quit.
    pub fn key(&mut self, key: Key) -> bool {
        if key.ctrl && key.code == KeyCode::Char('c') {
            return true;
        }
        match self.mode {
            Mode::Play => self.play_key(key),
            Mode::Type => {
                if key.kind != KeyKind::Release {
                    self.type_key(key);
                }
                false
            }
        }
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
            'i' => {
                self.drone = !self.drone;
                self.send(Cmd::Drone(self.drone));
            }
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
            '[' => self.set_tenseness(self.tenseness - 0.05),
            ']' => self.set_tenseness(self.tenseness + 0.05),
            '-' => self.set_vibrato(self.vibrato - 0.005),
            '=' => self.set_vibrato(self.vibrato + 0.005),
            ',' => self.rate = (self.rate - 0.1).max(0.4),
            '.' => self.rate = (self.rate + 0.1).min(2.5),
            _ => {}
        }
        false
    }

    fn play_special(&mut self, code: KeyCode) {
        let (mut i, mut d) = self.tongue;
        match code {
            KeyCode::Tab => {
                self.mode = Mode::Type;
                self.status = "type, Enter to speak · ↑↓ history · Esc/Tab back".into();
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
            KeyCode::Left => i -= 0.5,
            KeyCode::Right => i += 0.5,
            KeyCode::Up => d -= 0.08,
            KeyCode::Down => d += 0.08,
            _ => return,
        }
        self.set_tongue(i, d);
    }

    fn type_key(&mut self, key: Key) {
        match key.code {
            KeyCode::Esc | KeyCode::Tab => {
                self.mode = Mode::Play;
                self.status.clear();
            }
            KeyCode::Enter => {
                let t = self.text.trim().to_string();
                if !t.is_empty() {
                    self.speak(&t);
                    if self.history.last() != Some(&t) {
                        self.history.push(t);
                    }
                    self.history_pos = self.history.len();
                    self.text.clear();
                }
            }
            KeyCode::Backspace => {
                self.text.pop();
            }
            KeyCode::Up if self.history_pos > 0 => {
                self.history_pos -= 1;
                self.text.clone_from(&self.history[self.history_pos]);
            }
            KeyCode::Down => {
                self.history_pos = (self.history_pos + 1).min(self.history.len());
                self.text = self
                    .history
                    .get(self.history_pos)
                    .cloned()
                    .unwrap_or_default();
            }
            KeyCode::Char(c) if self.text.len() < 400 => self.text.push(c),
            _ => {}
        }
    }

    /// Stop singing and speaking; the voice fades out. Also ends a drone.
    fn silence(&mut self) {
        self.held = None;
        self.voiced = false;
        if self.drone {
            self.drone = false;
            self.send(Cmd::Drone(false));
        }
        self.send(Cmd::NoteOff);
        self.send(Cmd::StopSpeaking);
    }

    /// Switch between the classic and the physical sound model.
    fn toggle_engine(&mut self) {
        self.engine = match self.engine {
            Engine::Classic => Engine::Physical,
            Engine::Physical => Engine::Classic,
        };
        self.settings.engine = self.engine;
        self.send(Cmd::Model(self.engine));
        if let Some(v) = self.last_vowel {
            self.send(Cmd::Vowel(v));
        }
        self.status = format!("engine: {}", engine_name(self.engine));
    }

    /// Switch to the next (`+1`) or previous (`-1`) voice preset.
    fn cycle_preset(&mut self, dir: isize) {
        let all = Preset::ALL;
        let i = all.iter().position(|&p| p == self.preset).unwrap_or(0);
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let next = (i as isize + dir).rem_euclid(all.len() as isize) as usize;
        self.preset = all[next];
        self.settings = self.preset.settings().with_engine(self.engine);
        self.tenseness = self.settings.tenseness;
        self.vibrato = self.settings.vibrato;
        // Play at the note nearest the voice's speaking pitch.
        #[allow(clippy::cast_possible_truncation)]
        let note = (69.0 + 12.0 * (self.settings.pitch / 440.0).log2()).round() as i32;
        self.note = note;
        self.octave = (note / 12 - 1).clamp(1, 6);
        self.send(Cmd::Voice(self.settings));
        self.send(Cmd::Note(self.note_f32()));
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
        self.status = format!("“{text}” · {:.1} s", u.duration());
        self.send(Cmd::Speak(u));
    }

    fn vowel(&mut self, v: Phoneme) {
        // The sound model picks the shape; remember where it puts the
        // tongue so the handle and the arrow keys start from there.
        if let Some((i, d)) = v.tongue(self.engine) {
            self.tongue = (
                i.clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
                d.clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
            );
        }
        self.send(Cmd::Vowel(v));
        self.last_vowel = Some(v);
    }

    fn set_tongue(&mut self, i: f32, d: f32) {
        self.tongue = (
            i.clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end()),
            d.clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end()),
        );
        self.send(Cmd::Tongue(self.tongue.0, self.tongue.1));
    }

    fn set_tenseness(&mut self, t: f32) {
        self.tenseness = t.clamp(0.0, 1.0);
        self.send(Cmd::Tenseness(self.tenseness));
    }

    fn set_vibrato(&mut self, v: f32) {
        self.vibrato = v.clamp(0.0, 0.05);
        self.send(Cmd::Vibrato(self.vibrato));
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

    /// Where a vowel landmark sits on the tract view for the engine in use.
    #[must_use]
    pub fn landmark(&self, vowel: Phoneme) -> Option<(f32, f32)> {
        vowel.tongue(self.engine).map(tongue_handle)
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
        // The handle is drawn from what the voice reports.
        let (hx, hy) = tongue_handle(self.snap.tongue);
        if shift || ((x - hx).abs() <= 1.6 && (y - hy).abs() <= 0.45) {
            return Some(Grab::Tongue);
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
        // Outside the canvas the drag holds its last position.
        let Some((x, y)) = self.to_tract(col, row) else {
            return;
        };
        let index = x.clamp(2.0, N as f32 - 0.5);
        let span = TONGUE_DIAMETER.end() - TONGUE_DIAMETER.start();
        match grab {
            Grab::Constriction => {
                // Distance from the midline is the opening left; the finger
                // cannot leave the tract.
                let diameter = (y.abs().min(LUMEN) * 2.0 - 0.1).max(0.0);
                self.send(Cmd::Constriction(
                    SLOT_MOUSE,
                    Some(Constriction::new(index, diameter)),
                ));
            }
            Grab::Tongue => {
                // The handle hangs below the midline: put it under the mouse.
                let depth = ((y.abs() - 0.6) / 1.5).clamp(0.0, 1.0);
                self.set_tongue(index, TONGUE_DIAMETER.start() + depth * span);
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
            Grab::Control(_) => {}
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
            _ => {}
        }
    }

    /// Click a button.
    fn press(&mut self, c: Control) {
        match c {
            Control::OctaveDown => self.octave = (self.octave - 1).max(1),
            Control::OctaveUp => self.octave = (self.octave + 1).min(6),
            Control::Mode => match self.mode {
                Mode::Play => {
                    self.mode = Mode::Type;
                    self.status = "type, Enter to speak · ↑↓ history · Esc/Tab back".into();
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
            Control::Drone => {
                self.drone = !self.drone;
                self.send(Cmd::Drone(self.drone));
            }
            _ => {}
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
    // ----- mouse -------------------------------------------------------

    use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    /// An app with a tract canvas and a glottis panel laid out.
    fn laid_out() -> (App, Receiver<Cmd>) {
        let (mut app, rx) = app(true);
        app.tract_area = Rect::new(1, 2, 138, 26);
        app.controls = App::glottis_controls(Rect::new(1, 30, 44, 9));
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
        let handle = cell(&app, f64::from(hx), f64::from(hy));
        mouse(&mut app, DOWN, handle, PLAIN);
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
        // Where it was dropped is where the handle is now drawn.
        let (_, y) = tongue_handle((*i, *d));
        assert!((y + 1.5).abs() < 0.2, "{y}");
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
}
