//! Takes: a performance in the terminal app as the audio thread carried it
//! out, and the voice that carries it out again.
//!
//! The app records a take as two files with the same stem: a JSON Lines
//! log (`take-007.jsonl`) and the sound (`take-007.wav`). The log's first
//! line, the header, holds the state of every control at the first sample;
//! each further line is one command, stamped with the sample it was applied
//! before; the last line marks the end. The voice is deterministic, so
//! running the same commands at the same samples from the same state gives
//! the same sound, sample for sample.
//!
//! [`Performer`] is the part of the audio thread that turns commands into
//! sound; the app's audio callback and the library's `replay` example both
//! use it (the example includes this file), so a replay goes through the
//! very code the take was played with.

#![allow(
    dead_code, // the app and the replay example each use a part
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use serde_json::{Value, json};
use synthie_trombone::Gesture as ScoreGesture;
use synthie_trombone::gesture::{GestureId, Target, Window};
use synthie_trombone::inverse::Shape;
use synthie_trombone::mimic::Gesture;
use synthie_trombone::speech::{Prosody, Speaker, Utterance};
use synthie_trombone::{
    Constriction, Engine as SoundModel, MAX_CONSTRICTIONS, Phoneme, Trombone, VoiceSettings,
};

/// Names the layout of a take log; bumped when it changes.
pub const FORMAT: &str = "synthie-trombone take 1";
/// What the take's WAV file holds.
pub const SIGNAL: &str = "dry: the voice as the speaker renders it, before the reverb, \
                          the output gain of 0.9 and clipping; 32-bit float, mono";
/// Pitch and tenseness the app's voice starts with, before any command.
pub const START_PITCH: f32 = 110.0;
pub const START_TENSENESS: f32 = 0.6;
/// Reverb mix the app starts with.
pub const START_REVERB: f32 = 0.18;
/// Vibrato rate, Hz, of the app's vibrato control.
pub const VIBRATO_RATE: f32 = 5.5;
/// Articulator speed while mimicking: the person being copied has already
/// taken the time to move their own.
pub const MIMIC_SPEED: f32 = 8.0;

/// One command as the audio thread applied it. Speech is referred to by
/// number (the log line carries its text); a gesture is one the microphone
/// sent and mimic mode carried out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Logged {
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
    Lung(f32),
    Model(SoundModel),
    Vowel(Phoneme),
    Speak(u32),
    StopSpeaking,
    Mimic(bool),
    Shape(Shape),
    Gesture(Gesture),
    /// A gesture for the voice's gesture score, kept in a slot of the
    /// performer's so that it can be ended.
    Push(u8, ScoreGesture),
    /// End the gesture in a slot, over this many seconds.
    EndGesture(u8, f32),
    /// Push the lips forward by this many cm, or let them follow the
    /// rounding (`None`; `Trombone::set_lip_protrusion`).
    Protrusion(Option<f32>),
    /// Hold the larynx this many cm above its rest, negative lowered
    /// (`Trombone::set_larynx_height`).
    Larynx(f32),
}

/// Slots for gestures played by hand.
pub const GESTURE_SLOTS: usize = 4;

/// What last set the tract's vowel shape (the narrowing along the tongue,
/// which the voice does not report back).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shaping {
    Vowel(Phoneme),
    Shape(Shape),
}

/// Every control of the voice at one moment: enough to build it again.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // independent toggles
pub struct Controls {
    pub model: SoundModel,
    /// The voice settings last applied (`None`: never, the voice is the
    /// library's default one).
    pub voice: Option<VoiceSettings>,
    pub shaping: Option<Shaping>,
    /// Tongue body target `(index, diameter)`.
    pub tongue: (f32, f32),
    pub constrictions: [Option<Constriction>; MAX_CONSTRICTIONS],
    /// Velum target, `0‥1`.
    pub velum: f32,
    /// Tenseness of the instrument (speech uses its own while it talks).
    pub tenseness: f32,
    /// Lung pressure, cmH2O.
    pub lung: f32,
    /// Target pitch, Hz.
    pub frequency: f32,
    /// The voicing gate (a note held).
    pub voiced: bool,
    /// Vibrato depth and rate, Hz (`None`: the library's default).
    pub vibrato: Option<(f32, f32)>,
    pub wobble: bool,
    pub drone: bool,
    /// Reverb mix of the output (the take's sound is before it).
    pub reverb: f32,
    pub mimic: bool,
    /// Lip protrusion set by hand, cm (`None`: the lips follow the
    /// rounding).
    pub protrusion: Option<f32>,
    /// Larynx height set by hand, cm above rest.
    pub larynx: f32,
}

impl Controls {
    /// The app's controls before any command.
    #[must_use]
    pub fn initial() -> Self {
        Self {
            model: SoundModel::Classic,
            voice: None,
            shaping: None,
            tongue: Trombone::new(48_000.0).tongue(),
            constrictions: [None; MAX_CONSTRICTIONS],
            velum: 0.0,
            tenseness: START_TENSENESS,
            lung: Trombone::new(48_000.0).lung_pressure(),
            frequency: START_PITCH,
            voiced: false,
            vibrato: None,
            wobble: false,
            drone: false,
            reverb: START_REVERB,
            mimic: false,
            protrusion: None,
            larynx: 0.0,
        }
    }
}

/// A voice built from silence with `c`'s controls, its tract settled into
/// their shape. The app's audio thread starts every take on such a voice,
/// and a replay starts on the same one.
#[must_use]
pub fn build_voice(sample_rate: f32, c: &Controls) -> Trombone {
    let mut v = Trombone::new(sample_rate);
    v.set_frequency(START_PITCH);
    v.set_tenseness(START_TENSENESS);
    if let Some(settings) = &c.voice {
        settings.apply(&mut v);
    }
    v.set_engine(c.model);
    match &c.shaping {
        Some(Shaping::Vowel(p)) => v.articulate_phoneme(*p),
        Some(Shaping::Shape(s)) => v.set_shape(s),
        None => {}
    }
    v.set_tongue(c.tongue.0, c.tongue.1);
    for (slot, k) in c.constrictions.iter().enumerate() {
        v.set_constriction(slot, *k);
    }
    v.set_velum(c.velum);
    v.set_tenseness(c.tenseness);
    v.set_lung_pressure(c.lung);
    if let Some((amount, hz)) = c.vibrato {
        v.set_vibrato(amount, hz);
    }
    v.set_wobble(c.wobble);
    v.set_always_voice(c.drone);
    v.set_lip_protrusion(c.protrusion);
    v.set_larynx_height(c.larynx);
    if c.mimic {
        v.set_articulation_speed(MIMIC_SPEED);
        v.set_glide(f32::INFINITY);
        v.set_always_voice(false);
    }
    v.set_frequency(c.frequency);
    v.set_voiced(c.voiced);
    v.settle();
    v
}

/// The voice and the speaker, played by commands: what the app's audio
/// thread does with each command and sample, without the reverb and the
/// display. Never allocates; utterances it lets go of are handed back so
/// the caller can free them elsewhere.
pub struct Performer {
    voice: Trombone,
    speaker: Speaker,
    /// The controls the voice does not report back (the rest are read
    /// from it, see [`controls`](Self::controls)).
    controls: Controls,
    was_speaking: bool,
    /// Gestures played by hand, by slot.
    gestures: [Option<GestureId>; GESTURE_SLOTS],
}

impl Performer {
    /// The app's voice at start-up.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let mut voice = Trombone::new(sample_rate);
        voice.set_frequency(START_PITCH);
        voice.set_tenseness(START_TENSENESS);
        Self {
            voice,
            speaker: Speaker::new(),
            controls: Controls::initial(),
            was_speaking: false,
            gestures: [None; GESTURE_SLOTS],
        }
    }

    /// A voice built from silence with these controls (see [`build_voice`]).
    #[must_use]
    pub fn from_controls(sample_rate: f32, c: &Controls) -> Self {
        Self {
            voice: build_voice(sample_rate, c),
            speaker: Speaker::new(),
            controls: *c,
            was_speaking: false,
            gestures: [None; GESTURE_SLOTS],
        }
    }

    #[must_use]
    pub fn voice(&self) -> &Trombone {
        &self.voice
    }

    #[must_use]
    pub fn speaker(&self) -> &Speaker {
        &self.speaker
    }

    /// Every control as it stands.
    #[must_use]
    pub fn controls(&self) -> Controls {
        let v = &self.voice;
        let mut c = self.controls;
        c.model = v.engine();
        c.tongue = v.tongue();
        for (slot, k) in c.constrictions.iter_mut().enumerate() {
            *k = v.constriction(slot);
        }
        c.lung = v.lung_pressure();
        c.frequency = v.target_frequency();
        c.voiced = v.is_voiced();
        c
    }

    /// Reverb mix the output should use.
    #[must_use]
    pub fn reverb(&self) -> f32 {
        self.controls.reverb
    }

    /// Whether mimic mode is copying the microphone.
    #[must_use]
    pub fn mimicking(&self) -> bool {
        self.controls.mimic
    }

    /// Start again from silence with the controls as they stand: speech
    /// stops and the voice is rebuilt as [`build_voice`] builds it, so that
    /// what follows can be replayed exactly. Returns the utterance let go.
    pub fn restart(&mut self) -> Utterance {
        let old = self.speaker.stop(&mut self.voice);
        self.was_speaking = false;
        self.controls.shaping = Some(Shaping::Vowel(Phoneme::Sil));
        let c = self.controls();
        *self = Self::from_controls(self.voice.sample_rate(), &c);
        old
    }

    /// Carry out a command (all but [`Logged::Speak`], see
    /// [`speak`](Self::speak)). Returns an utterance let go, if any.
    #[allow(clippy::too_many_lines)] // one arm per command
    pub fn apply(&mut self, cmd: &Logged) -> Option<Utterance> {
        let v = &mut self.voice;
        let speaking = self.speaker.is_speaking();
        let c = &mut self.controls;
        match *cmd {
            Logged::NoteOn(note) => {
                let old = self.stop_speaking();
                self.voice.set_note(note);
                self.voice.note_on();
                return old;
            }
            Logged::NoteOff => v.note_off(),
            Logged::Note(note) => v.set_note(note),
            Logged::Tongue(i, d) => v.set_tongue(i, d),
            Logged::Constriction(slot, k) => v.set_constriction(slot, k),
            Logged::Velum(x) => {
                c.velum = x;
                v.set_velum(x);
            }
            Logged::Tenseness(t) => {
                c.tenseness = t;
                if !speaking {
                    v.set_tenseness(t);
                }
            }
            Logged::Vibrato(a) => {
                c.vibrato = Some((a, VIBRATO_RATE));
                v.set_vibrato(a, VIBRATO_RATE);
            }
            Logged::Wobble(w) => {
                c.wobble = w;
                v.set_wobble(w);
            }
            Logged::Drone(on) => {
                c.drone = on;
                if !speaking {
                    v.set_always_voice(on);
                }
            }
            Logged::Reverb(mix) => c.reverb = mix,
            Logged::Voice(settings) => {
                settings.apply(v);
                c.voice = Some(settings);
                c.tenseness = settings.tenseness;
                c.vibrato = Some((settings.vibrato, settings.vibrato_rate));
            }
            Logged::Lung(p) => v.set_lung_pressure(p),
            Logged::Model(model) => v.set_engine(model),
            Logged::Vowel(p) => {
                v.articulate_phoneme(p);
                v.set_velum(c.velum);
                c.shaping = Some(Shaping::Vowel(p));
            }
            Logged::Speak(_) => {}
            Logged::StopSpeaking => return self.stop_speaking(),
            Logged::Shape(shape) => {
                v.set_shape(&shape);
                c.shaping = Some(Shaping::Shape(shape));
            }
            Logged::Mimic(on) => {
                c.mimic = on;
                let old = self.stop_speaking();
                let v = &mut self.voice;
                if on {
                    v.set_articulation_speed(MIMIC_SPEED);
                    v.set_glide(f32::INFINITY);
                    v.set_always_voice(false);
                } else {
                    Gesture::REST.apply(v);
                    self.restore_instrument();
                }
                return old;
            }
            Logged::Gesture(g) => {
                self.follow(&g);
            }
            Logged::Push(slot, g) => {
                if let Some(at) = self.gestures.get_mut(usize::from(slot)) {
                    if let Some(old) = at.take() {
                        self.voice.end_gesture(old, 0.0);
                    }
                    *at = self.voice.push_gesture_tracked(g);
                }
            }
            Logged::Protrusion(cm) => {
                c.protrusion = cm;
                v.set_lip_protrusion(cm);
            }
            Logged::Larynx(cm) => {
                c.larynx = cm;
                v.set_larynx_height(cm);
            }
            Logged::EndGesture(slot, ramp) => {
                if let Some(id) = self
                    .gestures
                    .get_mut(usize::from(slot))
                    .and_then(Option::take)
                {
                    self.voice.end_gesture(id, ramp);
                }
            }
        }
        None
    }

    /// Start speaking `u`; returns the utterance it interrupts.
    pub fn speak(&mut self, u: Utterance) -> Utterance {
        Speaker::prepare(&mut self.voice);
        self.was_speaking = true;
        self.speaker.speak(u)
    }

    /// Carry out a gesture from the microphone, if mimic mode is copying
    /// it and nothing is being said. Whether it was carried out.
    pub fn follow(&mut self, g: &Gesture) -> bool {
        if !self.controls.mimic || self.speaker.is_speaking() {
            return false;
        }
        g.apply(&mut self.voice);
        if let Some(shape) = g.shape {
            self.controls.shaping = Some(Shaping::Shape(shape));
        }
        true
    }

    fn stop_speaking(&mut self) -> Option<Utterance> {
        if !self.speaker.is_speaking() {
            return None;
        }
        self.controls.shaping = Some(Shaping::Vowel(Phoneme::Sil));
        Some(self.speaker.stop(&mut self.voice))
    }

    /// Back to the instrument's settings after speech or mimicking.
    fn restore_instrument(&mut self) {
        let v = &mut self.voice;
        let c = &self.controls;
        v.reset_voicing_rates();
        v.set_articulation_speed(1.0);
        v.set_glide(0.1);
        v.set_wobble(c.wobble);
        v.set_tenseness(c.tenseness);
        v.set_velum(c.velum);
        v.set_always_voice(c.drone);
    }

    /// The next sample of the dry voice.
    pub fn sample(&mut self) -> f32 {
        let dry = self.speaker.process(&mut self.voice);
        if self.was_speaking && !self.speaker.is_speaking() {
            self.was_speaking = false;
            self.controls.shaping = Some(Shaping::Vowel(Phoneme::Sil));
            self.restore_instrument();
        }
        dry
    }
}

// ----- JSON -----------------------------------------------------------------

/// The text and intonation a [`Logged::Speak`] compiled its speech from.
#[derive(Clone, Debug, PartialEq)]
pub struct Speech {
    pub text: String,
    pub prosody: Prosody,
}

impl Speech {
    /// The speech as the app compiled it.
    #[must_use]
    pub fn utterance(&self) -> Utterance {
        Utterance::from_text(&self.text, &self.prosody)
    }
}

/// Write every number that holds an `f32` the short way it prints (`0.1`,
/// not `0.10000000149011612`) where that reads back as the same `f32`, so
/// the log is legible and still exact.
fn tidy(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            let Some(x) = n.as_f64() else { return };
            let narrow = x as f32;
            if f64::from(narrow).to_bits() != x.to_bits() {
                return;
            }
            let short = format!("{narrow}").parse::<f64>().ok();
            let back = short.and_then(|s| {
                let text = serde_json::to_string(&s).ok()?;
                serde_json::from_str::<f64>(&text).ok()
            });
            if let (Some(s), Some(b)) = (short, back)
                && (b as f32).to_bits() == narrow.to_bits()
                && let Some(n2) = serde_json::Number::from_f64(s)
            {
                *n = n2;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(tidy),
        Value::Object(o) => o.values_mut().for_each(tidy),
        _ => {}
    }
}

/// A JSON value with its `f32` numbers tidied (see [`tidy`]).
#[must_use]
pub fn tidied(mut v: Value) -> Value {
    tidy(&mut v);
    v
}

fn get<'a>(o: &'a Value, key: &str) -> Result<&'a Value, String> {
    o.get(key).ok_or_else(|| format!("missing {key:?}"))
}

fn num(o: &Value, key: &str) -> Result<f32, String> {
    get(o, key)?
        .as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| format!("{key:?} is not a number"))
}

fn flag(o: &Value, key: &str) -> Result<bool, String> {
    get(o, key)?
        .as_bool()
        .ok_or_else(|| format!("{key:?} is not true or false"))
}

fn pair(v: &Value, what: &str) -> Result<(f32, f32), String> {
    match v.as_array().map(Vec::as_slice) {
        Some([a, b]) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => Ok((a as f32, b as f32)),
            _ => Err(format!("{what} is not two numbers")),
        },
        _ => Err(format!("{what} is not two numbers")),
    }
}

fn decode<T: serde::de::DeserializeOwned>(v: &Value, what: &str) -> Result<T, String> {
    serde_json::from_value(v.clone()).map_err(|e| format!("{what}: {e}"))
}

fn to<T: serde::Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap_or(Value::Null)
}

fn constriction_json(c: Option<Constriction>) -> Value {
    c.map_or(
        Value::Null,
        |c| json!({"index": c.index, "diameter": c.diameter}),
    )
}

fn constriction_from(v: &Value) -> Result<Option<Constriction>, String> {
    if v.is_null() {
        return Ok(None);
    }
    Ok(Some(Constriction::new(
        num(v, "index")?,
        num(v, "diameter")?,
    )))
}

pub fn shape_json(s: &Shape) -> Value {
    let mut v = json!({
        "tongue_index": s.tongue_index,
        "tongue_diameter": s.tongue_diameter,
        "lips": s.lips,
        "narrowing": s.narrowing.map(|(at, d)| [at, d]),
        "pharynx": s.pharynx,
    });
    // The modal body's three numbers, only where a shape has them.
    if let (Some(q), Some(map)) = (s.modes, v.as_object_mut()) {
        map.insert("modes".into(), json!(q));
    }
    v
}

fn shape_from(v: &Value) -> Result<Shape, String> {
    let lips = get(v, "lips")?;
    let narrowing = get(v, "narrowing")?;
    Ok(Shape {
        tongue_index: num(v, "tongue_index")?,
        tongue_diameter: num(v, "tongue_diameter")?,
        lips: if lips.is_null() {
            None
        } else {
            Some(num(v, "lips")?)
        },
        narrowing: if narrowing.is_null() {
            None
        } else {
            Some(pair(narrowing, "narrowing")?)
        },
        // Takes from before the tongue root had a setting have it at rest.
        pharynx: if v.get("pharynx").is_some() {
            num(v, "pharynx")?
        } else {
            0.0
        },
        // Shapes of the cosine body have none.
        modes: match v.get("modes").and_then(Value::as_array).map(Vec::as_slice) {
            Some([a, b, c]) => match (a.as_f64(), b.as_f64(), c.as_f64()) {
                #[allow(clippy::cast_possible_truncation)]
                (Some(a), Some(b), Some(c)) => Some([a as f32, b as f32, c as f32]),
                _ => return Err("modes is not three numbers".into()),
            },
            Some(_) => return Err("modes is not three numbers".into()),
            None => None,
        },
    })
}

/// A gesture of the score as JSON: its tier, the variables its target sets
/// by name, its dynamics and its window (an end that never comes is
/// `null`).
pub fn score_gesture_json(g: &ScoreGesture) -> Value {
    let mut target = serde_json::Map::new();
    for (k, name) in g.tier.variables().iter().enumerate() {
        if let Some(x) = g.target.get(k) {
            target.insert((*name).to_string(), x.into());
        }
    }
    let w = g.window;
    json!({
        "tier": g.tier.name(),
        "target": target,
        "stiffness": g.stiffness,
        "damping": g.damping,
        "dominance": g.dominance,
        "window": [w.start, if w.end.is_finite() { Value::from(w.end) } else { Value::Null },
                   w.ramp_in, w.ramp_out],
        "flags": g.flags,
        "layer": g.layer,
    })
}

fn score_gesture_from(v: &Value) -> Result<ScoreGesture, String> {
    let name = get(v, "tier")?.as_str().ok_or("\"tier\" is not text")?;
    let tier = synthie_trombone::Tier::ALL
        .into_iter()
        .find(|t| t.name() == name)
        .ok_or_else(|| format!("unknown tier {name:?}"))?;
    let mut target = Target::NONE;
    let set = get(v, "target")?
        .as_object()
        .ok_or("\"target\" is not an object")?;
    for (name, x) in set {
        let k = tier
            .variables()
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("{tier:?} has no variable {name:?}"))?;
        target.set(k, x.as_f64().ok_or("a target is not a number")? as f32);
    }
    let window = get(v, "window")?
        .as_array()
        .ok_or("\"window\" is not a list")?;
    let at = |k: usize| window.get(k).and_then(Value::as_f64).map(|x| x as f32);
    let int = |key: &str| -> Result<u8, String> {
        get(v, key)?
            .as_u64()
            .and_then(|x| u8::try_from(x).ok())
            .ok_or_else(|| format!("{key:?} is not a small number"))
    };
    Ok(ScoreGesture::new(tier, target)
        .stiffness(num(v, "stiffness")?)
        .damping(num(v, "damping")?)
        .dominance(num(v, "dominance")?)
        .window(Window {
            start: at(0).unwrap_or(0.0),
            end: at(1).unwrap_or(f32::INFINITY),
            ramp_in: at(2).unwrap_or(0.0),
            ramp_out: at(3).unwrap_or(0.0),
        })
        .flags(int("flags")?)
        .layer(int("layer")?))
}

fn phoneme_from(v: &Value) -> Result<Phoneme, String> {
    v.as_str()
        .and_then(Phoneme::from_symbol)
        .ok_or_else(|| format!("{v} is not a phoneme"))
}

fn shaping_json(s: Option<&Shaping>) -> Value {
    match s {
        None => Value::Null,
        Some(Shaping::Vowel(p)) => json!({"vowel": p.symbol()}),
        Some(Shaping::Shape(s)) => json!({"shape": shape_json(s)}),
    }
}

fn shaping_from(v: &Value) -> Result<Option<Shaping>, String> {
    if v.is_null() {
        return Ok(None);
    }
    if let Some(p) = v.get("vowel") {
        return Ok(Some(Shaping::Vowel(phoneme_from(p)?)));
    }
    Ok(Some(Shaping::Shape(shape_from(get(v, "shape")?)?)))
}

impl Controls {
    #[must_use]
    pub fn json(&self) -> Value {
        tidied(json!({
            "model": to(&self.model),
            "voice": self.voice.as_ref().map_or(Value::Null, to),
            "shaping": shaping_json(self.shaping.as_ref()),
            "tongue": [self.tongue.0, self.tongue.1],
            "constrictions": self.constrictions.map(constriction_json).to_vec(),
            "velum": self.velum,
            "tenseness": self.tenseness,
            "lung": self.lung,
            "frequency": self.frequency,
            "voiced": self.voiced,
            "vibrato": self.vibrato.map(|(a, hz)| [a, hz]),
            "wobble": self.wobble,
            "drone": self.drone,
            "reverb": self.reverb,
            "mimic": self.mimic,
            "protrusion": self.protrusion,
            "larynx": self.larynx,
        }))
    }

    /// # Errors
    ///
    /// When a control is missing or malformed.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let mut constrictions = [None; MAX_CONSTRICTIONS];
        let list = get(v, "constrictions")?
            .as_array()
            .ok_or("\"constrictions\" is not a list")?;
        for (slot, c) in list.iter().enumerate().take(MAX_CONSTRICTIONS) {
            constrictions[slot] = constriction_from(c)?;
        }
        let voice = get(v, "voice")?;
        let vibrato = get(v, "vibrato")?;
        Ok(Self {
            model: decode(get(v, "model")?, "model")?,
            voice: if voice.is_null() {
                None
            } else {
                Some(decode(voice, "voice")?)
            },
            shaping: shaping_from(get(v, "shaping")?)?,
            tongue: pair(get(v, "tongue")?, "tongue")?,
            constrictions,
            velum: num(v, "velum")?,
            tenseness: num(v, "tenseness")?,
            lung: num(v, "lung")?,
            frequency: num(v, "frequency")?,
            voiced: flag(v, "voiced")?,
            vibrato: if vibrato.is_null() {
                None
            } else {
                Some(pair(vibrato, "vibrato")?)
            },
            wobble: flag(v, "wobble")?,
            drone: flag(v, "drone")?,
            reverb: num(v, "reverb")?,
            mimic: flag(v, "mimic")?,
            // Takes recorded before the lip horn and the larynx had them.
            protrusion: v
                .get("protrusion")
                .and_then(Value::as_f64)
                .map(|x| x as f32),
            larynx: v
                .get("larynx")
                .and_then(Value::as_f64)
                .map_or(0.0, |x| x as f32),
        })
    }
}

impl Logged {
    /// The command's name and arguments as a JSON object; speech gets its
    /// text and intonation from `speech`.
    #[must_use]
    pub fn json(&self, speech: Option<&Speech>) -> Value {
        let (name, args) = match *self {
            Logged::NoteOn(note) => ("note_on", json!({"note": note})),
            Logged::NoteOff => ("note_off", json!({})),
            Logged::Note(note) => ("note", json!({"note": note})),
            Logged::Tongue(i, d) => ("tongue", json!({"index": i, "diameter": d})),
            Logged::Constriction(slot, None) => ("release", json!({"slot": slot})),
            Logged::Constriction(slot, Some(c)) => (
                "constriction",
                json!({"slot": slot, "index": c.index, "diameter": c.diameter}),
            ),
            Logged::Velum(x) => ("velum", json!({"open": x})),
            Logged::Tenseness(t) => ("tenseness", json!({"value": t})),
            Logged::Vibrato(a) => ("vibrato", json!({"amount": a})),
            Logged::Wobble(on) => ("wobble", json!({"on": on})),
            Logged::Drone(on) => ("drone", json!({"on": on})),
            Logged::Reverb(mix) => ("reverb", json!({"mix": mix})),
            Logged::Voice(s) => ("voice", json!({"settings": to(&s)})),
            Logged::Lung(p) => ("lung", json!({"cm_h2o": p})),
            Logged::Model(m) => ("model", json!({"engine": to(&m)})),
            Logged::Vowel(p) => ("vowel", json!({"phoneme": p.symbol()})),
            Logged::Speak(id) => (
                "speak",
                json!({
                    "id": id,
                    "text": speech.map(|s| s.text.clone()),
                    "prosody": speech.map_or(Value::Null, |s| to(&s.prosody)),
                }),
            ),
            Logged::StopSpeaking => ("stop_speaking", json!({})),
            Logged::Mimic(on) => ("mimic", json!({"on": on})),
            Logged::Shape(s) => ("shape", json!({"shape": shape_json(&s)})),
            Logged::Push(slot, g) => ("push_gesture", {
                let mut v = score_gesture_json(&g);
                if let Value::Object(o) = &mut v {
                    o.insert("slot".into(), slot.into());
                }
                v
            }),
            Logged::EndGesture(slot, ramp) => {
                ("end_gesture", json!({"slot": slot, "ramp_out": ramp}))
            }
            Logged::Protrusion(cm) => ("protrusion", json!({"cm": cm})),
            Logged::Larynx(cm) => ("larynx", json!({"cm": cm})),
            Logged::Gesture(g) => (
                "gesture",
                json!({
                    "voiced": g.voiced,
                    "pitch": g.pitch,
                    "shape": g.shape.as_ref().map_or(Value::Null, shape_json),
                    "hiss": constriction_json(g.hiss),
                }),
            ),
        };
        // The name first; the arguments in the order JSON objects keep.
        let mut out = vec![("cmd".to_string(), Value::from(name))];
        if let Value::Object(args) = args {
            out.extend(args);
        }
        Value::Object(out.into_iter().collect())
    }

    /// A command from its JSON object, with the speech it names.
    ///
    /// # Errors
    ///
    /// When the command is unknown or an argument is missing or malformed.
    pub fn from_json(v: &Value) -> Result<(Self, Option<Speech>), String> {
        let name = get(v, "cmd")?.as_str().ok_or("\"cmd\" is not text")?;
        let slot = || -> Result<usize, String> {
            get(v, "slot")?
                .as_u64()
                .map(|s| s as usize)
                .ok_or_else(|| "\"slot\" is not a number".to_string())
        };
        let cmd = match name {
            "note_on" => Logged::NoteOn(num(v, "note")?),
            "note_off" => Logged::NoteOff,
            "note" => Logged::Note(num(v, "note")?),
            "tongue" => Logged::Tongue(num(v, "index")?, num(v, "diameter")?),
            "release" => Logged::Constriction(slot()?, None),
            "constriction" => Logged::Constriction(
                slot()?,
                Some(Constriction::new(num(v, "index")?, num(v, "diameter")?)),
            ),
            "velum" => Logged::Velum(num(v, "open")?),
            "tenseness" => Logged::Tenseness(num(v, "value")?),
            "vibrato" => Logged::Vibrato(num(v, "amount")?),
            "wobble" => Logged::Wobble(flag(v, "on")?),
            "drone" => Logged::Drone(flag(v, "on")?),
            "reverb" => Logged::Reverb(num(v, "mix")?),
            "voice" => Logged::Voice(decode(get(v, "settings")?, "settings")?),
            "lung" => Logged::Lung(num(v, "cm_h2o")?),
            "model" => Logged::Model(decode(get(v, "engine")?, "engine")?),
            "vowel" => Logged::Vowel(phoneme_from(get(v, "phoneme")?)?),
            "speak" => {
                let id = get(v, "id")?.as_u64().ok_or("\"id\" is not a number")?;
                let speech = match (get(v, "text")?.as_str(), get(v, "prosody")?) {
                    (Some(text), prosody) if !prosody.is_null() => Some(Speech {
                        text: text.to_string(),
                        prosody: decode(prosody, "prosody")?,
                    }),
                    _ => None,
                };
                return Ok((Logged::Speak(id as u32), speech));
            }
            "stop_speaking" => Logged::StopSpeaking,
            "mimic" => Logged::Mimic(flag(v, "on")?),
            "shape" => Logged::Shape(shape_from(get(v, "shape")?)?),
            "push_gesture" => {
                let slot = u8::try_from(slot()?).map_err(|e| e.to_string())?;
                Logged::Push(slot, score_gesture_from(v)?)
            }
            "end_gesture" => {
                let slot = u8::try_from(slot()?).map_err(|e| e.to_string())?;
                Logged::EndGesture(slot, num(v, "ramp_out")?)
            }
            "protrusion" => Logged::Protrusion(get(v, "cm")?.as_f64().map(|x| x as f32)),
            "larynx" => Logged::Larynx(num(v, "cm")?),
            "gesture" => {
                let shape = get(v, "shape")?;
                Logged::Gesture(Gesture {
                    voiced: flag(v, "voiced")?,
                    pitch: num(v, "pitch")?,
                    shape: if shape.is_null() {
                        None
                    } else {
                        Some(shape_from(shape)?)
                    },
                    hiss: constriction_from(get(v, "hiss")?)?,
                })
            }
            other => return Err(format!("unknown command {other:?}")),
        };
        Ok((cmd, None))
    }
}

/// Seconds at sample `at`, rounded to 0.1 ms: for reading the log by eye
/// (the sample count is the exact time).
fn seconds(at: u64, sample_rate: f32) -> f64 {
    (at as f64 / f64::from(sample_rate) * 1e4).round() / 1e4
}

/// One line of JSON with the fields in this order (a JSON object's fields
/// have none, but a person reads the log too): `first`, then the fields of
/// `rest` (an object).
fn ordered(first: &[(&str, Value)], rest: Value) -> String {
    let mut fields: Vec<(String, Value)> = first
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    if let Value::Object(rest) = rest {
        fields.extend(rest);
    }
    let body: Vec<String> = fields
        .into_iter()
        .map(|(k, v)| {
            let v = tidied(v);
            format!("{}:{}", Value::String(k), v)
        })
        .collect();
    format!("{{{}}}", body.join(","))
}

/// The header line: the take's name, the format, the sample rate and what
/// the sound file holds, the controls at its first sample, and what the
/// app knows about the voice (`app`, informational).
#[must_use]
pub fn header_line(name: &str, sample_rate: f32, controls: &Controls, app: Value) -> String {
    ordered(
        &[
            ("take", name.into()),
            ("format", FORMAT.into()),
            ("sample_rate", sample_rate.into()),
            ("signal", SIGNAL.into()),
            ("state", controls.json()),
            ("app", app),
        ],
        Value::Null,
    )
}

/// A command line: the sample it was applied before (`t`; `s` is the same
/// in seconds), and the command.
#[must_use]
pub fn command_line(at: u64, sample_rate: f32, cmd: &Logged, speech: Option<&Speech>) -> String {
    ordered(
        &[("t", at.into()), ("s", seconds(at, sample_rate).into())],
        cmd.json(speech),
    )
}

/// The last line: how long the take is, and what of it was lost on the way
/// to the disk (commands, and samples of sound written as silence).
#[must_use]
pub fn end_line(at: u64, sample_rate: f32, dropped_commands: u64, dropped_samples: u64) -> String {
    ordered(
        &[
            ("t", at.into()),
            ("s", seconds(at, sample_rate).into()),
            ("cmd", "end".into()),
            ("dropped_commands", dropped_commands.into()),
            ("dropped_samples", dropped_samples.into()),
        ],
        Value::Null,
    )
}

/// One command of a take.
#[derive(Clone, Debug, PartialEq)]
pub struct Stamped {
    /// The sample it was applied before, counted from the take's start.
    pub at: u64,
    pub cmd: Logged,
    /// For speech, what it said.
    pub speech: Option<Speech>,
}

/// A take as read from its log.
#[derive(Clone, Debug)]
pub struct Take {
    pub name: String,
    pub sample_rate: f32,
    pub controls: Controls,
    /// The header line as written.
    pub header: Value,
    pub commands: Vec<Stamped>,
    /// Length in samples; `None` if the log ends without its last line
    /// (the app stopped before the take was saved).
    pub end: Option<u64>,
    pub dropped_commands: u64,
    pub dropped_samples: u64,
}

impl Take {
    /// Read a take log.
    ///
    /// # Errors
    ///
    /// When a line is not JSON or not a header or command of this format.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty());
        let (_, first) = lines.next().ok_or("empty take log")?;
        let header: Value =
            serde_json::from_str(first).map_err(|e| format!("line 1: not JSON: {e}"))?;
        let format = get(&header, "format")?.as_str().unwrap_or_default();
        if format != FORMAT {
            return Err(format!("line 1: format {format:?}, expected {FORMAT:?}"));
        }
        let mut take = Self {
            name: get(&header, "take")?.as_str().unwrap_or_default().into(),
            sample_rate: num(&header, "sample_rate")?,
            controls: Controls::from_json(get(&header, "state")?)
                .map_err(|e| format!("line 1: {e}"))?,
            header,
            commands: Vec::new(),
            end: None,
            dropped_commands: 0,
            dropped_samples: 0,
        };
        for (k, line) in lines {
            let fail = |e: String| format!("line {}: {e}", k + 1);
            let v: Value = serde_json::from_str(line).map_err(|e| fail(e.to_string()))?;
            let at = get(&v, "t")
                .map_err(fail)?
                .as_u64()
                .ok_or_else(|| fail("\"t\" is not a sample count".into()))?;
            if get(&v, "cmd").map_err(fail)?.as_str() == Some("end") {
                take.end = Some(at);
                take.dropped_commands = v["dropped_commands"].as_u64().unwrap_or(0);
                take.dropped_samples = v["dropped_samples"].as_u64().unwrap_or(0);
                break;
            }
            let (cmd, speech) = Logged::from_json(&v).map_err(fail)?;
            take.commands.push(Stamped { at, cmd, speech });
        }
        Ok(take)
    }
}

/// Plays a take back on a voice built from its header.
pub struct Replay<'a> {
    take: &'a Take,
    performer: Performer,
    at: u64,
    next: usize,
    /// Speech whose text the log lacks, skipped.
    pub unspoken: usize,
}

impl<'a> Replay<'a> {
    #[must_use]
    pub fn new(take: &'a Take) -> Self {
        Self {
            take,
            performer: Performer::from_controls(take.sample_rate, &take.controls),
            at: 0,
            next: 0,
            unspoken: 0,
        }
    }

    /// The voice as it is now.
    #[must_use]
    pub fn performer(&self) -> &Performer {
        &self.performer
    }

    /// The sample about to be rendered.
    #[must_use]
    pub fn at(&self) -> u64 {
        self.at
    }

    /// Carry out the commands due at the current sample and render it.
    pub fn sample(&mut self) -> f32 {
        while let Some(c) = self.take.commands.get(self.next) {
            if c.at > self.at {
                break;
            }
            self.next += 1;
            match (&c.cmd, &c.speech) {
                (Logged::Speak(_), Some(speech)) => {
                    self.performer.speak(speech.utterance());
                }
                (Logged::Speak(_), None) => self.unspoken += 1,
                (cmd, _) => {
                    self.performer.apply(cmd);
                }
            }
        }
        self.at += 1;
        self.performer.sample()
    }
}

// ----- WAV ------------------------------------------------------------------

/// Bytes of the header [`wav_header`] writes; the samples follow it.
pub const WAV_HEADER_LEN: usize = 58;

/// Header of a mono 32-bit float WAV file of `samples` samples.
#[must_use]
pub fn wav_header(sample_rate: u32, samples: u32) -> [u8; WAV_HEADER_LEN] {
    let data = samples.saturating_mul(4);
    let mut h = [0u8; WAV_HEADER_LEN];
    let mut put = |at: usize, bytes: &[u8]| h[at..at + bytes.len()].copy_from_slice(bytes);
    put(0, b"RIFF");
    put(
        4,
        &(WAV_HEADER_LEN as u32 - 8)
            .saturating_add(data)
            .to_le_bytes(),
    );
    put(8, b"WAVEfmt ");
    put(16, &18u32.to_le_bytes());
    put(20, &3u16.to_le_bytes()); // IEEE float
    put(22, &1u16.to_le_bytes()); // mono
    put(24, &sample_rate.to_le_bytes());
    put(28, &sample_rate.saturating_mul(4).to_le_bytes());
    put(32, &4u16.to_le_bytes()); // bytes per frame
    put(34, &32u16.to_le_bytes()); // bits per sample
    put(36, &0u16.to_le_bytes()); // no extension
    put(38, b"fact");
    put(42, &4u32.to_le_bytes());
    put(46, &samples.to_le_bytes());
    put(50, b"data");
    put(54, &data.to_le_bytes());
    h
}

/// Write `samples` as a mono 32-bit float WAV file.
///
/// # Errors
///
/// When the file cannot be written.
pub fn write_wav(path: &std::path::Path, sample_rate: u32, samples: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let n = u32::try_from(samples.len()).unwrap_or(u32::MAX);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&wav_header(sample_rate, n))?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    f.flush()
}

/// The sample rate and samples of a mono WAV file (32-bit float or 16-bit
/// PCM). Reads to the end of the file if the data size was never filled
/// in.
///
/// # Errors
///
/// When it is not such a file.
pub fn read_wav(bytes: &[u8]) -> Result<(u32, Vec<f32>), String> {
    let u16_at = |at: usize| {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("not a WAV file".into());
    }
    let (mut format, mut rate, mut bits, mut channels) = (0, 0, 0, 0);
    let mut at = 12;
    while let (Some(id), Some(len)) = (bytes.get(at..at + 4), u32_at(at + 4)) {
        let body = at + 8;
        match id {
            b"fmt " => {
                format = u16_at(body).unwrap_or(0);
                channels = u16_at(body + 2).unwrap_or(0);
                rate = u32_at(body + 4).unwrap_or(0);
                bits = u16_at(body + 14).unwrap_or(0);
            }
            b"data" => {
                if channels != 1 {
                    return Err(format!("{channels} channels; only mono is read"));
                }
                let end = if len == 0 {
                    bytes.len()
                } else {
                    (body + len as usize).min(bytes.len())
                };
                let data = &bytes[body..end];
                let samples = match (format, bits) {
                    (3, 32) => data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                    (1, 16) => data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32_768.0)
                        .collect(),
                    _ => return Err(format!("format {format} with {bits} bits is not read")),
                };
                return Ok((rate, samples));
            }
            _ => {}
        }
        at = body + len as usize + (len as usize & 1);
    }
    Err("no data in the WAV file".into())
}
