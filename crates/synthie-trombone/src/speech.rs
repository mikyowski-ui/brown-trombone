//! Text-to-speech front end.
//!
//! [`Utterance::from_text`] turns English text into a timed articulation
//! track; a [`Speaker`] plays it on a [`Trombone`] from the audio thread.
//!
//! Text goes through:
//!
//! 1. **Normalisation:** lower case, numbers spelled out, punctuation kept as
//!    phrase breaks.
//! 2. **Letter to sound:** a small exceptions dictionary for common and
//!    irregular words, then context-sensitive English spelling rules
//!    (digraphs, magic *e*, soft *c*/*g*, vowel reduction). Write ARPAbet in
//!    brackets to bypass the rules: `"[HH AX L OW1] world"`.
//! 3. **Timing:** per-manner durations, lengthened under stress and at phrase
//!    ends; stops become closure, burst and aspiration; diphthongs glide.
//! 4. **Coarticulation:** consonants take the tongue body of the next vowel.
//! 5. **Intonation:** declining pitch over each phrase, accents on stressed
//!    syllables, a fall at `.` and a rise at `?`.
//!
//! Compiling allocates; do it off the audio thread and hand the
//! [`Utterance`] to the [`Speaker`].

#![allow(clippy::match_same_arms, clippy::if_same_then_else)] // rule tables

use crate::math::lerp;
use crate::phoneme::{Articulation, Manner, Phoneme};
use crate::vowel_targets::VowelKey;
use crate::{Engine, Trombone, Tuning};
use alloc::string::String;
use alloc::vec::Vec;

/// Voice settings for compiling speech.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Prosody {
    /// Baseline pitch in Hz.
    pub pitch: f32,
    /// Pitch excursion of accents and phrase ends, relative (`0.0` is a
    /// monotone robot, `1.0` is lively).
    pub expressiveness: f32,
    /// Speaking rate multiplier (`1.0` is about 4.5 syllables per second).
    pub rate: f32,
    /// Glottal tenseness while speaking.
    pub tenseness: f32,
    /// Velum opening during vowels and approximants (`0` oral, `1` very
    /// nasal).
    pub nasality: f32,
    /// How long sounds last and how early the articulators start for the
    /// next one (the timing fields of the voice's [`Tuning`]).
    pub tuning: Tuning,
    /// The sound model the speech is for. On the physical engine each
    /// vowel has a length of its own and drifts while it is held (pass C
    /// of the [`Tuning`]); the classic engine keeps every vowel alike.
    pub engine: Engine,
}

impl Default for Prosody {
    fn default() -> Self {
        Self {
            pitch: 110.0,
            expressiveness: 0.5,
            rate: 1.0,
            tenseness: 0.65,
            nasality: 0.0,
            tuning: Tuning::DEFAULT,
            engine: Engine::Classic,
        }
    }
}

/// One timed target of an [`Utterance`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// The phoneme this segment realises.
    pub phoneme: Phoneme,
    /// Tract target (tongue always filled in).
    pub articulation: Articulation,
    /// Length in seconds.
    pub duration: f32,
    /// Pitch at the start and end, Hz.
    pub pitch: (f32, f32),
    /// Glottal tenseness.
    pub tenseness: f32,
    /// Whether the folds vibrate.
    pub voiced: bool,
}

/// A compiled, ready-to-play articulation track.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Utterance {
    segments: Vec<Segment>,
    /// Head start the articulators get on each sound, and out of a
    /// closure, seconds (physical engine).
    lead: f32,
    release_lead: f32,
    /// Articulator speed while a vowel glides or drifts, relative
    /// (physical engine).
    glide_rate: f32,
}

/// A phoneme with its lexical context, the output of [`phonemize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    /// A sound; `stressed` marks the accented syllable of a content word.
    Phone {
        /// The sound.
        phoneme: Phoneme,
        /// Primary stress.
        stressed: bool,
    },
    /// Boundary between words.
    WordBreak,
    /// Phrase break from punctuation.
    Break(Break),
}

/// Kinds of phrase break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Break {
    /// `,` `;` `:` — short pause, pitch held.
    Comma,
    /// `.` — pitch falls.
    Period,
    /// `?` — pitch rises.
    Question,
    /// `!` — emphatic fall.
    Exclamation,
}

impl Utterance {
    /// Compile English text.
    #[must_use]
    pub fn from_text(text: &str, prosody: &Prosody) -> Self {
        Self::from_tokens(&phonemize(text), prosody)
    }

    /// An utterance from hand-made segments (see [`crate::sing`]).
    #[must_use]
    pub fn from_segments(segments: Vec<Segment>) -> Self {
        Self {
            segments,
            lead: Tuning::DEFAULT.lead,
            release_lead: Tuning::DEFAULT.release_lead,
            glide_rate: Tuning::DEFAULT.glide_rate,
        }
    }

    /// Compile a token stream (see [`phonemize`]).
    #[must_use]
    pub fn from_tokens(tokens: &[Token], prosody: &Prosody) -> Self {
        compile(tokens, prosody)
    }

    /// The segments, in order.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Total length in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Whether there is nothing to say.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Plays [`Utterance`]s on a [`Trombone`]. Never allocates.
#[derive(Clone, Debug, Default)]
pub struct Speaker {
    utterance: Utterance,
    index: usize,
    elapsed: f32,
    started: bool,
    /// The next segment's shape has already been sent (physical engine).
    ahead: bool,
    /// The articulator speed to return to after a glide, while one is
    /// under way (physical engine).
    gliding: Option<f32>,
}

// On the physical engine the articulators get a head start on each sound
// (`Tuning::lead`): their tissue has inertia there, and people begin moving
// towards a sound before it is due; without it every shape arrives late
// and speech slurs. Out of a closure the head start is short
// (`Tuning::release_lead`): stops and nasals need their closure held, and
// the release should fall on the boundary.

impl Speaker {
    /// An idle speaker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start speaking `utterance`, interrupting anything in progress. Returns
    /// the previous utterance so the caller can drop it off the audio thread.
    pub fn speak(&mut self, utterance: Utterance) -> Utterance {
        self.index = 0;
        self.elapsed = 0.0;
        self.started = false;
        self.ahead = false;
        core::mem::replace(&mut self.utterance, utterance)
    }

    /// Stop speaking (the voice fades out). Returns the utterance.
    pub fn stop(&mut self, voice: &mut Trombone) -> Utterance {
        if let Some(speed) = self.gliding.take() {
            voice.set_articulation_speed(speed);
        }
        voice.set_voiced(false);
        voice.articulate_phoneme(Phoneme::Sil);
        self.index = usize::MAX;
        core::mem::take(&mut self.utterance)
    }

    /// Whether an utterance is playing.
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.index < self.utterance.segments.len()
    }

    /// Index and segment being spoken.
    #[must_use]
    pub fn current(&self) -> Option<(usize, &Segment)> {
        self.utterance
            .segments
            .get(self.index)
            .map(|s| (self.index, s))
    }

    /// The utterance being spoken (or last spoken).
    #[must_use]
    pub fn utterance(&self) -> &Utterance {
        &self.utterance
    }

    /// Configure `voice` for speech: fast voicing, nimble articulators,
    /// quick pitch tracking, no wobble. Call once before speaking.
    pub fn prepare(voice: &mut Trombone) {
        voice.set_voicing_rates(40.0, 40.0);
        voice.set_articulation_speed(2.2);
        voice.set_glide(f32::INFINITY);
        voice.set_wobble(false);
        voice.set_always_voice(false);
    }

    /// Move `voice` to `seg`'s articulation. On the physical engine the
    /// articulators slow down (`Tuning::glide_rate`) for the glide of a
    /// diphthong or the drift of a vowel, so that it lasts the whole
    /// segment, and pick up their speed again for the next sound.
    fn articulate(&mut self, voice: &mut Trombone, seg: &Segment) {
        if voice.engine() == Engine::Physical {
            let glide = seg.articulation.vowel == Some(VowelKey::End(seg.phoneme));
            match (glide, self.gliding) {
                (true, None) => {
                    let speed = voice.articulation_speed();
                    self.gliding = Some(speed);
                    voice.set_articulation_speed(speed * self.utterance.glide_rate);
                }
                (false, Some(speed)) => {
                    self.gliding = None;
                    voice.set_articulation_speed(speed);
                }
                _ => {}
            }
        }
        voice.articulate(&seg.articulation);
    }

    /// Advance one output sample and render it.
    pub fn process(&mut self, voice: &mut Trombone) -> f32 {
        if let Some(&seg) = self.utterance.segments.get(self.index) {
            if !self.started {
                self.started = true;
                if !self.ahead {
                    self.articulate(voice, &seg);
                }
                self.ahead = false;
                voice.set_voiced(seg.voiced);
                voice.set_tenseness(seg.tenseness);
            }
            let t = (self.elapsed / seg.duration).min(1.0);
            voice.set_frequency(lerp(seg.pitch.0, seg.pitch.1, t));
            self.elapsed += 1.0 / voice.sample_rate();
            if !self.ahead
                && voice.engine() == Engine::Physical
                && let Some(&next) = self.utterance.segments.get(self.index + 1)
            {
                let closed = matches!(
                    seg.phoneme.manner(),
                    Manner::Stop | Manner::Affricate | Manner::Nasal
                ) && seg
                    .articulation
                    .constriction
                    .is_some_and(|c| c.diameter <= 0.3);
                let lead = if closed {
                    self.utterance.release_lead
                } else {
                    self.utterance.lead
                };
                if self.elapsed >= seg.duration - lead.min(0.5 * seg.duration) {
                    self.articulate(voice, &next);
                    self.ahead = true;
                }
            }
            if self.elapsed >= seg.duration {
                self.elapsed -= seg.duration;
                self.index += 1;
                self.started = false;
                if self.index == self.utterance.segments.len() {
                    if let Some(speed) = self.gliding.take() {
                        voice.set_articulation_speed(speed);
                    }
                    voice.set_voiced(false);
                    voice.articulate_phoneme(Phoneme::Sil);
                }
            }
        }
        voice.process()
    }

    /// Render a block.
    pub fn render(&mut self, voice: &mut Trombone, out: &mut [f32]) {
        for s in out {
            *s = self.process(voice);
        }
    }
}

// ===== text → tokens ======================================================

mod lexicon;
mod rules;

/// Convert English text (with optional `[ARPAbet]` spans) to tokens.
///
/// Words are looked up in the built-in dictionary, taken apart into known
/// stems and regular endings (`-s`, `-ed`, `-ing`, `-ly` ...), and spelled
/// out by English spelling rules otherwise.
#[must_use]
pub fn phonemize(text: &str) -> Vec<Token> {
    phonemize_words(text, &lexicon::builtin)
}

/// [`phonemize`] with `lexicon` consulted in place of the built-in
/// dictionary.
///
/// `lexicon` maps a lower-case word (apostrophes kept) to ARPAbet with `1`
/// after the accented vowel, `"K AX N UW1"`; a word with no `1` is spoken
/// unaccented, like a function word. Words it does not know go through the
/// same stems, endings and spelling rules as in [`phonemize`]. Useful for a
/// custom vocabulary, and for checking the spelling rules alone
/// (`|_| None`).
#[must_use]
pub fn phonemize_with<'a>(text: &str, lexicon: &dyn Fn(&str) -> Option<&'a str>) -> Vec<Token> {
    phonemize_words(text, &|w: &str| lexicon(w).map(lexicon::parse_arpabet))
}

fn phonemize_words(text: &str, lex: lexicon::Lexicon) -> Vec<Token> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut chars = text.chars().peekable();

    let flush = |word: &mut String, out: &mut Vec<Token>| {
        if !word.is_empty() {
            push_word(word, lex, out);
            word.clear();
        }
    };

    while let Some(c) = chars.next() {
        match c {
            '[' => {
                flush(&mut word, &mut out);
                let mut span = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    span.push(c);
                }
                push_arpabet(&span, &mut out);
            }
            c if c.is_alphabetic() || c == '\'' || c == '’' => {
                if c == '’' {
                    word.push('\'');
                } else {
                    word.extend(c.to_lowercase());
                }
            }
            c if c.is_ascii_digit() => {
                flush(&mut word, &mut out);
                let mut n = String::from(c);
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        n.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                for w in number_words(&n).split(' ').filter(|w| !w.is_empty()) {
                    push_word(w, lex, &mut out);
                }
            }
            _ => {
                flush(&mut word, &mut out);
                let brk = match c {
                    ',' | ';' | ':' | '—' | '–' => Some(Break::Comma),
                    '.' | '\n' => Some(Break::Period),
                    '?' => Some(Break::Question),
                    '!' => Some(Break::Exclamation),
                    _ => None,
                };
                if let Some(b) = brk {
                    // Collapse runs like "?!" or "..." into the first.
                    if !matches!(out.last(), Some(Token::Break(_))) {
                        if matches!(out.last(), Some(Token::WordBreak)) {
                            out.pop();
                        }
                        out.push(Token::Break(b));
                    }
                }
            }
        }
    }
    flush(&mut word, &mut out);
    if matches!(out.last(), Some(Token::WordBreak)) {
        out.pop();
    }
    out
}

/// An `[ARPAbet]` span. A `1` after a vowel marks the accent; with no `1`
/// the first vowel is accented, unless a vowel is marked `0` (`[DH AX0]`),
/// which keeps the word unaccented.
fn push_arpabet(span: &str, out: &mut Vec<Token>) {
    let mut any = false;
    let mut stressed_seen = false;
    let mut unstressed_marked = false;
    for sym in span.split_whitespace() {
        if let Some(p) = Phoneme::from_symbol(sym) {
            let stressed = sym.ends_with('1');
            stressed_seen |= stressed;
            unstressed_marked |= sym.ends_with('0');
            out.push(Token::Phone {
                phoneme: p,
                stressed,
            });
            any = true;
        }
    }
    if any {
        if !stressed_seen && !unstressed_marked {
            stress_first_vowel(out);
        }
        out.push(Token::WordBreak);
    }
}

fn push_word(word: &str, lex: lexicon::Lexicon, out: &mut Vec<Token>) {
    let word = word.trim_matches('\'');
    if word.is_empty() {
        return;
    }
    for (phoneme, stressed) in lexicon::pronounce(word, lex) {
        out.push(Token::Phone { phoneme, stressed });
    }
    if matches!(out.last(), Some(Token::Phone { .. })) {
        out.push(Token::WordBreak);
    }
}

fn stress_first_vowel(tokens: &mut [Token]) {
    // Within the current word: walk back to the last break.
    let start = tokens
        .iter()
        .rposition(|t| !matches!(t, Token::Phone { .. }))
        .map_or(0, |i| i + 1);
    for t in &mut tokens[start..] {
        if let Token::Phone { phoneme, stressed } = t
            && is_vowel(*phoneme)
        {
            *stressed = true;
            return;
        }
    }
}

fn is_vowel(p: Phoneme) -> bool {
    matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
}

fn number_words(digits: &str) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn below_thousand(n: usize, out: &mut String) {
        if n >= 100 {
            out.push_str(ONES[n / 100]);
            out.push_str(" hundred ");
        }
        let r = n % 100;
        if r >= 20 {
            out.push_str(TENS[r / 10]);
            out.push(' ');
            if !r.is_multiple_of(10) {
                out.push_str(ONES[r % 10]);
                out.push(' ');
            }
        } else if r > 0 || n == 0 {
            out.push_str(ONES[r]);
            out.push(' ');
        }
    }

    let mut out = String::new();
    match digits.parse::<usize>() {
        Ok(n) if digits.len() <= 9 && !(digits.len() > 1 && digits.starts_with('0')) => {
            let (m, rest) = (n / 1_000_000, n % 1_000_000);
            let (k, r) = (rest / 1000, rest % 1000);
            if m > 0 {
                below_thousand(m, &mut out);
                out.push_str("million ");
            }
            if k > 0 {
                below_thousand(k, &mut out);
                out.push_str("thousand ");
            }
            if r > 0 || n == 0 {
                below_thousand(r, &mut out);
            }
        }
        _ => {
            // Too long, or a code like "007": read digit by digit.
            for d in digits.bytes() {
                out.push_str(ONES[usize::from(d - b'0')]);
                out.push(' ');
            }
        }
    }
    out
}

// ===== tokens → timed segments ============================================

/// Share of a diphthong the classic engine holds on its first vowel.
const CLASSIC_NUCLEUS: f32 = 0.55;
/// The rest of it, the glide.
const CLASSIC_GLIDE: f32 = 0.45;

/// How long vowel or diphthong `p` lasts against others of its kind, on
/// the physical engine (pass C of the tuning). English vowels have
/// lengths of their own: lax /ɪ ɛ ʌ ʊ/ are about two thirds as long as
/// tense /i æ ɑ ɔ u/.
fn own_length(p: Phoneme, timing: &Tuning) -> f32 {
    use Phoneme as P;
    match p {
        P::Iy => timing.iy_length,
        P::Ih => timing.ih_length,
        P::Eh => timing.eh_length,
        P::Ae => timing.ae_length,
        P::Ah => timing.ah_length,
        P::Aa => timing.aa_length,
        P::Ao => timing.ao_length,
        P::Uh => timing.uh_length,
        P::Uw => timing.uw_length,
        P::Er => timing.er_length,
        P::Ax => timing.ax_length,
        P::Ey => timing.ey_length,
        P::Ay => timing.ay_length,
        P::Ow => timing.ow_length,
        P::Aw => timing.aw_length,
        P::Oy => timing.oy_length,
        _ => 1.0,
    }
}

/// Base durations in seconds at rate 1.
fn base_duration(p: Phoneme, stressed: bool, timing: &Tuning, physical: bool) -> f32 {
    let own = |d: f32| {
        if physical {
            d * own_length(p, timing)
        } else {
            d
        }
    };
    let d = match p.manner() {
        Manner::Vowel => own(timing.vowel_length),
        Manner::Diphthong => own(timing.diphthong_length),
        Manner::Stop | Manner::Affricate => timing.stop_length,
        Manner::Nasal => timing.nasal_length,
        Manner::Fricative => {
            if p.is_voiced() {
                timing.voiced_fricative_length
            } else {
                timing.fricative_length
            }
        }
        Manner::Approximant => timing.approximant_length,
        Manner::Aspirate => timing.aspirate_length,
        Manner::Pause => 0.1,
    };
    if stressed && is_vowel(p) {
        d * timing.stress_stretch
    } else {
        d
    }
}

/// Which part of a sound a draft is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// A whole sound, or the first part of a vowel.
    Head,
    /// A diphthong's glide to its end.
    Glide,
    /// A vowel's drift towards its end shape (physical engine).
    Drift,
}

#[derive(Clone, Copy)]
struct Draft {
    phoneme: Phoneme,
    articulation: Articulation,
    duration: f32,
    voiced: bool,
    tenseness: Option<f32>,
    /// Relative pitch target at the end of the segment (1.0 = baseline).
    accent: f32,
    vowel: bool,
    part: Part,
}

#[allow(clippy::too_many_lines)]
fn compile(tokens: &[Token], prosody: &Prosody) -> Utterance {
    let rate = prosody.rate.clamp(0.2, 5.0);
    let expr = prosody.expressiveness.clamp(0.0, 2.0);
    let tense = prosody.tenseness.clamp(0.0, 1.0);
    let nasality = prosody.nasality.clamp(0.0, 1.0);
    let timing = prosody.tuning.clamped();
    let physical = prosody.engine == Engine::Physical;

    // Split into phrases at breaks; each phrase gets its own contour.
    let mut drafts: Vec<Draft> = Vec::new();
    let mut phrase_start = 0;
    let mut i = 0;

    let push_phrase_end = |drafts: &mut Vec<Draft>, start: usize, brk: Option<Break>| {
        shape_phrase(&mut drafts[start..], brk, expr);
    };

    while i < tokens.len() {
        match tokens[i] {
            Token::Phone { phoneme, stressed } => {
                let next_vowel_or_approx = tokens[i + 1..]
                    .iter()
                    .find_map(|t| match t {
                        Token::Phone { phoneme, .. } => Some(*phoneme),
                        _ => None,
                    })
                    .is_some_and(|p| is_vowel(p) || p.manner() == Manner::Approximant);
                expand(
                    phoneme,
                    stressed,
                    next_vowel_or_approx,
                    rate,
                    &timing,
                    physical,
                    &mut drafts,
                );
            }
            Token::WordBreak => {}
            Token::Break(b) => {
                push_phrase_end(&mut drafts, phrase_start, Some(b));
                let pause = match b {
                    Break::Comma => 0.18,
                    _ => 0.38,
                } / rate;
                drafts.push(silence(pause));
                phrase_start = drafts.len();
            }
        }
        i += 1;
    }
    push_phrase_end(&mut drafts, phrase_start, Some(Break::Period));

    // Lengthen the last syllable of each phrase.
    for k in 1..drafts.len() {
        if drafts[k].phoneme == Phoneme::Sil
            && let Some(v) = drafts[..k].iter().rposition(|d| d.vowel)
        {
            drafts[v].duration *= 1.4;
            // On the physical engine the whole vowel, not only its end.
            let mut head = v;
            while physical && head > 0 && drafts[head].part != Part::Head {
                head -= 1;
                drafts[head].duration *= 1.4;
            }
        }
    }

    // A vowel that drifts keeps the pitch course it would have had held
    // still: its first part ends on the way to the accent, not on it.
    for k in 1..drafts.len() {
        if drafts[k].part == Part::Drift {
            let before = if k >= 2 { drafts[k - 2].accent } else { 1.0 };
            let (head, tail) = (drafts[k - 1].duration, drafts[k].duration);
            drafts[k - 1].accent = lerp(before, drafts[k].accent, head / (head + tail));
        }
    }

    // Coarticulation: consonants borrow the tongue of the next vowel (or
    // the previous one at the end of an utterance).
    let mut next_tongue = None;
    for d in drafts.iter_mut().rev() {
        if d.vowel {
            next_tongue = d.articulation.tongue;
        } else if d.articulation.tongue.is_none() {
            d.articulation.tongue = next_tongue;
        }
    }
    let mut prev_tongue = Phoneme::Ax.articulation().tongue;
    for d in &mut drafts {
        match d.articulation.tongue {
            Some(t) => prev_tongue = Some(t),
            None => d.articulation.tongue = prev_tongue,
        }
    }

    // Pitch: start each segment where the last ended.
    let mut segments = Vec::with_capacity(drafts.len() + 1);
    let mut last = prosody.pitch;
    for mut d in drafts {
        if d.vowel || d.phoneme.manner() == Manner::Approximant {
            d.articulation.velum = d.articulation.velum.max(nasality);
        }
        let target = prosody.pitch * d.accent;
        segments.push(Segment {
            phoneme: d.phoneme,
            articulation: d.articulation,
            duration: d.duration,
            pitch: (last, target),
            tenseness: d.tenseness.unwrap_or(tense),
            voiced: d.voiced,
        });
        last = target;
    }
    // Trim a trailing pause to a short release.
    if let Some(s) = segments.last_mut()
        && s.phoneme == Phoneme::Sil
    {
        s.duration = 0.05;
    }
    Utterance {
        segments,
        lead: timing.lead,
        release_lead: timing.release_lead,
        glide_rate: timing.glide_rate,
    }
}

fn silence(duration: f32) -> Draft {
    Draft {
        phoneme: Phoneme::Sil,
        articulation: Phoneme::Sil.articulation(),
        duration,
        voiced: false,
        tenseness: None,
        accent: 1.0,
        vowel: false,
        part: Part::Head,
    }
}

fn expand(
    p: Phoneme,
    stressed: bool,
    before_vowel: bool,
    rate: f32,
    timing: &Tuning,
    physical: bool,
    out: &mut Vec<Draft>,
) {
    let a = p.articulation();
    let dur = base_duration(p, stressed, timing, physical) / rate;
    let accent = if stressed { 1.0 } else { 0.0 };
    let draft = |phoneme: Phoneme, articulation: Articulation, duration: f32, voiced: bool| Draft {
        phoneme,
        articulation,
        duration,
        voiced,
        tenseness: articulation.tenseness,
        // Marks stress for `shape_phrase`; replaced by a pitch ratio there.
        accent,
        vowel: is_vowel(phoneme),
        part: Part::Head,
    };

    match p.manner() {
        Manner::Diphthong => {
            // Start on the diphthong's own shape (OW starts on /o/, not on
            // AO's more open "aw"), then glide to its end (on the physical
            // engine the diphthong's own, on the classic its second vowel).
            let end = p.glide_end().expect("diphthong");
            let (head, glide) = if physical {
                let n = timing.diphthong_nucleus;
                (n, 1.0 - n)
            } else {
                (CLASSIC_NUCLEUS, CLASSIC_GLIDE)
            };
            out.push(draft(p, a, dur * head, true));
            let mut tail = draft(p, end, dur * glide, true);
            tail.accent = 0.0;
            tail.part = Part::Glide;
            out.push(tail);
        }
        Manner::Vowel if physical && timing.vowel_drift > 0.0 => {
            // Held, then on the way to where the vowel drifts: lax vowels
            // towards the middle of the mouth, tense ones further out.
            let end = p.drift_end().expect("a vowel");
            let drift = timing.vowel_drift;
            out.push(draft(p, a, dur * (1.0 - drift), true));
            let mut tail = draft(p, end, dur * drift, true);
            tail.part = Part::Drift;
            out.push(tail);
        }
        Manner::Stop => {
            out.push(draft(p, a, dur, a.voiced));
            if !a.voiced && before_vowel {
                // Aspiration after the burst: open tract, breathy folds.
                let mut asp = Phoneme::Hh.articulation();
                asp.tongue = a.tongue;
                out.push(draft(p, asp, timing.aspiration / rate, true));
            }
        }
        Manner::Affricate => {
            out.push(draft(p, a, dur * 0.6, a.voiced));
            let fric = if a.voiced { Phoneme::Zh } else { Phoneme::Sh };
            out.push(draft(p, fric.articulation(), dur, a.voiced));
        }
        _ => out.push(draft(p, a, dur, a.voiced)),
    }
}

/// Intonation for one phrase: gentle declination, accents on stressed
/// vowels, and a boundary tone.
#[allow(clippy::cast_precision_loss)]
fn shape_phrase(drafts: &mut [Draft], brk: Option<Break>, expr: f32) {
    let total: f32 = drafts.iter().map(|d| d.duration).sum();
    if total <= 0.0 {
        return;
    }
    let last_vowel = drafts.iter().rposition(|d| d.vowel);
    let mut t = 0.0;
    for (k, d) in drafts.iter_mut().enumerate() {
        t += d.duration;
        let pos = t / total;
        let declination = 1.0 + expr * (0.12 - 0.22 * pos);
        let accent = if d.accent > 0.0 && d.vowel {
            1.0 + 0.18 * expr
        } else {
            1.0
        };
        let boundary = if Some(k) == last_vowel {
            match brk {
                Some(Break::Question) => 1.0 + 0.6 * expr,
                Some(Break::Period) => 1.0 - 0.18 * expr,
                Some(Break::Exclamation) => 1.0 - 0.25 * expr,
                Some(Break::Comma) => 1.0 + 0.08 * expr,
                None => 1.0,
            }
        } else {
            1.0
        };
        d.accent = declination * accent * boundary;
    }
    // After a question rise, keep the remaining consonants up.
    if let (Some(v), Some(Break::Question)) = (last_vowel, brk) {
        let top = drafts[v].accent;
        for d in &mut drafts[v + 1..] {
            d.accent = top;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Phoneme as P;
    use alloc::vec;

    fn phones(text: &str) -> Vec<Phoneme> {
        phonemize(text)
            .into_iter()
            .filter_map(|t| match t {
                Token::Phone { phoneme, .. } => Some(phoneme),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn dictionary_words() {
        assert_eq!(
            phones("hello world"),
            vec![P::Hh, P::Ax, P::L, P::Ow, P::W, P::Er, P::L, P::D]
        );
    }

    #[test]
    fn spelling_rules() {
        assert_eq!(phones("cat"), vec![P::K, P::Ae, P::T]);
        assert_eq!(phones("make"), vec![P::M, P::Ey, P::K]);
        assert_eq!(phones("time"), vec![P::T, P::Ay, P::M]);
        assert_eq!(phones("ship"), vec![P::Sh, P::Ih, P::P]);
        assert_eq!(phones("night"), vec![P::N, P::Ay, P::T]);
        assert_eq!(phones("city"), vec![P::S, P::Ih, P::T, P::Iy]);
        assert_eq!(phones("sing"), vec![P::S, P::Ih, P::Ng]);
        assert_eq!(phones("nation"), vec![P::N, P::Ey, P::Sh, P::Ax, P::N]);
        assert_eq!(phones("dogs"), vec![P::D, P::Aa, P::G, P::Z]);
        assert_eq!(phones("bell"), vec![P::B, P::Eh, P::L]);
        assert_eq!(phones("knight"), vec![P::N, P::Ay, P::T]);
        assert_eq!(phones("queen"), vec![P::K, P::W, P::Iy, P::N]);
        assert_eq!(phones("boat"), vec![P::B, P::Ow, P::T]);
    }

    #[test]
    fn arpabet_spans_bypass_rules() {
        assert_eq!(phones("[K AE1 T]"), vec![P::K, P::Ae, P::T]);
        assert_eq!(phones("[nonsense]"), vec![]);
    }

    #[test]
    fn numbers_are_spelled_out() {
        assert_eq!(number_words("0").trim(), "zero");
        assert_eq!(number_words("42").trim(), "forty two");
        assert_eq!(
            number_words("1984").trim(),
            "one thousand nine hundred eighty four"
        );
        assert_eq!(number_words("007").trim(), "zero zero seven");
        assert_ne!(phones("route 66"), vec![]);
    }

    #[test]
    fn punctuation_makes_breaks() {
        let t = phonemize("hi, you?! ok.");
        let breaks: Vec<_> = t
            .iter()
            .filter_map(|t| match t {
                Token::Break(b) => Some(*b),
                _ => None,
            })
            .collect();
        assert_eq!(breaks, vec![Break::Comma, Break::Question, Break::Period]);
    }

    #[test]
    fn content_words_are_stressed() {
        let stressed: Vec<_> = phonemize("the cat")
            .into_iter()
            .filter_map(|t| match t {
                Token::Phone {
                    phoneme,
                    stressed: true,
                } => Some(phoneme),
                _ => None,
            })
            .collect();
        assert_eq!(stressed, vec![P::Ae]);
    }

    #[test]
    fn questions_rise_statements_fall() {
        let p = Prosody::default();
        let q = Utterance::from_text("you are here?", &p);
        let s = Utterance::from_text("you are here.", &p);
        let last_vowel_pitch = |u: &Utterance| {
            u.segments()
                .iter()
                .rev()
                .find(|s| is_vowel(s.phoneme))
                .unwrap()
                .pitch
                .1
        };
        assert!(last_vowel_pitch(&q) > p.pitch * 1.1);
        assert!(last_vowel_pitch(&s) < p.pitch);
    }

    #[test]
    fn consonants_borrow_the_next_vowel_tongue() {
        let u = Utterance::from_text("[S IY]", &Prosody::default());
        let s = u.segments()[0];
        assert_eq!(s.phoneme, P::S);
        assert_eq!(s.articulation.tongue, P::Iy.articulation().tongue);
    }

    #[test]
    fn faster_rate_is_shorter() {
        let slow = Utterance::from_text(
            "pink trombone",
            &Prosody {
                rate: 0.5,
                ..Prosody::default()
            },
        );
        let fast = Utterance::from_text(
            "pink trombone",
            &Prosody {
                rate: 2.0,
                ..Prosody::default()
            },
        );
        assert!(slow.duration() > 3.0 * fast.duration());
    }

    #[test]
    fn speaker_plays_and_finishes() {
        let mut voice = Trombone::new(16_000.0);
        Speaker::prepare(&mut voice);
        let mut sp = Speaker::new();
        let u = Utterance::from_text("Hello, pink trombone!", &Prosody::default());
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = ((u.duration() + 0.3) * 16_000.0) as usize;
        sp.speak(u);
        assert!(sp.is_speaking());
        let mut buf = vec![0.0; n];
        sp.render(&mut voice, &mut buf);
        assert!(!sp.is_speaking());
        assert!(buf.iter().all(|s| s.is_finite()));
        let energy: f32 = buf.iter().map(|s| s * s).sum();
        assert!(energy > 1.0, "{energy}");
    }

    #[test]
    fn never_panics_on_odd_text() {
        for t in [
            "",
            "   ",
            "!!!",
            "[",
            "]]",
            "ÄÖÜ ß",
            "a",
            "e",
            "y",
            "x",
            "123456789012",
            "'''",
            "—",
        ] {
            let u = Utterance::from_text(t, &Prosody::default());
            assert!(
                u.segments()
                    .iter()
                    .all(|s| s.duration > 0.0 && s.duration.is_finite())
            );
        }
    }
}
