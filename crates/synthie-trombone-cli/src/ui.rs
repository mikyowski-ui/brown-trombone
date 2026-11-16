//! Rendering: the tract as a Braille side view, glottis meters, scope,
//! and the speech line.

use crate::app::{
    App, BAR_WIDTH, Control, DIALS, LANDMARKS, LOUD_RANGE, Mode, NOSE_Y, VOWEL_KEYS, X_BOUNDS,
    Y_BOUNDS, engine_name, lerp_cell,
};
use crate::audio::SCOPE_LEN;
use crate::line::MAX_CHARS;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::widgets::canvas::Points;
use ratatui::widgets::canvas::{Canvas, Context, Line as CLine};
use synthie_trombone::inverse;
use synthie_trombone::{N, NOSE_LEN, NOSE_START};
use synthie_ui::theme::{self, BG, CYAN, DIM, FAINT, FG, MAGENTA, PURPLE, RED, YELLOW, bold, fg};
use synthie_ui::widgets::{key_hints, keycap, logo, meter, paint_background, panel, vu};

const NOTE_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];

pub fn draw(frame: &mut Frame, app: &mut App) {
    paint_background(frame);
    app.controls.clear();
    let [header, tract, mid, speech, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(12),
        Constraint::Length(11),
        Constraint::Length(4),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    draw_header(frame, header, app);
    draw_tract(frame, tract, app);
    let [glottis, persona, scope] = Layout::horizontal([
        Constraint::Length(54),
        Constraint::Length(38),
        Constraint::Min(12),
    ])
    .areas(mid);
    draw_glottis(frame, glottis, app);
    draw_persona(frame, persona, app);
    draw_sound(frame, scope, app);
    draw_speech(frame, speech, app);
    draw_footer(frame, footer, app);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &mut App) {
    let mut spans = logo().spans;
    spans.push(Span::styled("  P I N K   T R O M B O N E  ", bold(MAGENTA)));
    let (label, color) = match app.mode {
        Mode::Play => (" PLAY ", CYAN),
        Mode::Type => (" TYPE ", YELLOW),
    };
    // The mode badge is a button.
    let before = u16::try_from(Line::from(spans.clone()).width()).unwrap_or(0);
    let width = u16::try_from(label.len()).unwrap_or(0);
    app.controls
        .push((Rect::new(area.x + before, area.y, width, 1), Control::Mode));
    spans.push(Span::styled(
        label,
        Style::default()
            .fg(BG)
            .bg(color)
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(format!("  {}", app.status), fg(DIM)));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
    // While a take is recorded, its marker goes over the right end of the
    // status, clear of the mode badge.
    if let Some(marker) = rec_marker(app) {
        let room = area.width.saturating_sub(before + width + 1);
        let fits = [
            marker.clone(),
            marker.replace(" REC ", " "),
            " ● REC ".to_string(),
            "●".to_string(),
        ]
        .into_iter()
        .find(|m| u16::try_from(m.chars().count()).is_ok_and(|n| n <= room));
        if let Some(m) = fits {
            let n = u16::try_from(m.chars().count()).unwrap_or(0);
            let r = Rect::new(area.right() - n, area.y, n, 1);
            let style = Style::default().fg(BG).bg(RED).add_modifier(Modifier::BOLD);
            frame.render_widget(Paragraph::new(Span::styled(m, style)), r);
        }
    }
}

/// The recording marker, `● REC take-007 0:03`, while a take is recorded
/// (with `!` once anything was lost on the way to the disk).
fn rec_marker(app: &App) -> Option<String> {
    let name = app.recording()?;
    let clock = app.snap.take.unwrap_or_default();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    let seconds = (clock.samples as f64 / f64::from(app.sample_rate())) as u64;
    let lost = if clock.dropped > 0 { " !" } else { "" };
    Some(format!(
        " ● REC {name} {}:{:02}{lost} ",
        seconds / 60,
        seconds % 60
    ))
}

fn amp_color(a: f32) -> Color {
    theme::heat((a * 1.6).clamp(0.0, 1.0))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
fn draw_tract(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = panel(
        "vocal tract  ·  drag: constrict  ·  drag ◆ or shift-drag: tongue  ·  click a vowel or the nose",
        true,
    );
    app.tract_area = block.inner(area);
    let s = app.snap;
    let marks: Vec<(f64, f64, &str)> = LANDMARKS
        .iter()
        .filter_map(|&(name, vowel)| {
            app.landmark(vowel)
                .map(|(x, y)| (f64::from(x), f64::from(y), name))
        })
        .collect();

    let canvas = Canvas::default()
        .block(block)
        .background_color(BG)
        .marker(Marker::Braille)
        .x_bounds(X_BOUNDS)
        .y_bounds(Y_BOUNDS)
        .paint(move |ctx: &mut Context| {
            // Air column, coloured by recent pressure.
            let mut x = 0.0;
            while x <= (N - 1) as f64 {
                let d = lerp_cell(&s.diameters, x) / 2.0;
                let a = lerp_cell(&s.amplitudes, x);
                #[allow(clippy::cast_possible_truncation)]
                let color = amp_color(a as f32);
                ctx.draw(&CLine::new(x, -d, x, d, color));
                x += 0.2;
            }
            let mut x = NOSE_START as f64;
            while x <= (N - 1) as f64 {
                let k = x - NOSE_START as f64;
                let d = lerp_cell(&s.nose, k) / 2.0;
                let a = lerp_cell(&s.nose_amplitudes, k);
                #[allow(clippy::cast_possible_truncation)]
                let color = amp_color(a as f32);
                ctx.draw(&CLine::new(x, NOSE_Y - d, x, NOSE_Y + d, color));
                x += 0.2;
            }

            // Target shape, faint.
            for i in 0..N - 1 {
                let (a, b) = (
                    f64::from(s.targets[i]) / 2.0,
                    f64::from(s.targets[i + 1]) / 2.0,
                );
                let (x0, x1) = (i as f64, (i + 1) as f64);
                ctx.draw(&CLine::new(x0, a, x1, b, FAINT));
                ctx.draw(&CLine::new(x0, -a, x1, -b, FAINT));
            }
            // Walls.
            for i in 0..N - 1 {
                let (a, b) = (
                    f64::from(s.diameters[i]) / 2.0,
                    f64::from(s.diameters[i + 1]) / 2.0,
                );
                let (x0, x1) = (i as f64, (i + 1) as f64);
                ctx.draw(&CLine::new(x0, a, x1, b, CYAN));
                ctx.draw(&CLine::new(x0, -a, x1, -b, PURPLE));
            }
            for k in 0..NOSE_LEN - 1 {
                let (a, b) = (f64::from(s.nose[k]) / 2.0, f64::from(s.nose[k + 1]) / 2.0);
                let (x0, x1) = ((NOSE_START + k) as f64, (NOSE_START + k + 1) as f64);
                ctx.draw(&CLine::new(x0, NOSE_Y + a, x1, NOSE_Y + b, CYAN));
                ctx.draw(&CLine::new(x0, NOSE_Y - a, x1, NOSE_Y - b, CYAN));
            }
            // Velum: the door between mouth and nose.
            let top = f64::from(s.diameters[NOSE_START]) / 2.0;
            let gap = f64::from(s.velum);
            let vx = NOSE_START as f64;
            let vcol = if gap > 0.2 { YELLOW } else { DIM };
            ctx.draw(&CLine::new(
                vx,
                top,
                vx,
                NOSE_Y - f64::from(s.nose[0]) / 2.0,
                vcol,
            ));
            ctx.draw(&CLine::new(
                vx + 0.6,
                top,
                vx + 0.6 + gap * 0.8,
                top + 0.8,
                vcol,
            ));
            ctx.layer();

            // Where the vowels sit, faintly. The tongue handle goes on top
            // once the canvas is drawn.
            for &(x, y, name) in &marks {
                ctx.print(x, y, Span::styled(name, fg(DIM)));
            }
            // Each constriction is marked where it presses on the wall.
            for (k, c) in s.constrictions.iter().enumerate() {
                if let Some(c) = c {
                    let label = match k {
                        0 | 1 => "✕",
                        2 => "●",
                        _ => "○",
                    };
                    let wall = f64::from((c.diameter - 0.3).max(0.0)) / 2.0;
                    ctx.print(
                        f64::from(c.index),
                        (wall + 0.25).min(1.9),
                        Span::styled(label, bold(MAGENTA)),
                    );
                }
            }
            ctx.print(-0.8, -2.4, Span::styled("glottis", fg(DIM)));
            ctx.print(40.5, -2.4, Span::styled("lips", fg(DIM)));
            ctx.print(41.0, NOSE_Y + 1.1, Span::styled("nose", fg(DIM)));
            ctx.print(NOSE_START as f64 - 3.5, 2.4, Span::styled("velum", fg(DIM)));
        });
    frame.render_widget(canvas, area);
    // The tongue handle, in the cell the walls are drawn around and the
    // mouse finds it in: inside the mouth whatever shape the tract has.
    if let Some((col, row)) = app.tongue_cell() {
        let room = usize::from(app.tract_area.right().saturating_sub(col));
        frame
            .buffer_mut()
            .set_stringn(col, row, "◆ tongue", room, bold(YELLOW));
    }
}

fn note_name(note: i32) -> String {
    #[allow(clippy::cast_sign_loss)]
    let name = NOTE_NAMES[note.rem_euclid(12) as usize];
    format!("{name}{}", note.div_euclid(12) - 1)
}

fn row<'a>(label: &'a str, bar: Vec<Span<'a>>, value: impl std::fmt::Display) -> Line<'a> {
    let mut spans = vec![Span::styled(format!(" {label:<8}"), fg(DIM))];
    spans.extend(bar);
    spans.push(Span::styled(format!(" {value}"), fg(FG)));
    Line::from(spans)
}

/// One persona dial as a panel row; the dial the keys work is marked.
fn dial(app: &App, k: usize) -> Line<'static> {
    let (name, lo, hi) = DIALS[k];
    let value = app.dials()[k];
    let chosen = app.dial == k;
    let mut spans = vec![Span::styled(
        format!("{}{name:<8}", if chosen { "›" } else { " " }),
        if chosen { bold(MAGENTA) } else { fg(DIM) },
    )];
    spans.extend(meter((value - lo) / (hi - lo), usize::from(BAR_WIDTH)));
    spans.push(Span::styled(format!(" {value:.2}"), fg(FG)));
    Line::from(spans)
}

/// The persona panel: the voice described the way a person would, four
/// dials that each move several parameters together.
fn draw_persona(frame: &mut Frame, area: Rect, app: &mut App) {
    // Turning a dial with nothing sounding sounds the voice for a moment.
    let block = panel("persona  ·  drag to hear", false);
    app.controls
        .extend(App::persona_controls(block.inner(area)));
    let app = &*app;
    let v = &app.settings;
    let lines = vec![
        dial(app, 0),
        dial(app, 1),
        dial(app, 2),
        dial(app, 3),
        Line::default(),
        Line::from(vec![
            Span::styled(" pitch ", fg(DIM)),
            Span::styled(format!("{:.0} Hz", v.pitch), fg(FG)),
            Span::styled("  tract ", fg(DIM)),
            Span::styled(format!("×{:.2}", v.tract_length), fg(FG)),
        ]),
        Line::from(vec![
            Span::styled(" lungs ", fg(DIM)),
            Span::styled(
                format!("{:.1} cmH2O", v.tuning.anatomy.lung_pressure),
                fg(FG),
            ),
            Span::styled("  tense ", fg(DIM)),
            Span::styled(format!("{:.2}", v.tenseness), fg(FG)),
        ]),
        Line::from(Span::styled(
            " ' next  PgUp/PgDn turn  Del neutral",
            fg(DIM),
        )),
    ];
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_glottis(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = panel("glottis  ·  click or drag", false);
    app.controls
        .extend(App::glottis_controls(block.inner(area)));
    let app = &*app;
    let s = &app.snap;
    let w = usize::from(BAR_WIDTH);
    let on_off = |b: bool| {
        if b {
            "on".to_string()
        } else {
            "off".to_string()
        }
    };
    let vowel = app.last_vowel.map_or("—", |v| v.symbol());
    let lines = vec![
        Line::from(vec![
            Span::styled(" note    ", fg(DIM)),
            Span::styled(format!("{:<4}", note_name(app.note)), bold(YELLOW)),
            Span::styled(format!(" {:6.1} Hz", s.frequency), fg(FG)),
            Span::styled("  ", fg(DIM)),
            Span::styled("[-]", bold(CYAN)),
            Span::styled(format!(" oct {} ", app.octave), fg(FG)),
            Span::styled("[+]", bold(CYAN)),
            Span::styled(format!(" {vowel} "), fg(DIM)),
            Span::styled(app.preset.name(), bold(MAGENTA)),
        ]),
        row(
            "voice",
            meter(s.intensity, w),
            format!("{:.2}", s.intensity),
        ),
        row(
            "tense",
            meter(s.tenseness, w),
            format!("{:.2}", s.tenseness),
        ),
        row(
            "tongue",
            meter((s.tongue.0 - 12.0) / 17.0, w),
            format!("{:.1}/{:.2}", s.tongue.0, s.tongue.1),
        ),
        row(
            "velum",
            meter(s.velum, w),
            format!("{:.2}{}", s.velum, if app.nasal { " (n)" } else { "" }),
        ),
        row(
            "vibrato",
            meter(app.vibrato / 0.05, w),
            format!("{:.3}", app.vibrato),
        ),
        row(
            "reverb",
            meter(app.reverb / 0.4, w),
            format!("{:.2}", app.reverb),
        ),
        row(
            "loud",
            meter((app.loud - LOUD_RANGE.0) / (LOUD_RANGE.1 - LOUD_RANGE.0), w),
            if app.engine == synthie_trombone::Engine::Physical {
                format!("{:.0} cmH2O", app.loud)
            } else {
                "physical only".to_string()
            },
        ),
        Line::from(vec![
            Span::styled(" wobble ", fg(DIM)),
            Span::styled(format!("{:<4}", on_off(app.wobble)), fg(FG)),
            Span::styled(" drone ", fg(DIM)),
            Span::styled(format!("{:<4}", on_off(app.drone)), fg(FG)),
            Span::styled(" engine ", fg(DIM)),
            Span::styled(format!("{:<8}", engine_name(app.engine)), bold(CYAN)),
            Span::styled(format!(" ×{:.1}", app.rate), fg(DIM)),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

#[allow(clippy::cast_precision_loss)]
fn draw_scope(frame: &mut Frame, area: Rect, app: &App) {
    let s = &app.snap.scope;
    // Trigger on a rising zero crossing so the waveform stands still.
    let show = SCOPE_LEN / 2;
    let start = (1..SCOPE_LEN - show)
        .find(|&i| s[i - 1] < 0.0 && s[i] >= 0.0)
        .unwrap_or(0);
    let gain = 0.9 / app.snap.level.max(0.05);
    let samples: Vec<f64> = s[start..start + show]
        .iter()
        .map(|v| f64::from(v * gain))
        .collect();
    let title = format!("scope  ·  {:.0} kHz", app.sample_rate() / 1000.0);
    let mut lvl = vec![Span::raw(" ")];
    lvl.extend(vu(app.snap.level, 12));
    let block = panel(&title, false).title_bottom(Line::from(lvl));
    let canvas = Canvas::default()
        .block(block)
        .background_color(BG)
        .marker(Marker::Braille)
        .x_bounds([0.0, show as f64])
        .y_bounds([-1.0, 1.0])
        .paint(move |ctx| {
            ctx.draw(&CLine::new(0.0, 0.0, show as f64, 0.0, FAINT));
            for (i, w) in samples.windows(2).enumerate() {
                let x = i as f64;
                ctx.draw(&CLine::new(
                    x,
                    w[0],
                    x + 1.0,
                    w[1],
                    theme::gradient(i as f32 / show as f32),
                ));
            }
        });
    frame.render_widget(canvas, area);
}

/// Columns of the speech panel's prompt, `" ❯ "`, before the text.
const PROMPT_WIDTH: usize = 3;

fn draw_speech(frame: &mut Frame, area: Rect, app: &mut App) {
    let focused = app.mode == Mode::Type;
    let inner = usize::from(area.width.saturating_sub(2));
    // The text scrolls sideways under the prompt to keep the cursor in
    // view; the column left at the right end shows `…` while there is more
    // text past it, and the prompt's space shows it for text before.
    let room = inner.saturating_sub(PROMPT_WIDTH + 1);
    let window = app.line.window(room);
    let (cursor, len) = (app.line.cursor(), app.line.len());
    let more = |b: bool| if b { "…" } else { " " };
    let mut prompt = vec![
        Span::styled(" ❯", bold(if focused { YELLOW } else { DIM })),
        Span::styled(more(window.start > 0), fg(DIM)),
    ];
    let text = app.line.text();
    let shown: String = text.chars().skip(window.start).take(window.len()).collect();
    if focused {
        // The cursor is the cell it sits on, in reverse.
        let at = cursor - window.start;
        let split = |k: usize| shown.char_indices().nth(k).map_or(shown.len(), |(b, _)| b);
        let (a, b) = (split(at), split(at + 1));
        let under = if cursor < len { &shown[a..b] } else { " " };
        prompt.push(Span::styled(shown[..a].to_string(), fg(FG)));
        prompt.push(Span::styled(
            under.to_string(),
            Style::default().fg(BG).bg(YELLOW),
        ));
        prompt.push(Span::styled(
            shown[b.min(shown.len())..].to_string(),
            fg(FG),
        ));
    } else {
        prompt.push(Span::styled(shown, fg(FG)));
    }
    if window.end < len {
        let used: usize = Line::from(prompt.clone()).width();
        prompt.push(Span::raw(
            " ".repeat((PROMPT_WIDTH + room).saturating_sub(used)),
        ));
        prompt.push(Span::styled("…", fg(DIM)));
    }
    if app.line.is_empty() && !focused {
        prompt.push(Span::styled(
            "press Tab and type or paste something to say",
            fg(FAINT),
        ));
    }
    let strip = phoneme_strip(app, inner);
    // How long the text is, once it no longer fits on the line.
    let title = if app.line.is_full() {
        format!("speak  ·  full: {MAX_CHARS} characters")
    } else if window.len() < len {
        format!("speak  ·  {len} / {MAX_CHARS}")
    } else {
        "speak".to_string()
    };
    frame.render_widget(
        Paragraph::new(vec![Line::from(prompt), strip]).block(panel(&title, focused)),
        area,
    );
}

/// The phonemes of the last utterance, the one being spoken highlighted,
/// in `width` columns. A long utterance runs past the panel: the strip
/// then follows the speech, with the sound being spoken a third of the way
/// in, and only what is in view is laid out.
fn phoneme_strip(app: &App, width: usize) -> Line<'static> {
    let spoken = &app.spoken;
    let current = app.snap.speaking;
    // Columns a phoneme takes, with its space (a pause is a `·`).
    let label = |sym: &str| if sym == "SIL" { 1 } else { sym.len() } + 1;
    let start = current.map_or(0, |c| {
        let at = spoken.partition_point(|(_, r)| r.end <= c);
        let mut first = at.min(spoken.len());
        let mut before = 0;
        while first > 0 && before + label(spoken[first - 1].0) <= width / 3 {
            first -= 1;
            before += label(spoken[first].0);
        }
        first
    });
    let mut strip = vec![Span::styled(
        if start > 0 { "  …" } else { "   " },
        fg(FAINT),
    )];
    let mut used = 3;
    for (sym, range) in &spoken[start..] {
        if used >= width {
            break;
        }
        let style = if current.is_some_and(|c| range.contains(&c)) {
            Style::default()
                .fg(BG)
                .bg(MAGENTA)
                .add_modifier(Modifier::BOLD)
        } else if *sym == "SIL" {
            fg(FAINT)
        } else {
            fg(PURPLE)
        };
        used += label(sym);
        let sym = if *sym == "SIL" { "·" } else { sym };
        strip.push(Span::styled(format!("{sym} "), style));
    }
    Line::from(strip)
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let line = match app.mode {
        Mode::Play => {
            let l = key_hints(&[
                ("a‥;", "sing"),
                ("Space", "stop"),
                ("z x", "octave"),
                ("←→↑↓", arrows_hint(app)),
                ("[ ]", "tense"),
                ("n", "nasal"),
                ("b", "pop"),
                ("i", "drone"),
                ("Tab", "type/paste"),
                ("q", "quit"),
            ]);
            let mut v = vec![Span::styled(" vowels ", fg(DIM))];
            for (k, p) in VOWEL_KEYS {
                v.push(keycap_owned(k));
                v.push(Span::styled(format!(" {}  ", p.symbol()), fg(DIM)));
            }
            let settings = key_hints(&[
                ("F5", "record"),
                ("v V", "voice"),
                ("- =", "vibrato"),
                ("m", "wobble"),
                ("r", "reverb"),
                (", .", "speech rate"),
                ("c", "clear"),
                ("\\", "engine"),
                ("/", "formant lock"),
            ]);
            vec![l, Line::from(v), settings]
        }
        Mode::Type => vec![
            key_hints(&[
                ("Enter", "speak"),
                ("↑↓", "history"),
                ("[HH AH0 L OW1]", "ARPAbet"),
                ("Esc", "stop / back"),
            ]),
            key_hints(&[
                ("←→ Home End", "move"),
                ("Ctrl ←→", "word"),
                ("Ctrl W", "delete word"),
                ("Ctrl U", "clear"),
            ]),
            key_hints(&[
                ("Ctrl Shift V", "paste"),
                ("Shift middle-click", "paste the selection"),
                ("F5", "record"),
            ]),
        ],
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn keycap_owned(c: char) -> Span<'static> {
    let style = keycap("").style;
    Span::styled(format!(" {c} "), style)
}

// ----- Formant Lock -------------------------------------------------------

/// What the arrow keys do, for the footer.
fn arrows_hint(app: &App) -> &'static str {
    if app.formant_lock.is_some() {
        "F2 F1"
    } else {
        "tongue"
    }
}

/// The panel beside the persona: the scope, or with Formant Lock on the
/// vowel chart; either way with the button that switches between them.
fn draw_sound(frame: &mut Frame, area: Rect, app: &mut App) {
    if app.formant_lock.is_some() {
        draw_chart(frame, area, app);
    } else {
        app.chart_area = Rect::default();
        draw_scope(frame, area, app);
    }
    lock_button(frame, area, app);
}

/// The Formant Lock button, on the right of the panel's bottom edge.
fn lock_button(frame: &mut Frame, area: Rect, app: &mut App) {
    let label = if app.formant_lock.is_some() {
        "[/ scope]"
    } else {
        "[/ vowels]"
    };
    // Room for what the bottom edge shows on the left.
    let label = if area.width >= 26 {
        label
    } else if area.width >= 6 {
        "[/]"
    } else {
        return;
    };
    let width = u16::try_from(label.chars().count()).unwrap_or(0);
    let r = Rect::new(
        area.right().saturating_sub(width + 1),
        area.bottom().saturating_sub(1),
        width,
        1,
    );
    frame.render_widget(Paragraph::new(Span::styled(label, bold(CYAN))), r);
    app.controls.push((r, Control::FormantLock));
}

/// Hz as a short axis label: `590`, `2.6k`.
fn short_hz(hz: f32) -> String {
    if hz >= 1000.0 {
        format!("{:.1}k", hz / 1000.0)
    } else {
        format!("{hz:.0}")
    }
}

/// The vowel chart of Formant Lock: F2 across (high on the left), F1 down
/// (low at the top), on logarithmic axes spanning what the voice's tract
/// can reach. The reachable region is shaded and outlined, the vowels are
/// marked where this voice makes them, and `◆` is where the voice is.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn draw_chart(frame: &mut Frame, area: Rect, app: &mut App) {
    let range = app.chart_range();
    let length = app.settings.tract_length;
    let now = app.lock_formants();
    // The longest title that fits between the corners, glyph and padding.
    let room = usize::from(area.width.saturating_sub(6));
    let title = now.map_or_else(
        || "formant lock".to_string(),
        |(f1, f2)| {
            let fits = [
                format!("F1 {f1:.0}  F2 {f2:.0} Hz"),
                format!("F1 {f1:.0} F2 {f2:.0}"),
                format!("{f1:.0}/{f2:.0}"),
            ];
            fits.iter()
                .find(|t| t.chars().count() <= room)
                .cloned()
                .unwrap_or_else(|| format!("{f1:.0}"))
        },
    );
    let mut block = panel(&title, true);
    // The F2 axis on the bottom edge, where the lock button leaves room.
    if area.width >= 28 {
        let axis = format!(" F2 {}←→{} ", short_hz(range.f2.1), short_hz(range.f2.0));
        block = block.title_bottom(Line::from(Span::styled(axis, fg(DIM))));
    }
    let inner = block.inner(area);
    if inner.width < 8 || inner.height < 3 {
        // Too small to draw: the title still says where the voice is.
        app.chart_area = Rect::default();
        frame.render_widget(block, area);
        return;
    }
    app.chart_area = inner;

    // The reachable region, sampled once per Braille dot.
    let (w, h) = (usize::from(inner.width) * 2, usize::from(inner.height) * 4);
    let mut reach = vec![false; w * h];
    for j in 0..h {
        for i in 0..w {
            let (f1, f2) = range.hz((i as f32 + 0.5) / w as f32, (j as f32 + 0.5) / h as f32);
            reach[j * w + i] = inverse::reachable_for(f1, f2, length);
        }
    }
    let at = |i: isize, j: isize| {
        i >= 0
            && j >= 0
            && (i as usize) < w
            && (j as usize) < h
            && reach[j as usize * w + i as usize]
    };
    let (mut edge, mut fill) = (Vec::new(), Vec::new());
    for j in 0..h {
        for i in 0..w {
            if !reach[j * w + i] {
                continue;
            }
            let p = (
                (i as f64 + 0.5) / w as f64,
                1.0 - (j as f64 + 0.5) / h as f64,
            );
            let (si, sj) = (i as isize, j as isize);
            if !(at(si - 1, sj) && at(si + 1, sj) && at(si, sj - 1) && at(si, sj + 1)) {
                edge.push(p);
            } else if i % 2 == 0 && j % 4 == 1 {
                fill.push(p);
            }
        }
    }
    let canvas = Canvas::default()
        .block(block)
        .background_color(BG)
        .marker(Marker::Braille)
        .x_bounds([0.0, 1.0])
        .y_bounds([0.0, 1.0])
        .paint(move |ctx| {
            ctx.draw(&Points {
                coords: &fill,
                color: FAINT,
            });
            ctx.draw(&Points {
                coords: &edge,
                color: PURPLE,
            });
        });
    frame.render_widget(canvas, area);

    // Vowel marks, then the voice on top, in the cells the mouse finds
    // them in.
    let marks = app.chart_marks();
    let cursor = now.and_then(|(f1, f2)| app.chart_cell(f1, f2));
    let buf = frame.buffer_mut();
    for (name, _, cell) in marks {
        buf[cell].set_symbol(name).set_style(bold(FG));
    }
    if let Some(cell) = cursor {
        buf[cell].set_symbol("◆").set_style(bold(YELLOW));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{Cmd, Link};
    use crossbeam_channel::{Receiver, bounded};
    use ratatui::{Terminal, backend::TestBackend};
    use synthie_trombone::speech::{Prosody, Speaker, Utterance};
    use synthie_trombone::{Constriction, Phoneme, Trombone};

    /// Render one frame of a live-looking voice to text, so layout changes
    /// can be eyeballed with `cargo test -- --nocapture`.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn renders_a_frame() {
        let (cmd, cmd_rx) = bounded::<Cmd>(1024);
        let (_snap_tx, snapshots) = bounded(1);
        let (_trash_tx, trash) = bounded(1);
        let link = Link {
            cmd,
            snapshots,
            trash,
            sample_rate: 48_000.0,
            microphone: false,
            microphone_name: None,
            recorder: None,
        };
        let mut app = App::new(link, true, synthie_trombone::Engine::Classic);
        app.preset = synthie_trombone::Preset::Critter;

        // Drive a real voice into an interesting state.
        let mut voice = Trombone::new(48_000.0);
        voice.articulate_phoneme(Phoneme::Aa);
        voice.set_velum(0.6);
        voice.set_constriction(2, Some(Constriction::new(36.0, 0.5)));
        voice.note_on_midi(45.0);
        let mut buf = vec![0.0; 24_000];
        voice.render(&mut buf);
        let mut snap = crate::audio::Snapshot::default();
        snap.diameters = *voice.diameters();
        snap.targets = *voice.target_diameters();
        snap.amplitudes = *voice.amplitudes();
        snap.nose = *voice.nose_diameters();
        snap.nose_amplitudes = *voice.nose_amplitudes();
        snap.velum = voice.velum();
        snap.tongue = voice.tongue();
        snap.intensity = voice.intensity();
        snap.frequency = voice.frequency();
        snap.constrictions[2] = voice.constriction(2);
        snap.scope.copy_from_slice(&buf[buf.len() - SCOPE_LEN..]);
        snap.level = 0.4;
        snap.speaking = Some(3);
        app.snap = snap;
        app.line.set("hello there");
        let u = Utterance::from_text("Hello, pink trombone!", &Prosody::default());
        for (k, s) in u.segments().iter().enumerate() {
            app.spoken.push((s.phoneme.symbol(), k..k + 1));
        }
        let _ = Speaker::new();
        drop(cmd_rx);

        let mut term = Terminal::new(TestBackend::new(120, 34)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push('\n');
        }
        println!("{text}");
        assert!(text.contains("vocal tract"));
        assert!(app.tract_area.width > 0);
    }

    /// A frame as text.
    fn frame_text(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push('\n');
        }
        text
    }

    /// The vowel chart of Formant Lock at the usual size and in smaller
    /// terminals; look at it with `cargo test renders_the_vowel_chart --
    /// --nocapture`.
    #[test]
    fn renders_the_vowel_chart() {
        let (cmd, cmd_rx) = bounded::<Cmd>(1024);
        let (_snap_tx, snapshots) = bounded(1);
        let (_trash_tx, trash) = bounded(1);
        let link = Link {
            cmd,
            snapshots,
            trash,
            sample_rate: 48_000.0,
            microphone: false,
            microphone_name: None,
            recorder: None,
        };
        let mut app = App::new(link, true, synthie_trombone::Engine::Physical);
        app.key(synthie_ui::Key::press(synthie_ui::KeyCode::Char('3')));
        app.key(synthie_ui::Key::press(synthie_ui::KeyCode::Char('/')));
        // Let a real voice take up the shape, so the tract view shows it.
        let shape = app.formant_lock.expect("formant lock is on");
        let mut voice = Trombone::new(48_000.0);
        voice.set_engine(synthie_trombone::Engine::Physical);
        voice.set_shape(&shape);
        voice.note_on_midi(45.0);
        let mut buf = vec![0.0; 24_000];
        voice.render(&mut buf);
        app.snap.diameters = *voice.diameters();
        app.snap.targets = *voice.target_diameters();
        app.snap.tongue = voice.tongue();
        drop(cmd_rx);

        for (w, h) in [(120, 34), (100, 30), (80, 24), (60, 20)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| draw(f, &mut app)).unwrap();
            let text = frame_text(&term);
            println!("{w}×{h}, chart {:?}\n{text}", app.chart_area);
            let lock = app
                .controls
                .iter()
                .find(|(_, c)| *c == Control::FormantLock);
            assert!(lock.is_some(), "{w}×{h}: no lock button");
            if w >= 80 {
                // Down to 80 columns the persona panel says a turn is heard.
                assert!(text.contains("persona  ·  drag to hear"), "{w}×{h}: {text}");
            }
            if (w, h) == (120, 34) {
                assert_eq!(app.chart_area, Rect::new(93, 17, 26, 9));
                assert!(text.contains("F1 ") && text.contains("[/ scope]"), "{text}");
                // The marks are drawn in the cells a click finds them in.
                let marks = app.chart_marks();
                assert!(marks.len() >= 8, "{marks:?}");
                let buf = term.backend().buffer();
                for (name, _, cell) in marks {
                    let shown = buf[cell].symbol();
                    assert!(shown == name || shown == "◆", "{name} at {cell:?}: {shown}");
                }
            }
            let a = app.chart_area;
            assert!(a.right() <= w && a.bottom() <= h, "{w}×{h}: {a:?}");
        }
        // Off again, the scope is back and the chart takes no clicks.
        app.key(synthie_ui::Key::press(synthie_ui::KeyCode::Char('/')));
        let mut term = Terminal::new(TestBackend::new(120, 34)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let text = frame_text(&term);
        println!("{text}");
        assert!(
            text.contains("scope") && text.contains("[/ vowels]"),
            "{text}"
        );
        assert_eq!(app.chart_area, Rect::default());
    }

    /// While a take is recorded the header ends in its marker, even at
    /// 80×24, and nothing else on screen moves.
    #[test]
    fn the_recording_marker_fits_at_80_by_24() {
        use crate::app::RECORD_KEY;
        use crate::record::{Clock, Recorder};
        let dir = std::env::temp_dir().join(format!("synthie-takes-{}-marker", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // A recorder whose writer never reads: the files are created, and
        // nothing is written to them.
        let (events, _unread) = bounded(16);
        let (_saved_tx, saved) = bounded(1);
        let (cmd, _cmd_rx) = bounded::<Cmd>(1024);
        let (_snap_tx, snapshots) = bounded(1);
        let (_trash_tx, trash) = bounded(1);
        let link = Link {
            cmd,
            snapshots,
            trash,
            sample_rate: 48_000.0,
            microphone: false,
            microphone_name: None,
            recorder: Some(Recorder::new(dir.clone(), events, saved)),
        };
        let mut app = App::new(link, true, synthie_trombone::Engine::Classic);
        let rows = |app: &mut App, w: u16, h: u16| -> Vec<String> {
            app.status = "hold a piano key to sing".into();
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| draw(f, app)).unwrap();
            frame_text(&term).lines().map(str::to_string).collect()
        };
        let idle = rows(&mut app, 80, 24);
        assert!(!idle[0].contains("REC"), "{}", idle[0]);
        assert!(idle.iter().any(|r| r.contains(" F5  record")), "{idle:?}");

        app.key(synthie_ui::Key::press(RECORD_KEY));
        assert_eq!(app.recording(), Some("take-001"));
        app.snap.take = Some(Clock {
            samples: 3 * 48_000 + 2_400,
            dropped: 0,
        });
        let rec = rows(&mut app, 80, 24);
        println!("{}", rec.join("\n"));
        assert!(rec[0].ends_with(" ● REC take-001 0:03 "), "{}", rec[0]);
        assert_eq!(rec[0].chars().count(), 80);
        assert!(
            rec[0].contains("P I N K") && rec[0].contains(" PLAY "),
            "{}",
            rec[0]
        );
        assert_eq!(idle[1..], rec[1..], "the layout moved");
        // Anything lost on the way to the disk is flagged.
        app.snap.take = Some(Clock {
            samples: 75 * 48_000,
            dropped: 2,
        });
        let lost = rows(&mut app, 80, 24);
        assert!(lost[0].ends_with(" ● REC take-001 1:15 ! "), "{}", lost[0]);
        // Narrower, the marker shortens rather than cover the mode badge.
        for w in [70, 64, 60] {
            let narrow = rows(&mut app, w, 24);
            assert!(narrow[0].contains(" PLAY "), "{w}: {}", narrow[0]);
            assert!(narrow[0].contains("●"), "{w}: {}", narrow[0]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An app with no sound behind it, and the queue to it.
    fn quiet_app() -> (App, Receiver<Cmd>) {
        let (cmd, cmd_rx) = bounded::<Cmd>(1024);
        let (_snap_tx, snapshots) = bounded(1);
        let (_trash_tx, trash) = bounded(1);
        let link = Link {
            cmd,
            snapshots,
            trash,
            sample_rate: 48_000.0,
            microphone: false,
            microphone_name: None,
            recorder: None,
        };
        (
            App::new(link, true, synthie_trombone::Engine::Classic),
            cmd_rx,
        )
    }

    /// The row of the text line, the cursor's cell on it (the one in
    /// reverse) and what the row shows.
    fn text_row(term: &Terminal<TestBackend>) -> (u16, u16, String) {
        let buf = term.backend().buffer();
        for y in 0..buf.area.height {
            let row: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect();
            if row.contains("❯") {
                let cursor = (0..buf.area.width)
                    .find(|&x| buf[(x, y)].bg == YELLOW)
                    .expect("no cursor on the text line");
                return (y, cursor, row);
            }
        }
        panic!("no text line:\n{}", frame_text(term));
    }

    /// A long pasted text at 80×24: the line scrolls so that the cursor is
    /// always in view, wherever it goes, and says how long the text is.
    #[test]
    fn a_long_text_scrolls_with_the_cursor_at_80_by_24() {
        use synthie_ui::{Key, KeyCode};
        let (mut app, _rx) = quiet_app();
        let words: Vec<String> = (0..120).map(|k| format!("w{k}")).collect();
        app.paste(&words.join(" "));
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let mut look = |app: &mut App| {
            term.draw(|f| draw(f, app)).unwrap();
            let (y, x, row) = text_row(&term);
            let under = term.backend().buffer()[(x, y)].symbol().to_string();
            (x, under, row, frame_text(&term))
        };
        // At the end: the cursor after the last word, the start cut off.
        let (x, under, row, frame) = look(&mut app);
        println!("{frame}");
        assert!(row.contains("❯…") && row.contains("w119"), "{row}");
        assert!(!row.contains(" w1 "), "{row}");
        assert_eq!(under, " ");
        assert!(x < 79, "{x}");
        assert!(
            frame.contains(&format!("speak  ·  {} / ", app.line.len())),
            "{frame}"
        );
        // Home: the start in view, the rest cut off at the right.
        app.terminal_key(
            crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Home),
            false,
        );
        let (_, under, row, _) = look(&mut app);
        assert!(
            row.contains("❯ w0 w1 w2") && row.trim_end().ends_with("…│"),
            "{row}"
        );
        assert_eq!(under, "w");
        // Word by word to the right: the cursor never leaves the line, and
        // the character under it is the one it is on.
        let word_right = Key {
            ctrl: true,
            ..Key::press(KeyCode::Right)
        };
        for k in 0..120 {
            app.key(word_right);
            let (x, under, row, _) = look(&mut app);
            assert!((3..79).contains(&x), "word {k}: cursor at {x} in {row}");
            let want = app.line.text().chars().nth(app.line.cursor());
            assert_eq!(
                under,
                want.map_or(" ".into(), String::from),
                "word {k}: {row}"
            );
        }
        // And back with Backspace held down to nothing.
        for _ in 0..app.line.len() {
            app.key(Key::press(KeyCode::Backspace));
            let (x, ..) = look(&mut app);
            assert!((3..79).contains(&x));
        }
        let (_, _, row, frame) = look(&mut app);
        assert!(row.contains("❯  "), "{row}");
        assert!(frame.contains("◆ speak "), "{frame}");
    }

    /// While a long text is spoken, the strip of phonemes follows the one
    /// being said.
    #[test]
    fn the_phoneme_strip_follows_long_speech() {
        let (mut app, rx) = quiet_app();
        app.paste("One sentence is said first. Then another one is said. A third follows them.");
        app.key(synthie_ui::Key::press(synthie_ui::KeyCode::Enter));
        let segments = rx
            .try_iter()
            .find_map(|c| match c {
                Cmd::Speak(u, _) => Some(u.segments().len()),
                _ => None,
            })
            .unwrap();
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        for k in (0..segments).step_by(7) {
            app.snap.speaking = Some(k);
            term.draw(|f| draw(f, &mut app)).unwrap();
            let buf = term.backend().buffer();
            let lit: Vec<(u16, u16)> = (0..24)
                .flat_map(|y| (0..80).map(move |x| (x, y)))
                .filter(|&c| buf[c].bg == MAGENTA)
                .collect();
            assert!(!lit.is_empty(), "segment {k} of {segments} is not in view");
            let sym = app
                .spoken
                .iter()
                .find(|(_, r)| r.contains(&k))
                .map(|(s, _)| if *s == "SIL" { "·" } else { s })
                .unwrap();
            let shown: String = lit.iter().map(|&c| buf[c].symbol()).collect();
            assert_eq!(shown.trim(), sym, "segment {k}");
        }
    }

    /// Rows of the tract view with dots of one wall in column `col`, the
    /// wall drawn as `draw_tract` draws it (`side` 1 for the upper wall,
    /// -1 for the lower).
    #[allow(clippy::cast_precision_loss)]
    fn wall_rows(area: Rect, diameters: &[f32; N], side: f64, col: u16) -> Vec<u16> {
        let canvas = Canvas::default()
            .marker(Marker::Braille)
            .x_bounds(X_BOUNDS)
            .y_bounds(Y_BOUNDS)
            .paint(|ctx| {
                for i in 0..N - 1 {
                    let (a, b) = (f64::from(diameters[i]), f64::from(diameters[i + 1]));
                    let (x0, x1) = (i as f64, (i + 1) as f64);
                    ctx.draw(&CLine::new(x0, side * a / 2.0, x1, side * b / 2.0, CYAN));
                }
            });
        let mut buf = ratatui::buffer::Buffer::empty(area);
        ratatui::widgets::Widget::render(canvas, area, &mut buf);
        (area.y..area.bottom())
            .filter(|&r| buf[(col, r)].symbol() != " ")
            .collect()
    }

    /// Draw `app` in a `size` terminal: the tongue handle is drawn once, in
    /// its cell, between the walls there, and a press on it takes hold of
    /// the tongue without moving it.
    fn check_handle(app: &mut App, rx: &Receiver<Cmd>, size: (u16, u16), what: &str) {
        use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
        let mut term = Terminal::new(TestBackend::new(size.0, size.1)).unwrap();
        term.draw(|f| draw(f, app)).unwrap();
        let buf = term.backend().buffer();
        let a = app.tract_area;
        let shown: Vec<(u16, u16)> = (a.y..a.bottom())
            .flat_map(|r| (a.x..a.right()).map(move |c| (c, r)))
            .filter(|&p| buf[p].symbol() == "◆")
            .collect();
        let handle = app.tongue_cell();
        assert_eq!(shown, Vec::from_iter(handle), "{what}");
        let (col, row) = handle.unwrap();
        let upper = wall_rows(a, &app.snap.diameters, 1.0, col);
        let lower = wall_rows(a, &app.snap.diameters, -1.0, col);
        assert!(
            upper.first().is_some_and(|&u| row >= u),
            "{what}: handle in row {row}, above the upper wall {upper:?}"
        );
        assert!(
            lower.last().is_some_and(|&l| row <= l),
            "{what}: handle in row {row}, below the lower wall {lower:?}"
        );
        while rx.try_recv().is_ok() {}
        for kind in [
            MouseEventKind::Down(MouseButton::Left),
            MouseEventKind::Up(MouseButton::Left),
        ] {
            app.mouse(MouseEvent {
                kind,
                column: col,
                row,
                modifiers: KeyModifiers::NONE,
            });
        }
        let cmds: Vec<Cmd> = rx.try_iter().collect();
        let [Cmd::Tongue(i, d)] = cmds[..] else {
            panic!("{what}: a press on the handle sent {cmds:?}");
        };
        let (ti, td) = app.snap.tongue;
        assert!(
            (i - ti).abs() < 1e-3 && (d - td).abs() < 1e-3,
            "{what}: the press moved the tongue to {:?}",
            (i, d)
        );
    }

    /// Wherever the voice puts the tongue, for any vowel, body size and
    /// engine and at any terminal size, the handle is drawn between the
    /// walls of the mouth, and a press on it takes hold of the tongue.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn the_tongue_handle_stays_in_the_mouth() {
        use synthie_trombone::{Engine, Persona, Preset, TONGUE_DIAMETER, TONGUE_INDEX};

        // The shortest and longest tracts the size dial gives any voice.
        let (_, small, large) = DIALS[0];
        let lengths: Vec<f32> = Preset::ALL
            .iter()
            .flat_map(|p| {
                [small, large].map(|size| {
                    let persona = Persona {
                        size,
                        ..Persona::NEUTRAL
                    };
                    p.settings().with_persona(&persona).tract_length
                })
            })
            .collect();
        let shortest = lengths.iter().copied().fold(f32::INFINITY, f32::min);
        let longest = lengths.iter().copied().fold(0.0, f32::max);
        // Every tongue position on a grid over its whole range, and every
        // vowel the keys and the tract view set.
        let (i0, i1) = (*TONGUE_INDEX.start(), *TONGUE_INDEX.end());
        let (d0, d1) = (*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end());
        let mut tongues = Vec::new();
        for k in 0..=17 {
            for j in 0..=8 {
                tongues.push((
                    i0 + (i1 - i0) * k as f32 / 17.0,
                    d0 + (d1 - d0) * j as f32 / 8.0,
                ));
            }
        }
        let vowels: Vec<Phoneme> = VOWEL_KEYS
            .iter()
            .map(|&(_, v)| v)
            .chain(LANDMARKS.iter().map(|&(_, v)| v))
            .collect();

        let mut checked = 0;
        for engine in [Engine::Classic, Engine::Physical] {
            let (cmd, rx) = bounded::<Cmd>(1024);
            let (_snap_tx, snapshots) = bounded(1);
            let (_trash_tx, trash) = bounded(1);
            let link = Link {
                cmd,
                snapshots,
                trash,
                sample_rate: 48_000.0,
                microphone: false,
                microphone_name: None,
                recorder: None,
            };
            let mut app = App::new(link, true, engine);
            for length in [shortest, 1.0, longest] {
                let shapes = tongues
                    .iter()
                    .map(|&t| (Some(t), None))
                    .chain(vowels.iter().map(|&v| (None, Some(v))));
                for (tongue, vowel) in shapes {
                    let mut voice = Trombone::new(48_000.0);
                    voice.set_engine(engine);
                    voice.set_tract_length(length);
                    if let Some(v) = vowel {
                        voice.articulate_phoneme(v);
                    }
                    if let Some((i, d)) = tongue {
                        voice.set_tongue(i, d);
                    }
                    voice.settle();
                    app.snap.diameters = *voice.diameters();
                    app.snap.targets = *voice.target_diameters();
                    app.snap.tongue = voice.tongue();
                    for (w, h) in [(80, 24), (120, 34), (200, 60)] {
                        let what = format!(
                            "{engine:?}, length {length}, {tongue:?} {vowel:?}, {w}×{h}, tongue at {:?}",
                            app.snap.tongue
                        );
                        check_handle(&mut app, &rx, (w, h), &what);
                        checked += 1;
                    }
                }
            }
        }
        println!("tongue handle checked in {checked} frames");
    }
}
