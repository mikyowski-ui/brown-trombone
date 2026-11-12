//! `synthie-trombone`: play a Pink Trombone vocal tract in the terminal,
//! type text and hear it spoken, or sing the melody of a MIDI file.

mod app;
mod audio;
mod line;
mod record;
mod sing;
mod take;
mod ui;

#[cfg(test)]
mod screenshots;
#[cfg(test)]
mod take_tests;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
        supports_keyboard_enhancement,
    },
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{io, panic, path::PathBuf, time::Duration};
use synthie_trombone::{Engine, Persona, Preset, VoiceSettings};

/// Pink Trombone in the terminal: play a vocal tract, or have it speak and sing.
#[derive(Debug, Parser)]
#[command(name = "synthie-trombone", version, about)]
struct Cli {
    /// Sound model: `classic` is Pink Trombone as written; `physical` is
    /// the same tract in real units, with self-oscillating vocal folds,
    /// lung pressure, yielding walls and airflow-driven noise.
    #[arg(long, global = true, default_value = "classic", value_parser = parse_engine)]
    engine: Engine,
    /// Voice to start with (a preset name; `v` cycles them in the app).
    #[arg(long, value_parser = parse_voice)]
    voice: Option<Preset>,
    /// Voice profile to lay over the voice: a JSON file as written by the
    /// tuner or by `match_voice.py`.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Body size relative to the voice, 0.6 to 1.4 (1 leaves it alone).
    #[arg(long, default_value_t = Persona::NEUTRAL.size)]
    size: f32,
    /// Vocal effort, 0 (murmur) to 1 (calling out).
    #[arg(long, default_value_t = Persona::NEUTRAL.effort)]
    effort: f32,
    /// Breathiness, 0 (pressed) to 1 (breathy).
    #[arg(long, default_value_t = Persona::NEUTRAL.breathiness)]
    breathiness: f32,
    /// Clarity of articulation, 0 (casual) to 1 (careful).
    #[arg(long, default_value_t = Persona::NEUTRAL.clarity)]
    clarity: f32,
    /// Copy what the microphone hears: pitch, vowels and hiss (physical
    /// engine). Wear headphones, or the voice will hear itself; `` ` ``
    /// switches it off and on while playing.
    #[arg(long)]
    mimic: bool,
    /// Microphone for `--mimic`: part of an input device's name (the
    /// system's default, then a sound server, are tried otherwise).
    #[arg(long, requires = "mimic")]
    mic: Option<String>,
    /// Directory for takes: F5 starts and stops recording one (the
    /// commands as the voice carried them out, and its sound), numbered
    /// take-001, take-002, … Created on the first take.
    #[arg(long, value_name = "DIR", default_value = "takes")]
    record: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

fn parse_voice(name: &str) -> Result<Preset, String> {
    Preset::from_name(name).ok_or_else(|| {
        let names: Vec<_> = Preset::ALL.iter().map(|p| p.name()).collect();
        format!("unknown voice {name:?}; try one of {}", names.join(", "))
    })
}

impl Cli {
    /// The voice the flags describe: preset, base settings (the profile
    /// laid over the preset) and persona dials.
    fn voice(&self) -> Result<(Preset, VoiceSettings, Persona)> {
        let preset = self.voice.unwrap_or(Preset::Deep);
        let mut base = preset.settings().with_engine(self.engine);
        if let Some(path) = &self.profile {
            let json = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            base = base
                .with_profile(&json)
                .with_context(|| format!("in {}", path.display()))?;
        }
        let persona = Persona {
            size: self.size,
            effort: self.effort,
            breathiness: self.breathiness,
            clarity: self.clarity,
        };
        Ok((preset, base, persona))
    }
}

fn parse_engine(name: &str) -> Result<Engine, String> {
    match name.to_ascii_lowercase().as_str() {
        "classic" => Ok(Engine::Classic),
        "physical" => Ok(Engine::Physical),
        other => Err(format!("unknown engine {other:?}; try classic or physical")),
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Sing the melody of a MIDI file into a WAV file.
    Sing(sing::SingArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Sing(args)) => sing::run(&args, cli.engine),
        None => {
            let (preset, mut base, persona) = cli.voice()?;
            if cli.mimic {
                // The map from formants to shapes is the physical engine's.
                base.engine = Engine::Physical;
            }
            // A profile may name its own sound model.
            tui(
                base.engine,
                (preset, base, persona),
                cli.mimic,
                cli.mic.as_deref(),
                cli.record,
            )
        }
    }
}

fn tui(
    engine: Engine,
    (preset, base, persona): (Preset, VoiceSettings, Persona),
    mimic: bool,
    microphone: Option<&str>,
    takes: PathBuf,
) -> Result<()> {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore();
        original_hook(info);
    }));

    let (streams, link) = audio::start(mimic, microphone, takes)?;
    let _streams = streams;

    enable_raw_mode()?;
    let result = session(engine, (preset, base, persona), link);
    // Whatever happened, the terminal is given back as it was.
    restore()?;
    result
}

/// The app in the terminal, once it is in raw mode.
fn session(
    engine: Engine,
    (preset, base, persona): (Preset, VoiceSettings, Persona),
    link: audio::Link,
) -> Result<()> {
    let mut stdout = io::stdout();
    // Bracketed paste: a paste arrives as one piece of text, not as keys
    // (where a line break in it would be Enter and speak half of it).
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )?;
    // Key releases (kitty protocol) let notes stop when keys come up.
    // Releases of text keys (the piano letters) are only reported when all
    // keys arrive as escape codes; alternate keys keep Shift+letter typing
    // working in that mode.
    let release = supports_keyboard_enhancement().unwrap_or(false)
        && execute!(
            stdout,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
            )
        )
        .is_ok();

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    let mut app = app::App::new(link, release, engine);
    if (preset, base, persona)
        != (
            Preset::Deep,
            Preset::Deep.settings().with_engine(engine),
            Persona::NEUTRAL,
        )
    {
        app.set_voice(preset, base, persona);
    }
    let result = run(&mut terminal, &mut app);
    // A take still recording is saved while the sound still runs.
    app.finish();

    if release {
        let _ = execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags);
    }
    terminal.show_cursor()?;
    result
}

/// Text keys in one read of the terminal from which on the keys are taken
/// to be a paste the terminal did not bracket: nobody types that fast.
const BURST_KEYS: usize = 3;

/// Whether `events`, read in one go, hold a burst of typed text.
fn is_burst(events: &[Event]) -> bool {
    let typed = events
        .iter()
        .filter(|e| {
            matches!(e, Event::Key(k) if k.kind != KeyEventKind::Release
                && matches!(k.code, KeyCode::Char(_))
                && !k.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT))
        })
        .count();
    typed >= BURST_KEYS
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut app::App) -> Result<()> {
    loop {
        app.tick();
        terminal.draw(|f| ui::draw(f, app))?;
        if event::poll(Duration::from_millis(16))? {
            // Drain everything queued so drags stay responsive.
            let mut events = vec![event::read()?];
            while event::poll(Duration::ZERO)? {
                events.push(event::read()?);
            }
            let burst = is_burst(&events);
            for e in events {
                match e {
                    Event::Key(k) => {
                        if app.terminal_key(k, burst) {
                            return Ok(());
                        }
                    }
                    Event::Mouse(m) => app.mouse(m),
                    Event::Paste(text) => app.paste(&text),
                    _ => {}
                }
            }
        }
    }
}

fn restore() -> Result<()> {
    disable_raw_mode()?;
    execute!(
        io::stdout(),
        DisableBracketedPaste,
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    Ok(())
}
