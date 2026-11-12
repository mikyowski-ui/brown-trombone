//! `synthie-trombone`: play a Pink Trombone vocal tract in the terminal,
//! type text and hear it spoken, or sing the melody of a MIDI file.

mod app;
mod audio;
mod sing;
mod ui;

#[cfg(test)]
mod screenshots;

use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
        supports_keyboard_enhancement,
    },
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{io, panic, time::Duration};
use synthie_trombone::Engine;

/// Pink Trombone in the terminal: play a vocal tract, or have it speak and sing.
#[derive(Debug, Parser)]
#[command(name = "synthie-trombone", version, about)]
struct Cli {
    /// Sound model: `classic` is Pink Trombone as written; `physical` is
    /// the same tract in real units, with self-oscillating vocal folds,
    /// lung pressure, yielding walls and airflow-driven noise.
    #[arg(long, global = true, default_value = "classic", value_parser = parse_engine)]
    engine: Engine,
    #[command(subcommand)]
    command: Option<Command>,
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
        None => tui(cli.engine),
    }
}

fn tui(engine: Engine) -> Result<()> {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore();
        original_hook(info);
    }));

    let (stream, link) = audio::start()?;
    let _stream = stream;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
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
    let result = run(&mut terminal, &mut app);

    if release {
        let _ = execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags);
    }
    restore()?;
    terminal.show_cursor()?;
    result
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut app::App) -> Result<()> {
    loop {
        app.tick();
        terminal.draw(|f| ui::draw(f, app))?;
        if event::poll(Duration::from_millis(16))? {
            // Drain everything queued so drags stay responsive.
            loop {
                match event::read()? {
                    Event::Key(k) => {
                        if app.key(k.into()) {
                            return Ok(());
                        }
                    }
                    Event::Mouse(m) => app.mouse(m),
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
    }
}

fn restore() -> Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}
