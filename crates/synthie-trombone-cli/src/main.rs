//! `synthie-trombone`: play a Pink Trombone vocal tract in the terminal,
//! type text and hear it spoken, or sing the melody of a MIDI file.

mod app;
mod audio;
mod line;
mod record;
mod sing;
mod take;
mod telemetry;
mod ui;

// The formant meter of the library's examples, which `replay --trace` uses
// too: the instrument panel shows the same numbers.
#[path = "../../synthie-trombone/examples/support/spectrum.rs"]
mod spectrum;
#[path = "../../synthie-trombone/examples/support/survey.rs"]
mod survey;

#[cfg(test)]
mod cost_tests;
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
use std::io::Write as _;
use std::{io, panic, path::PathBuf, time::Duration};
use synthie_trombone::{Engine, Persona, Preset, Stereo, TongueModel, TractModel, VoiceSettings};

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
    /// The physical engine's vocal tract, laid over the voice after the
    /// profile: `anatomical` (the default's: lip horn, moving larynx, the
    /// measured tube above the folds) or `original` (Pink Trombone's tube
    /// in real units). Left out, the voice keeps its own.
    #[arg(long, value_parser = parse_tract)]
    tract: Option<TractModel>,
    /// The physical engine's tongue body, laid over the voice after the
    /// profile: `cosine` (the default's) or `mri` (the measured modes of
    /// one man's vowels). `--tract original --tongue-model mri` is
    /// `profiles/mri-tongue.json`.
    #[arg(long, value_parser = parse_tongue)]
    tongue_model: Option<TongueModel>,
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
    /// Write every key event as the terminal reports it (time, press,
    /// repeat or release, key) to this file: for finding out why keys
    /// misbehave in a terminal. It holds whatever is typed.
    #[arg(long, value_name = "FILE")]
    key_log: Option<PathBuf>,
    /// Play the physical engine in stereo, as the two ears of a listener a
    /// metre in front of the talker (each port of the tract on its own
    /// path round both heads); `--stereo pair` for a plain microphone pair
    /// instead, for loudspeakers.
    #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "binaural", value_parser = parse_stereo)]
    stereo: Option<Stereo>,
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
        if let Some(t) = self.tract {
            base.tuning = base.tuning.with_tract(t);
        }
        if let Some(t) = self.tongue_model {
            base.tuning = base.tuning.with_tongue(t);
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

fn parse_stereo(name: &str) -> Result<Stereo, String> {
    match name.to_ascii_lowercase().as_str() {
        "binaural" | "ears" => Ok(Stereo::Binaural),
        "pair" => Ok(Stereo::Pair),
        "off" | "mono" => Ok(Stereo::Off),
        other => Err(format!("unknown stereo {other:?}; try binaural or pair")),
    }
}

fn parse_engine(name: &str) -> Result<Engine, String> {
    match name.to_ascii_lowercase().as_str() {
        "classic" => Ok(Engine::Classic),
        "physical" => Ok(Engine::Physical),
        other => Err(format!("unknown engine {other:?}; try classic or physical")),
    }
}

fn parse_tract(name: &str) -> Result<TractModel, String> {
    TractModel::from_name(name)
        .ok_or_else(|| format!("unknown tract {name:?}; try anatomical or original"))
}

fn parse_tongue(name: &str) -> Result<TongueModel, String> {
    TongueModel::from_name(name)
        .ok_or_else(|| format!("unknown tongue model {name:?}; try cosine or mri"))
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
                cli.key_log.as_deref(),
                cli.stereo.unwrap_or(Stereo::Off),
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
    key_log: Option<&std::path::Path>,
    stereo: Stereo,
) -> Result<()> {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore();
        original_hook(info);
    }));

    let (streams, link) = audio::start(mimic, microphone, takes, stereo)?;
    let _streams = streams;

    enable_raw_mode()?;
    let key_log = key_log
        .map(|path| {
            std::fs::File::create(path)
                .with_context(|| format!("creating {}", path.display()))
                .map(io::BufWriter::new)
        })
        .transpose()?;
    let result = session(engine, (preset, base, persona), link, key_log);
    // Whatever happened, the terminal is given back as it was.
    restore()?;
    result
}

/// The app in the terminal, once it is in raw mode.
fn session(
    engine: Engine,
    (preset, base, persona): (Preset, VoiceSettings, Persona),
    link: audio::Link,
    mut key_log: Option<KeyLog>,
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
        app.set_voice(preset, &base, persona);
    }
    if let Some(log) = &mut key_log {
        writeln!(log, "terminal says it reports key releases: {release}")?;
    }
    let result = run(&mut terminal, &mut app, key_log);
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

/// Where `--key-log` writes.
type KeyLog = io::BufWriter<std::fs::File>;

fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut app::App,
    mut key_log: Option<KeyLog>,
) -> Result<()> {
    let start = std::time::Instant::now();
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
                        if let Some(log) = &mut key_log {
                            writeln!(
                                log,
                                "{:9.3} s  {:7?}  {:?}  {:?}",
                                start.elapsed().as_secs_f64(),
                                k.kind,
                                k.code,
                                k.modifiers
                            )?;
                            log.flush()?;
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use synthie_trombone::Tuning;

    fn voice(args: &[&str]) -> VoiceSettings {
        let mut all = vec!["synthie-trombone"];
        all.extend_from_slice(args);
        Cli::try_parse_from(all)
            .expect("flags")
            .voice()
            .expect("a voice")
            .1
    }

    const PROFILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../profiles/mri-tongue.json"
    );

    #[test]
    fn tract_and_tongue_flags_parse() {
        for (flag, value) in [
            ("--tract", "anatomical"),
            ("--tract", "Original"),
            ("--tongue-model", "cosine"),
            ("--tongue-model", "mri"),
            ("--tongue-model", "measured"),
        ] {
            assert!(Cli::try_parse_from(["synthie-trombone", flag, value]).is_ok());
        }
        for (flag, value) in [("--tract", "wide"), ("--tongue-model", "flat")] {
            assert!(Cli::try_parse_from(["synthie-trombone", flag, value]).is_err());
        }
        // The `sing` subcommand's own --tract is the tract length.
        let cli = Cli::try_parse_from(["synthie-trombone", "--tract", "original", "sing", "x.mid"])
            .expect("flags");
        assert_eq!(cli.tract, Some(TractModel::Original));
    }

    #[test]
    fn the_flags_select_what_they_say() {
        let plain = voice(&["--engine", "physical"]);
        // Left out or given as the default's, nothing changes.
        assert_eq!(plain.tuning, Tuning::DEFAULT);
        assert_eq!(
            voice(&[
                "--engine",
                "physical",
                "--tract",
                "anatomical",
                "--tongue-model",
                "cosine"
            ]),
            plain
        );
        let original = voice(&["--engine", "physical", "--tract", "original"]);
        assert_eq!(
            original.tuning,
            Tuning::DEFAULT.with_tract(TractModel::Original)
        );
        assert_eq!(TractModel::of(&original.tuning), TractModel::Original);
        let mri = voice(&["--engine", "physical", "--tongue-model", "mri"]);
        assert_eq!(mri.tuning.anatomy.tongue_model, 1.0);
        assert_eq!(TractModel::of(&mri.tuning), TractModel::Anatomical);
        // Both together are the profile, field for field.
        assert_eq!(
            voice(&[
                "--engine",
                "physical",
                "--tract",
                "original",
                "--tongue-model",
                "mri"
            ]),
            voice(&["--engine", "physical", "--profile", PROFILE])
        );
    }
}
