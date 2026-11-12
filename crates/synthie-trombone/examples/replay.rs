//! Replay a take recorded in the terminal app (`F5` there): rebuild the
//! voice from the take's header, carry out its commands at the samples the
//! app's audio thread applied them at, and render it offline.
//!
//! ```sh
//! cargo run -p synthie-trombone --example replay --release -- takes/take-007.jsonl \
//!     [--out replay.wav] [--trace trace.tsv]
//! ```
//!
//! The replay runs through the same code as the app's audio thread (this
//! example includes the app's `take.rs`), so it reproduces the take's WAV
//! file sample for sample; it says how closely it did when that file is
//! next to the log. `--out` writes the replay (the dry voice, 32-bit float
//! mono). `--trace` writes a row every 5 ms, tab-separated: the controls'
//! targets, what the tract actually does, the voice, and the first three
//! formants of the tract at that moment (peaks of its transfer function,
//! as `tune_physical` measures them, on the shape of the current 10.7 ms
//! parameter block); the columns are named in its first row.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/survey.rs"]
mod survey;
#[path = "../../synthie-trombone-cli/src/take.rs"]
mod take;

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use synthie_trombone::{BLADE_START, LIP_START, MAX_CONSTRICTIONS, N, TIP_START, Trombone};
use take::{Replay, Take};

/// Time between trace rows, seconds.
const TRACE_STEP: f32 = 0.005;
/// Seconds of impulse response the formant meter analyses (vowel
/// resonances have died away well before).
const FORMANT_SECONDS: f32 = 0.04;
/// Seconds rendered past the last command of a take whose log has no end
/// line and no WAV file to say how long it was.
const TAIL: f32 = 1.0;

struct Args {
    log: PathBuf,
    out: Option<PathBuf>,
    trace: Option<PathBuf>,
}

fn args() -> Result<Args, String> {
    let usage = "usage: replay TAKE.jsonl [--out replay.wav] [--trace trace.tsv]";
    let mut log = None;
    let (mut out, mut trace) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = Some(it.next().ok_or(usage)?.into()),
            "--trace" => trace = Some(it.next().ok_or(usage)?.into()),
            "-h" | "--help" => return Err(usage.into()),
            _ if log.is_none() && !a.starts_with("--") => log = Some(a.into()),
            _ => return Err(format!("unexpected {a:?}\n{usage}")),
        }
    }
    Ok(Args {
        log: log.ok_or(usage)?,
        out,
        trace,
    })
}

/// The narrowest cell of `d` in `range`, and its diameter.
fn narrowest(d: &[f32; N], range: std::ops::Range<usize>) -> (usize, f32) {
    range
        .map(|k| (k, d[k]))
        .fold((0, f32::INFINITY), |a, b| if b.1 < a.1 { b } else { a })
}

/// Formants of the voice's tract as it is, remembered while its shape
/// stays the same (the tract only changes shape once per parameter block).
struct Formants {
    meter: survey::Meter,
    key: Vec<u32>,
    last: Option<[f32; 3]>,
}

impl Formants {
    fn of(&mut self, voice: &Trombone) -> Option<[f32; 3]> {
        let key: Vec<u32> = voice
            .diameters()
            .iter()
            .chain(voice.nose_diameters())
            .chain([voice.velum(), voice.tract_length()].iter())
            .map(|x| x.to_bits())
            .chain([voice.engine() as u32])
            .collect();
        if key != self.key {
            self.key = key;
            self.last = self.meter.formants(voice);
        }
        self.last
    }
}

fn trace_header() -> String {
    let mut h = String::from(
        "time\tnote_hz\tgate\ttongue_index\ttongue_diameter\tvelum_target\ttenseness\tlung",
    );
    for k in 0..MAX_CONSTRICTIONS {
        let _ = write!(h, "\tc{k}_index\tc{k}_diameter");
    }
    h += "\tbody_cell\tbody_diameter\ttip_diameter\tlips_diameter";
    for k in 0..MAX_CONSTRICTIONS {
        let _ = write!(h, "\tc{k}_actual");
    }
    h += "\tvelum\tintensity\tpitch_hz\tfold_pitch_hz\tlevel_db\tspeaking\tF1\tF2\tF3";
    for k in 0..N {
        let _ = write!(h, "\td{k}");
    }
    h
}

/// One trace row at `time`, with the RMS level of the last rows' worth of
/// sound.
#[allow(clippy::many_single_char_names)]
fn trace_row(replay: &Replay, time: f32, level: f32, formants: Option<[f32; 3]>) -> String {
    let p = replay.performer();
    let v = p.voice();
    let c = p.controls();
    let d = v.diameters();
    let mut row = format!(
        "{time:.3}\t{:.2}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
        v.target_frequency(),
        u8::from(v.is_voiced() || c.drone),
        c.tongue.0,
        c.tongue.1,
        c.velum,
        v.tenseness(),
        v.lung_pressure(),
    );
    for k in &c.constrictions {
        match k {
            Some(k) => {
                let _ = write!(row, "\t{:.3}\t{:.3}", k.index, k.diameter);
            }
            None => row += "\t\t",
        }
    }
    let (body, body_d) = narrowest(d, BLADE_START..TIP_START);
    let (_, tip) = narrowest(d, TIP_START..LIP_START);
    let (_, lips) = narrowest(d, LIP_START..N);
    let _ = write!(row, "\t{body}\t{body_d:.3}\t{tip:.3}\t{lips:.3}");
    for k in &c.constrictions {
        match k {
            Some(k) => {
                let cell = (k.index.round() as usize).min(N - 1);
                let _ = write!(row, "\t{:.3}", d[cell]);
            }
            None => row += "\t",
        }
    }
    let speaking = p
        .speaker()
        .current()
        .map_or("-", |(_, seg)| seg.phoneme.symbol());
    let _ = write!(
        row,
        "\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.1}\t{speaking}",
        v.velum(),
        v.intensity(),
        v.frequency(),
        v.gauges().fold_pitch,
        20.0 * level.max(1e-6).log10(),
    );
    match formants {
        Some([f1, f2, f3]) => {
            let _ = write!(row, "\t{f1:.0}\t{f2:.0}\t{f3:.0}");
        }
        None => row += "\t\t\t",
    }
    for x in d {
        let _ = write!(row, "\t{x:.3}");
    }
    row
}

/// The take's own sound, if its WAV file is next to the log.
fn recorded(log: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(log.with_extension("wav")).ok()?;
    match take::read_wav(&bytes) {
        Ok((_, samples)) => Some(samples),
        Err(e) => {
            eprintln!("{}: {e}", log.with_extension("wav").display());
            None
        }
    }
}

/// Say how the replay `out` compares with the take's own sound.
fn compare(path: &Path, wav: &[f32], out: &[f32]) {
    let n = wav.len().min(out.len());
    let pairs = || wav[..n].iter().zip(&out[..n]);
    let diff = pairs().map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let differ = pairs().filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if differ == 0 && wav.len() == out.len() {
        println!("  identical to {}", path.display());
    } else {
        println!(
            "  against {}: {differ} of {n} samples differ, by up to {diff:e} (peak {peak:.3}); \
             lengths {} and {}",
            path.display(),
            wav.len(),
            out.len()
        );
    }
}

fn run() -> Result<(), String> {
    let args = args()?;
    let text =
        std::fs::read_to_string(&args.log).map_err(|e| format!("{}: {e}", args.log.display()))?;
    let take = Take::parse(&text).map_err(|e| format!("{}: {e}", args.log.display()))?;
    let wav = recorded(&args.log);
    let rate = take.sample_rate;
    let length = take.end.unwrap_or_else(|| {
        let last = take.commands.last().map_or(0, |c| c.at);
        wav.as_ref()
            .map_or(last + (TAIL * rate) as u64, |w| w.len() as u64)
    });
    if take.end.is_none() {
        eprintln!(
            "{}: the log has no end line (the app stopped while recording); replaying {:.2} s",
            take.name,
            length as f32 / rate
        );
    }
    let mut trace = match &args.trace {
        Some(path) => {
            let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut w = std::io::BufWriter::new(f);
            writeln!(w, "{}", trace_header()).map_err(|e| e.to_string())?;
            Some(w)
        }
        None => None,
    };
    let mut formants = Formants {
        meter: survey::Meter::with_seconds(FORMANT_SECONDS),
        key: Vec::new(),
        last: None,
    };
    let step = ((TRACE_STEP * rate).round() as u64).max(1);
    let started = std::time::Instant::now();
    let mut replay = Replay::new(&take);
    let mut out = Vec::with_capacity(length as usize);
    let mut energy = 0.0f64;
    for at in 0..length {
        let s = replay.sample();
        out.push(s);
        energy += f64::from(s) * f64::from(s);
        if (at + 1) % step == 0
            && let Some(w) = &mut trace
        {
            let level = (energy / step as f64).sqrt() as f32;
            energy = 0.0;
            let f = formants.of(replay.performer().voice());
            let time = (at + 1) as f32 / rate;
            writeln!(w, "{}", trace_row(&replay, time, level, f)).map_err(|e| e.to_string())?;
        }
    }
    if let Some(mut w) = trace {
        w.flush().map_err(|e| e.to_string())?;
    }

    let speech = take
        .commands
        .iter()
        .filter(|c| matches!(c.cmd, take::Logged::Speak(_)))
        .count();
    println!(
        "{}: {:.2} s at {rate} Hz, {} commands ({speech} speech), rendered in {:.1} s",
        take.name,
        length as f32 / rate,
        take.commands.len(),
        started.elapsed().as_secs_f32()
    );
    if take.dropped_commands > 0 || take.dropped_samples > 0 {
        println!(
            "  the take lost {} commands and {} samples on the way to the disk: \
             the replay cannot match it",
            take.dropped_commands, take.dropped_samples
        );
    }
    if replay.unspoken > 0 {
        println!(
            "  {} speech commands have no text in the log and were skipped",
            replay.unspoken
        );
    }
    if let Some(wav) = &wav {
        compare(&args.log.with_extension("wav"), wav, &out);
    }
    if let Some(path) = &args.out {
        take::write_wav(path, rate.round() as u32, &out)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        println!("  wrote {}", path.display());
    }
    if let Some(path) = &args.trace {
        println!("  wrote {} (a row every {TRACE_STEP} s)", path.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
