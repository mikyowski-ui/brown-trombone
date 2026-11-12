//! Render the same material through both engines, side by side.
//!
//! ```sh
//! cargo run -p synthie-trombone --example ab --release            # into report/
//! cargo run -p synthie-trombone --example ab --release -- out/dir
//! ```
//!
//! Writes pairs of WAV files and an `index.html` that plays them next to
//! their spectrograms (make those with `ffmpeg`, see the page's footer).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/wav.rs"]
mod wav;

use std::fmt::Write as _;
use synthie_trombone::{Engine, GlottalSource, Phoneme, Preset, Talker, Trombone, normalize};

const SR: u32 = 48_000;

fn engines() -> [(&'static str, Engine, GlottalSource); 3] {
    [
        ("classic", Engine::Classic, GlottalSource::Folds),
        ("physical", Engine::Physical, GlottalSource::Folds),
        ("physical-pulse", Engine::Physical, GlottalSource::Pulse),
    ]
}

fn instrument(engine: Engine, source: GlottalSource) -> Trombone {
    let mut t = Trombone::new(SR as f32);
    t.set_engine(engine);
    t.set_glottal_source(source);
    t
}

fn render(t: &mut Trombone, seconds: f32, out: &mut Vec<f32>) {
    let start = out.len();
    out.resize(start + (seconds * SR as f32) as usize, 0.0);
    t.render(&mut out[start..]);
}

fn vowels(engine: Engine, source: GlottalSource) -> Vec<f32> {
    let mut t = instrument(engine, source);
    let mut out = Vec::new();
    t.set_frequency(110.0);
    for p in [
        Phoneme::Iy,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Aa,
        Phoneme::Ao,
        Phoneme::Uw,
        Phoneme::Er,
    ] {
        t.articulate_phoneme(p);
        t.note_on();
        render(&mut t, 0.55, &mut out);
        t.note_off();
        render(&mut t, 0.2, &mut out);
    }
    out
}

fn scale(engine: Engine, source: GlottalSource) -> Vec<f32> {
    let mut t = instrument(engine, source);
    let mut out = Vec::new();
    t.articulate_phoneme(Phoneme::Aa);
    for note in [
        45.0, 47.0, 49.0, 50.0, 52.0, 54.0, 56.0, 57.0, 57.0, 52.0, 45.0,
    ] {
        t.note_on_midi(note);
        render(&mut t, 0.4, &mut out);
    }
    t.note_off();
    render(&mut t, 0.4, &mut out);
    out
}

fn breathiness(engine: Engine, source: GlottalSource) -> Vec<f32> {
    let mut t = instrument(engine, source);
    let mut out = Vec::new();
    t.articulate_phoneme(Phoneme::Aa);
    t.set_frequency(120.0);
    for tense in [0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.0] {
        t.set_tenseness(tense);
        t.note_on();
        render(&mut t, 0.6, &mut out);
        t.note_off();
        render(&mut t, 0.15, &mut out);
    }
    out
}

fn syllables(engine: Engine, source: GlottalSource) -> Vec<f32> {
    let mut talker = Talker::new(SR as f32, Preset::Deep.settings().with_engine(engine));
    talker.voice_mut().set_glottal_source(source);
    talker.bake("[P AA1] [B AA1] [T AA1] [D AA1] [K AA1] [G AA1]. [S AA1] [SH AA1] [F AA1] [TH AA1] [Z AA1] [V AA1]. [M AA1] [N AA1] [L AA1] [R AA1] [W AA1] [Y AA1].")
}

fn spoken(engine: Engine, source: GlottalSource, preset: Preset, text: &str) -> Vec<f32> {
    let mut talker = Talker::new(SR as f32, preset.settings().with_engine(engine));
    talker.voice_mut().set_glottal_source(source);
    talker.bake(text)
}

const SENTENCES: &[(&str, &str)] = &[
    ("hello", "Hello there, how are you today?"),
    ("shells", "She sells sea shells by the sea shore."),
    ("dog", "The big dog barked at the old man."),
];

fn main() -> std::io::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "report".to_string());
    std::fs::create_dir_all(&dir)?;
    let mut items: Vec<(String, String)> = Vec::new();

    let mut emit = |name: &str,
                    title: &str,
                    make: &dyn Fn(Engine, GlottalSource) -> Vec<f32>|
     -> std::io::Result<()> {
        for (label, engine, source) in engines() {
            let mut x = make(engine, source);
            normalize(&mut x, 0.89);
            wav::write(&format!("{dir}/{name}-{label}.wav"), SR, &x)?;
        }
        items.push((name.to_string(), title.to_string()));
        println!("{name}");
        Ok(())
    };

    emit(
        "vowels",
        "Seven vowels at 110 Hz: ee, eh, a (cat), ah, aw, oo, er",
        &vowels,
    )?;
    emit("scale", "A scale on \"ah\"", &scale)?;
    emit(
        "breathiness",
        "\"ah\" from pressed to breathy to whisper",
        &breathiness,
    )?;
    emit(
        "syllables",
        "Consonants before \"ah\": pa ba ta da ka ga, sa sha fa tha za va, ma na la ra wa ya",
        &syllables,
    )?;
    for &(tag, text) in SENTENCES {
        for preset in [Preset::Deep, Preset::Bright, Preset::Elder] {
            let name = format!("{tag}-{}", preset.name());
            let title = format!("\"{text}\" ({} voice)", preset.name());
            emit(&name, &title, &|e, s| spoken(e, s, preset, text))?;
        }
    }
    emit(
        "whisper",
        "\"Hello there, how are you today?\" (whisper voice)",
        &|e, s| spoken(e, s, Preset::Whisper, SENTENCES[0].1),
    )?;

    let mut html = String::from(
        "<!doctype html><meta charset=\"utf-8\"><title>Trombone A/B</title>\n<style>\
body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#16171a;color:#e6e6e6}\
h1{font-size:1.4rem}h2{font-size:1.05rem;margin:2rem 0 .4rem}\
.row{display:grid;grid-template-columns:9rem 18rem 1fr;gap:.8rem;align-items:center;margin:.3rem 0}\
img{width:100%;height:110px;object-fit:fill;border-radius:4px}audio{width:100%}\
.tag{font-weight:600}p{color:#b8b8b8}</style>\n\
<h1>Trombone: classic engine against the physical engine</h1>\n\
<p>Each item is the same input through three models. <b>classic</b> is Pink Trombone as shipped. \
<b>physical</b> is the new engine with self-oscillating vocal folds. <b>physical-pulse</b> is the new \
engine with a prescribed glottal pulse instead of the folds. Spectrograms run 0 to 8 kHz. \
Levels are peak-normalised.</p>\n",
    );
    for (name, title) in &items {
        let _ = writeln!(html, "<h2>{title}</h2>");
        for (label, _, _) in engines() {
            let _ = writeln!(
                html,
                "<div class=\"row\"><span class=\"tag\">{label}</span><audio controls preload=\"none\" src=\"{name}-{label}.wav\"></audio><img loading=\"lazy\" src=\"{name}-{label}.png\" alt=\"spectrogram\"></div>"
            );
        }
    }
    html.push_str("<p>Spectrograms: <code>for f in *.wav; do ffmpeg -y -loglevel error -i \"$f\" -lavfi showspectrumpic=s=1200x220:legend=0:stop=8000:scale=log \"${f%.wav}.png\"; done</code></p>\n");
    std::fs::write(format!("{dir}/index.html"), html)?;
    println!("wrote {dir}/index.html");
    Ok(())
}
