//! The articulators' frames (hybrid roadmap, stage 2), phoneme by phoneme
//! and through a sentence.
//!
//! ```sh
//! cargo run -p synthie-trombone --example lipsync --release               # every phoneme held
//! cargo run -p synthie-trombone --example lipsync --release -- "Papa."    # a text, frame by frame
//! ```

#![allow(clippy::cast_precision_loss)]

use synthie_trombone::speech::Speaker;
use synthie_trombone::{ArticulatorFrame, Engine, Phoneme, Preset, Talker, Trombone};

fn row(name: &str, f: &ArticulatorFrame) {
    let c = f.to_array();
    println!(
        "{name:6} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2}",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]
    );
}

fn main() {
    let text: Vec<String> = std::env::args().skip(1).collect();
    println!("        jaw  close funnl puckr press tng_up tng_bk velum larnx glott");
    let settings = Preset::Deep.settings().with_engine(Engine::Physical);
    if text.is_empty() {
        for p in Phoneme::ALL {
            let mut v = Trombone::new(48_000.0);
            settings.apply(&mut v);
            Speaker::prepare(&mut v);
            v.articulate_phoneme(*p);
            v.settle();
            let mut block = vec![0.0f32; 4800];
            v.render(&mut block);
            row(&p.to_string(), &v.articulators());
        }
        return;
    }
    let mut talker = Talker::new(48_000.0, settings);
    let (out, frames) = talker.bake_with_articulators(&text.join(" "));
    for (k, f) in frames.iter().enumerate() {
        let from = k * 512;
        let level = out[from..(from + 512).min(out.len())]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        row(&format!("{:5.2}", k as f32 * ArticulatorFrame::PERIOD), f);
        if level < 1e-3 {
            println!("       (quiet)");
        }
    }
}
