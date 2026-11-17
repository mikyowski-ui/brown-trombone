//! Where the steps and clicks of `tests/physical_gates.rs`'s
//! `stop_releases_are_bursts_not_clicks` are, for every voice: the largest
//! step at a release (with its stop and the sounds around it) and the
//! largest click (with the sound it falls in), against each voice's loudest
//! vowel sample.
//!
//! ```sh
//! cargo run -p synthie-trombone --example clicks --release
//! cargo run -p synthie-trombone --example clicks --release -- --lengthened
//! ```

#![allow(clippy::cast_precision_loss)]

#[path = "support/stops.rs"]
mod stops;

use synthie_trombone::{Preset, TractModel, VoiceSettings};

const TEXTS: [&str; 3] = [
    "As for etchings they are of two kinds British and foreign.",
    "The judge checked the big black bag at the gate.",
    "Pick a cup and dig deep to get the chips.",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tract = if args.iter().any(|a| a == "--lengthened") {
        Some(TractModel::Anatomical)
    } else if args.iter().any(|a| a == "--original") {
        Some(TractModel::Original)
    } else {
        None
    };
    for preset in Preset::ALL {
        let mut settings: VoiceSettings = preset.settings();
        if let Some(t) = tract {
            settings.tuning = settings.tuning.with_tract(t);
        }
        let (mut click, mut at_release) = ((0.0f32, String::new()), (0.0f32, String::new()));
        for text in TEXTS {
            let r = stops::record(&settings, text);
            let vowel = stops::vowel_peak(&r);
            let (c, i) = stops::largest_click_at(&r);
            if c / vowel > click.0 {
                let k = r.segment[i.min(r.segment.len() - 1)];
                let around: Vec<&str> = (k.saturating_sub(2)..(k + 3).min(r.phonemes.len()))
                    .map(|j| r.phonemes[j].symbol())
                    .collect();
                click = (
                    c / vowel,
                    format!(
                        "in {} ({}) at {:.3} s of \"{text}\"",
                        r.phonemes[k].symbol(),
                        around.join(" "),
                        i as f32 / 48_000.0
                    ),
                );
            }
            for (n, b) in stops::bursts(&r).iter().enumerate() {
                if b.step / vowel > at_release.0 {
                    at_release = (
                        b.step / vowel,
                        format!("{} (burst {n}) of \"{text}\"", b.phoneme.symbol()),
                    );
                }
            }
        }
        println!(
            "{preset}: click {:.2} {}; at a release {:.2} {}",
            click.0, click.1, at_release.0, at_release.1
        );
    }
}
