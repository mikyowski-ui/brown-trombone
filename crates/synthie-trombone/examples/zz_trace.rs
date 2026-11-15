#![allow(clippy::all, clippy::pedantic)]
//! TEMPORARY (r11-canoe): trace velum, lips, tongue minimum, glottal area and output level
//! every ms while saying a text. zz_trace "text" [profile.json]
//! ZZ_CELLS=1: diameters every other cell. ZZ_AT=ms,ms,...: transfer-function peaks at those times.
#[path = "support/spectrum.rs"]
mod spectrum;
use synthie_trombone::{Engine, Preset, Talker};
fn main() {
    let text = std::env::args().nth(1).unwrap();
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    if let Some(p) = std::env::args().nth(2) {
        settings = settings.with_profile(&std::fs::read_to_string(p).unwrap()).unwrap();
    }
    let at: Vec<usize> = std::env::var("ZZ_AT")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_default();
    let quiet = !at.is_empty();
    let mut t = Talker::new(48_000.0, settings);
    if let Ok(s) = std::env::var("ZZ_UWP") {
        let f: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
        let voice = t.voice_mut();
        let length = voice.tract_length();
        let scale = synthie_trombone::vowel_targets::ADULT_MALE_LENGTH / length;
        let key = synthie_trombone::vowel_targets::VowelKey::Fronted(synthie_trombone::Phoneme::Uw);
        let near = key.articulation().and_then(|a| a.physical_shape(length));
        let shape = synthie_trombone::inverse::solve_for(
            f[0] * scale,
            f[1] * scale,
            Some(f[2] * scale),
            near.as_ref(),
            length,
        );
        voice.set_vowel_shape(key, Some(&shape));
    }
    t.say(&text);
    let mut k = 0usize;
    let mut acc = 0.0f32;
    if !quiet {
        println!("ms velum lip41 lip42 min20_38 at min39_43 glot psupra out_db");
    }
    while t.is_speaking() || k < 4800 {
        let s = t.process();
        acc += s * s;
        k += 1;
        if k % 48 == 0 {
            let ms = k / 48;
            let v = t.voice_mut();
            let d = *v.diameters();
            if at.contains(&ms) {
                let mag = spectrum::transfer(v, 0.2, 5000.0, 2.0);
                let r = spectrum::resonances(&mag, 2.0);
                print!("{ms} ms velum {:.2}: ", v.velum());
                for x in r.iter().take(5) {
                    print!("{:5.0}/{:3.0}/{:5.1}  ", x.hz, x.bandwidth, x.level);
                }
                println!();
                print!("   cells:");
                for i in (8..44).step_by(2) {
                    print!(" {:.2}", d[i]);
                }
                println!();
            }
            if !quiet {
                let (mut mi, mut mv) = (0, 9.0f32);
                for i in 20..39 {
                    if d[i] < mv {
                        mv = d[i];
                        mi = i;
                    }
                }
                let ml = d[39..44].iter().fold(9.0f32, |a, &b| a.min(b));
                let g = v.gauges();
                println!(
                    "{} {:.3} {:.2} {:.2} {:.2} {} {:.2} {:.3} {:.1} {:.1}",
                    ms,
                    v.velum(),
                    d[41],
                    d[42],
                    mv,
                    mi,
                    ml,
                    g.glottal_area,
                    g.supraglottal_pressure,
                    10.0 * (acc / 48.0 + 1e-12).log10()
                );
                if std::env::var("ZZ_CELLS").is_ok() {
                    print!("   cells:");
                    for i in (8..44).step_by(2) {
                        print!(" {:.2}", d[i]);
                    }
                    println!();
                }
            }
            acc = 0.0;
        }
        if k > 48_000 * 20 {
            break;
        }
    }
}
