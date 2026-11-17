//! The tract and the tongue body chosen by name (`say --tract`,
//! `--tongue-model`, the app's flags of the same names) are the numbers a
//! profile would give, and the defaults are left alone.

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Preset, TongueModel, TractModel, Trombone, Tuning, VoiceSettings};

const PROFILE: &str = include_str!("../../../profiles/mri-tongue.json");

fn render(settings: &VoiceSettings, text: &str) -> Vec<f32> {
    let mut voice = Trombone::new(48_000.0);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let mut speaker = Speaker::new();
    speaker.speak(Utterance::from_text(text, &settings.prosody()));
    let mut out = Vec::new();
    while speaker.is_speaking() {
        out.push(speaker.process(&mut voice));
    }
    out
}

#[test]
fn names_parse() {
    assert_eq!(
        TractModel::from_name("Anatomical"),
        Some(TractModel::Anatomical)
    );
    assert_eq!(
        TractModel::from_name("original"),
        Some(TractModel::Original)
    );
    assert_eq!(TractModel::from_name("wide"), None);
    assert_eq!(TongueModel::from_name("cosine"), Some(TongueModel::Cosine));
    assert_eq!(TongueModel::from_name("MRI"), Some(TongueModel::Measured));
    assert_eq!(
        TongueModel::from_name("measured"),
        Some(TongueModel::Measured)
    );
    assert_eq!(TongueModel::from_name("flat"), None);
    for t in [TractModel::Anatomical, TractModel::Original] {
        assert_eq!(TractModel::from_name(t.name()), Some(t));
    }
    for t in [TongueModel::Cosine, TongueModel::Measured] {
        assert_eq!(TongueModel::from_name(t.name()), Some(t));
    }
}

#[test]
fn the_defaults_are_the_anatomical_tract_and_the_cosine_tongue() {
    let named = Tuning::DEFAULT
        .with_tract(TractModel::Anatomical)
        .with_tongue(TongueModel::Cosine);
    assert_eq!(named, Tuning::DEFAULT);
    assert_eq!(TractModel::of(&Tuning::DEFAULT), TractModel::Anatomical);
    assert_eq!(Tuning::DEFAULT.anatomy.tongue_model, 0.0);
    let mri = Tuning::DEFAULT.with_tongue(TongueModel::Measured);
    assert_eq!(mri.anatomy.tongue_model, 1.0);
    // Only the tongue moved.
    assert_eq!(mri.with_tongue(TongueModel::Cosine), Tuning::DEFAULT);
}

#[test]
fn the_named_mri_tongue_on_the_original_tract_is_the_profile() {
    let deep = Preset::Deep.settings().with_engine(Engine::Physical);
    let by_profile = deep
        .with_profile(PROFILE)
        .expect("profiles/mri-tongue.json");
    let mut by_name = deep;
    by_name.tuning = by_name
        .tuning
        .with_tract(TractModel::Original)
        .with_tongue(TongueModel::Measured);
    assert_eq!(by_name, by_profile);
    assert_eq!(TractModel::of(&by_name.tuning), TractModel::Original);
    // And it sounds the same, sample for sample, and not as the default.
    let text = "The juice of lemons.";
    let a = render(&by_name, text);
    assert_eq!(a, render(&by_profile, text));
    assert_ne!(a, render(&deep, text));
}
