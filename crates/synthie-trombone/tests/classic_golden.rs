//! The classic (Pink Trombone) engine is the baseline every physical-engine
//! stage is compared against, so its output is pinned bit for bit.
//!
//! The hashes depend on the platform's `sin`/`exp`/`powf`, so the check only
//! runs where they were recorded. To re-record after an intended change to
//! the classic engine:
//!
//! ```sh
//! GOLDEN_PRINT=1 cargo test -p synthie-trombone --test classic_golden -- --nocapture
//! ```

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use synthie_trombone::{Constriction, Phoneme, Preset, SLOT_MAIN, Talker, Trombone};

/// FNV-1a over the sample bit patterns.
fn hash(samples: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn sung_vowels() -> Vec<f32> {
    let mut t = Trombone::with_seed(44_100.0, 1234);
    let mut out = vec![0.0; 44_100];
    let mut pos = 0;
    for (p, note) in [
        (Phoneme::Aa, 45.0),
        (Phoneme::Iy, 52.0),
        (Phoneme::Uw, 57.0),
    ] {
        t.articulate_phoneme(p);
        t.note_on_midi(note);
        t.render(&mut out[pos..pos + 14_700]);
        pos += 14_700;
    }
    out
}

fn consonants() -> Vec<f32> {
    let mut t = Trombone::with_seed(48_000.0, 99);
    let mut out = vec![0.0; 48_000];
    t.set_tract_length(0.8);
    t.note_on();
    t.render(&mut out[..12_000]);
    t.set_constriction(SLOT_MAIN, Some(Constriction::new(41.0, 0.0)));
    t.render(&mut out[12_000..20_000]);
    t.set_constriction(SLOT_MAIN, None);
    t.render(&mut out[20_000..28_000]);
    t.note_off();
    t.articulate_phoneme(Phoneme::S);
    t.render(&mut out[28_000..38_000]);
    t.articulate_phoneme(Phoneme::M);
    t.note_on();
    t.render(&mut out[38_000..]);
    out
}

fn spoken(preset: Preset) -> Vec<f32> {
    Talker::new(22_050.0, preset).bake("Hello there, how are you? Pink trombone, 42.")
}

const GOLDEN: &[(&str, u64)] = &[
    ("sung_vowels", 0xab25_c845_031f_eab8),
    ("consonants", 0x3214_4abb_a372_26db),
    ("deep", 0xd089_6210_11ce_e5a9),
    ("bright", 0x01d3_1df4_dfe2_cc06),
    ("child", 0x5f1b_7845_5f35_6717),
    ("elder", 0xa685_b0c2_88c8_5361),
    ("giant", 0x2dc4_e729_cd55_6c13),
    ("critter", 0x6484_3619_3cfb_f8ea),
    ("robot", 0xcd7b_473d_289e_396b),
    ("whisper", 0xc0e5_b974_5e5c_72b3),
    ("android", 0xc510_2258_92ff_b635),
];

#[test]
fn classic_engine_is_bit_identical() {
    let mut got = vec![
        ("sung_vowels", hash(&sung_vowels())),
        ("consonants", hash(&consonants())),
    ];
    for p in Preset::ALL {
        got.push((p.name(), hash(&spoken(p))));
    }
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for (name, h) in &got {
            println!("    (\"{name}\", 0x{h:016x}),");
        }
        return;
    }
    assert_eq!(got.len(), GOLDEN.len());
    for ((name, h), (gname, gh)) in got.iter().zip(GOLDEN) {
        assert_eq!(name, gname);
        assert_eq!(h, gh, "classic engine output changed for `{name}`");
    }
}
