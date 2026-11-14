//! How the physical engine gets from one shape to the next, measured from
//! the tract's own transfer function every 5 ms while it speaks (never LPC
//! on the output).
//!
//! * Out of an /n t d/ into /u/ the second formant does not fall from a
//!   palatal height onto the vowel: that fall is /ju/, and "canoe" was
//!   heard as "can you" while the lips of /u/ only began to round as the
//!   vowel did.
//! * From /l/ into /ɪ/ the first formant does not rise above both the
//!   /l/'s and the vowel's (an /aɪ/: "slid" heard as "slide"), and the /l/
//!   keeps the low second formant of an /l/.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/place.rs"]
mod place;
#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use place::Spoken;
use synthie_trombone::{Phoneme, Preset, VoiceSettings};

fn voice() -> VoiceSettings {
    Preset::Deep.settings()
}

/// The frames from sample `from` to sample `to` whose tract is open and
/// whose nose is shut (with the velum open the peaks are the nose's too).
fn oral(s: &Spoken, from: usize, to: usize) -> Vec<(usize, [f32; 3])> {
    (Spoken::frame_of(from)..Spoken::frame_of(to).min(s.frames.len()))
        .filter_map(|k| {
            let f = &s.frames[k];
            (f.velum < 0.1).then_some(())?;
            f.formants.map(|x| (k, x))
        })
        .collect()
}

/// The last segment of phoneme `p` (its first one is `k`) and the next.
fn after(s: &Spoken, p: Phoneme) -> (usize, usize) {
    let k = s.find(p, 1).unwrap_or_else(|| panic!("no {p}"));
    let (_, end) = s.consonant_span(k);
    (k, end + 1)
}

#[test]
fn the_second_formant_does_not_fall_onto_u_after_a_coronal() {
    for (text, c) in [
        ("a canoe", Phoneme::N),
        ("the new", Phoneme::N),
        ("to do", Phoneme::D),
        ("two", Phoneme::T),
    ] {
        let s = place::speak(&voice(), text, true);
        let (_, v) = after(&s, c);
        assert_eq!(s.phonemes[v], Phoneme::Uw, "{text}: {:?}", s.phonemes);
        let (v0, v1) = s.ranges[v];
        let mid = usize::midpoint(v0, v1);
        // The stop's own transition, its first 20 ms, falls from the
        // alveolar locus as a speaker's does; what follows is the vowel.
        let settled = v0 + (0.02 * place::SR) as usize;
        let frames = oral(&s, settled, mid + 2 * place::FRAME);
        assert!(frames.len() >= 4, "{text}: {} oral frames", frames.len());
        let target = frames.last().expect("frames").1[1];
        let highest = frames.iter().map(|f| f.1[1]).fold(0.0f32, f32::max);
        // Before round 8 the first oral frame of the /u/ of "canoe" stood
        // at 1816 Hz and the vowel's middle near 1150: a fall of 650 Hz
        // over 100 ms, while the lips rounded. A speaker's /u/ after /n d/
        // holds its F2 (the LibriSpeech reader's "the new", "do": flat,
        // fronted, near 2200 Hz).
        assert!(
            highest - target <= 200.0,
            "{text}: F2 falls from {highest:.0} to {target:.0} Hz into the /u/"
        );
    }
}

#[test]
fn the_first_formant_runs_flat_from_l_into_ih() {
    for text in ["slid", "lid", "a lid"] {
        let s = place::speak(&voice(), text, true);
        let (l, v) = after(&s, Phoneme::L);
        assert_eq!(s.phonemes[v], Phoneme::Ih, "{text}: {:?}", s.phonemes);
        let (l0, l1) = (s.ranges[l].0, s.ranges[v - 1].1);
        let (v0, v1) = s.ranges[v];
        let l_mid = usize::midpoint(l0, l1);
        let v_mid = usize::midpoint(v0, v1);
        let lateral = s.formants_at(l_mid, 0.0).expect("an open /l/");
        let vowel = s.formants_at(v_mid, 0.0).expect("an open /ɪ/");
        let path = oral(&s, l_mid, v_mid);
        let peak = path.iter().map(|f| f.1[0]).fold(0.0f32, f32::max);
        // Under the old /l/ body: 472 Hz in the /l/, 585 between, 456 in
        // the /ɪ/. The LibriSpeech reader's runs flat ("sleep": 311 to 328
        // Hz) or rises some 60 Hz above both ("exactly": 420, 490, 431).
        assert!(
            peak <= lateral[0].max(vowel[0]) + 60.0,
            "{text}: F1 {:.0} in the /l/, {peak:.0} between, {:.0} in the /ɪ/",
            lateral[0],
            vowel[0]
        );
        // An /l/, not a vowel: the dark second formant, and a rise of
        // its own into the front vowel.
        assert!(lateral[1] < 1450.0, "{text}: /l/ F2 {:.0}", lateral[1]);
        assert!(
            vowel[1] - lateral[1] > 500.0,
            "{text}: F2 {:.0} to {:.0}",
            lateral[1],
            vowel[1]
        );
    }
}
