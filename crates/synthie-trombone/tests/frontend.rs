//! The text front end: each class of English spelling rule on several
//! words, and the words of the judge's reading material that the old front
//! end got wrong.
//!
//! Rule tests go through the spelling rules alone (no dictionary), so they
//! check the rules rather than the dictionary. The expected sounds agree
//! with eSpeak NG (`en-us`) up to the notation the judge ignores.

#![cfg(feature = "alloc")]

use synthie_trombone::speech::{Token, phonemize, phonemize_with};

/// ARPAbet of `text` with `1` on accented vowels.
fn marked(tokens: &[Token]) -> String {
    tokens
        .iter()
        .filter_map(|t| match *t {
            Token::Phone { phoneme, stressed } => {
                Some(format!("{phoneme}{}", if stressed { "1" } else { "" }))
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The spelling rules' sounds for `word`, no accent marks.
fn rules(word: &str) -> String {
    marked(&phonemize_with(word, &|_| None)).replace('1', "")
}

/// The full front end's sounds for `word`, with accent marks.
fn say(word: &str) -> String {
    marked(&phonemize(word))
}

fn check_rules(cases: &[(&str, &str)]) {
    for &(word, expected) in cases {
        assert_eq!(rules(word), expected, "{word}");
    }
}

/// Past `-ed`: t after a voiceless sound, d after a voiced one, id after t or d; a dropped e or a doubled consonant is undone first.
#[test]
fn past_tense_ed() {
    check_rules(&[
        ("chopped", "CH AA P T"),
        ("parked", "P AA R K T"),
        ("missed", "M IH S T"),
        ("hoped", "HH OW P T"),
        ("wanted", "W AA N T IH D"),
        ("needed", "N IY D IH D"),
        ("played", "P L EY D"),
        ("loved", "L AH V D"),
        ("planned", "P L AE N D"),
        ("started", "S T AA R T IH D"),
        ("added", "AE D IH D"),
        ("laughed", "L AE F T"),
        ("judged", "JH AH JH D"),
    ]);
}

/// Plural and third-person `-s`: s, z, or iz after a hissing sound.
#[test]
fn plural_and_third_person_s() {
    check_rules(&[
        ("cats", "K AE T S"),
        ("dogs", "D AA G Z"),
        ("horses", "HH AO R S IH Z"),
        ("boxes", "B AA K S IH Z"),
        ("kids", "K IH D Z"),
        ("cars", "K AA R Z"),
        ("wishes", "W IH SH IH Z"),
        ("judges", "JH AH JH IH Z"),
        ("plays", "P L EY Z"),
        ("tries", "T R AY Z"),
    ]);
}

/// `th` is voiced in `-ther`, final `-the` and `-thm`, voiceless elsewhere.
#[test]
fn voiced_and_voiceless_th() {
    check_rules(&[
        ("brother", "B R AH DH ER"),
        ("mother", "M AH DH ER"),
        ("weather", "W EH DH ER"),
        ("bathe", "B EY DH"),
        ("breathe", "B R IY DH"),
        ("rhythm", "R IH DH AX M"),
        ("thin", "TH IH N"),
        ("thick", "TH IH K"),
        ("method", "M EH TH AX D"),
        ("author", "AO TH ER"),
        ("farther", "F AA R DH ER"),
    ]);
}

/// A final e is silent and makes the vowel before one consonant long, also before neutral endings.
#[test]
fn silent_e_and_vowel_consonant_e() {
    check_rules(&[
        ("make", "M EY K"),
        ("time", "T AY M"),
        ("hope", "HH OW P"),
        ("cute", "K Y UW T"),
        ("hopeless", "HH OW P L AX S"),
        ("statement", "S T EY T M AX N T"),
        ("lately", "L EY T L IY"),
        ("note", "N OW T"),
        ("ride", "R AY D"),
    ]);
}

/// Vowels before r: er, ir, ur, ear, or after w, -ire, -ure, -are, -ore.
#[test]
fn r_coloured_vowels() {
    check_rules(&[
        ("fire", "F AY ER"),
        ("heard", "HH ER D"),
        ("work", "W ER K"),
        ("word", "W ER D"),
        ("bird", "B ER D"),
        ("turn", "T ER N"),
        ("fear", "F IH R"),
        ("near", "N IH R"),
        ("earth", "ER TH"),
        ("learn", "L ER N"),
        ("care", "K EH R"),
        ("more", "M AO R"),
        ("pure", "P Y UH R"),
        ("nature", "N EY CH ER"),
        ("store", "S T AO R"),
        ("tire", "T AY ER"),
        ("worse", "W ER S"),
    ]);
}

/// Vowel digraphs: oo, ou, ow, ea, ui, oe, ie, ei, au, aw, ai, oa, oi, oy.
#[test]
fn vowel_digraphs() {
    check_rules(&[
        ("book", "B UH K"),
        ("food", "F UW D"),
        ("out", "AW T"),
        ("cow", "K AW"),
        ("snow", "S N OW"),
        ("eat", "IY T"),
        ("head", "HH EH D"),
        ("fruit", "F R UW T"),
        ("toe", "T OW"),
        ("field", "F IY L D"),
        ("receive", "R IH S IY V"),
        ("cause", "K AO Z"),
        ("saw", "S AO"),
        ("boat", "B OW T"),
        ("rain", "R EY N"),
        ("coin", "K OY N"),
        ("boy", "B OY"),
    ]);
}

/// `wh` is w, or h before o.
#[test]
fn wh() {
    check_rules(&[
        ("when", "W EH N"),
        ("which", "W IH CH"),
        ("while", "W AY L"),
        ("whip", "W IH P"),
        ("whole", "HH OW L"),
        ("white", "W AY T"),
        ("whale", "W EY L"),
    ]);
}

/// c and g are soft before e, i and y.
#[test]
fn soft_c_and_g() {
    check_rules(&[
        ("city", "S IH T IY"),
        ("cent", "S EH N T"),
        ("cycle", "S AY K AX L"),
        ("gem", "JH EH M"),
        ("giant", "JH AY AX N T"),
        ("cage", "K EY JH"),
        ("magic", "M AE JH IH K"),
        ("cat", "K AE T"),
        ("cut", "K AH T"),
        ("gift", "G IH F T"),
        ("gentle", "JH EH N T AX L"),
    ]);
}

/// Silent letters (kn, wr, mb, gh) and the endings -tion, -sion, -ture, -stle, -sten, -gn.
#[test]
fn silent_letters_and_endings() {
    check_rules(&[
        ("knee", "N IY"),
        ("knife", "N AY F"),
        ("write", "R AY T"),
        ("wrong", "R AA NG"),
        ("lamb", "L AE M"),
        ("climb", "K L AY M"),
        ("night", "N AY T"),
        ("nation", "N EY SH AX N"),
        ("vision", "V IH ZH AX N"),
        ("picture", "P IH K CH ER"),
        ("castle", "K AE S AX L"),
        ("listen", "L IH S IH N"),
        ("sign", "S AY N"),
        ("hour", "AW ER"),
    ]);
}

/// A doubled consonant is one sound and keeps the vowel before it short.
#[test]
fn doubled_consonants() {
    check_rules(&[
        ("letter", "L EH T ER"),
        ("happy", "HH AE P IY"),
        ("butter", "B AH T ER"),
        ("running", "R AH N IH NG"),
        ("hopping", "HH AA P IH NG"),
        ("summer", "S AH M ER"),
        ("dinner", "D IH N ER"),
    ]);
}

/// The accent follows endings, prefixes and syllable weight; unaccented vowels reduce.
#[test]
fn accent_and_reduced_vowels() {
    check_rules(&[
        ("attack", "AX T AE K"),
        ("begin", "B IH G IH N"),
        ("decide", "D IH S AY D"),
        ("ability", "AX B IH L IH T IY"),
        ("photography", "F AX T AA G R AX F IY"),
        ("conversation", "K AA N V ER S EY SH AX N"),
        ("military", "M IH L IH T EH R IY"),
        ("develop", "D IH V EH L AX P"),
        ("remember", "R IH M EH M B ER"),
        ("example", "EH G Z AE M P AX L"),
    ]);
}

/// Prefixes and suffixes keep their own sounds and leave the accent on the stem.
#[test]
fn prefixes_and_suffixes() {
    check_rules(&[
        ("unhappy", "AH N HH AE P IY"),
        ("nonstop", "N AA N S T AA P"),
        ("happiness", "HH AE P IY N AX S"),
        ("careless", "K EH R L AX S"),
        ("quickly", "K W IH K L IY"),
        ("enjoyment", "EH N JH OY M IH N T"),
        ("sadness", "S AE D N AX S"),
        ("kindness", "K AY N D N AX S"),
    ]);
}

#[test]
fn accent_falls_where_the_rules_put_it() {
    for (word, expected) in [
        ("nation", "N EY1 SH AX N"),
        ("ability", "AX B IH1 L IH T IY"),
        ("conversation", "K AA N V ER S EY1 SH AX N"),
        ("begin", "B IH G IH1 N"),
        ("develop", "D IH V EH1 L AX P"),
        ("photography", "F AX T AA1 G R AX F IY"),
        ("hello", "HH AX L OW1"),
        ("trombone", "T R AA M B OW1 N"),
    ] {
        assert_eq!(say(word), expected, "{word}");
    }
}

/// Irregular words come from the built-in dictionary.
#[test]
fn dictionary_words() {
    for (word, expected) in [
        ("canoe", "K AX N UW1"),
        ("one", "W AH1 N"),
        ("two", "T UW1"),
        ("said", "S EH1 D"),
        ("does", "D AH Z"),
        ("women", "W IH1 M AX N"),
        ("juice", "JH UW1 S"),
        ("whose", "HH UW1 Z"),
        ("the", "DH AX"),
    ] {
        assert_eq!(say(word), expected, "{word}");
    }
}

#[test]
fn a_custom_lexicon_replaces_the_dictionary() {
    let lexicon = |w: &str| (w == "synthie").then_some("S IH1 N TH IY");
    assert_eq!(
        marked(&phonemize_with("synthie", &lexicon)),
        "S IH1 N TH IY"
    );
    // Its words still take regular endings.
    assert_eq!(
        marked(&phonemize_with("synthies", &lexicon)),
        "S IH1 N TH IY Z"
    );
}

#[test]
fn arpabet_zero_keeps_a_word_unaccented() {
    assert_eq!(marked(&phonemize("[AA0 R]")), "AA R");
    assert_eq!(marked(&phonemize("[AA R]")), "AA1 R");
}

#[test]
fn contractions_and_possessives() {
    assert_eq!(rules("cat's"), "K AE T S");
    assert_eq!(rules("dog's"), "D AA G Z");
    assert_eq!(rules("horse's"), "HH AO R S IH Z");
    assert_eq!(say("couldn't"), "K UH1 D AX N T");
    assert_eq!(say("we'll"), "W IY L");
}

// ---- regression: words of the judge's material the old front end got wrong ----

/// Sounds that count as the same: unaccented vowel quality, final -y, the
/// cot-caught merger, and the flap (`DX`) for t or d.
fn same(a: &str, b: &str) -> bool {
    let benign: [&[&str]; 5] = [
        &["AX", "AH", "IH"],
        &["IY", "IH"],
        &["AA", "AO"],
        &["T", "DX"],
        &["D", "DX"],
    ];
    a == b
        || benign
            .iter()
            .any(|set| set.contains(&a) && set.contains(&b))
}

/// The judge's notation: r-coloured schwa written as one symbol, tense and
/// lax vowels before r.
fn canonical(arpabet: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let syms: Vec<&str> = arpabet
        .split_whitespace()
        .map(|s| s.trim_end_matches('1'))
        .collect();
    for (k, &s) in syms.iter().enumerate() {
        if s == "R"
            && out
                .last()
                .is_some_and(|p| matches!(p.as_str(), "ER" | "AX" | "AH"))
        {
            *out.last_mut().unwrap() = "ER".into();
            continue;
        }
        let lax = match (s, syms.get(k + 1)) {
            ("IY", Some(&"R")) => "IH",
            ("UW", Some(&"R")) => "UH",
            ("OW", Some(&"R")) => "AO",
            ("EY", Some(&"R")) => "EH",
            _ => s,
        };
        out.push(lax.into());
    }
    out
}

/// Edit distance with the judge's benign substitutions free.
fn distance(reference: &[String], said: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=said.len()).collect();
    for (row, r) in reference.iter().enumerate() {
        let mut cur = vec![row + 1; said.len() + 1];
        for (col, s) in said.iter().enumerate() {
            let cost = usize::from(!same(r, s));
            cur[col + 1] = (prev[col] + cost).min(prev[col + 1] + 1).min(cur[col] + 1);
        }
        prev = cur;
    }
    prev[said.len()]
}

/// eSpeak NG's phonemes (as the judge folds them) for material words the
/// old front end said wrong: canoe, smooth, fled, fire, heard, work, juice,
/// whose, brother, chopped ...
const FORMERLY_WRONG: &[(&str, &str)] = &[
    ("thin", "TH IH N"),
    ("pull", "P UH L"),
    ("water", "W AO DX ER"),
    ("mother", "M AH DH ER"),
    ("father", "F AA DH ER"),
    ("house", "HH AW S"),
    ("school", "S K UW L"),
    ("measure", "M EH ZH ER"),
    ("judge", "JH AH JH"),
    ("young", "Y AH NG"),
    ("around", "ER AW N D"),
    ("money", "M AH N IY"),
    ("table", "T EY B AX L"),
    ("paper", "P EY P ER"),
    ("question", "K W EH S CH AX N"),
    ("language", "L AE NG G W IH JH"),
    ("together", "T AX G EH DH ER"),
    ("between", "B IH T W IY N"),
    ("beautiful", "B Y UW DX IY F AX L"),
    ("canoe", "K AX N UW"),
    ("smooth", "S M UW DH"),
    ("easy", "IY Z IY"),
    ("rare", "R EH R"),
    ("bowls", "B OW L Z"),
    ("juice", "JH UW S"),
    ("makes", "M EY K S"),
    ("thrown", "TH R OW N"),
    ("beside", "B IH S AY D"),
    ("parked", "P AA R K T"),
    ("chopped", "CH AA P T"),
    ("garbage", "G AA R B IH JH"),
    ("hours", "AW ER Z"),
    ("steady", "S T EH D IY"),
    ("work", "W ER K"),
    ("us", "AH S"),
    ("used", "Y UW Z D"),
    ("salmon", "S AE M AX N"),
    ("source", "S AO R S"),
    ("huge", "HH Y UW JH"),
    ("woman", "W UH M AX N"),
    ("evening", "IY V N IH NG"),
    ("fires", "F AY ER Z"),
    ("cushion", "K UH SH AX N"),
    ("salt", "S AO L T"),
    ("across", "AX K R AA S"),
    ("sold", "S OW L D"),
    ("apostle", "AX P AA S AX L"),
    ("welcome", "W EH L K AH M"),
    ("general", "JH EH N ER AX L"),
    ("principles", "P R IH N S IH P AX L Z"),
    ("writes", "R AY T S"),
    ("equal", "IY K W AX L"),
    ("lucidity", "L UW S IH D IH DX IY"),
    ("kinds", "K AY N D Z"),
    ("foreign", "F AO R AX N"),
    ("fire", "F AY ER"),
    ("fred", "F R EH D"),
    ("unfortunately", "AH N F AO R CH AX N AX T L IY"),
    ("own", "OW N"),
    ("missed", "M IH S T"),
    ("even", "IY V AX N"),
    ("harry", "HH AE R IY"),
    ("asleep", "AX S L IY P"),
    ("heard", "HH ER D"),
    ("opened", "OW P AX N D"),
    ("door", "D AO R"),
    ("fled", "F L EH D"),
    ("friends", "F R EH N D Z"),
    ("begged", "B EH G D"),
    ("ago", "AX G OW"),
    ("away", "AX W EY"),
    ("also", "AO L S OW"),
    ("brother", "B R AH DH ER"),
    ("steadily", "S T EH D IY L IY"),
    ("replied", "R IH P L AY D"),
    ("doesn't", "D AH Z AX N T"),
    ("shaggy", "SH AE G IY"),
    ("nothing", "N AH TH IH NG"),
    ("whose", "HH UW Z"),
    ("great", "G R EY T"),
    ("worries", "W ER IY Z"),
    ("busy", "B IH Z IY"),
    ("exactly", "EH G Z AE K T L IY"),
    ("returned", "R IH T ER N D"),
    ("head", "HH EH D"),
    ("dead", "D EH D"),
    ("hood", "HH UH D"),
    ("who'd", "HH UW D"),
    ("suit", "S UW T"),
    ("sofa", "S OW F AX"),
    ("died", "D AY D"),
    ("hoed", "HH OW D"),
    ("bowl", "B OW L"),
    ("cold", "K OW L D"),
    ("gold", "G OW L D"),
    ("decree", "D IH K R IY"),
    ("degree", "D IH G R IY"),
    ("racer", "R EY S ER"),
    ("razor", "R EY Z ER"),
    ("bus", "B AH S"),
    ("rifle", "R AY F AX L"),
    ("rival", "R AY V AX L"),
    ("safer", "S EY F ER"),
    ("saver", "S EY V ER"),
    ("half", "HH AE F"),
];

#[test]
fn formerly_wrong_material_words() {
    let wrong: Vec<String> = FORMERLY_WRONG
        .iter()
        .filter(|(word, reference)| distance(&canonical(reference), &canonical(&say(word))) > 0)
        .map(|(word, reference)| format!("{word}: {} (eSpeak {reference})", say(word)))
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}
