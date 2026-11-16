//! Text arriving piece by piece, cut into the pieces that are compiled on
//! their own: whole sentences (or phrases), never a part of one.

/// Where streamed text is cut for compiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Granularity {
    /// At the end of a sentence (`.` `!` `?`, or a line break). A sentence
    /// is compiled once it has all arrived, so its intonation and rhythm
    /// are exactly those of the sentence spoken whole; speech starts once
    /// the first sentence has ended.
    #[default]
    Sentence,
    /// At the end of a phrase as well (`,` `;` `:` and dashes): speech
    /// starts sooner, and each phrase still gets the contour it would get
    /// inside its sentence (the compiler gives every phrase its own), but
    /// the shape the tongue takes in the pause after a comma, which looks
    /// ahead to the next phrase, is not known yet.
    Phrase,
}

fn ends_sentence(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

fn ends_phrase(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '—' | '–')
}

/// What may follow a closing mark and still belong to it.
fn closes(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '”' | '’' | '»')
}

/// The end (byte index) of the first complete piece of `text`, if any: a
/// run of closing marks and the space after them must have arrived, so that
/// `3.5` or `...` still being typed is not cut. A line break always ends a
/// piece.
pub(crate) fn first_piece(text: &str, granularity: Granularity) -> Option<usize> {
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if c == '\n' {
            return Some(at + c.len_utf8());
        }
        let ends = ends_sentence(c) || (granularity == Granularity::Phrase && ends_phrase(c));
        if !ends {
            continue;
        }
        // The whole run of marks and closing quotes, then a space.
        let mut end = at + c.len_utf8();
        while let Some(&(k, d)) = chars.peek() {
            if ends_sentence(d) || ends_phrase(d) || closes(d) {
                end = k + d.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        match chars.peek() {
            Some(&(_, d)) if d.is_whitespace() => return Some(end),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str, g: Granularity) -> std::vec::Vec<&str> {
        let mut out = std::vec::Vec::new();
        let mut rest = text;
        while let Some(end) = first_piece(rest, g) {
            out.push(&rest[..end]);
            rest = &rest[end..];
        }
        out.push(rest);
        out
    }

    #[test]
    fn sentences_are_cut_once_they_have_ended() {
        let s = Granularity::Sentence;
        assert_eq!(pieces("Hello there", s), ["Hello there"]);
        assert_eq!(pieces("Hello there.", s), ["Hello there."]);
        assert_eq!(pieces("Hello there. Who", s), ["Hello there.", " Who"]);
        assert_eq!(pieces("It is 3.5 m. Yes", s), ["It is 3.5 m.", " Yes"]);
        assert_eq!(
            pieces("Wait... what?! No", s),
            ["Wait...", " what?!", " No"]
        );
        assert_eq!(
            pieces("He said \"go.\" Then", s),
            ["He said \"go.\"", " Then"]
        );
        assert_eq!(pieces("one, two. three", s), ["one, two.", " three"]);
        assert_eq!(pieces("a line\nnext", s), ["a line\n", "next"]);
        let p = Granularity::Phrase;
        assert_eq!(pieces("one, two; three", p), ["one,", " two;", " three"]);
        assert_eq!(pieces("1,000 men", p), ["1,000 men"]);
    }
}
