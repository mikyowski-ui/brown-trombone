//! The text line of type mode: what was typed or pasted, where the cursor
//! is, and the part of it that is in view.

use std::ops::Range;
use unicode_width::UnicodeWidthChar;

/// Most characters the text line holds: a few paragraphs, a few minutes of
/// speech. Compiling that much takes a few milliseconds on the UI thread.
pub const MAX_CHARS: usize = 4000;

/// A line of text with a cursor. Positions count characters, not bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextLine {
    text: String,
    /// Cursor, in characters from the start.
    cursor: usize,
    /// Characters in `text`.
    len: usize,
    /// First character in view (see [`window`](Self::window)).
    scroll: usize,
}

impl TextLine {
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Cursor position, in characters.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Length in characters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the line holds [`MAX_CHARS`] and takes no more.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len >= MAX_CHARS
    }

    /// Replace the text (cut to [`MAX_CHARS`]); the cursor goes to the end.
    pub fn set(&mut self, text: &str) {
        self.text = text.chars().take(MAX_CHARS).collect();
        self.len = self.text.chars().count();
        self.cursor = self.len;
    }

    /// Empty the line.
    pub fn clear(&mut self) {
        self.set("");
        self.scroll = 0;
    }

    /// Byte offset of character `k`.
    fn byte(&self, k: usize) -> usize {
        self.text
            .char_indices()
            .nth(k)
            .map_or(self.text.len(), |(b, _)| b)
    }

    fn char_at(&self, k: usize) -> Option<char> {
        self.text.chars().nth(k)
    }

    /// Insert `s` at the cursor, as much of it as fits; the cursor ends up
    /// after it. Returns how many characters went in.
    pub fn insert(&mut self, s: &str) -> usize {
        let room = MAX_CHARS.saturating_sub(self.len);
        let taken: String = s.chars().take(room).collect();
        let n = taken.chars().count();
        let at = self.byte(self.cursor);
        self.text.insert_str(at, &taken);
        self.len += n;
        self.cursor += n;
        n
    }

    /// Insert pasted text, already [`normalise`]d, at the cursor without
    /// doubling the space on either side of it. Returns how many characters
    /// of `s` were left out because the line was full.
    pub fn insert_pasted(&mut self, s: &str) -> usize {
        let mut s = s;
        let before = self.cursor.checked_sub(1).and_then(|k| self.char_at(k));
        if before.is_none_or(|c| c == ' ') {
            s = s.trim_start_matches(' ');
        }
        if self.char_at(self.cursor) == Some(' ') {
            s = s.trim_end_matches(' ');
        }
        let want = s.chars().count();
        want - self.insert(s)
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.len);
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.len;
    }

    /// Where the word before the cursor starts: back over spaces, then
    /// over everything up to the next space (as a shell's Ctrl+W counts
    /// words).
    fn word_start(&self) -> usize {
        let chars: Vec<char> = self.text.chars().take(self.cursor).collect();
        let mut k = chars.len();
        while k > 0 && chars[k - 1] == ' ' {
            k -= 1;
        }
        while k > 0 && chars[k - 1] != ' ' {
            k -= 1;
        }
        k
    }

    /// Where the word after the cursor ends: on over spaces, then over
    /// everything up to the next space.
    fn word_end(&self) -> usize {
        let mut k = self.cursor;
        let mut rest = self.text.chars().skip(self.cursor).peekable();
        while rest.next_if_eq(&' ').is_some() {
            k += 1;
        }
        while rest.next_if(|&c| c != ' ').is_some() {
            k += 1;
        }
        k
    }

    pub fn word_left(&mut self) {
        self.cursor = self.word_start();
    }

    pub fn word_right(&mut self) {
        self.cursor = self.word_end();
    }

    /// Remove the characters `from..to`.
    fn remove(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let (a, b) = (self.byte(from), self.byte(to));
        self.text.replace_range(a..b, "");
        self.len -= to - from;
        if self.cursor > from {
            self.cursor = from.max(self.cursor.saturating_sub(to - from));
        }
    }

    /// Delete the character before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.remove(self.cursor - 1, self.cursor);
        }
    }

    /// Delete the character under the cursor.
    pub fn delete(&mut self) {
        self.remove(self.cursor, (self.cursor + 1).min(self.len));
    }

    /// Delete the word before the cursor.
    pub fn delete_word(&mut self) {
        self.remove(self.word_start(), self.cursor);
    }

    /// Delete from the cursor to the end.
    pub fn delete_to_end(&mut self) {
        self.remove(self.cursor, self.len);
    }

    /// The characters to show in `width` columns, with the cursor in view
    /// (it takes a column of its own at the end of the text). The window
    /// stays where it was while the cursor is inside it, moves as little as
    /// it must otherwise, and shows as much as fits.
    pub fn window(&mut self, width: usize) -> Range<usize> {
        let widths: Vec<usize> = self.text.chars().map(column_width).collect();
        // The cell the cursor sits on: a character, or the blank after the
        // text.
        let cell = |k: usize| widths.get(k).copied().unwrap_or(1).max(1);
        let span = |from: usize, to: usize| widths[from..to].iter().sum::<usize>();
        let width = width.max(1);
        let c = self.cursor;
        // Left of the window: bring it back with a quarter of the room
        // to spare before the cursor, to see what is being edited.
        if c < self.scroll {
            let mut s = c;
            while s > 0 && span(s - 1, c) + cell(c) <= width / 4 {
                s -= 1;
            }
            self.scroll = s;
        }
        // Right of it: move on until the cursor fits.
        while self.scroll < c && span(self.scroll, c) + cell(c) > width {
            self.scroll += 1;
        }
        // Room left at the end (the text got shorter): show more before.
        let mut tail = span(self.scroll, self.len) + usize::from(c == self.len);
        while self.scroll > 0 && tail + widths[self.scroll - 1] <= width {
            self.scroll -= 1;
            tail += widths[self.scroll];
        }
        let mut end = self.scroll;
        let mut used = 0;
        while end < self.len && used + widths[end] <= width {
            used += widths[end];
            end += 1;
        }
        self.scroll..end
    }
}

/// Columns a character takes on screen (control characters none).
#[must_use]
pub fn column_width(c: char) -> usize {
    c.width().unwrap_or(0)
}

/// Tidy pasted text for the text line and the speech front end: line
/// breaks, tabs and other spaces become one space, control and invisible
/// formatting characters go, typographic quotes, dashes and the ellipsis
/// become what the front end reads, and accented Latin letters lose their
/// accents (the front end drops letters it does not know, so "café" would
/// be said "caf"). Letters, digits (which are read out as numbers),
/// punctuation and `[ARPAbet]` brackets are kept as they are.
#[must_use]
pub fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let mapped: &str = match c {
            // Invisible: zero-width spaces and joiners, soft hyphen, word
            // joiner, byte order mark, direction marks.
            '\u{00AD}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'
            | '\u{2066}'..='\u{2069}'
            | '\u{FEFF}' => continue,
            c if c.is_whitespace() => " ",
            c if c.is_control() => continue,
            '‘' | '’' | '‚' | '‛' | '′' | 'ʼ' => "'",
            '“' | '”' | '„' | '‟' | '″' | '«' | '»' => "\"",
            // The front end makes a comma's pause of an em or en dash.
            '‒' | '―' => "—",
            '‐' | '‑' | '−' => "-",
            '…' => "...",
            '&' => " and ",
            c => {
                let Some(plain) = fold(c) else {
                    push_char(&mut out, c);
                    continue;
                };
                plain
            }
        };
        for c in mapped.chars() {
            push_char(&mut out, c);
        }
    }
    out
}

/// Append `c`, never two spaces in a row.
fn push_char(out: &mut String, c: char) {
    if !(c == ' ' && out.ends_with(' ')) {
        out.push(c);
    }
}

/// The plain letters for an accented Latin letter.
#[rustfmt::skip]
fn fold(c: char) -> Option<&'static str> {
    Some(match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' | 'Ă' | 'Ą' => "A",
        'æ' => "ae", 'Æ' => "Ae", 'œ' => "oe", 'Œ' => "Oe", 'ß' => "ss",
        'ç' | 'ć' | 'č' => "c", 'Ç' | 'Ć' | 'Č' => "C",
        'ď' | 'đ' | 'ð' => "d", 'Ď' | 'Đ' | 'Ð' => "D",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' | 'ě' => "e",
        'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ė' | 'Ę' | 'Ě' => "E",
        'ğ' => "g", 'Ğ' => "G",
        'ì' | 'í' | 'î' | 'ï' | 'ī' | 'į' | 'ı' => "i",
        'Ì' | 'Í' | 'Î' | 'Ï' | 'Ī' | 'Į' | 'İ' => "I",
        'ł' | 'ľ' => "l", 'Ł' | 'Ľ' => "L",
        'ñ' | 'ń' | 'ň' => "n", 'Ñ' | 'Ń' | 'Ň' => "N",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' => "o",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ő' => "O",
        'ř' => "r", 'Ř' => "R",
        'ś' | 'š' | 'ş' => "s", 'Ś' | 'Š' | 'Ş' => "S",
        'ť' | 'ţ' => "t", 'Ť' | 'Ţ' => "T",
        'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'ų' => "u",
        'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ū' | 'Ů' | 'Ű' | 'Ų' => "U",
        'ý' | 'ÿ' => "y", 'Ý' | 'Ÿ' => "Y",
        'ź' | 'ż' | 'ž' => "z", 'Ź' | 'Ż' | 'Ž' => "Z",
        'þ' => "th", 'Þ' => "Th",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line holding `text`, cursor at the end.
    fn line(text: &str) -> TextLine {
        let mut line = TextLine::default();
        line.set(text);
        line
    }

    #[test]
    fn pasted_text_is_tidied_for_the_front_end() {
        assert_eq!(
            normalise("One.\r\nTwo,\tthree.\n\n  Four\u{7}\u{1b}!\n"),
            "One. Two, three. Four! "
        );
        assert_eq!(
            normalise("“It’s ‘fine’” — she said… 2–3 times"),
            "\"It's 'fine'\" — she said... 2–3 times"
        );
        assert_eq!(normalise("café naïve Ærø"), "cafe naive Aero");
        assert_eq!(
            normalise("zero\u{200B}width\u{00A0}space"),
            "zerowidth space"
        );
        assert_eq!(normalise("salt & pepper"), "salt and pepper");
        // What the front end reads is left alone: digits, ARPAbet, ASCII
        // punctuation.
        assert_eq!(
            normalise("route 66? [HH AH0 L OW1]!"),
            "route 66? [HH AH0 L OW1]!"
        );
        // Every character the front end reads as a phrase break survives.
        for (pasted, kept) in [("a, b", ','), ("a; b", ';'), ("a: b", ':'), ("a. b", '.')] {
            assert!(normalise(pasted).contains(kept));
        }
    }

    #[test]
    fn editing_moves_by_characters_and_words() {
        let mut l = line("héllo big world");
        assert_eq!((l.cursor(), l.len()), (15, 15));
        l.word_left();
        assert_eq!(l.cursor(), 10);
        l.word_left();
        l.word_left();
        assert_eq!(l.cursor(), 0);
        l.word_right();
        assert_eq!(l.cursor(), 5);
        l.right();
        l.delete_word();
        assert_eq!(l.text(), "big world");
        assert_eq!(l.cursor(), 0);
        l.delete();
        assert_eq!(l.text(), "ig world");
        l.end();
        l.backspace();
        l.home();
        l.insert("é");
        assert_eq!(l.text(), "éig worl");
        l.delete_word();
        assert_eq!(l.text(), "ig worl");
        l.end();
        l.word_left();
        l.delete_to_end();
        assert_eq!(l.text(), "ig ");
        l.clear();
        assert!(l.is_empty());
    }

    #[test]
    fn the_line_holds_up_to_its_limit() {
        let mut l = line(&"a".repeat(MAX_CHARS + 10));
        assert_eq!(l.len(), MAX_CHARS);
        assert!(l.is_full());
        assert_eq!(l.insert("b"), 0);
        l.backspace();
        assert_eq!(l.insert_pasted("xyz"), 2);
        assert!(l.text().ends_with('x'));
    }

    #[test]
    fn pasting_does_not_double_spaces() {
        let mut l = line("say ");
        l.insert_pasted(" hello ");
        assert_eq!(l.text(), "say hello ");
        l.home();
        l.word_right();
        l.insert_pasted(" there ");
        assert_eq!(l.text(), "say there hello ");
    }

    #[test]
    fn the_window_keeps_the_cursor_in_view() {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(8);
        let mut l = line(&text[..200]);
        let w = l.window(20);
        // At the end: the last 19 characters and the cursor's cell.
        assert_eq!(w, 181..200);
        l.home();
        assert_eq!(l.window(20), 0..20);
        for _ in 0..25 {
            l.right();
        }
        let w = l.window(20);
        assert!(w.contains(&25) && w.len() == 20, "{w:?}");
        // Moving back inside the window leaves it where it is.
        l.left();
        assert_eq!(l.window(20), w);
        // A wide character counts double.
        let mut wide = line("日本語のテキスト");
        let w = wide.window(7);
        let cols: usize = wide
            .text()
            .chars()
            .skip(w.start)
            .take(w.len())
            .map(column_width)
            .sum();
        assert!(cols < 7, "{w:?}");
        // Text shorter than the room is shown whole.
        let mut short = line("hi");
        short.home();
        assert_eq!(short.window(20), 0..2);
    }
}
