//! A single-producer single-consumer ring of 32-bit words, lock-free and
//! allocation-free, in safe code: every word is an atomic, the producer
//! publishes what it wrote with a release store of its count, and the
//! consumer frees what it read with a release store of its own. Messages
//! are runs of words: a header (kind, length, whether timed), an optional
//! sample time, and the payload.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Longest message, in words (header and time included).
pub(crate) const MAX_MESSAGE: usize = 32;

/// How far each end has got: words written and words read, counted from
/// the start (they wrap around together).
#[derive(Debug, Default)]
pub(crate) struct Counters {
    written: AtomicUsize,
    read: AtomicUsize,
}

impl Counters {
    pub(crate) const fn new() -> Self {
        Self {
            written: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
        }
    }

    /// Start again, empty (both ends are held exclusively).
    pub(crate) fn clear(&self) {
        self.written.store(0, Ordering::Relaxed);
        self.read.store(0, Ordering::Relaxed);
    }

    /// Words free for the producer.
    pub(crate) fn free(&self, words: &[AtomicU32]) -> usize {
        let written = self.written.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        words.len() - written.wrapping_sub(read)
    }

    /// Words waiting for the consumer.
    pub(crate) fn waiting(&self) -> usize {
        let read = self.read.load(Ordering::Relaxed);
        self.written.load(Ordering::Acquire).wrapping_sub(read)
    }

    /// Append `message` (producer only). `false`, with nothing written, if
    /// there is not room for all of it. `words.len()` is a power of two.
    pub(crate) fn write(&self, words: &[AtomicU32], message: &[u32]) -> bool {
        debug_assert!(words.len().is_power_of_two());
        let mask = words.len() - 1;
        if self.free(words) < message.len() {
            return false;
        }
        let at = self.written.load(Ordering::Relaxed);
        for (k, &w) in message.iter().enumerate() {
            words[at.wrapping_add(k) & mask].store(w, Ordering::Relaxed);
        }
        self.written
            .store(at.wrapping_add(message.len()), Ordering::Release);
        true
    }

    /// Take the next message (consumer only) into `out`; its length, or
    /// `None` if nothing is waiting. `len_of` reads a message's length from
    /// its header word.
    pub(crate) fn read(
        &self,
        words: &[AtomicU32],
        out: &mut [u32; MAX_MESSAGE],
        len_of: impl Fn(u32) -> usize,
    ) -> Option<usize> {
        let mask = words.len() - 1;
        let waiting = self.waiting();
        if waiting == 0 {
            return None;
        }
        let at = self.read.load(Ordering::Relaxed);
        let header = words[at & mask].load(Ordering::Relaxed);
        let len = len_of(header).clamp(1, MAX_MESSAGE).min(waiting);
        for (k, o) in out.iter_mut().enumerate().take(len) {
            *o = words[at.wrapping_add(k) & mask].load(Ordering::Relaxed);
        }
        self.read.store(at.wrapping_add(len), Ordering::Release);
        Some(len)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::sync::Arc;
    use std::vec::Vec;

    fn ring(n: usize) -> Vec<AtomicU32> {
        (0..n).map(|_| AtomicU32::new(0)).collect()
    }

    #[test]
    fn messages_wrap_around_and_a_full_ring_refuses() {
        let words = ring(8);
        let c = Counters::new();
        let mut out = [0u32; MAX_MESSAGE];
        let len = |h: u32| h as usize;
        for round in 0..20u32 {
            assert!(c.write(&words, &[3, round, round + 1]));
            assert!(c.write(&words, &[2, round]));
            // 5 of 8 used: a message of 4 does not fit, and nothing of it
            // is written.
            assert!(!c.write(&words, &[4, 0, 0, 0]));
            assert_eq!(c.read(&words, &mut out, len), Some(3));
            assert_eq!(out[..3], [3, round, round + 1]);
            assert_eq!(c.read(&words, &mut out, len), Some(2));
            assert_eq!(out[..2], [2, round]);
            assert_eq!(c.read(&words, &mut out, len), None);
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn words_cross_threads_in_order() {
        let words: Arc<Vec<AtomicU32>> = Arc::new(ring(64));
        let c = Arc::new(Counters::new());
        let (w, k) = (Arc::clone(&words), Arc::clone(&c));
        let n = 100_000u32;
        let producer = std::thread::spawn(move || {
            let mut i = 0;
            while i < n {
                if k.write(&w, &[2, i]) {
                    i += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        });
        let mut out = [0u32; MAX_MESSAGE];
        let mut next = 0;
        while next < n {
            if let Some(len) = c.read(&words, &mut out, |h| h as usize) {
                assert_eq!((len, out[1]), (2, next));
                next += 1;
            } else {
                std::thread::yield_now();
            }
        }
        producer.join().unwrap();
    }
}
