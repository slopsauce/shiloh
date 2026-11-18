// Canonical prefix codes built from code lengths in table order, as FFmpeg's
// `ff_vlc_init_from_lengths` does: codes are handed out in the order the (symbol, length) pairs are listed,
// each one the next free left-aligned code of its length. With `VLC_INIT_LE` the first bit read from the
// little-endian stream is the code's most significant bit.
//
// Decoding: a table indexed by the next `PRIMARY` stream bits gives (length, symbol) for every code of at most
// that length; longer codes (rare) are matched bit by bit.
//
// This file is part of the qdmc crate, a Rust port of FFmpeg's QDMC decoder.
// Copyright (c) 2017 Paul B Mahol (original C code)
// Licensed under the GNU Lesser General Public License, version 2.1 or later (see COPYING).

use crate::bits::BitReader;

const PRIMARY: u32 = 10;

pub(crate) struct Vlc {
    /// indexed by the next PRIMARY bits as read (LSB = first bit): (code length or 0 = longer code, symbol)
    table: Vec<(u8, i32)>,
    bits: u32,
    /// per code length (index = length): (code value MSB-first, symbol)
    by_len: Vec<Vec<(u32, i32)>>,
    max_len: usize,
}

impl Vlc {
    /// `pairs` = (symbol, length); `offset` is added to every symbol (FFmpeg passes -1, so symbol 0 = escape).
    pub fn new(pairs: &[(u8, u8)], offset: i32) -> Vlc {
        let max_len = pairs.iter().map(|p| p.1 as usize).max().unwrap_or(0);
        let bits = PRIMARY.min(max_len as u32);
        let mut by_len = vec![Vec::new(); max_len + 1];
        let mut table = vec![(0u8, 0i32); 1 << bits];
        let mut code: u64 = 0;
        for &(sym, len) in pairs {
            let len = len as u32;
            debug_assert!(code & ((1u64 << (32 - len)) - 1) == 0, "misaligned code");
            let c = (code >> (32 - len)) as u32;
            let s = sym as i32 + offset;
            by_len[len as usize].push((c, s));
            if len <= bits {
                // the code as it appears in the stream: its first bit lowest
                let pattern = c.reverse_bits() >> (32 - len);
                for hi in 0..1u32 << (bits - len) {
                    table[(pattern | hi << len) as usize] = (len as u8, s);
                }
            }
            code += 1u64 << (32 - len);
        }
        debug_assert!(code <= 1u64 << 32, "overdetermined tree");
        Vlc { table, bits, by_len, max_len }
    }

    /// Decode one symbol; `None` for a bit sequence that is no code (never for the complete QDMC trees).
    #[inline]
    pub fn read(&self, br: &mut BitReader) -> Option<i32> {
        let (len, s) = self.table[br.peek(self.bits) as usize];
        if len > 0 {
            br.skip(len as u32);
            return Some(s);
        }
        let mut acc = 0u32;
        for len in 1..=self.max_len {
            acc = (acc << 1) | br.bits(1);
            if let Some(&(_, s)) = self.by_len[len].iter().find(|(c, _)| *c == acc) {
                return Some(s);
            }
        }
        None
    }
}
