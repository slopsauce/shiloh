// Little-endian bit reader (bits are consumed from the least significant bit of each byte first),
// matching FFmpeg's `BITSTREAM_READER_LE` as used by libavcodec/qdmc.c.
//
// This file is part of the qdmc crate, a Rust port of FFmpeg's QDMC decoder.
// Copyright (c) 2017 Paul B Mahol (original C code)
// Licensed under the GNU Lesser General Public License, version 2.1 or later (see COPYING).

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// position in bits
    pos: usize,
    /// FFmpeg caps the position at size_in_bits + 8 (checked reader)
    cap: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, cap: data.len() * 8 + 8 }
    }

    /// Bits left (may go negative like `get_bits_left`).
    pub fn left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }

    /// The next `n` (0..=32) bits without consuming them; past the end FFmpeg reads the (zeroed) input padding.
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        let byte = self.pos >> 3;
        let mut w = [0u8; 8];
        if let Some(s) = self.data.get(byte..) {
            let k = s.len().min(8);
            w[..k].copy_from_slice(&s[..k]);
        }
        let v = u64::from_le_bytes(w) >> (self.pos & 7);
        (v & ((1u64 << n) - 1)) as u32
    }

    #[inline]
    pub fn skip(&mut self, n: u32) {
        self.pos = (self.pos + n as usize).min(self.cap);
    }

    /// Read `n` bits (0..=32); the first bit read is the least significant bit of the result.
    #[inline]
    pub fn bits(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.skip(n);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bit by bit, the definition
    fn slow(d: &[u8], pos: usize, n: u32) -> u32 {
        (0..n).map(|i| d.get((pos + i as usize) >> 3).map_or(0, |b| ((b >> ((pos + i as usize) & 7)) & 1) as u32) << i).sum()
    }

    #[test]
    fn lsb_first() {
        let d = [0b1010_0110u8, 0xff, 0x01, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9a];
        let mut br = BitReader::new(&d);
        assert_eq!(br.bits(1), 0);
        assert_eq!(br.bits(3), 0b011);
        let mut pos = 4;
        for n in [8u32, 32, 5, 0, 17, 3] {
            assert_eq!(br.bits(n), slow(&d, pos, n), "at {pos} n {n}");
            pos += n as usize;
        }
        assert_eq!(br.left(), 72 - pos as isize);
        br.skip(30);
        assert_eq!(br.bits(16), 0); // padding
        assert_eq!(br.left(), -8); // capped at the end + 8
    }
}
