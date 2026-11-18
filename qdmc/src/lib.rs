//! QDesign Music Codec 1 ('QDMC') decoder.
//!
//! A line-by-line Rust port of FFmpeg's `libavcodec/qdmc.c` ("QDMC compatible decoder",
//! Copyright (c) 2017 Paul B Mahol), so it is licensed like the original: GNU Lesser General Public
//! License, version 2.1 or (at your option) any later version — see `COPYING`.
//!
//! The codec codes each frame as noise levels per band plus a set of tones (frequency, amplitude,
//! phase) in five groups of decreasing duration; both are written into a spectrum that is turned into
//! 16-bit PCM by overlapping inverse FFTs of twice the subframe length.
//!
//! ```no_run
//! # fn demo(extradata: &[u8], packets: &[&[u8]]) -> Result<(), qdmc::Error> {
//! let mut dec = qdmc::Decoder::new(extradata)?;
//! let mut pcm = Vec::new(); // interleaved i16, `dec.channels()` channels at `dec.sample_rate()` Hz
//! for p in packets {
//!     dec.decode_packet(p, &mut pcm)?;
//! }
//! # Ok(()) }
//! ```

mod bits;
mod fft;
mod vlc;

use bits::BitReader;
use fft::{Complex, InverseFft};
use std::fmt;
use std::sync::OnceLock;
use vlc::Vlc;

/// Why a stream or a frame cannot be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The codec configuration (the 'QDCA' atom of the sample description) is missing or invalid.
    Config(String),
    /// A frame is truncated, has a bad label / checksum or an invalid bitstream; the decoder was reset.
    InvalidFrame(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Config(s) => write!(f, "QDMC configuration: {s}"),
            Error::InvalidFrame(s) => write!(f, "QDMC frame: {s}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Default)]
struct Tone {
    // the field widths of the C struct matter: values are stored truncated
    mode: u8,
    phase: u8,
    offset: u8,
    freq: i16,
    amplitude: i16,
}

const CODE_PREFIX: [u32; 65] = [
    0x0, 0x1, 0x2, 0x3, 0x4, 0x6, 0x8, 0xA, 0xC, 0x10, 0x14, 0x18, 0x1C, 0x24, 0x2C, 0x34, 0x3C, 0x4C, 0x5C, 0x6C,
    0x7C, 0x9C, 0xBC, 0xDC, 0xFC, 0x13C, 0x17C, 0x1BC, 0x1FC, 0x27C, 0x2FC, 0x37C, 0x3FC, 0x4FC, 0x5FC, 0x6FC, 0x7FC,
    0x9FC, 0xBFC, 0xDFC, 0xFFC, 0x13FC, 0x17FC, 0x1BFC, 0x1FFC, 0x27FC, 0x2FFC, 0x37FC, 0x3FFC, 0x4FFC, 0x5FFC, 0x6FFC,
    0x7FFC, 0x9FFC, 0xBFFC, 0xDFFC, 0xFFFC, 0x13FFC, 0x17FFC, 0x1BFFC, 0x1FFFC, 0x27FFC, 0x2FFFC, 0x37FFC, 0x3FFFC,
];

const AMPLITUDE_TAB: [f32; 64] = [
    1.18750000, 1.68359380, 2.37500000, 3.36718750, 4.75000000, 6.73437500, 9.50000000, 13.4687500, 19.0000000,
    26.9375000, 38.0000000, 53.8750000, 76.0000000, 107.750000, 152.000000, 215.500000, 304.000000, 431.000000,
    608.000000, 862.000000, 1216.00000, 1724.00000, 2432.00000, 3448.00000, 4864.00000, 6896.00000, 9728.00000,
    13792.0000, 19456.0000, 27584.0000, 38912.0000, 55168.0000, 77824.0000, 110336.000, 155648.000, 220672.000,
    311296.000, 441344.000, 622592.000, 882688.000, 1245184.00, 1765376.00, 2490368.00, 3530752.00, 4980736.00,
    7061504.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

const NODES: [u16; 112] = [
    0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 56, 64, 80, 96, 120, 144, 176, 208, 240, 256, //
    0, 2, 4, 8, 16, 24, 32, 48, 56, 64, 80, 104, 128, 160, 208, 256, 0, 0, 0, 0, 0, //
    0, 2, 4, 8, 16, 32, 48, 64, 80, 112, 160, 208, 256, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 4, 8, 16, 32, 48, 64, 96, 144, 208, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 4, 16, 32, 64, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0,
];

const NOISE_BANDS_SIZE: [usize; 7] = [19, 14, 11, 9, 4, 2, 0];
const NOISE_BANDS_SELECTOR: [usize; 7] = [4, 3, 2, 1, 0, 0, 0];

/// (symbol, code length) pairs of the six trees, in code order; symbols are stored +1 (0 = escape).
const HUFFTAB: [(u8, u8); 132] = [
    // Noise value - 27 entries
    (1, 2), (10, 7), (26, 9), (22, 9), (24, 9), (14, 9), (8, 6), (6, 5), (7, 5), (9, 7), (30, 9), (32, 10),
    (13, 10), (20, 9), (28, 9), (12, 7), (15, 11), (36, 12), (0, 12), (34, 10), (18, 9), (11, 9), (16, 9), (5, 3),
    (2, 3), (4, 3), (3, 2),
    // Noise segment length - 12 entries
    (1, 1), (2, 2), (3, 4), (8, 9), (9, 10), (0, 10), (13, 8), (7, 7), (6, 6), (17, 5), (4, 4), (5, 4),
    // Amplitude - 28 entries
    (18, 3), (16, 3), (22, 7), (8, 10), (4, 10), (3, 9), (2, 8), (23, 8), (10, 8), (11, 7), (21, 5), (20, 4),
    (1, 7), (7, 10), (5, 10), (9, 9), (6, 10), (25, 11), (26, 12), (27, 13), (0, 13), (24, 9), (12, 6), (13, 5),
    (14, 4), (19, 3), (15, 3), (17, 2),
    // Frequency differences - 47 entries
    (2, 4), (14, 6), (26, 7), (31, 8), (32, 9), (35, 9), (7, 5), (10, 5), (22, 7), (27, 7), (19, 7), (20, 7),
    (4, 5), (13, 5), (17, 6), (15, 6), (8, 5), (5, 4), (28, 7), (33, 9), (36, 11), (38, 12), (42, 14), (45, 16),
    (44, 18), (0, 18), (46, 17), (43, 15), (40, 13), (37, 11), (39, 12), (41, 12), (34, 8), (16, 6), (11, 5), (9, 4),
    (1, 2), (3, 4), (30, 7), (29, 7), (23, 6), (24, 6), (18, 6), (6, 4), (12, 5), (21, 6), (25, 6),
    // Amplitude differences - 9 entries
    (1, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (0, 8), (2, 1),
    // Phase differences - 9 entries
    (2, 2), (1, 2), (3, 4), (7, 4), (6, 5), (5, 6), (0, 6), (4, 4), (8, 2),
];

const HUFF_SIZES: [usize; 6] = [27, 12, 28, 47, 9, 9];

/// The noise generator: r' = 214013 r + 2531011 (mod 2^32), and its k-step jumps (another affine map).
#[derive(Clone, Copy)]
struct Lcg {
    a: u32,
    c: u32,
}

impl Lcg {
    #[inline]
    fn apply(self, r: u32) -> u32 {
        self.a.wrapping_mul(r).wrapping_add(self.c)
    }
    /// `self` then `o`
    const fn then(self, o: Lcg) -> Lcg {
        Lcg { a: o.a.wrapping_mul(self.a), c: o.a.wrapping_mul(self.c).wrapping_add(o.c) }
    }
    const fn pow(self, mut k: usize) -> Lcg {
        let mut acc = Lcg { a: 1, c: 0 };
        let mut sq = self;
        while k > 0 {
            if k & 1 == 1 {
                acc = acc.then(sq);
            }
            sq = sq.then(sq);
            k >>= 1;
        }
        acc
    }
}

const LCG: Lcg = Lcg { a: 214013, c: 2531011 };
const LCG4: Lcg = LCG.pow(4);

/// Two distinct rows of the spectrum buffer, mutably.
fn two_rows(buf: &mut [f32], a: usize, b: usize) -> (&mut [f32], &mut [f32]) {
    debug_assert!(a < b);
    let (lo, hi) = buf.split_at_mut(b * FFT_ROW);
    (&mut lo[a * FFT_ROW..(a + 1) * FFT_ROW], &mut hi[..FFT_ROW])
}

struct Statics {
    vlc: Vec<Vlc>,
    sin_table: [f32; 512],
}

fn statics() -> &'static Statics {
    static S: OnceLock<Statics> = OnceLock::new();
    S.get_or_init(|| {
        let mut vlc = Vec::new();
        let mut at = 0;
        for n in HUFF_SIZES {
            vlc.push(Vlc::new(&HUFFTAB[at..at + n], -1));
            at += n;
        }
        let mut sin_table = [0f32; 512];
        for (i, s) in sin_table.iter_mut().enumerate() {
            // C: sin(2.0f * i * M_PI * 0.001953125f), in double
            *s = (((2.0f32 * i as f32) as f64) * std::f64::consts::PI * 0.001953125f32 as f64).sin() as f32;
        }
        Statics { vlc, sin_table }
    })
}

/// Size of the persistent spectrum rows (two frames of the largest size).
const FFT_ROW: usize = 8192 * 2;
const OUT_BUF: usize = 2 * 32768;

/// A QDMC decoder for one audio track.
pub struct Decoder {
    frame_bits: u32,
    band_index: usize,
    frame_size: usize,
    subframe_size: usize,
    fft_offset: usize,
    buffer_offset: usize,
    nb_channels: usize,
    checksum_size: usize,
    sample_rate: u32,
    bit_rate: u32,

    noise: [[[u8; 17]; 19]; 2],
    tones: Vec<Vec<Tone>>,
    cur_tone: [usize; 5],
    alt_sin: [[f32; 31]; 5],
    /// rows 0/1: imaginary parts of channel 0/1, rows 2/3: real parts
    fft_buffer: Vec<f32>,
    noise2_buffer: Vec<f32>,
    noise_buffer: Vec<f32>,
    buffer: Vec<f32>,
    rndval: u32,
    /// the generator's run for one subframe of one channel, and its values
    lcg_all: Lcg,
    rnd_vals: Vec<f32>,

    /// FFT work buffer
    work: Vec<Complex>,
    fft: InverseFft,
}

const MAX_TONES: usize = 8192;

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

impl Decoder {
    /// A decoder from the codec's extradata: the sample description bytes holding the `frma`/`QDMC`
    /// atom followed by the `QDCA` atom (FFmpeg's extradata for this codec: the 'wave' atom contents).
    /// The whole stsd entry may be passed; the `frma QDMC` marker is searched for.
    pub fn new(extradata: &[u8]) -> Result<Decoder, Error> {
        let st = statics();
        if extradata.len() < 48 {
            return Err(Error::Config("extradata missing or truncated".into()));
        }
        // skip to after "frmaQDMC" (or to the end - 8 when it is absent, as the C code does)
        let mut p = 0;
        while extradata.len() - p > 8 {
            if &extradata[p..p + 8] == b"frmaQDMC" {
                break;
            }
            p += 1;
        }
        p += 8;
        let left = extradata.len().saturating_sub(p);
        if left < 36 {
            return Err(Error::Config(format!("not enough extradata ({left})")));
        }
        let size = be32(extradata, p) as usize;
        if size > left - 4 {
            return Err(Error::Config(format!("extradata size too small, {} < {size}", left - 4)));
        }
        if &extradata[p + 4..p + 8] != b"QDCA" {
            return Err(Error::Config("invalid extradata, expecting QDCA".into()));
        }
        let q = p + 12; // + size, tag, version
        let nb_channels = be32(extradata, q) as i32;
        if !(1..=2).contains(&nb_channels) {
            return Err(Error::Config("invalid number of channels".into()));
        }
        let nb_channels = nb_channels as usize;
        let sample_rate = be32(extradata, q + 4);
        let bit_rate = be32(extradata, q + 8);
        let fft_size = be32(extradata, q + 16);
        let fft_order = if fft_size == 0 { 0 } else { 31 - fft_size.leading_zeros() + 1 };
        let checksum_size = be32(extradata, q + 20) as usize;
        if checksum_size >= 1 << 28 {
            return Err(Error::Config(format!("data block size too large ({checksum_size})")));
        }
        let (mut x, frame_bits) = if sample_rate >= 32000 {
            (28000, 13)
        } else if sample_rate >= 16000 {
            (20000, 12)
        } else {
            (16000, 11)
        };
        let frame_size = 1usize << frame_bits;
        let subframe_size = frame_size >> 5;
        if nb_channels == 2 {
            x = 3 * x / 2;
        }
        // llrint(floor(bit_rate * 3.0 / x + 0.5)), at most 6
        let sel = ((bit_rate as f64 * 3.0 / x as f64 + 0.5).floor() as i64).clamp(i64::MIN, 6);
        let band_index = NOISE_BANDS_SELECTOR[sel.max(0) as usize];
        if !(7..=9).contains(&fft_order) {
            return Err(Error::Config(format!("unknown FFT order {fft_order}")));
        }
        if fft_size != 1 << (fft_order - 1) {
            return Err(Error::Config(format!("FFT size {fft_size} not power of 2")));
        }
        if checksum_size < 6 {
            return Err(Error::Config(format!("frame size {checksum_size} too small")));
        }

        let mut alt_sin = [[0f32; 31]; 5];
        for g in (1..=5usize).rev() {
            for j in 0..(1usize << g) - 1 {
                alt_sin[5 - g][j] = st.sin_table[((j + 1) << (8 - g)) & 0x1FF];
            }
        }

        let mut d = Decoder {
            frame_bits,
            band_index,
            frame_size,
            subframe_size,
            fft_offset: 0,
            buffer_offset: 0,
            nb_channels,
            checksum_size,
            sample_rate,
            bit_rate,
            noise: [[[0; 17]; 19]; 2],
            tones: vec![Vec::new(); 5],
            cur_tone: [0; 5],
            alt_sin,
            fft_buffer: vec![0.0; 4 * FFT_ROW],
            noise2_buffer: vec![0.0; 4096 * 2],
            noise_buffer: vec![0.0; 4096 * 2],
            buffer: vec![0.0; OUT_BUF],
            rndval: 0,
            lcg_all: LCG.pow(2 * (subframe_size - 3)),
            rnd_vals: vec![0.0; 2 * subframe_size],
            work: vec![Complex::default(); 512],
            fft: InverseFft::new(1 << fft_order),
        };
        d.make_noises();
        Ok(d)
    }

    pub fn channels(&self) -> usize {
        self.nb_channels
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn bit_rate(&self) -> u32 {
        self.bit_rate
    }
    /// Bytes of one coded frame.
    pub fn frame_bytes(&self) -> usize {
        self.checksum_size
    }
    /// Samples per channel of one decoded frame.
    pub fn frame_samples(&self) -> usize {
        self.frame_size
    }

    /// Back to the initial state (FFmpeg's flush; the noise generator keeps running).
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.fft_buffer.fill(0.0);
        self.fft_offset = 0;
        self.buffer_offset = 0;
    }

    /// Decode every whole frame of `data` (a demuxed packet may hold several), appending interleaved
    /// samples to `out`. A trailing partial frame is an error. On a bad frame the decoder is reset (as FFmpeg
    /// does) and the error is returned; the frames before it stay in `out`.
    pub fn decode_packet(&mut self, data: &[u8], out: &mut Vec<i16>) -> Result<(), Error> {
        let mut rest = data;
        while !rest.is_empty() {
            let used = self.decode_frame(rest, out)?;
            rest = &rest[used..];
        }
        Ok(())
    }

    /// Decode the frame at the start of `data`, appending `frame_samples() * channels()` interleaved samples
    /// to `out`; returns the bytes used (`frame_bytes()`).
    pub fn decode_frame(&mut self, data: &[u8], out: &mut Vec<i16>) -> Result<usize, Error> {
        if data.len() < self.checksum_size {
            // FFmpeg rejects it before touching the state (no flush)
            return Err(Error::InvalidFrame("packet shorter than a frame"));
        }
        let mut br = BitReader::new(&data[..self.checksum_size]);
        for t in &mut self.tones {
            t.clear();
        }
        self.cur_tone = [0; 5];
        let start = out.len();
        out.resize(start + self.frame_size * self.nb_channels, 0);
        match self.frame(&mut br, data, &mut out[start..]) {
            Ok(()) => Ok(self.checksum_size),
            Err(e) => {
                out.truncate(start);
                self.reset();
                Err(e)
            }
        }
    }

    fn make_noises(&mut self) {
        let bi = self.band_index;
        for j in 0..NOISE_BANDS_SIZE[bi] {
            let n0 = NODES[j + 21 * bi] as usize;
            let n1 = NODES[j + 21 * bi + 1] as usize;
            let n2 = NODES[j + 21 * bi + 2] as usize;
            let base = 256 * j;
            for i in 0..n1.saturating_sub(n0) {
                self.noise_buffer[base + i] = i as f32 / (n1 - n0) as f32;
            }
            let mut diff = n2 as i32 - n1 as i32;
            let mut at = base + n1 - n0;
            for _ in n1..n2 {
                self.noise_buffer[at] = diff as f32 / (n2 - n1) as f32;
                at += 1;
                diff -= 1;
            }
        }
    }

    fn get_vlc(br: &mut BitReader, table: usize, flag: bool) -> Result<i32, Error> {
        if br.left() < 1 {
            return Err(Error::InvalidFrame("bitstream overread"));
        }
        let mut v = statics().vlc[table].read(br).unwrap_or(-1);
        if v < 0 {
            let n = br.bits(3) + 1;
            v = br.bits(n) as i32;
        }
        if flag {
            if v as usize >= CODE_PREFIX.len() {
                return Err(Error::InvalidFrame("invalid code prefix"));
            }
            v = (CODE_PREFIX[v as usize] + br.bits((v >> 2) as u32)) as i32;
        }
        Ok(v)
    }

    /// The 'QMC\1' label and the checksum of the frame (`data` = the whole frame bytes).
    fn skip_label(&self, br: &mut BitReader, data: &[u8]) -> bool {
        let label = br.bits(32);
        let checksum = br.bits(16) as u16;
        if label != u32::from_le_bytes([b'Q', b'M', b'C', 1]) {
            return false;
        }
        let mut sum: u16 = 226;
        for &b in &data[6..self.checksum_size] {
            sum = sum.wrapping_add(b as u16);
        }
        sum == checksum
    }

    fn read_noise_data(&mut self, br: &mut BitReader) -> Result<(), Error> {
        for ch in 0..self.nb_channels {
            for band in 0..NOISE_BANDS_SIZE[self.band_index] {
                let mut v = Self::get_vlc(br, 0, false)?;
                v = if v & 1 != 0 { v + 1 } else { -v };
                let mut lastval = v / 2;
                self.noise[ch][band][0] = (lastval - 1) as u8;
                let mut j = 0i32;
                while j < 15 {
                    let len = Self::get_vlc(br, 1, true)? + 1;
                    let v = Self::get_vlc(br, 0, false)?;
                    let newval = if v & 1 != 0 { lastval + (v + 1) / 2 } else { lastval - v / 2 };
                    let mut idx = j + 1;
                    if len + idx > 16 {
                        return Err(Error::InvalidFrame("noise segment too long"));
                    }
                    let mut k = 1;
                    while idx <= j + len {
                        self.noise[ch][band][idx as usize] = (lastval + k * (newval - lastval) / len - 1) as u8;
                        k += 1;
                        idx += 1;
                    }
                    lastval = newval;
                    j += len;
                }
            }
        }
        Ok(())
    }

    fn add_tone(&mut self, group: usize, offset: i32, freq: i32, mode: i32, amplitude: i32, phase: i32) {
        if self.tones[group].len() >= MAX_TONES {
            return; // FFmpeg: "Too many tones already in buffer, ignoring tone!"
        }
        self.tones[group].push(Tone {
            offset: offset as u8,
            freq: freq as i16,
            mode: mode as u8,
            amplitude: amplitude as i16,
            phase: phase as u8,
        });
    }

    fn read_wave_data(&mut self, br: &mut BitReader) -> Result<(), Error> {
        let mut stereo_mode = 0i32;
        for group in 0..5usize {
            let group_size = 1i32 << (self.frame_bits as usize - group - 1);
            let group_bits = 4 - group as i32;
            let mut pos2 = 0i32;
            let mut off = 0i32;
            let mut i = 1i32;
            loop {
                let v = Self::get_vlc(br, 3, true)?;
                let mut freq = i + v;
                while freq >= group_size - 1 {
                    freq += 2 - group_size;
                    pos2 += group_size;
                    off += 1 << group_bits;
                }
                if pos2 >= self.frame_size as i32 {
                    break;
                }
                if self.nb_channels > 1 {
                    stereo_mode = br.bits(2) as i32;
                }
                let amp = Self::get_vlc(br, 2, false)?;
                let phase = br.bits(3) as i32;
                let (mut amp2, mut phase2) = (0, 0);
                if stereo_mode > 1 {
                    amp2 = amp - Self::get_vlc(br, 4, false)?;
                    phase2 = phase - Self::get_vlc(br, 5, false)?;
                    if phase2 < 0 {
                        phase2 += 8;
                    }
                }
                if (freq >> group_bits) + 1 < self.subframe_size as i32 {
                    self.add_tone(group, off, freq, stereo_mode & 1, amp, phase);
                    if stereo_mode > 1 {
                        self.add_tone(group, off, freq, !stereo_mode & 1, amp2, phase2);
                    }
                }
                i = freq + 1;
            }
        }
        Ok(())
    }

    fn lin_calc(&mut self, amplitude: f32, node1: usize, node2: usize, index: usize) {
        let scale = 0.5 * amplitude;
        let sub = self.subframe_size.min(node2);
        if sub <= node1 {
            return;
        }
        let len = sub - node1;
        let nb = &self.noise_buffer[256 * index..256 * index + len];
        for (d, n) in self.noise2_buffer[node1..sub].iter_mut().zip(nb) {
            *d += scale * n;
        }
    }

    fn add_noise(&mut self, ch: usize, current_subframe: usize) {
        let ss = self.subframe_size;
        let base = self.fft_offset + ss * current_subframe;
        self.noise2_buffer[..ss].fill(0.0);
        let bi = self.band_index;
        let mut any = false;
        for i in 0..NOISE_BANDS_SIZE[bi] {
            if NODES[i + 21 * bi] as usize > ss - 1 {
                break;
            }
            let aindex = self.noise[ch][i][current_subframe / 2] as i32;
            let amplitude = if aindex > 0 { AMPLITUDE_TAB[(aindex & 0x3F) as usize] } else { 0.0 };
            // a silent band only adds zeros (FFmpeg adds them; the result is the same)
            if amplitude != 0.0 {
                any = true;
                self.lin_calc(amplitude, NODES[21 * bi + i] as usize, NODES[21 * bi + i + 2] as usize, i);
            }
        }
        // the generator runs 2 steps per bin j in 2..ss-1 whatever the levels
        let steps = 2 * (ss - 3);
        if !any {
            self.rndval = self.lcg_all.apply(self.rndval);
            return;
        }
        // the generator's values, from four interleaved streams of stride 4 (one serial chain would bound the loop)
        let vals = &mut self.rnd_vals[..steps];
        let mut s = [0u32; 4];
        let mut r = self.rndval;
        for q in &mut s {
            r = LCG.apply(r);
            *q = r;
        }
        let f = |r: u32| ((r & 0x7FFF) as f32 - 16384.0) * 0.000030517578;
        let mut k = 0;
        while k + 4 <= steps {
            for q in 0..4 {
                vals[k + q] = f(s[q]);
                s[q] = LCG4.apply(s[q]);
            }
            k += 4;
        }
        for q in 0..steps - k {
            vals[k + q] = f(s[q]);
        }
        self.rndval = self.lcg_all.apply(self.rndval);

        let (im, re) = two_rows(&mut self.fft_buffer, ch, ch + 2);
        let (im, re) = (&mut im[base..base + ss], &mut re[base..base + ss]);
        // bin j gets +rnd(j) and -rnd(j-1), in FFmpeg's order: (x - rnd(j-1)) + rnd(j)
        let (mut prev_im, mut prev_re) = (0.0f32, 0.0f32);
        for (((v, &n), di), dr) in vals
            .chunks_exact(2)
            .zip(&self.noise2_buffer[2..ss - 1])
            .zip(&mut im[2..ss - 1])
            .zip(&mut re[2..ss - 1])
        {
            let (rnd_im, rnd_re) = (v[0] * n, v[1] * n);
            *di = (*di - prev_im) + rnd_im;
            *dr = (*dr - prev_re) + rnd_re;
            prev_im = rnd_im;
            prev_re = rnd_re;
        }
        im[ss - 1] -= prev_im;
        re[ss - 1] -= prev_re;
    }

    fn add_wave(&mut self, t: Tone, group: usize) {
        let st = statics();
        let stereo_mode = if self.nb_channels == 1 { 0 } else { t.mode as usize };
        let (freqs, amp, phase) = (t.freq as i32, t.amplitude as i32, t.phase as i32);
        let group_bits = 4 - group as i32;
        let pos = (freqs >> group_bits) as usize;
        let amplitude = AMPLITUDE_TAB[(amp & 0x3F) as usize];
        let (ss, end) = (self.subframe_size, 2 * self.frame_size);
        let mut at = self.fft_offset + ss * t.offset as usize + pos;
        let mut pindex = (phase << 6) - ((2 * (freqs >> group_bits) + 1) << 7);
        let step = (2 * freqs + 1) << (7 - group_bits);
        let alt = &self.alt_sin[group][..(1usize << (group_bits + 1)) - 1];
        let (im_rows, re_rows) = two_rows(&mut self.fft_buffer, stereo_mode, stereo_mode + 2);
        let (im_rows, re_rows) = (&mut im_rows[..end], &mut re_rows[..end]);
        for &a in alt {
            pindex += step;
            let level = amplitude * a;
            let im = level * st.sin_table[(pindex & 0x1FF) as usize];
            let re = level * st.sin_table[((pindex + 128) & 0x1FF) as usize];
            im_rows[at] += im;
            im_rows[at + 1] -= im;
            re_rows[at] += re;
            re_rows[at + 1] -= re;
            at += ss;
            if at >= end {
                at = pos;
            }
        }
    }

    fn add_wave0(&mut self, t: Tone) {
        let st = statics();
        let stereo_mode = if self.nb_channels == 1 { 0 } else { t.mode as usize };
        let phase = t.phase as i32;
        let level = AMPLITUDE_TAB[(t.amplitude as i32 & 0x3F) as usize];
        let im = level * st.sin_table[((phase << 6) & 0x1FF) as usize];
        let re = level * st.sin_table[(((phase << 6) + 128) & 0x1FF) as usize];
        let pos = self.fft_offset + t.freq as usize + self.subframe_size * t.offset as usize;
        let (im_row, re_row) = (stereo_mode * FFT_ROW, (2 + stereo_mode) * FFT_ROW);
        self.fft_buffer[im_row + pos] += im;
        self.fft_buffer[re_row + pos] += re;
        self.fft_buffer[im_row + pos + 1] -= im;
        self.fft_buffer[re_row + pos + 1] -= re;
    }

    fn add_waves(&mut self, current_subframe: usize) {
        for g in 0..5 {
            let mut w = self.cur_tone[g];
            while w < self.tones[g].len() {
                let t = self.tones[g][w];
                if current_subframe < t.offset as usize {
                    break;
                }
                if g < 4 {
                    self.add_wave(t, g);
                } else {
                    self.add_wave0(t);
                }
                w += 1;
            }
            self.cur_tone[g] = w;
        }
    }

    /// One subframe's spectra (at `base` in the spectrum rows) → time samples overlap-added into `buffer` at `r`
    /// (interleaved, 2 * subframe_size frames). The usual case: the FFT is twice the subframe, so each spectrum fills
    /// its lower half. Only the real part of the inverse transform is used, which is the inverse transform of the
    /// spectrum's Hermitian part H[k] = (X[k] + conj(X[-k])) / 2 — real. In stereo the two channels' Hermitian parts
    /// go into one transform as H0 + i*H1, whose output is y0 + i*y1: one FFT for both channels. In mono the upper
    /// half stays zero, which the FFT skips.
    fn synth_fast(&mut self, base: usize, r: usize) {
        let ss = self.subframe_size;
        let n = 2 * ss;
        let w = &mut self.work[..n];
        let row = |k: usize| &self.fft_buffer[k * FFT_ROW + base..k * FFT_ROW + base + ss];
        if self.nb_channels == 2 {
            let (i0, i1, r0, r1) = (row(0), row(1), row(2), row(3));
            w[0] = Complex { re: r0[0], im: r1[0] };
            w[ss] = Complex::default();
            for k in 1..ss {
                let (x0r, x0i, x1r, x1i) = (r0[k], i0[k], r1[k], i1[k]);
                w[k] = Complex { re: 0.5 * (x0r - x1i), im: 0.5 * (x0i + x1r) };
                w[n - k] = Complex { re: 0.5 * (x0r + x1i), im: 0.5 * (x1r - x0i) };
            }
            self.fft.run(w);
            let rev = &self.fft.rev;
            for (o, &p) in self.buffer[r..r + 2 * n].chunks_exact_mut(2).zip(rev.iter()) {
                let y = w[p as usize];
                o[0] += y.re;
                o[1] += y.im;
            }
        } else {
            let (i0, r0) = (row(0), row(2));
            for ((c, &re), &im) in w[..ss].iter_mut().zip(r0).zip(i0) {
                *c = Complex { re, im };
            }
            self.fft.run_half(w);
            let rev = &self.fft.rev;
            for (o, &p) in self.buffer[r..r + n].iter_mut().zip(rev.iter()) {
                *o += w[p as usize].re;
            }
        }
    }

    /// Any other FFT size (a stream whose QDCA subframe field disagrees with its sample rate): FFmpeg's layout as is.
    fn synth_generic(&mut self, base: usize, r: usize) {
        let (ss, nch, nf) = (self.subframe_size, self.nb_channels, self.fft.len());
        for ch in 0..nch {
            let w = &mut self.work[..nf];
            w.fill(Complex::default());
            for i in 0..ss.min(nf) {
                w[i] = Complex {
                    re: self.fft_buffer[(ch + 2) * FFT_ROW + base + i],
                    im: self.fft_buffer[ch * FFT_ROW + base + i],
                };
            }
            self.fft.run(w);
            for i in 0..2 * ss {
                if i < nf {
                    self.buffer[r + i * nch + ch] += w[self.fft.rev[i] as usize].re;
                }
            }
        }
    }

    fn frame(&mut self, br: &mut BitReader, data: &[u8], out: &mut [i16]) -> Result<(), Error> {
        if !self.skip_label(br, data) {
            return Err(Error::InvalidFrame("bad label or checksum"));
        }
        self.fft_offset = self.frame_size - self.fft_offset;
        let nch = self.nb_channels;
        let ss = self.subframe_size;
        let bp = nch * self.buffer_offset;

        self.read_noise_data(br)?;
        self.read_wave_data(br)?;

        let n_fft = self.fft.len();
        for n in 0..32 {
            for ch in 0..nch {
                self.add_noise(ch, n);
            }
            self.add_waves(n);

            let base = self.fft_offset + n * ss;
            let r = bp + nch * n * ss;
            if n_fft == 2 * ss {
                self.synth_fast(base, r);
            } else {
                self.synth_generic(base, r);
            }
            let r = bp + n * ss * nch;
            for (i, o) in out[n * ss * nch..(n + 1) * ss * nch].iter_mut().enumerate() {
                // av_clipf, then the implicit float -> int16 conversion (truncation)
                *o = self.buffer[r + i].clamp(-32768.0, 32767.0) as i16;
            }
            for ch in 0..nch {
                self.fft_buffer[ch * FFT_ROW + base..ch * FFT_ROW + base + ss].fill(0.0);
                self.fft_buffer[(ch + 2) * FFT_ROW + base..(ch + 2) * FFT_ROW + base + ss].fill(0.0);
            }
            let z = nch * (n * ss + self.frame_size + self.buffer_offset);
            self.buffer[z..z + ss * nch].fill(0.0);
        }

        self.buffer_offset += self.frame_size;
        if self.buffer_offset >= 32768 - self.frame_size {
            let from = nch * self.buffer_offset;
            self.buffer.copy_within(from..from + self.frame_size * nch, 0);
            self.buffer_offset = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trees_are_complete() {
        let mut at = 0;
        for n in HUFF_SIZES {
            let k: f64 = HUFFTAB[at..at + n].iter().map(|p| 2f64.powi(-(p.1 as i32))).sum();
            assert_eq!(k, 1.0);
            at += n;
        }
    }

    /// Every code of every tree, written into a stream (first code bit first, LSB-first packing) and followed by
    /// other bits, decodes to its symbol and consumes exactly its length — through the table and the long-code path.
    #[test]
    fn every_code_decodes() {
        let mut at = 0;
        for (t, n) in HUFF_SIZES.into_iter().enumerate() {
            let mut code: u64 = 0;
            for &(sym, len) in &HUFFTAB[at..at + n] {
                let c = (code >> (32 - len)) as u32;
                code += 1u64 << (32 - len);
                for tail in [0u32, 0x5555_5555, 0xffff_ffff] {
                    // stream bits: the code MSB first, then the tail
                    let mut v: u64 = 0;
                    for i in 0..len as u32 {
                        v |= (((c >> (len as u32 - 1 - i)) & 1) as u64) << i;
                    }
                    v |= (tail as u64) << len;
                    let bytes = v.to_le_bytes();
                    let mut br = BitReader::new(&bytes);
                    assert_eq!(statics().vlc[t].read(&mut br), Some(sym as i32 - 1), "tree {t} sym {sym}");
                    assert_eq!(br.left(), 64 - len as isize, "tree {t} sym {sym}");
                }
            }
            at += n;
        }
    }

    #[test]
    fn lcg_jumps() {
        for k in [0usize, 1, 2, 3, 4, 5, 250, 506] {
            let mut r = 12345u32;
            for _ in 0..k {
                r = r.wrapping_mul(214013).wrapping_add(2531011);
            }
            assert_eq!(LCG.pow(k).apply(12345), r, "k {k}");
        }
    }

    #[test]
    fn rejects_bad_extradata() {
        assert!(Decoder::new(&[0u8; 10]).is_err());
        assert!(Decoder::new(&[0u8; 64]).is_err());
    }
}
