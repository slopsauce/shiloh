# qdmc

> This directory is licensed separately from the rest of the repository: **LGPL-2.1-or-later** (see
> `COPYING` here), not MIT/Apache-2.0. Shiloh uses it only with the optional `qdmc` feature of `port/`.

A decoder for **QDesign Music Codec 1** (`QDMC`), the audio codec of QuickTime 3 era movies (1998–2000):
pure Rust, no dependencies, no `unsafe`.

It is a port of FFmpeg's `libavcodec/qdmc.c` (Copyright (c) 2017 Paul B Mahol) and is therefore licensed
like the original: **GNU Lesser General Public License, version 2.1 or (at your option) any later version**
(see `COPYING`).

## How the codec works

A frame (2048, 4096 or 8192 samples per channel, by sample rate) starts with a `QMC\1` label and a checksum,
then carries:

- **noise**: per channel, a level for each of up to 19 bands, as a piecewise linear curve over the frame's 16
  subframe pairs; synthesised as pseudo-random spectral values shaped by triangular band windows;
- **tones**: in five groups of halving duration (from 31 subframes down to one), each with a frequency
  (differential), an amplitude (index into a table in steps of √2, i.e. 3 dB), a 3-bit phase and, in stereo, a mode that puts it in one
  channel or both (with amplitude/phase differences for the second one); a tone is spread over the subframes it spans
  with a sine-shaped envelope.

Every subframe, the noise and tones written into a spectrum of the subframe size become PCM through an inverse FFT of
twice that length, overlapped by half. Huffman trees are little-endian bit-reversed canonical codes; numbers may
escape to raw bits.

## API

```rust
let mut dec = qdmc::Decoder::new(extradata)?; // stsd entry bytes (holds `frma QDMC` + the `QDCA` atom)
let mut pcm = Vec::new();                      // interleaved i16
for packet in packets {
    dec.decode_packet(packet, &mut pcm)?;      // whole frames of dec.frame_bytes() bytes
}
println!("{} Hz, {} ch", dec.sample_rate(), dec.channels());
```

## qdmc2wav

```sh
cargo run --release --bin qdmc2wav -- IN.mov OUT.wav [--compare REF.s16]
```

Decodes the first QDMC track of a QuickTime file (whole media, edit lists ignored) to WAV. `--compare` checks the
result against a raw reference made with
`ffmpeg -ignore_editlist 1 -i IN.mov -map 0:a:0 -f s16le -c:a pcm_s16le REF.s16`.

## Validation against FFmpeg 9.0.2

Same length on every file; samples differ by at most 1 LSB (float rounding: FFmpeg's FFT vs this one, then the same
truncation to 16 bit):

| file | format | frames | samples off by 1 |
|---|---|---|---|
| X-Files Game `XS/72106.MUS`, `72107.MUS`, `72108.MUS`, `80788.MUS` | 44.1 kHz stereo, 48 kbit/s | 4795 | 0.009–0.050 % |
| X-Files Game `XV/64421.XMV`, `64657.XMV`, `XS/89487.AMV` | 44.1 kHz stereo, 48 kbit/s | 83 | 0.010–0.052 % |
| X-Files Game `XS/92289.DMV` | 11.025 kHz mono, 24 kbit/s | 28 | 0.016 % |
| MPlayer samples `rumcoke.mov`, `slick.mov` | 44.1 kHz stereo, 24 kbit/s | 2871 | 0.040 % |
| MPlayer sample `tidemo1-24bit-rle.mov` | 22.05 kHz mono, 64 kbit/s | 378 | 0.009 % |

`cargo run --release --example fuzz -- FILE.mov` decodes 30 000 mutated frames (bits flipped, checksum fixed):
bad frames are rejected or decoded, never a panic.

## Speed

One core of a 2-vCPU Linux VM, best of 5, `qdmc2wav` decode time:

| file | length | first port | now | |
|---|---|---|---|---|
| `XS/72106.MUS` (stereo) | 5 min 45 | 981 ms | 476 ms | 2.1× |
| `XS/80788.MUS` (stereo) | 1 min 37 | 274 ms | 124 ms | 2.2× |
| `XV/64421.XMV` (stereo) | 6.3 s | 17 ms | 7.6 ms | 2.2× |
| `rumcoke.mov` (stereo) | 4 min 27 | 681 ms | 316 ms | 2.2× |
| `tidemo1-24bit-rle.mov` (mono) | 1 min 10 | 54 ms | 31 ms | 1.7× |

What made it faster: in stereo the two channels share one complex inverse FFT (only the real part of each channel's
transform is needed: the transform of its spectrum's Hermitian part, real, so H0 + i·H1 gives both); the FFT is a
radix-2² decimation in frequency in f32 that skips the zero upper half in mono; Huffman codes are read through
10-bit lookup tables with a word-at-a-time bit reader; the noise generator's values come from four interleaved
jump-ahead streams and silent bands are skipped. Output is unchanged by all but the FFT (rounding).
