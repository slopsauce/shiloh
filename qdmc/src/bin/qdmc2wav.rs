//! qdmc2wav — decode the first QDMC sound track of a QuickTime file to WAV (whole media, edit lists ignored,
//! like `ffmpeg -ignore_editlist 1`), and optionally compare it with a raw s16le reference.
//!
//!   qdmc2wav IN.mov OUT.wav [--compare REF.s16]
//!
//! Part of the qdmc crate (LGPL-2.1-or-later).

use std::process::exit;

fn be16(b: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([b[at], b[at + 1]])
}
fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn be64(b: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
}

/// (type, payload) of the atoms in `b`.
fn atoms(b: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut v = Vec::new();
    let mut p = 0;
    while p + 8 <= b.len() {
        let mut size = be32(b, p) as usize;
        let ty: [u8; 4] = b[p + 4..p + 8].try_into().unwrap();
        let mut hdr = 8;
        if size == 1 && p + 16 <= b.len() {
            size = be64(b, p + 8) as usize;
            hdr = 16;
        } else if size == 0 {
            size = b.len() - p;
        }
        if size < hdr || p + size > b.len() {
            break;
        }
        v.push((ty, &b[p + hdr..p + size]));
        p += size;
    }
    v
}

fn child<'a>(b: &'a [u8], ty: &[u8; 4]) -> Option<&'a [u8]> {
    atoms(b).into_iter().find(|(t, _)| t == ty).map(|(_, d)| d)
}

struct Track<'a> {
    stsd_entry: &'a [u8],
    stbl: &'a [u8],
}

fn find_qdmc_track(moov: &[u8]) -> Option<Track<'_>> {
    for (t, trak) in atoms(moov) {
        if &t != b"trak" {
            continue;
        }
        let stbl = child(trak, b"mdia").and_then(|m| child(m, b"minf")).and_then(|m| child(m, b"stbl"))?;
        let stsd = child(stbl, b"stsd")?;
        // version/flags (4), entry count (4), then entries: size (4), format (4), ...
        if stsd.len() < 16 || &stsd[12..16] != b"QDMC" {
            continue;
        }
        let size = be32(stsd, 8) as usize;
        return Some(Track { stsd_entry: &stsd[8..(8 + size).min(stsd.len())], stbl });
    }
    None
}

/// The track's coded bytes, chunk after chunk (QDMC frames have a fixed size, so frame boundaries follow).
fn track_bytes(file: &[u8], tr: &Track) -> Result<Vec<u8>, String> {
    let e = tr.stsd_entry;
    // sound description: 8 (size, format) + 6 reserved + 2 dref index, then version at 16
    let version = be16(e, 16);
    let (spp, bpf) = if version == 1 && e.len() >= 52 { (be32(e, 36), be32(e, 44)) } else { (0, 0) };
    let stsc = child(tr.stbl, b"stsc").ok_or("no stsc")?;
    let stsz = child(tr.stbl, b"stsz").ok_or("no stsz")?;
    let offsets: Vec<u64> = if let Some(c) = child(tr.stbl, b"stco") {
        (0..be32(c, 4) as usize).map(|i| be32(c, 8 + 4 * i) as u64).collect()
    } else if let Some(c) = child(tr.stbl, b"co64") {
        (0..be32(c, 4) as usize).map(|i| be64(c, 8 + 8 * i)).collect()
    } else {
        return Err("no stco/co64".into());
    };
    let n_stsc = be32(stsc, 4) as usize;
    let stsc_e: Vec<(u32, u32)> = (0..n_stsc).map(|i| (be32(stsc, 8 + 12 * i), be32(stsc, 12 + 12 * i))).collect();
    let sample_size = be32(stsz, 4);
    let n_samples = be32(stsz, 8) as usize;
    let sizes: Vec<u32> = if sample_size == 0 { (0..n_samples).map(|i| be32(stsz, 12 + 4 * i)).collect() } else { vec![] };

    let mut out = Vec::new();
    let mut sample = 0usize;
    for (ci, &off) in offsets.iter().enumerate() {
        let chunk = ci as u32 + 1;
        let per = stsc_e.iter().rev().find(|(first, _)| *first <= chunk).map(|e| e.1).unwrap_or(0) as usize;
        let bytes = if sample_size == 0 {
            sizes[sample..sample + per].iter().map(|&s| s as usize).sum::<usize>()
        } else if spp > 0 {
            per / spp as usize * bpf as usize
        } else {
            per * sample_size as usize
        };
        sample += per;
        let (a, b) = (off as usize, off as usize + bytes);
        if b > file.len() {
            return Err(format!("chunk {chunk} past the end of the file"));
        }
        out.extend_from_slice(&file[a..b]);
    }
    Ok(out)
}

fn wav(path: &str, pcm: &[i16], ch: u16, rate: u32) -> std::io::Result<()> {
    let data_len = (pcm.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + pcm.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&ch.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * ch as u32 * 2).to_le_bytes());
    b.extend_from_slice(&(ch * 2).to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        b.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: qdmc2wav IN.mov OUT.wav [--compare REF.s16]");
        exit(2);
    }
    let die = |m: String| -> ! {
        eprintln!("qdmc2wav: {m}");
        exit(1)
    };
    let file = std::fs::read(&args[1]).unwrap_or_else(|e| die(format!("{}: {e}", args[1])));
    let moov = child(&file, b"moov").unwrap_or_else(|| die("no moov atom".into()));
    let tr = find_qdmc_track(moov).unwrap_or_else(|| die("no QDMC track".into()));
    let mut dec = qdmc::Decoder::new(tr.stsd_entry).unwrap_or_else(|e| die(e.to_string()));
    let coded = track_bytes(&file, &tr).unwrap_or_else(|e| die(e));
    let fb = dec.frame_bytes();
    if coded.len() % fb != 0 {
        eprintln!("qdmc2wav: note: {} coded bytes is not a multiple of the frame size {fb}", coded.len());
    }
    let t0 = std::time::Instant::now();
    let mut pcm = Vec::new();
    let mut bad = 0;
    for f in coded.chunks_exact(fb) {
        if let Err(e) = dec.decode_frame(f, &mut pcm) {
            bad += 1;
            eprintln!("qdmc2wav: {e}");
        }
    }
    let el = t0.elapsed();
    let ch = dec.channels();
    let secs = pcm.len() as f64 / ch as f64 / dec.sample_rate() as f64;
    println!(
        "{}: {} Hz, {} ch, {} bit/s, {} frames of {} bytes ({} bad) -> {:.3} s, decoded in {:.1} ms ({:.0}x real time)",
        args[1],
        dec.sample_rate(),
        ch,
        dec.bit_rate(),
        coded.len() / fb,
        fb,
        bad,
        secs,
        el.as_secs_f64() * 1e3,
        secs / el.as_secs_f64().max(1e-9)
    );
    wav(&args[2], &pcm, ch as u16, dec.sample_rate()).unwrap_or_else(|e| die(format!("{}: {e}", args[2])));
    if args.len() >= 5 && args[3] == "--compare" {
        let r = std::fs::read(&args[4]).unwrap_or_else(|e| die(format!("{}: {e}", args[4])));
        let reference: Vec<i16> = r.chunks_exact(2).map(|x| i16::from_le_bytes([x[0], x[1]])).collect();
        let n = reference.len().min(pcm.len());
        let (mut maxd, mut ndiff, mut sq) = (0i32, 0usize, 0f64);
        for i in 0..n {
            let d = (pcm[i] as i32 - reference[i] as i32).abs();
            if d > 0 {
                ndiff += 1;
            }
            maxd = maxd.max(d);
            sq += (d * d) as f64;
        }
        let peak = reference.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
        println!(
            "  vs reference: {} / {} samples, {} differ ({:.4} %), max |diff| {}, rms diff {:.4}, reference peak {}",
            pcm.len(),
            reference.len(),
            ndiff,
            100.0 * ndiff as f64 / n.max(1) as f64,
            maxd,
            (sq / n.max(1) as f64).sqrt(),
            peak
        );
        if pcm.len() != reference.len() || maxd > 1 {
            exit(3);
        }
    }
}
