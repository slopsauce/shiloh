// mutate real frames (bits flipped, checksum fixed) and decode: must never panic
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let file = std::fs::read(&a[1]).unwrap();
    let i = file.windows(4).position(|w| w == b"stsd").unwrap() + 12;
    let size = u32::from_be_bytes(file[i..i + 4].try_into().unwrap()) as usize;
    let extra = file[i..i + size].to_vec();
    let mut dec = qdmc::Decoder::new(&extra).unwrap();
    let fb = dec.frame_bytes();
    // frames: find 'QMC\x01' labels
    let starts: Vec<usize> = file.windows(4).enumerate().filter(|(_, w)| *w == b"QMC\x01").map(|(i, _)| i).take(200).collect();
    let mut rng = 0x1234_5678u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let (mut ok, mut bad) = (0, 0);
    let mut out = Vec::new();
    for it in 0..30000 {
        let s = starts[it % starts.len()];
        let mut f = file[s..s + fb].to_vec();
        for _ in 0..(1 + next() % 40) {
            let p = 6 + (next() as usize) % (fb - 6);
            f[p] ^= 1 << (next() % 8);
        }
        let sum = f[6..].iter().fold(226u16, |a, &b| a.wrapping_add(b as u16));
        f[4..6].copy_from_slice(&sum.to_le_bytes());
        out.clear();
        match dec.decode_frame(&f, &mut out) { Ok(_) => ok += 1, Err(_) => bad += 1 }
    }
    println!("{}: {ok} decoded, {bad} rejected, no panic", a[1]);
}
