// Unnormalised inverse complex FFT (exp(+2*pi*i*n*k/N)), the transform FFmpeg's QDMC decoder gets from
// `av_tx_init(AV_TX_FLOAT_FFT, inv = 1, scale = 1)`.
//
// Radix-2 decimation in frequency, in place, in f32 with twiddles computed in f64: natural-order input,
// bit-reversed output (read it through `rev`). The decoder's spectra have their upper half zero (mono) — that
// first stage is then a plain copy and a multiply (`run_half`).
//
// This file is part of the qdmc crate, a Rust port of FFmpeg's QDMC decoder.
// Licensed under the GNU Lesser General Public License, version 2.1 or later (see COPYING).

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct Complex {
    pub re: f32,
    pub im: f32,
}

#[inline(always)]
fn mul(a: Complex, w: Complex) -> Complex {
    Complex { re: a.re * w.re - a.im * w.im, im: a.re * w.im + a.im * w.re }
}

#[inline(always)]
fn neg(w: Complex) -> Complex {
    Complex { re: -w.re, im: -w.im }
}

pub(crate) struct InverseFft {
    n: usize,
    /// per stage (size n, n/2, … 2): its twiddles exp(+2*pi*i*k/size), k < size/2, one after the other
    tw: Vec<Complex>,
    /// bit reversal: the output of index i is at `rev[i]`
    pub rev: Vec<u32>,
}

impl InverseFft {
    pub fn new(n: usize) -> InverseFft {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let mut tw = Vec::with_capacity(n);
        let mut size = n;
        while size >= 2 {
            for k in 0..size / 2 {
                let a = 2.0 * std::f64::consts::PI * k as f64 / size as f64;
                tw.push(Complex { re: a.cos() as f32, im: a.sin() as f32 });
            }
            size /= 2;
        }
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        InverseFft { n, tw, rev }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// In place: `buf[rev[k]] = sum_j buf[j] * exp(+2*pi*i*j*k/n)`.
    pub fn run(&self, buf: &mut [Complex]) {
        self.stages(buf, self.n, 0);
    }

    /// As [`run`](Self::run) for an input whose upper half (`buf[n/2..]`) is zero (it need not be cleared).
    pub fn run_half(&self, buf: &mut [Complex]) {
        let half = self.n / 2;
        let (lo, hi) = buf[..self.n].split_at_mut(half);
        for ((a, b), w) in lo.iter().zip(hi.iter_mut()).zip(&self.tw[..half]) {
            *b = Complex { re: a.re * w.re - a.im * w.im, im: a.re * w.im + a.im * w.re };
        }
        self.stages(buf, half, half);
    }

    /// The DIF stages from block size `size` down to 2; `tw_at` = where that stage's twiddles start. Two radix-2
    /// stages are done at once (radix 2^2: per 4 points 3 twiddle multiplies and one pass over the data instead of 4
    /// and two), a last odd stage alone.
    fn stages(&self, buf: &mut [Complex], mut size: usize, mut tw_at: usize) {
        let buf = &mut buf[..self.n];
        while size >= 8 {
            let q = size / 4;
            // twiddles of size `size`: w^k, k < size/2 (w^2k and w^3k are taken from the same table, w^3k via -w^(3k-size/2))
            let tw = &self.tw[tw_at..tw_at + size / 2];
            for block in buf.chunks_exact_mut(size) {
                let (b01, b23) = block.split_at_mut(2 * q);
                let (b0, b1) = b01.split_at_mut(q);
                let (b2, b3) = b23.split_at_mut(q);
                for k in 0..q {
                    let (x0, x1, x2, x3) = (b0[k], b1[k], b2[k], b3[k]);
                    let (s02r, s02i, d02r, d02i) = (x0.re + x2.re, x0.im + x2.im, x0.re - x2.re, x0.im - x2.im);
                    let (s13r, s13i, d13r, d13i) = (x1.re + x3.re, x1.im + x3.im, x1.re - x3.re, x1.im - x3.im);
                    b0[k] = Complex { re: s02r + s13r, im: s02i + s13i };
                    // z1 = (s02 - s13) w^2k
                    let w2 = if 2 * k < size / 2 { tw[2 * k] } else { neg(tw[2 * k - size / 2]) };
                    b1[k] = mul(Complex { re: s02r - s13r, im: s02i - s13i }, w2);
                    // z2 = (d02 + i d13) w^k, z3 = (d02 - i d13) w^3k (inverse transform: w^(size/4) = +i)
                    b2[k] = mul(Complex { re: d02r - d13i, im: d02i + d13r }, tw[k]);
                    let w3 = if 3 * k < size / 2 { tw[3 * k] } else { neg(tw[3 * k - size / 2]) };
                    b3[k] = mul(Complex { re: d02r + d13i, im: d02i - d13r }, w3);
                }
            }
            // the next size: size / 4; skip the twiddles of size / 2
            tw_at += size / 2 + size / 4;
            size /= 4;
        }
        while size >= 2 {
            let half = size / 2;
            let tw = &self.tw[tw_at..tw_at + half];
            if half == 1 {
                for p in buf.chunks_exact_mut(2) {
                    let (a, b) = (p[0], p[1]);
                    p[0] = Complex { re: a.re + b.re, im: a.im + b.im };
                    p[1] = Complex { re: a.re - b.re, im: a.im - b.im };
                }
            } else {
                for block in buf.chunks_exact_mut(size) {
                    let (lo, hi) = block.split_at_mut(half);
                    for ((a, b), w) in lo.iter_mut().zip(hi.iter_mut()).zip(tw) {
                        let (dr, di) = (a.re - b.re, a.im - b.im);
                        a.re += b.re;
                        a.im += b.im;
                        b.re = dr * w.re - di * w.im;
                        b.im = dr * w.im + di * w.re;
                    }
                }
            }
            tw_at += half;
            size = half;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(inp: &[Complex]) -> Vec<(f64, f64)> {
        let n = inp.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, x) in inp.iter().enumerate() {
                    let a = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    re += x.re as f64 * a.cos() - x.im as f64 * a.sin();
                    im += x.re as f64 * a.sin() + x.im as f64 * a.cos();
                }
                (re, im)
            })
            .collect()
    }

    #[test]
    fn matches_direct_dft() {
        for n in [2usize, 4, 16, 128, 512] {
            let inp: Vec<Complex> =
                (0..n).map(|i| Complex { re: (i as f32 * 0.37).sin(), im: (i as f32 * 1.3).cos() }).collect();
            let want = direct(&inp);
            let f = InverseFft::new(n);
            let mut b = inp.clone();
            f.run(&mut b);
            for k in 0..n {
                let o = b[f.rev[k] as usize];
                assert!((o.re as f64 - want[k].0).abs() < 1e-3 && (o.im as f64 - want[k].1).abs() < 1e-3, "n {n} bin {k}");
            }
            // upper half zero
            let mut h = inp.clone();
            h[n / 2..].fill(Complex::default());
            let want = direct(&h);
            h[n / 2..].fill(Complex { re: 9.0, im: 9.0 }); // garbage: run_half must not read it
            f.run_half(&mut h);
            for k in 0..n {
                let o = h[f.rev[k] as usize];
                assert!((o.re as f64 - want[k].0).abs() < 1e-3 && (o.im as f64 - want[k].1).abs() < 1e-3, "half n {n} bin {k}");
            }
        }
    }
}
