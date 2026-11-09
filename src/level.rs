//! Normalisasi level keluaran: loudness (BS.1770) + pembatas puncak.
//!
//! Keluaran mentah mesin ini sekitar −28 LUFS dengan puncak −10 dBFS: terlalu
//! pelan untuk didengar langsung, dan kalau digain mentah-mentah puncaknya
//! menembus 0 dBFS. Jadi dua langkah: naikkan sampai target loudness, lalu
//! batasi puncak dengan lookahead 5 ms supaya tidak pecah.

#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

use std::collections::VecDeque;

/// Koefisien K-weighting BS.1770 untuk sample rate apa pun.
///
/// Spesifikasi hanya memberi angka jadi pada 48 kHz; di sini filter diturunkan
/// ulang dari parameter aslinya (high-shelf 1681,97 Hz +3,9998 dB Q 0,7072 dan
/// high-pass 38,1355 Hz Q 0,5003) sehingga berlaku untuk 24 kHz juga.
fn k_weighting(fs: f64) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    // Tahap 1: high shelf, diturunkan dengan prewarp K = tan(pi f0 / fs).
    let f0 = 1_681.974_450_955_533_f64;
    let gain = 3.999_843_853_973_347_f64;
    let q = 0.707_175_236_955_419_6_f64;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(gain / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let shelf_b = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
    ];
    let shelf_a = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0];

    // Tahap 2: high-pass 38 Hz.
    let f0 = 38.135_470_876_024_44_f64;
    let q = 0.500_327_037_323_877_3_f64;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let hp_b = [1.0, -2.0, 1.0];
    let hp_a = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0];
    (shelf_b, shelf_a, hp_b, hp_a)
}

fn biquad(x: &[f32], b: [f64; 3], a: [f64; 3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &v in x {
        let xi = v as f64;
        let yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2;
        x2 = x1;
        x1 = xi;
        y2 = y1;
        y1 = yi;
        out.push(yi as f32);
    }
    out
}

/// Loudness terintegrasi (LUFS) menurut BS.1770: blok 400 ms, hop 100 ms,
/// gerbang absolut −70 LUFS dan gerbang relatif −10 LU.
pub fn integrated_lufs(x: &[f32], fs: f64) -> f32 {
    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    let (sb, sa, hb, ha) = k_weighting(fs);
    let y = biquad(&biquad(x, sb, sa), hb, ha);

    let blok = (0.4 * fs) as usize;
    let hop = (0.1 * fs) as usize;
    if y.len() < blok || hop == 0 {
        return f32::NEG_INFINITY;
    }
    let mut daya: Vec<f64> = Vec::new();
    let mut i = 0;
    while i + blok <= y.len() {
        let mut s = 0.0f64;
        for v in &y[i..i + blok] {
            s += (*v as f64) * (*v as f64);
        }
        daya.push(s / blok as f64);
        i += hop;
    }
    let lufs = |z: f64| -0.691 + 10.0 * z.max(1e-20).log10();

    // Gerbang absolut.
    let lolos: Vec<f64> = daya.iter().copied().filter(|z| lufs(*z) > -70.0).collect();
    if lolos.is_empty() {
        return f32::NEG_INFINITY;
    }
    // Gerbang relatif terhadap rata-rata yang sudah lolos gerbang absolut.
    let rata = lolos.iter().sum::<f64>() / lolos.len() as f64;
    let ambang = lufs(rata) - 10.0;
    let akhir: Vec<f64> = lolos
        .iter()
        .copied()
        .filter(|z| lufs(*z) > ambang)
        .collect();
    if akhir.is_empty() {
        return f32::NEG_INFINITY;
    }
    lufs(akhir.iter().sum::<f64>() / akhir.len() as f64) as f32
}

/// Puncak sampel dalam dBFS.
pub fn peak_dbfs(x: &[f32]) -> f32 {
    let p = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if p <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * p.log10()
    }
}

/// Naikkan ke target loudness, lalu batasi puncak.
///
/// Kembalikan (gain dB, LUFS sebelum, puncak sesudah).
pub fn normalize(x: &mut [f32], fs: f64, target_lufs: f32, ceiling_db: f32) -> (f32, f32, f32) {
    let awal = integrated_lufs(x, fs);
    if !awal.is_finite() {
        return (0.0, awal, peak_dbfs(x));
    }
    let ceiling = 10f32.powf(ceiling_db / 20.0);
    let mut total = 0.0f32;
    // Pembatasan puncak menurunkan loudness, jadi gain-nya diulang sampai
    // target tercapai (biasanya cukup dua putaran).
    for _ in 0..3 {
        let sekarang = integrated_lufs(x, fs);
        if !sekarang.is_finite() {
            break;
        }
        let perlu = (target_lufs - sekarang).clamp(-12.0, 36.0);
        if perlu.abs() < 0.3 {
            break;
        }
        let g = 10f64.powf(perlu as f64 / 20.0) as f32;
        for v in x.iter_mut() {
            *v *= g;
        }
        total += perlu;
        if peak_dbfs(x) > ceiling_db {
            limit(x, fs, ceiling);
        }
    }
    (total, awal, peak_dbfs(x))
}

/// Pembatas puncak dengan lookahead 5 ms dan pelepasan 50 ms.
fn limit(x: &mut [f32], fs: f64, ceiling: f32) {
    let n = x.len();
    let la = ((0.005 * fs) as usize).max(1);
    // Kurva gain yang dibutuhkan tiap sampel.
    let butuh: Vec<f32> = x
        .iter()
        .map(|v| {
            let a = v.abs();
            if a > ceiling {
                ceiling / a
            } else {
                1.0
            }
        })
        .collect();

    // Minimum bergerak selebar lookahead: gain reduction "melihat" ke depan.
    let mut e = vec![1.0f32; n];
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        while let Some(&b) = dq.back() {
            if butuh[b] >= butuh[i] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(i);
        while let Some(&f) = dq.front() {
            if f + la < i {
                dq.pop_front();
            } else {
                break;
            }
        }
        e[i] = butuh[*dq.front().expect("deque tidak kosong")];
    }

    let rel = (-1.0 / (0.050 * fs)).exp() as f32;
    let mut cur = 1.0f32;
    for i in 0..n {
        if e[i] < cur {
            cur = e[i];
        } else {
            cur += (e[i] - cur) * (1.0 - rel);
        }
        x[i] = (x[i] * cur).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Di 48 kHz koefisien ini harus sama dengan angka baku BS.1770.
    #[test]
    fn koefisien_sesuai_spesifikasi_48k() {
        let (sb, sa, hb, ha) = k_weighting(48_000.0);
        assert!((sb[0] - 1.535_124_859_586_97).abs() < 1e-6, "{sb:?}");
        assert!((sb[1] + 2.691_696_189_406_38).abs() < 1e-6, "{sb:?}");
        assert!((sb[2] - 1.198_392_810_852_85).abs() < 1e-6, "{sb:?}");
        assert!((sa[1] + 1.690_659_293_182_41).abs() < 1e-6, "{sa:?}");
        assert!((sa[2] - 0.732_480_774_215_85).abs() < 1e-6, "{sa:?}");
        assert!((hb[0] - 1.0).abs() < 1e-6, "{hb:?}");
        assert!((hb[1] + 2.0).abs() < 1e-6, "{hb:?}");
        assert!((ha[1] + 1.990_047_454_833_98).abs() < 1e-6, "{ha:?}");
        assert!((ha[2] - 0.990_072_250_366_21).abs() < 1e-6, "{ha:?}");
    }

    /// Sinus 1 kHz amplitudo 0,1: RMS −23,0 dBFS, K-weighting +0,7 dB di 1 kHz,
    /// jadi loudness-nya ≈ −23,0 LUFS. Angka ini juga cocok dengan
    /// `ffmpeg -af ebur128` pada berkas yang sama.
    #[test]
    fn sinus_seribu_hertz_sesuai_hitungan_tangan() {
        let fs = 48_000.0;
        let amp = 0.1f32;
        let x: Vec<f32> = (0..(fs as usize * 3))
            .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / fs as f32).sin())
            .collect();
        let l = integrated_lufs(&x, fs);
        assert!((l + 23.0).abs() < 0.7, "lufs={l}");
    }

    #[test]
    fn normalisasi_menaikkan_dan_tidak_melewati_ceiling() {
        let fs = 24_000.0;
        let mut x: Vec<f32> = (0..(fs as usize * 2))
            .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / fs as f32).sin())
            .collect();
        let (gain, awal, akhir) = normalize(&mut x, fs, -16.0, -1.5);
        assert!(gain > 10.0, "gain={gain} awal={awal}");
        assert!(akhir <= -1.4, "puncak={akhir}");
        let sesudah = integrated_lufs(&x, fs);
        assert!((sesudah + 16.0).abs() < 1.5, "sesudah={sesudah}");
    }

    #[test]
    fn audio_senyap_tidak_meledak() {
        let fs = 24_000.0;
        let mut x = vec![0.0f32; 24_000];
        let (gain, _, akhir) = normalize(&mut x, fs, -16.0, -1.5);
        assert_eq!(gain, 0.0);
        assert_eq!(akhir, f32::NEG_INFINITY);
    }
}
