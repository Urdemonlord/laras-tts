//! I/O audio: baca WAV apa pun, jadikan mono 24 kHz, tulis WAV 16-bit.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Laju sampel model.
pub const SAMPLE_RATE: u32 = 24_000;

/// Panjang maksimum prompt suara yang dipakai (detik).
pub const MAX_VOICE_SECS: u32 = 30;

/// Baca berkas audio dan kembalikan sampel mono float32 pada 24 kHz.
///
/// WAV stereo dirata-rata jadi mono, laju sampel lain di-resample linear,
/// dan amplitudo di atas 1,0 dinormalisasi agar tidak clipping.
pub fn read_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("gagal membuka {} (hanya WAV yang didukung)", path.display()))?;
    let spec = reader.spec();

    let mut mono: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => {
            let all: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            all.context("gagal membaca sampel float")?
        }
        SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(8);
            let scale = 1.0 / (1u32 << (bits - 1)) as f32;
            let all: Result<Vec<i32>, _> = reader.samples::<i32>().collect();
            all.context("gagal membaca sampel int")?
                .into_iter()
                .map(|v| v as f32 * scale)
                .collect()
        }
    };

    let ch = spec.channels.max(1) as usize;
    if ch > 1 {
        let frames = mono.len() / ch;
        let mut out = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut acc = 0f32;
            for c in 0..ch {
                acc += mono[i * ch + c];
            }
            out.push(acc / ch as f32);
        }
        mono = out;
    }

    let mut samples = if spec.sample_rate != SAMPLE_RATE {
        resample(&mono, spec.sample_rate, SAMPLE_RATE)
    } else {
        mono
    };

    // Batas panjang prompt suara: model dilatih dengan konteks pendek.
    let max = (MAX_VOICE_SECS * SAMPLE_RATE) as usize;
    if samples.len() > max {
        samples.truncate(max);
    }

    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 1.0 {
        let k = 1.0 / peak;
        for v in &mut samples {
            *v *= k;
        }
    }

    if samples.is_empty() {
        return Err(anyhow!("berkas audio kosong"));
    }
    Ok(samples)
}

/// Resample linear dengan aritmetika f64.
///
/// Sengaja meniru `np.interp` (presisi ganda, `linspace` sebagai sumbu sumber)
/// agar hasilnya identik bit-per-bit dengan worker Python acuan. Graf int8
/// sangat sensitif: selisih 4e-7 pada prompt suara bisa menjadi 0,02 pada
/// conditioning, jadi kesamaan di sini penting untuk verifikasi.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = ((x.len() as f64) * (to as f64) / (from as f64))
        .round()
        .max(1.0) as usize;
    if n < 2 {
        return vec![x[0]];
    }
    let step = (x.len() - 1) as f64 / (n - 1) as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pos = i as f64 * step;
        let i0 = (pos.floor() as usize).min(x.len() - 1);
        let frac = pos - i0 as f64;
        let a = x[i0] as f64;
        let b = x[(i0 + 1).min(x.len() - 1)] as f64;
        out.push((a + (b - a) * frac) as f32);
    }
    out
}

/// Tulis WAV mono 16-bit 24 kHz.
pub fn write_mono(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec)
        .with_context(|| format!("gagal menulis {}", path.display()))?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::resample;

    #[test]
    fn resample_panjang_masuk_akal() {
        let x: Vec<f32> = (0..480).map(|i| i as f32 / 480.0).collect();
        let y = resample(&x, 24_000, 48_000);
        assert_eq!(y.len(), 960);
        assert!((y[0] - 0.0).abs() < 1e-6);
    }
}
