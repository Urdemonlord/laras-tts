//! I/O audio: baca WAV apa pun, jadikan mono 24 kHz, tulis WAV 16-bit.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use anyhow::{anyhow, bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Laju sampel model.
pub const SAMPLE_RATE: u32 = 24_000;

/// Panjang maksimum prompt suara yang dipakai (detik).
pub const MAX_VOICE_SECS: u32 = 30;

/// Baca berkas audio apa pun dan kembalikan sampel mono float32 pada 24 kHz.
///
/// WAV dibaca langsung. Format lain (mp3, m4a, ogg, flac, …) dikonversi dulu
/// lewat `ffmpeg` ke WAV 24 kHz mono di direktori sementara, lalu dibaca. Hasil
/// konversi dipakai ulang selama berkas sumbernya tidak berubah (nama berkas
/// sementara diturunkan dari path, ukuran, dan mtime), jadi permintaan
/// berikutnya tidak membayar biaya konversi lagi.
///
/// Kalau `ffmpeg` tidak terpasang, galatnya menyebut perintah konversi manual —
/// jangan biarkan pengguna menebak kenapa MP3-nya ditolak.
pub fn read_mono_auto(path: &Path) -> Result<Vec<f32>> {
    if !path.exists() {
        return Err(anyhow!(
            "referensi {} tidak ada — ambil contoh dengan `scripts/unduh-model.sh`, \
             atau tunjuk berkas lain lewat REF_WAV",
            path.display()
        ));
    }
    if let Ok(samples) = read_mono(path) {
        return Ok(samples);
    }
    let wav = konversi_ke_wav(path)?;
    read_mono(&wav).with_context(|| format!("membaca hasil konversi {}", wav.display()))
}

/// Konversi `src` ke WAV 24 kHz mono di direktori sementara; kembalikan jalurnya.
fn konversi_ke_wav(src: &Path) -> Result<PathBuf> {
    let meta =
        std::fs::metadata(src).with_context(|| format!("membaca metadata {}", src.display()))?;
    let mut h = DefaultHasher::new();
    src.hash(&mut h);
    meta.len().hash(&mut h);
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .hash(&mut h);
    let out = std::env::temp_dir().join(format!("laras-ref-{:016x}.wav", h.finish()));
    if out.exists() {
        return Ok(out);
    }

    let sr = SAMPLE_RATE.to_string();
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(src)
        .args(["-ar", sr.as_str(), "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&out)
        .status();

    match status {
        Ok(s) if s.success() => Ok(out),
        Ok(s) => {
            let _ = std::fs::remove_file(&out);
            bail!(
                "ffmpeg keluar dengan kode {s} saat mengonversi {}",
                src.display()
            )
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
            "{} bukan WAV dan `ffmpeg` tidak terpasang. Konversi manual:\n  \
             ffmpeg -i {} -ar {SAMPLE_RATE} -ac 1 {}",
            src.display(),
            src.display(),
            src.with_extension("wav").display()
        ),
        Err(e) => Err(e).with_context(|| format!("menjalankan ffmpeg untuk {}", src.display())),
    }
}

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
    use super::*;

    #[test]
    fn resample_panjang_masuk_akal() {
        let x: Vec<f32> = (0..480).map(|i| i as f32 / 480.0).collect();
        let y = resample(&x, 24_000, 48_000);
        assert_eq!(y.len(), 960);
        assert!((y[0] - 0.0).abs() < 1e-6);
    }

    fn tulis_wav(jalur: &Path, n: usize) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(jalur, spec).expect("buat wav");
        for i in 0..n {
            w.write_sample(((i % 100) as i16 - 50) * 100)
                .expect("tulis");
        }
        w.finalize().expect("finalisasi");
    }

    #[test]
    fn read_mono_auto_berkas_hilang_pesan_jelas() {
        let err = read_mono_auto(Path::new("/tmp/laras-tidak-ada-xyz.mp3")).unwrap_err();
        let pesan = format!("{err:#}");
        assert!(pesan.contains("tidak ada"), "{pesan}");
        assert!(pesan.contains("unduh-model.sh"), "{pesan}");
    }

    #[test]
    fn read_mono_auto_wav_dibaca_langsung() {
        let p = std::env::temp_dir().join("laras-uji-auto-wav.wav");
        tulis_wav(&p, 240);
        let s = read_mono_auto(&p).expect("baca wav");
        assert_eq!(s.len(), 240);
        let _ = std::fs::remove_file(&p);
    }

    /// MP3 harus jalan tanpa disuruh konversi manual. Dilewati kalau ffmpeg
    /// tidak terpasang di mesin uji.
    #[test]
    fn read_mono_auto_mp3_lewat_ffmpeg() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            eprintln!("ffmpeg tidak ada, tes dilewati");
            return;
        }
        let p = std::env::temp_dir().join("laras-uji-auto-mp3.mp3");
        let dibuat = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.3",
                "-ar",
                "48000",
                "-ac",
                "1",
            ])
            .arg(&p)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(dibuat, "gagal membuat mp3 uji");

        let s = read_mono_auto(&p).expect("baca mp3");
        if let Ok(c) = konversi_ke_wav(&p) {
            let _ = std::fs::remove_file(c);
        }
        let _ = std::fs::remove_file(&p);
        // 0,3 detik pada 24 kHz ≈ 7.200 sampel.
        assert!((s.len() as i64 - 7_200).abs() < 600, "panjang {}", s.len());
        assert!(s.iter().any(|x| x.abs() > 0.01), "sampel kosong");
    }

    /// Konversi diulang tidak boleh menumpuk berkas baru di /tmp.
    #[test]
    fn konversi_dipakai_ulang() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let p = std::env::temp_dir().join("laras-uji-ulang.mp3");
        let ok = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=duration=0.1",
            ])
            .arg(&p)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return;
        }
        let a = konversi_ke_wav(&p).expect("konversi pertama");
        let b = konversi_ke_wav(&p).expect("konversi kedua");
        assert_eq!(a, b, "jalur konversi harus sama");
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&p);
    }
}
