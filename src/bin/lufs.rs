//! Ukur loudness dan puncak sebuah wav mono PCM16.
//!
//! Dipakai untuk membandingkan hasil `laras::level` dengan `ffmpeg -af ebur128`
//! pada berkas yang sama.

use std::fs;

fn baca_wav(path: &str) -> Result<(Vec<f32>, f64), String> {
    let b = fs::read(path).map_err(|e| format!("baca {path}: {e}"))?;
    if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("bukan wav".into());
    }
    let mut i = 12;
    let mut fs_hz = 24_000u32;
    let mut kanal = 1u16;
    let mut bits = 16u16;
    let mut data: Option<(usize, usize)> = None;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let ukuran = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let mulai = i + 8;
        match id {
            b"fmt " => {
                kanal = u16::from_le_bytes([b[mulai + 2], b[mulai + 3]]);
                fs_hz =
                    u32::from_le_bytes([b[mulai + 4], b[mulai + 5], b[mulai + 6], b[mulai + 7]]);
                bits = u16::from_le_bytes([b[mulai + 14], b[mulai + 15]]);
            }
            b"data" => data = Some((mulai, ukuran.min(b.len() - mulai))),
            _ => {}
        }
        i = mulai + ukuran + (ukuran & 1);
    }
    let (mulai, ukuran) = data.ok_or("tidak ada chunk data")?;
    if bits != 16 {
        return Err(format!("hanya PCM16, ini {bits} bit"));
    }
    let mut x = Vec::with_capacity(ukuran / 2 / kanal as usize);
    let mut j = mulai;
    while j + 2 <= mulai + ukuran {
        let v = i16::from_le_bytes([b[j], b[j + 1]]) as f32 / 32768.0;
        x.push(v);
        j += 2 * kanal as usize;
    }
    Ok((x, fs_hz as f64))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("pakai: lufs <berkas.wav> [target_lufs] [ceiling_db]");
        std::process::exit(2);
    });
    let (mut x, fs_hz) = match baca_wav(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let awal = laras::level::integrated_lufs(&x, fs_hz);
    println!(
        "{path}: {:.1} LUFS, puncak {:.1} dBFS, {:.2} s @ {fs_hz:.0} Hz",
        awal,
        laras::level::peak_dbfs(&x),
        x.len() as f64 / fs_hz
    );
    if let Some(t) = args.next() {
        let target: f32 = t.parse().unwrap_or(-16.0);
        let ceiling: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(-1.5);
        let (gain, sebelum, akhir) = laras::level::normalize(&mut x, fs_hz, target, ceiling);
        println!(
            "  dinormalkan: {sebelum:.1} -> {:.1} LUFS, puncak {akhir:.1} dBFS (gain {gain:+.1} dB)",
            laras::level::integrated_lufs(&x, fs_hz)
        );
    }
}
