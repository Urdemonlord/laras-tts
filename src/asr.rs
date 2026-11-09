//! Pemilih take berbasis ASR — "opsi A" tanpa Python.
//!
//! Ide: hasil sintesis tidak bisa dinilai dari statistik internal (sudah
//! diuji, tidak ada sinyal), tapi bisa **didengar**. Whisper (whisper.cpp,
//! CPU) mentranskrip tiap take, lalu take dengan WER terendah terhadap teks
//! asli yang dipilih.
//!
//! Tidak ada Python, tidak ada subprocess: satu binary, satu proses.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Sampel rate keluaran mesin TTS.
const TTS_RATE: usize = 24_000;
/// Sampel rate yang diharapkan whisper.cpp.
const ASR_RATE: usize = 16_000;

pub struct Asr {
    /// State dipakai ulang: membuat state baru mengalokasikan ~400 MB buffer
    /// tiap kali, yang jauh lebih mahal daripada inferensinya sendiri.
    state: Mutex<WhisperState>,
    threads: usize,
    /// `None` = jendela penuh 30 s bawaan whisper.cpp.
    audio_ctx: Option<i32>,
}

impl Asr {
    pub fn new(model: &Path, _threads: usize) -> Result<Self> {
        // Diukur pada 4 vCPU: 1 thread 10,1 s vs 4 thread 12,2 s untuk audio
        // 2,5 s — overhead thread pool ggml lebih besar daripada manfaatnya.
        // `ASR_THREADS` disediakan kalau suatu saat mau diubah.
        let threads = std::env::var("ASR_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize);
        let p = model
            .to_str()
            .ok_or_else(|| anyhow!("path model ASR bukan UTF-8"))?;
        let ctx = WhisperContext::new_with_params(p, WhisperContextParameters::default())
            .map_err(|e| anyhow!("memuat model ASR {}: {e}", model.display()))?;
        let state = ctx.create_state().map_err(|e| anyhow!("state ASR: {e}"))?;
        let audio_ctx = std::env::var("ASR_CTX")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v >= 128);
        Ok(Self {
            state: Mutex::new(state),
            threads,
            audio_ctx,
        })
    }

    /// Transkrip audio 24 kHz mono menjadi teks.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let pcm = resample(audio);
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("state ASR terkunci"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads.max(1) as i32);
        params.set_language(Some("id"));
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Tanpa timestamp & satu segmen: jumlah token turun drastis, dan
        // itu yang menentukan lama inferensi pada audio pendek.
        params.set_no_timestamps(true);
        params.set_single_segment(true);
        params.set_temperature(0.0);
        // Batas token: tanpa ini whisper.cpp bisa terus memancarkan token
        // kosong sampai batas konteks (16k) — satu-satunya penyebab lambat.
        params.set_max_tokens(64);
        // `ASR_CTX` memotong jendela encoder 30 s bawaan whisper.cpp. Diukur:
        // 1500 -> 10 s, 768 -> 5,6 s, 512 -> 4 s (akurasi mulai rusak di 512).
        if let Some(ctx) = self.audio_ctx {
            params.set_audio_ctx(ctx);
        }
        // Tanpa fallback temperatur: satu lintasan saja per take.
        params.set_temperature_inc(0.0);
        state
            .full(params, &pcm)
            .map_err(|e| anyhow!("inferensi ASR: {e}"))?;
        let n = state
            .full_n_segments()
            .map_err(|e| anyhow!("segmen ASR: {e}"))?;
        let mut out = String::new();
        for i in 0..n {
            if let Ok(t) = state.full_get_segment_text(i) {
                out.push_str(&t);
                out.push(' ');
            }
        }
        Ok(out)
    }

    /// WER antara teks referensi (sudah dinormalisasi) dan hasil transkrip.
    pub fn wer(&self, audio: &[f32], reference: &str) -> Result<f32> {
        let hyp = self.transcribe(audio)?;
        Ok(word_error_rate(reference, &hyp))
    }
}

/// Resample linear 24 kHz -> 16 kHz.
fn resample(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let n = (audio.len() * ASR_RATE).div_ceil(TTS_RATE);
    let step = TTS_RATE as f64 / ASR_RATE as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 * step;
            let i0 = x.floor() as usize;
            let i1 = (i0 + 1).min(audio.len() - 1);
            let w = (x - i0 as f64) as f32;
            audio[i0.min(audio.len() - 1)] * (1.0 - w) + audio[i1] * w
        })
        .collect()
}

/// Token kata: huruf/angka kecil saja, tanda baca dibuang.
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// WER = jarak edit kata / jumlah kata referensi.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let r = words(reference);
    let h = words(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f32 / r.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_sempurna_nol() {
        assert_eq!(
            word_error_rate("selamat pagi dunia", "Selamat pagi, dunia!"),
            0.0
        );
    }

    #[test]
    fn wer_satu_dari_tiga() {
        let w = word_error_rate("selamat pagi dunia", "selamat malam dunia");
        assert!((w - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn wer_semua_salah() {
        assert_eq!(word_error_rate("satu dua", "tiga empat"), 1.0);
    }

    #[test]
    fn resample_panjang_benar() {
        let a = vec![0.0f32; 24_000];
        assert_eq!(resample(&a).len(), 16_000);
    }
}
