//! Backend inferensi native (ONNX Runtime) — pengganti `worker.py`.
//!
//! Semua di dalam proses: tidak ada Python, tidak ada subprocess. Voice
//! referensi di-cache per path supaya permintaan berikutnya tidak perlu
//! meng-encode ulang audio referensi.
//!
//! Kalau fitur `asr` aktif dan model Whisper tersedia, take dipilih dengan
//! mendengarkan hasilnya (WER terendah), bukan dengan statistik internal —
//! statistik internal sudah diuji dan tidak berkorelasi dengan kualitas.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::lang::{self, Lang};
use crate::onnx::{wav, Best, Diag, Engine, Params};
use anyhow::{Context, Result};

#[cfg(feature = "asr")]
use crate::asr::Asr;

/// Batas jumlah voice referensi yang disimpan di memori.
const VOICE_CACHE_MAX: usize = 8;

pub struct Native {
    engine: Engine,
    models: PathBuf,
    params: Params,
    voice_key: Option<String>,
    voices: HashMap<String, Vec<f32>>,
    lang: &'static dyn Lang,
    #[cfg(feature = "asr")]
    asr: Option<Asr>,
    /// Ambang WER untuk berhenti lebih awal (ASR_ACCEPT); 0 = selalu `takes`.
    asr_accept: f32,
}

impl Native {
    pub fn new(
        models: PathBuf,
        threads: usize,
        params: Params,
        lang_code: &str,
        #[cfg(feature = "asr")] asr: Option<Asr>,
    ) -> Result<Self> {
        let engine = Engine::new(&models, threads)
            .with_context(|| format!("memuat model ONNX dari {}", models.display()))?;
        let lang = lang::get(lang_code)
            .ok_or_else(|| anyhow::anyhow!("bahasa \"{lang_code}\" belum didukung"))?;
        Ok(Self {
            engine,
            models,
            params,
            voice_key: None,
            voices: HashMap::new(),
            lang,
            #[cfg(feature = "asr")]
            asr,
            asr_accept: std::env::var("ASR_ACCEPT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.15),
        })
    }

    pub fn models(&self) -> &Path {
        &self.models
    }

    /// Apakah pemilihan take memakai ASR.
    pub fn asr_aktif(&self) -> bool {
        #[cfg(feature = "asr")]
        {
            self.asr.is_some()
        }
        #[cfg(not(feature = "asr"))]
        {
            false
        }
    }

    /// Suara referensi yang sedang aktif.
    pub fn active_voice(&self) -> Option<&str> {
        self.voice_key.as_deref()
    }

    /// Pastikan referensi `path` sudah menjadi conditioning aktif.
    pub fn use_voice(&mut self, path: &str) -> Result<()> {
        if self.voice_key.as_deref() == Some(path) {
            return Ok(());
        }
        if !self.voices.contains_key(path) {
            if self.voices.len() >= VOICE_CACHE_MAX {
                self.voices.clear();
            }
            // `read_mono_auto` menerima WAV apa pun dan mengonversi mp3/m4a/ogg
            // lewat ffmpeg, supaya referensi tidak harus disiapkan manual dulu.
            let samples = wav::read_mono_auto(Path::new(path))
                .with_context(|| format!("menyiapkan referensi {path}"))?;
            if samples.is_empty() {
                anyhow::bail!("referensi {path} kosong");
            }
            self.voices.insert(path.to_string(), samples);
        }
        let samples = self.voices.get(path).expect("baru dimasukkan").clone();
        self.engine.set_voice(&samples)?;
        self.voice_key = Some(path.to_string());
        Ok(())
    }

    /// Sintesis dan tulis WAV ke `out`. Mengembalikan (durasi, take terpilih, skor).
    pub fn synth_to(
        &mut self,
        text: &str,
        ref_wav: &str,
        takes: usize,
        out: &Path,
    ) -> Result<(f64, usize, f32)> {
        self.use_voice(ref_wav)?;
        let t0 = std::time::Instant::now();

        // Teks ternormalisasi dipakai dua-duanya: rujukan WER dan hitungan kata.
        let rujukan = self.lang.prepare(text);
        let kata = rujukan.split_whitespace().count();

        // Pemilihan adaptif: satu take dulu, didengarkan; kalau WER-nya sudah
        // di bawah `ASR_ACCEPT`, sisa take tidak perlu dibuat. Kalau tidak,
        // take berikutnya ditambah satu per satu sampai batas `takes`.
        let n_maks = takes.max(1);
        let mut cands: Vec<(Diag, Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_maks);
        let mut skor: Vec<Option<f32>> = Vec::with_capacity(n_maks);
        let mut t_synth = 0.0f32;

        for k in 0..n_maks {
            let t = std::time::Instant::now();
            let c = self.engine.synthesize_kth(text, &self.params, k, None)?;
            t_synth += t.elapsed().as_secs_f32();
            // Satu take tidak ada yang bisa dipilih: ASR hanya membuang waktu.
            #[cfg(feature = "asr")]
            let w = match &self.asr {
                Some(asr) if n_maks > 1 => asr.wer(&c.1, &rujukan).ok(),
                _ => None,
            };
            #[cfg(not(feature = "asr"))]
            let w: Option<f32> = None;

            if std::env::var("LARAS_TRACE").is_ok() {
                if let Some(w) = w {
                    eprintln!("[asr] take {k}: WER {w:.3}");
                }
            }
            cands.push(c);
            skor.push(w);
            // `ASR_ACCEPT` negatif = jangan berhenti awal (selalu `takes` take).
            if let Some(w) = w {
                if self.asr_accept >= 0.0 && w <= self.asr_accept {
                    break;
                }
            }
        }

        let index = if skor.iter().all(|s| s.is_some()) && !skor.is_empty() {
            let nilai: Vec<f32> = skor.iter().map(|s| s.unwrap_or(f32::INFINITY)).collect();
            nilai
                .iter()
                .enumerate()
                .fold(0usize, |bi, (i, w)| if *w < nilai[bi] { i } else { bi })
        } else {
            // Tanpa skor ASR, pakai medoid latent seperti bawaan mesin.
            crate::onnx::medoid_index(&cands)
        };
        let best = Best { index, cands };
        let dipakai = best.cands.len();

        let skor_akhir = skor
            .get(index)
            .copied()
            .flatten()
            .unwrap_or_else(|| best.diag().score(kata));
        let audio = best.audio();

        if std::env::var("LARAS_TRACE").is_ok() {
            eprintln!(
                "[waktu] synth {:.0} ms ({} take) + asr {:.0} ms = {:.0} ms",
                t_synth * 1000.0,
                dipakai,
                (t0.elapsed().as_secs_f32() - t_synth).max(0.0) * 1000.0,
                t0.elapsed().as_secs_f32() * 1000.0
            );
        }

        wav::write_mono(out, audio)?;
        let dur = audio.len() as f64 / wav::SAMPLE_RATE as f64;
        Ok((dur, index, skor_akhir))
    }
}
