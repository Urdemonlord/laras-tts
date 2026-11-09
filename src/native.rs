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
use anyhow::{Context, Result};
use laras::onnx::{wav, Engine, Params};

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
    fn use_voice(&mut self, path: &str) -> Result<()> {
        if self.voice_key.as_deref() == Some(path) {
            return Ok(());
        }
        if !self.voices.contains_key(path) {
            if self.voices.len() >= VOICE_CACHE_MAX {
                self.voices.clear();
            }
            let samples = wav::read_mono(Path::new(path))
                .with_context(|| format!("membaca referensi {path}"))?;
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
        let best = self
            .engine
            .synthesize_best(text, &self.params, takes, None)?;

        let kata = text.split_whitespace().count();

        // Pemilihan: kalau ASR tersedia, dengarkan tiap take dan ambil WER
        // terendah. Kalau tidak, pakai pilihan medoid bawaan mesin.
        #[cfg(feature = "asr")]
        let (audio, index, skor) = match &self.asr {
            Some(asr) if best.cands.len() > 1 => {
                let rujukan = self.lang.prepare(text);
                let mut pilih = best.index;
                let mut terbaik = f32::INFINITY;
                for (i, (_, audio, _)) in best.cands.iter().enumerate() {
                    if let Ok(w) = asr.wer(audio, &rujukan) {
                        if w < terbaik {
                            terbaik = w;
                            pilih = i;
                        }
                    }
                }
                (best.cands[pilih].1.as_slice(), pilih, terbaik)
            }
            _ => (best.audio(), best.index, best.diag().score(kata)),
        };

        #[cfg(not(feature = "asr"))]
        let (audio, index, skor) = (best.audio(), best.index, best.diag().score(kata));

        wav::write_mono(out, audio)?;
        let dur = audio.len() as f64 / wav::SAMPLE_RATE as f64;
        Ok((dur, index, skor))
    }
}
