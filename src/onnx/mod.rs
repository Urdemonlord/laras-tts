//! Mesin inferensi Laras: Pocket TTS Bahasa Indonesia di ONNX Runtime, CPU-only.
//!
//! Alur (terverifikasi terhadap implementasi PyTorch acuan):
//!
//! ```text
//! wav ──mimi_encoder──► prompt [1,T,1024] ──┐ (prepend bos_before_voice)
//!                                           ▼
//! teks ──tokenizer──► ids ──text_conditioner──► [1,N,1024]
//!                                           ▼
//!                        flow_lm_main  (2 pass conditioning, lalu 1 frame/panggilan)
//!                                           ▼
//!                          flow_lm_flow (ODE 1 langkah) ──► latent [1,32]
//!                                           ▼
//!                       mimi_decoder (streaming 1 frame/panggilan) ──► audio 24 kHz
//! ```
//!
//! Dua hal yang wajib benar dan mudah salah:
//!
//! * **Decoder harus streaming satu frame per panggilan.** Memanggilnya dengan
//!   seluruh latent sekaligus menghasilkan audio rusak (korelasi 0,38 vs 0,996).
//! * **`bos_before_voice` wajib di-prepend.** Tanpa itu EOS molor dari 42 frame
//!   ke 259 frame.

pub mod npy;
pub mod tokenizer;
pub mod wav;

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use half::f16;
use ort::environment::{init as init_env, Environment};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Shape, TensorElementType, Value, ValueType};

use tokenizer::Tokenizer;

pub use wav::SAMPLE_RATE;

/// Dimensi latent per frame.
const LATENT: usize = 32;
/// Batas atas jumlah frame agar tidak pernah menggantung.
const MAX_FRAMES: usize = 400;
/// Jumlah frame minimum sebelum EOS dianggap sah.
const MIN_FRAMES: usize = 10;

/// Panjang acuan saat menyamakan sekuens latent antar-take untuk pemilihan.
const MEDOID_FRAMES: usize = 64;

/// Statistik satu take, dipakai pemilih tanpa ASR.
#[derive(Debug, Clone)]
pub struct Diag {
    /// Jumlah frame latent yang dihasilkan.
    pub frames: usize,
    /// Frame tempat EOS pertama kali menyala (None = mentok batas).
    pub eos_frame: Option<usize>,
    /// Logit EOS tertinggi sepanjang generasi.
    pub max_eos: f32,
    /// Logit EOS pada frame saat EOS menyala.
    pub eos_conf: f32,
    /// Rata-rata |latent|.
    pub latent_abs: f32,
    /// Simpangan baku seluruh nilai latent.
    pub latent_std: f32,
    /// Puncak amplitudo audio.
    pub peak: f32,
    /// RMS audio.
    pub rms: f32,
    /// Durasi audio (detik).
    pub dur: f32,
    /// Waktu sintesis (milidetik).
    pub ms: f64,
}

impl Diag {
    /// Skor kelayakan satu take (makin tinggi makin baik).
    ///
    /// Tidak memakai ASR: hanya geometri trayektori model.
    pub fn score(&self, words: usize) -> f32 {
        let mut s = 0.0f32;

        // 1) EOS harus benar-benar menyala. Take yang mentok batas hampir
        //    selalu rusak, jadi diberi penalti besar.
        s += if self.eos_frame.is_some() { 2.0 } else { -4.0 };

        // 2) Keyakinan EOS: makin tinggi makin tegas akhir kalimatnya.
        s += (self.eos_conf.clamp(-10.0, 10.0)) * 0.25;

        // 3) Kecepatan bicara wajar: 0,22-0,55 detik per kata.
        let w = words.max(1) as f32;
        let per_word = self.dur / w;
        let center = 0.36;
        s -= ((per_word - center) / 0.13).powi(2).min(9.0);

        // 4) Audio tidak boleh hampir senyap atau pecah.
        if self.rms < 0.005 {
            s -= 3.0;
        }
        if self.peak > 0.99 {
            s -= 0.5;
        }

        // 5) Latent yang kolaps (hampir konstan) menandakan generasi mati.
        if self.latent_std < 0.05 {
            s -= 2.0;
        }

        s
    }
}

/// Parameter sintesis.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Temperatur sampling latent.
    pub temperature: f32,
    /// Ambang logit EOS; makin tinggi makin cepat berhenti.
    pub eos_threshold: f32,
    /// Seed RNG (deterministik per seed).
    pub seed: u64,
    /// Frame tambahan setelah EOS terdeteksi.
    pub eos_extra: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            temperature: 0.10,
            eos_threshold: -4.0,
            seed: 1234,
            eos_extra: 3,
        }
    }
}

/// Tipe data tensor yang dipakai state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F16,
    F32,
    I64,
    Bool,
}

/// Spesifikasi satu state (nama, bentuk awal, tipe).
#[derive(Debug, Clone)]
struct Spec {
    name: String,
    shape: Vec<usize>,
    ty: DType,
}

/// Isi buffer state.
#[derive(Debug, Clone)]
enum Buf {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

/// State yang dibawa antar panggilan model.
#[derive(Debug, Clone)]
struct State {
    name: String,
    shape: Vec<usize>,
    buf: Buf,
}

impl State {
    fn zeros(spec: &Spec) -> Self {
        let n: usize = spec.shape.iter().product();
        let buf = match spec.ty {
            DType::F16 => Buf::F16(vec![f16::ZERO; n]),
            DType::F32 => Buf::F32(vec![0.0; n]),
            DType::I64 => Buf::I64(vec![0; n]),
            DType::Bool => Buf::Bool(vec![false; n]),
        };
        Self {
            name: spec.name.clone(),
            shape: spec.shape.clone(),
            buf,
        }
    }

    fn to_value(&self) -> Result<DynValue> {
        let sh = Shape::from(self.shape.clone());
        Ok(match &self.buf {
            Buf::F16(d) => Value::from_array((sh, d.clone()))?.into_dyn(),
            Buf::F32(d) => Value::from_array((sh, d.clone()))?.into_dyn(),
            Buf::I64(d) => Value::from_array((sh, d.clone()))?.into_dyn(),
            Buf::Bool(d) => Value::from_array((sh, d.clone()))?.into_dyn(),
        })
    }

    fn read(name: &str, v: &DynValue) -> Result<Self> {
        let ty = match v.dtype() {
            ValueType::Tensor { ty, .. } => map_dtype(ty)?,
            other => return Err(anyhow!("state {name} bukan tensor: {other:?}")),
        };
        let (shape, buf) = match ty {
            DType::F16 => {
                let (s, d) = v.try_extract_tensor::<f16>()?;
                (dims(s), Buf::F16(d.to_vec()))
            }
            DType::F32 => {
                let (s, d) = v.try_extract_tensor::<f32>()?;
                (dims(s), Buf::F32(d.to_vec()))
            }
            DType::I64 => {
                let (s, d) = v.try_extract_tensor::<i64>()?;
                (dims(s), Buf::I64(d.to_vec()))
            }
            DType::Bool => {
                let (s, d) = v.try_extract_tensor::<bool>()?;
                (dims(s), Buf::Bool(d.to_vec()))
            }
        };
        Ok(Self {
            name: name.to_string(),
            shape,
            buf,
        })
    }
}

fn dims(s: &Shape) -> Vec<usize> {
    s.iter()
        .map(|d| if *d < 0 { 0 } else { *d as usize })
        .collect()
}

fn map_dtype(t: &TensorElementType) -> Result<DType> {
    match t {
        TensorElementType::Float16 => Ok(DType::F16),
        TensorElementType::Float32 => Ok(DType::F32),
        TensorElementType::Int64 => Ok(DType::I64),
        TensorElementType::Bool => Ok(DType::Bool),
        other => Err(anyhow!("tipe tensor belum didukung: {other:?}")),
    }
}

/// RNG deterministik (SplitMix64 + Box–Muller), tanpa dependensi eksternal.
struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / 16_777_216.0
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Mesin TTS siap pakai.
pub struct Engine {
    enc: Session,
    txt: Session,
    main: Session,
    flow: Session,
    dec: Session,

    enc_in: String,
    enc_out: String,
    txt_in: String,
    txt_out: String,
    flow_ins: [String; 4],
    flow_out: String,
    main_cond: String,
    main_eos: String,
    dec_in: String,
    dec_out: String,

    main_specs: Vec<Spec>,
    dec_specs: Vec<Spec>,

    tok: Tokenizer,
    bos: Vec<f32>,

    voice: Option<Vec<f32>>,
    voice_frames: usize,
    hidden: usize,

    /// Harus hidup selama sesi; diletakkan terakhir agar di-drop paling akhir.
    _env: Environment,
}

impl Engine {
    /// Muat seluruh graf dan tokenizer dari direktori model.
    pub fn new(models: &Path, threads: usize) -> Result<Self> {
        let t0 = Instant::now();

        let env = init_env()
            .with_name("laras")
            .with_telemetry(false)
            .build()
            .map_err(|e| anyhow!("inisialisasi ONNX Runtime: {e}"))?;

        let enc = open(models, "mimi_encoder.onnx", threads, &env)?;
        let txt = open(models, "text_conditioner.onnx", threads, &env)?;
        let main = open(models, "flow_lm_main_int8.onnx", threads, &env)?;
        let flow = open(models, "flow_lm_flow_int8.onnx", threads, &env)?;
        let dec = open(models, "mimi_decoder_int8.onnx", threads, &env)?;

        let tok = Tokenizer::from_file(&models.join("tokenizer.model"))?;
        let bos = npy::read_f32(&models.join("bos_before_voice.npy"))
            .context("gagal memuat bos_before_voice.npy")?
            .data;

        let enc_in = first_input(&enc);
        let enc_out = first_output(&enc);
        let txt_in = first_input(&txt);
        let txt_out = first_output(&txt);

        let flow_names: Vec<String> = flow.inputs().iter().map(|o| o.name().to_string()).collect();
        let flow_ins = [
            pick(&flow_names, "c", 0),
            pick(&flow_names, "s", 1),
            pick(&flow_names, "t", 2),
            pick(&flow_names, "x", 3),
        ];
        let flow_out = first_output(&flow);

        let main_outs: Vec<String> = main
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let main_cond = pick(&main_outs, "conditioning", 0);
        let main_eos = pick(&main_outs, "eos_logit", 1);

        let dec_in = first_input(&dec);
        let dec_out = first_output(&dec);

        let main_specs = state_specs(&main)?;
        let dec_specs = state_specs(&dec)?;

        if main_specs.is_empty() || dec_specs.is_empty() {
            return Err(anyhow!("graf tidak punya state_* — berkas model salah?"));
        }

        eprintln!(
            "[laras] model dimuat dalam {:.2}s ({} state utama, {} state decoder)",
            t0.elapsed().as_secs_f32(),
            main_specs.len(),
            dec_specs.len()
        );

        Ok(Self {
            enc,
            txt,
            main,
            flow,
            dec,
            enc_in,
            enc_out,
            txt_in,
            txt_out,
            flow_ins,
            flow_out,
            main_cond,
            main_eos,
            dec_in,
            dec_out,
            main_specs,
            dec_specs,
            tok,
            bos,
            voice: None,
            voice_frames: 0,
            hidden: 1024,
            _env: env,
        })
    }

    /// Tokenizer yang dipakai (untuk debugging).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Set prompt suara dari sampel mono 24 kHz.
    pub fn set_voice(&mut self, samples: &[f32]) -> Result<()> {
        if samples.len() < SAMPLE_RATE as usize / 10 {
            return Err(anyhow!("referensi suara terlalu pendek (< 0,1 s)"));
        }
        let n = samples.len();
        if let Ok(path) = std::env::var("LARAS_DUMP_PCM") {
            let raw: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, raw)?;
        }
        let input = Value::from_array((Shape::from(vec![1, 1, n]), samples.to_vec()))?.into_dyn();
        let outs = self.enc.run(vec![(self.enc_in.clone(), input)])?;
        let v = outs
            .get(self.enc_out.as_str())
            .ok_or_else(|| anyhow!("output encoder hilang"))?;
        let (sh, d) = v.try_extract_tensor::<f32>()?;
        let dims = dims(sh);
        if dims.len() != 3 {
            return Err(anyhow!("bentuk prompt tak terduga: {dims:?}"));
        }
        let (frames, hidden) = (dims[1], dims[2]);
        if hidden != self.bos.len() {
            return Err(anyhow!(
                "dimensi prompt ({hidden}) tidak cocok dengan bos ({})",
                self.bos.len()
            ));
        }

        let mut voice = Vec::with_capacity((frames + 1) * hidden);
        voice.extend_from_slice(&self.bos);
        voice.extend_from_slice(d);

        if let Ok(path) = std::env::var("LARAS_DUMP_VOICE") {
            let raw: Vec<u8> = voice.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, raw)?;
        }

        self.voice = Some(voice);
        self.voice_frames = frames + 1;
        self.hidden = hidden;
        Ok(())
    }

    /// Jumlah frame prompt suara (termasuk BOS).
    pub fn voice_frames(&self) -> usize {
        self.voice_frames
    }

    /// Sintesis teks menjadi sampel mono 24 kHz.
    ///
    /// `noise` opsional: deret normal baku `[frames][32]` untuk uji kesetaraan
    /// numerik. Bila `None`, dipakai RNG internal dari `params.seed`.
    pub fn synthesize(
        &mut self,
        text: &str,
        params: &Params,
        noise: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Diag, Vec<f32>)> {
        let t_start = Instant::now();
        let voice = self
            .voice
            .clone()
            .ok_or_else(|| anyhow!("suara referensi belum di-set"))?;
        let voice_frames = self.voice_frames;
        let hidden = self.hidden;

        let (prepared, eos_extra) = prepare_text(text);
        let ids = self.tok.encode(&prepared);
        if ids.is_empty() {
            return Err(anyhow!("teks tidak menghasilkan token"));
        }
        let n_ids = ids.len();
        let ids_i64: Vec<i64> = ids.iter().map(|&i| i as i64).collect();

        // 1) kondisi teks
        let temb = {
            let t = Value::from_array((Shape::from(vec![1, n_ids]), ids_i64))?.into_dyn();
            let outs = self.txt.run(vec![(self.txt_in.clone(), t)])?;
            let v = outs
                .get(self.txt_out.as_str())
                .ok_or_else(|| anyhow!("output text_conditioner hilang"))?;
            v.try_extract_tensor::<f32>()?.1.to_vec()
        };

        if let Ok(path) = std::env::var("LARAS_DUMP_TEMB") {
            let raw: Vec<u8> = temb.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, raw)?;
        }

        // 2) dua pass conditioning pada flow_lm_main
        let mut states: Vec<State> = self.main_specs.iter().map(State::zeros).collect();

        let empty_seq =
            Value::from_array((Shape::from(vec![1, 0, LATENT]), Vec::<f32>::new()))?.into_dyn();
        let voice_v =
            Value::from_array((Shape::from(vec![1, voice_frames, hidden]), voice))?.into_dyn();
        run_main(
            &mut self.main,
            &mut states,
            &self.main_cond,
            &self.main_eos,
            empty_seq,
            voice_v,
        )?;

        let empty_seq =
            Value::from_array((Shape::from(vec![1, 0, LATENT]), Vec::<f32>::new()))?.into_dyn();
        let te_v = Value::from_array((Shape::from(vec![1, n_ids, hidden]), temb))?.into_dyn();
        run_main(
            &mut self.main,
            &mut states,
            &self.main_cond,
            &self.main_eos,
            empty_seq,
            te_v,
        )?;

        // 3) loop autoregresif
        let t_ar = Instant::now();
        let std = params.temperature.max(0.0).sqrt();
        let mut rng = Rng::new(params.seed);
        let mut latents: Vec<f32> = Vec::new();
        let mut frames = 0usize;
        let mut eos_frame: Option<usize> = None;
        let mut cur = vec![f32::NAN; LATENT];
        let mut eos_logits: Vec<f32> = Vec::with_capacity(MAX_FRAMES);

        while frames < MAX_FRAMES {
            let seq = Value::from_array((Shape::from(vec![1, 1, LATENT]), cur.clone()))?.into_dyn();
            let empty_te =
                Value::from_array((Shape::from(vec![1, 0, hidden]), Vec::<f32>::new()))?.into_dyn();
            let (cond, eos) = run_main(
                &mut self.main,
                &mut states,
                &self.main_cond,
                &self.main_eos,
                seq,
                empty_te,
            )?;

            if frames == 0 {
                if let Ok(path) = std::env::var("LARAS_DUMP_COND") {
                    let raw: Vec<u8> = cond.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(path, raw)?;
                }
            }

            eos_logits.push(eos);
            if eos_frame.is_none() && eos > params.eos_threshold && frames >= MIN_FRAMES {
                eos_frame = Some(frames);
            }
            if let Some(ef) = eos_frame {
                if frames >= ef + eos_extra {
                    break;
                }
            }

            let n: Vec<f32> = match noise {
                Some(buf) => {
                    let s = frames * LATENT;
                    if s + LATENT > buf.len() {
                        return Err(anyhow!("berkas noise kurang panjang"));
                    }
                    buf[s..s + LATENT].to_vec()
                }
                None => (0..LATENT).map(|_| rng.normal() * std).collect(),
            };

            let x = Value::from_array((Shape::from(vec![1, LATENT]), n.clone()))?.into_dyn();
            let c = Value::from_array((Shape::from(vec![cond.len()]), cond))?.into_dyn();
            let s0 = Value::from_array((Shape::from(vec![1usize, 1]), vec![0.0f32]))?.into_dyn();
            let t1 = Value::from_array((Shape::from(vec![1usize, 1]), vec![1.0f32]))?.into_dyn();

            let v = {
                let outs = self.flow.run(vec![
                    (self.flow_ins[0].clone(), c),
                    (self.flow_ins[1].clone(), s0),
                    (self.flow_ins[2].clone(), t1),
                    (self.flow_ins[3].clone(), x),
                ])?;
                outs.get(self.flow_out.as_str())
                    .ok_or_else(|| anyhow!("output flow hilang"))?
                    .try_extract_tensor::<f32>()?
                    .1
                    .to_vec()
            };
            if frames == 0 {
                if let Ok(path) = std::env::var("LARAS_DUMP_FLOW0") {
                    let raw: Vec<u8> = v.iter().flat_map(|b| b.to_le_bytes()).collect();
                    std::fs::write(path, raw)?;
                }
            }

            if v.len() != LATENT {
                return Err(anyhow!("output flow berukuran {} (harap 32)", v.len()));
            }

            cur = n.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
            latents.extend_from_slice(&cur);
            frames += 1;
        }

        if frames == 0 {
            return Err(anyhow!("model tidak menghasilkan frame"));
        }

        if let Ok(path) = std::env::var("LARAS_DUMP_LATENTS") {
            let raw: Vec<u8> = latents.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, raw)?;
        }

        // Jalur uji: lewati loop AR, pakai latent dari berkas (isolasi decoder).
        if let Ok(path) = std::env::var("LARAS_LATENTS_IN") {
            let raw = std::fs::read(&path)?;
            latents = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            frames = latents.len() / LATENT;
            if frames == 0 {
                return Err(anyhow!("berkas latent kosong"));
            }
            eprintln!("[laras] latent eksternal: {frames} frame dari {path}");
        }

        // 4) decoder streaming: satu frame per panggilan
        let t_dec = Instant::now();
        let mut dstates: Vec<State> = self.dec_specs.iter().map(State::zeros).collect();
        let mut audio: Vec<f32> = Vec::with_capacity(frames * 1920);
        for f in 0..frames {
            let lat = latents[f * LATENT..(f + 1) * LATENT].to_vec();
            let lv = Value::from_array((Shape::from(vec![1, 1, LATENT]), lat))?.into_dyn();
            let chunk = run_dec(&mut self.dec, &mut dstates, &self.dec_in, &self.dec_out, lv)?;
            audio.extend_from_slice(&chunk);
        }

        if std::env::var("LARAS_TRACE").is_ok() {
            eprintln!(
                "[laras] {} token | {} frame AR {:.2}s | decode {:.2}s | total {:.2}s",
                n_ids,
                frames,
                t_ar.elapsed().as_secs_f32(),
                t_dec.elapsed().as_secs_f32(),
                t_start.elapsed().as_secs_f32()
            );
        }

        let abs_mean = if latents.is_empty() {
            0.0
        } else {
            latents.iter().map(|v| v.abs()).sum::<f32>() / latents.len() as f32
        };
        let mean = if latents.is_empty() {
            0.0
        } else {
            latents.iter().sum::<f32>() / latents.len() as f32
        };
        let var = if latents.is_empty() {
            0.0
        } else {
            latents.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / latents.len() as f32
        };
        let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rms = if audio.is_empty() {
            0.0
        } else {
            (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt()
        };
        let max_eos = eos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let eos_conf = eos_frame
            .and_then(|f| eos_logits.get(f).copied())
            .unwrap_or(max_eos);

        let diag = Diag {
            frames,
            eos_frame,
            max_eos,
            eos_conf,
            latent_abs: abs_mean,
            latent_std: var.sqrt(),
            peak,
            rms,
            dur: audio.len() as f32 / SAMPLE_RATE as f32,
            ms: t_start.elapsed().as_secs_f64() * 1000.0,
        };

        Ok((audio, diag, latents))
    }

    /// Jalankan `takes` kandidat dan pilih yang skornya tertinggi.
    ///
    /// Seed naik per take (seed, seed+1, ...), jadi hasilnya deterministik
    /// untuk kombinasi teks + seed yang sama.
    pub fn synthesize_best(
        &mut self,
        text: &str,
        params: &Params,
        takes: usize,
        noise: Option<&[f32]>,
    ) -> Result<Best> {
        let words = text.split_whitespace().count();
        let n = takes.max(1);
        let mut cands: Vec<(Diag, Vec<f32>, Vec<f32>)> = Vec::with_capacity(n);

        for k in 0..n {
            let mut p = *params;
            p.seed = params.seed.wrapping_add(k as u64);
            let (audio, diag, lat) = self.synthesize(text, &p, noise)?;
            if std::env::var("LARAS_TRACE").is_ok() {
                eprintln!(
                    "[laras] take {k}: skor {:+.2} | {} frame | eos {:?} ({:+.2}) | {:.2}s",
                    diag.score(words),
                    diag.frames,
                    diag.eos_frame,
                    diag.eos_conf,
                    diag.dur
                );
            }
            cands.push((diag, audio, lat));
        }

        // Pemilihan tanpa ASR: ambil medoid di ruang latent.
        //
        // Take yang rusak hampir selalu menyimpang dari take lain (isi
        // fonem hilang, tempo melenceng), jadi take yang paling dekat
        // dengan "pusat" kandidat biasanya yang paling wajar.
        let index = medoid_index(&cands);

        Ok(Best { index, cands })
    }
}

/// Samakan panjang sekuens latent ke `frames` baris dengan interpolasi linear.
fn resample_latents(lat: &[f32], frames: usize) -> Vec<f32> {
    let n = lat.len() / LATENT;
    let mut out = vec![0.0f32; frames * LATENT];
    if n == 0 {
        return out;
    }
    for i in 0..frames {
        let pos = if frames <= 1 {
            0.0
        } else {
            i as f32 * (n - 1) as f32 / (frames - 1) as f32
        };
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let w = pos - lo as f32;
        for d in 0..LATENT {
            let a = lat[lo * LATENT + d];
            let b = lat[hi * LATENT + d];
            out[i * LATENT + d] = a * (1.0 - w) + b * w;
        }
    }
    out
}

/// Jarak kosinus antar dua vektor (0 = identik).
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 1.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

/// Indeks take yang paling dekat ke pusat kandidat (medoid kosinus).
fn medoid_index(cands: &[(Diag, Vec<f32>, Vec<f32>)]) -> usize {
    if cands.len() <= 1 {
        return 0;
    }
    let norm: Vec<Vec<f32>> = cands
        .iter()
        .map(|c| resample_latents(&c.2, MEDOID_FRAMES))
        .collect();

    let mut best_i = 0usize;
    let mut best_sum = f32::INFINITY;
    for i in 0..norm.len() {
        let mut sum = 0.0f32;
        for j in 0..norm.len() {
            if i != j {
                sum += cosine_dist(&norm[i], &norm[j]);
            }
        }
        if sum < best_sum - 1e-6 {
            best_sum = sum;
            best_i = i;
        }
    }
    best_i
}

/// Hasil pemilihan take terbaik beserta seluruh kandidat.
pub struct Best {
    /// Indeks take yang dipilih.
    pub index: usize,
    /// (statistik, audio, latent) untuk setiap take.
    pub cands: Vec<(Diag, Vec<f32>, Vec<f32>)>,
}

impl Best {
    /// Audio take terpilih.
    pub fn audio(&self) -> &[f32] {
        &self.cands[self.index].1
    }

    /// Statistik take terpilih.
    pub fn diag(&self) -> &Diag {
        &self.cands[self.index].0
    }

    /// Statistik semua take.
    pub fn diags(&self) -> impl Iterator<Item = &Diag> {
        self.cands.iter().map(|(d, _, _)| d)
    }

    /// Latent take terpilih (frame x 32, baris-mayor).
    pub fn latents(&self) -> &[f32] {
        &self.cands[self.index].2
    }

    /// Latent semua take.
    pub fn all_latents(&self) -> impl Iterator<Item = &[f32]> {
        self.cands.iter().map(|(_, _, l)| l.as_slice())
    }
}

fn open(models: &Path, file: &str, threads: usize, env: &Environment) -> Result<Session> {
    let p = models.join(file);
    if !p.is_file() {
        return Err(anyhow!("berkas model tidak ada: {}", p.display()));
    }
    let sess = Session::builder(env)
        .map_err(|e| anyhow!("builder ONNX: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("optimasi ONNX: {e}"))?
        .with_intra_threads(threads.max(1))
        .map_err(|e| anyhow!("thread ONNX: {e}"))?
        .commit_from_file(&p)
        .with_context(|| format!("gagal memuat {}", p.display()))?;
    Ok(sess)
}

fn first_input(sess: &Session) -> String {
    sess.inputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_default()
}

fn first_output(sess: &Session) -> String {
    sess.outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_default()
}

fn pick(names: &[String], want: &str, idx: usize) -> String {
    if let Some(n) = names.iter().find(|n| n.as_str() == want) {
        return n.clone();
    }
    names.get(idx).cloned().unwrap_or_else(|| want.to_string())
}

fn state_specs(sess: &Session) -> Result<Vec<Spec>> {
    let mut out = Vec::new();
    for o in sess.inputs() {
        let name = o.name();
        if !name.starts_with("state_") {
            continue;
        }
        match o.dtype() {
            ValueType::Tensor { ty, shape, .. } => out.push(Spec {
                name: name.to_string(),
                shape: dims(shape),
                ty: map_dtype(ty)?,
            }),
            other => return Err(anyhow!("state {name} bukan tensor: {other:?}")),
        }
    }
    Ok(out)
}

fn run_main(
    sess: &mut Session,
    states: &mut [State],
    cond_name: &str,
    eos_name: &str,
    seq: DynValue,
    te: DynValue,
) -> Result<(Vec<f32>, f32)> {
    let mut feeds: Vec<(String, DynValue)> = Vec::with_capacity(states.len() + 2);
    feeds.push(("sequence".to_string(), seq));
    feeds.push(("text_embeddings".to_string(), te));
    for s in states.iter() {
        feeds.push((s.name.clone(), s.to_value()?));
    }

    let outs = sess.run(feeds)?;

    let cond = outs
        .get(cond_name)
        .ok_or_else(|| anyhow!("output {cond_name} hilang"))?
        .try_extract_tensor::<f32>()?
        .1
        .to_vec();
    let eos = *outs
        .get(eos_name)
        .ok_or_else(|| anyhow!("output {eos_name} hilang"))?
        .try_extract_tensor::<f32>()?
        .1
        .first()
        .ok_or_else(|| anyhow!("logit EOS kosong"))?;

    for s in states.iter_mut() {
        let oname = format!("out_{}", s.name);
        if let Some(v) = outs.get(oname.as_str()) {
            let name = s.name.clone();
            *s = State::read(&name, v)?;
        }
    }

    Ok((cond, eos))
}

fn run_dec(
    sess: &mut Session,
    states: &mut [State],
    latent_name: &str,
    audio_name: &str,
    latent: DynValue,
) -> Result<Vec<f32>> {
    let mut feeds: Vec<(String, DynValue)> = Vec::with_capacity(states.len() + 1);
    feeds.push((latent_name.to_string(), latent));
    for s in states.iter() {
        feeds.push((s.name.clone(), s.to_value()?));
    }

    let outs = sess.run(feeds)?;

    let chunk = outs
        .get(audio_name)
        .ok_or_else(|| anyhow!("output audio hilang"))?
        .try_extract_tensor::<f32>()?
        .1
        .to_vec();

    for s in states.iter_mut() {
        let oname = format!("out_{}", s.name);
        if let Some(v) = outs.get(oname.as_str()) {
            let name = s.name.clone();
            *s = State::read(&name, v)?;
        }
    }

    Ok(chunk)
}

/// Siapkan teks agar sesuai distribusi latih model.
///
/// Mengembalikan teks siap-tokenisasi dan jumlah frame tambahan setelah EOS
/// (kalimat pendek butuh ekor lebih panjang agar tidak terpotong).
pub fn prepare_text(text: &str) -> (String, usize) {
    let cleaned = text.replace(['"', '`'], "").trim().to_string();
    let nwords = cleaned.split_whitespace().count();

    let mut out = String::with_capacity(cleaned.len() + 9);
    let mut chars = cleaned.chars();
    if let Some(first) = chars.next() {
        if first.is_lowercase() {
            out.extend(first.to_uppercase());
        } else {
            out.push(first);
        }
        out.push_str(chars.as_str());
    }
    if out.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        out.push('.');
    }
    if nwords < 5 {
        out.insert_str(0, "        ");
    }

    (out, if nwords <= 4 { 5 } else { 3 })
}

#[cfg(test)]
mod tests {
    use super::{cosine_dist, medoid_index, prepare_text, resample_latents, Diag, Rng, LATENT};

    #[test]
    fn teks_pendek_diberi_awalan_dan_titik() {
        let (t, extra) = prepare_text("halo dunia");
        assert_eq!(t, "        Halo dunia.");
        assert_eq!(extra, 5);
    }

    #[test]
    fn teks_panjang_tanpa_awalan() {
        let (t, extra) = prepare_text("Selamat pagi, nama saya Laras");
        assert_eq!(t, "Selamat pagi, nama saya Laras.");
        assert_eq!(extra, 3);
    }

    #[test]
    fn rng_deterministik_dan_wajar() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let mut sum = 0f64;
        for _ in 0..10_000 {
            let x = a.normal();
            assert_eq!(x.to_bits(), b.normal().to_bits());
            sum += x as f64;
        }
        let mean = sum / 10_000.0;
        assert!(mean.abs() < 0.05, "mean {mean}");
    }

    fn diag_polos() -> Diag {
        Diag {
            frames: 10,
            eos_frame: Some(8),
            max_eos: 0.0,
            eos_conf: 0.0,
            latent_abs: 0.5,
            latent_std: 0.5,
            peak: 0.8,
            rms: 0.1,
            dur: 2.0,
            ms: 100.0,
        }
    }

    #[test]
    fn resample_latents_menjaga_panjang() {
        let a: Vec<f32> = (0..10 * LATENT).map(|i| i as f32).collect();
        let b = resample_latents(&a, 64);
        assert_eq!(b.len(), 64 * LATENT);
        // titik awal & akhir dipertahankan
        assert!((b[0] - a[0]).abs() < 1e-4);
        assert!((b[b.len() - 1] - a[a.len() - 1]).abs() < 1e-4);
        let c = resample_latents(&a, 10);
        assert_eq!(c, a);
    }

    #[test]
    fn cosine_dist_nol_untuk_vektor_sama() {
        let a = vec![0.3f32, -1.2, 4.0];
        assert!(cosine_dist(&a, &a).abs() < 1e-6);
        let b = vec![-0.3f32, 1.2, -4.0];
        assert!(cosine_dist(&a, &b) > 1.9);
    }

    #[test]
    fn medoid_membuang_pencilan() {
        // tiga take hampir sama, satu menyimpang jauh
        let dasar: Vec<f32> = (0..20 * LATENT).map(|i| ((i % 7) as f32) - 3.0).collect();
        let mut pencilan = dasar.clone();
        for v in pencilan.iter_mut() {
            *v = -*v * 3.0;
        }
        let cands = vec![
            (diag_polos(), vec![], pencilan),
            (diag_polos(), vec![], dasar.clone()),
            (diag_polos(), vec![], dasar.clone()),
            (diag_polos(), vec![], dasar),
        ];
        let i = medoid_index(&cands);
        assert_ne!(i, 0, "take pencilan tidak boleh terpilih");
    }
}
