//! laras — engine TTS Bahasa Indonesia, CPU-first, dengan kloning suara.
//!
//! Tanggung jawab (semua di Rust):
//!   * normalisasi teks lewat profil bahasa (`src/lang/`)
//!   * cache hasil berdasarkan hash teks + voice + takes
//!   * sintesis lewat backend native ONNX Runtime (fitur `onnx`)
//!
//! Dengan `--features onnx` tidak ada proses Python sama sekali: model dimuat
//! sekali ke dalam proses ini, lalu `takes` kandidat digenerate dan yang
//! terbaik dipilih tanpa ASR (medoid di ruang latent).
//!
//! Tanpa fitur `onnx`, build tetap jalan memakai `worker.py` sebagai backend
//! cadangan supaya kontrak HTTP bisa diuji tanpa runtime ONNX.

mod lang;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[cfg(feature = "onnx")]
mod native;

#[cfg(feature = "asr")]
mod asr;

// --------------------------------------------------------------- backend

#[cfg(not(feature = "onnx"))]
mod worker {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

    pub struct Worker {
        pub child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    impl Worker {
        pub fn spawn(python: &str, script: &str, envs: &[(&str, String)]) -> Self {
            let mut cmd = Command::new(python);
            cmd.arg(script)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
            for (k, v) in envs {
                cmd.env(k, v);
            }
            let mut child = cmd.spawn().expect("gagal spawn worker");
            let stdin = child.stdin.take().expect("stdin worker");
            let mut stdout = BufReader::new(child.stdout.take().expect("stdout worker"));
            let mut hello = String::new();
            if matches!(stdout.read_line(&mut hello), Ok(n) if n > 0) {
                eprintln!("worker siap: {}", hello.trim());
            }
            Self {
                child,
                stdin,
                stdout,
            }
        }

        /// Kirim satu permintaan; kembalikan (path, durasi, wer).
        pub fn synth(
            &mut self,
            key: &str,
            text: &str,
            ref_wav: &str,
            takes: u32,
        ) -> Result<(String, Option<f64>, Option<f64>), String> {
            let payload = serde_json::json!({
                "id": key, "text": text, "ref": ref_wav, "takes": takes,
            });
            let mut line = payload.to_string();
            line.push('\n');
            self.stdin
                .write_all(line.as_bytes())
                .and_then(|_| self.stdin.flush())
                .map_err(|e| format!("worker tidak bisa ditulis: {e}"))?;

            let mut resp = String::new();
            match self.stdout.read_line(&mut resp) {
                Ok(0) => return Err("worker mati".into()),
                Ok(_) => {}
                Err(e) => return Err(format!("baca worker gagal: {e}")),
            }
            let v: serde_json::Value = serde_json::from_str(resp.trim())
                .map_err(|e| format!("respons worker bukan JSON: {e}: {resp}"))?;
            if v.get("ok").and_then(|x| x.as_bool()) != Some(true) {
                let err = v.get("err").and_then(|x| x.as_str()).unwrap_or("unknown");
                return Err(format!("worker error: {err}"));
            }
            Ok((
                v.get("path").and_then(|x| x.as_str()).unwrap_or("").into(),
                v.get("dur").and_then(|x| x.as_f64()),
                v.get("wer").and_then(|x| x.as_f64()),
            ))
        }
    }
}

enum Backend {
    #[cfg(feature = "onnx")]
    Native(native::Native),
    #[cfg(not(feature = "onnx"))]
    Worker(worker::Worker),
}

struct Synth {
    dur: f64,
    wer: Option<f64>,
    take: Option<usize>,
    /// Jumlah potongan kalimat (>1 kalau teks dipecah).
    chunks: Option<usize>,
    /// Jumlah take yang benar-benar dipakai.
    takes: Option<u32>,
}

impl Backend {
    /// Sintesis ke `out`, kembalikan metadata.
    fn synth(
        &mut self,
        key: &str,
        text: &str,
        ref_wav: &str,
        takes: u32,
        out: &std::path::Path,
    ) -> Result<Synth, String> {
        match self {
            #[cfg(feature = "onnx")]
            Backend::Native(n) => {
                let _ = key;
                let (dur, take, _score) = n
                    .synth_to(text, ref_wav, takes as usize, out)
                    .map_err(|e| format!("sintesis gagal: {e:#}"))?;
                Ok(Synth {
                    dur,
                    wer: None,
                    take: Some(take),
                    chunks: None,
                    takes: Some(takes),
                })
            }
            #[cfg(not(feature = "onnx"))]
            Backend::Worker(w) => {
                let (path, dur, wer) = w.synth(key, text, ref_wav, takes)?;
                std::fs::copy(&path, out).map_err(|e| format!("salin wav gagal: {e}"))?;
                Ok(Synth {
                    dur: dur.unwrap_or(0.0),
                    wer,
                    take: None,
                    chunks: None,
                    takes: Some(takes),
                })
            }
        }
    }

    fn describe(&self) -> serde_json::Value {
        match self {
            #[cfg(feature = "onnx")]
            Backend::Native(n) => serde_json::json!({
                "kind": "onnx-native",
                "models": n.models().to_string_lossy(),
                "voice": n.active_voice(),
                "asr": n.asr_aktif(),
            }),
            #[cfg(not(feature = "onnx"))]
            Backend::Worker(w) => serde_json::json!({
                "kind": "python-worker",
                "pid": w.child.id(),
            }),
        }
    }
}

struct AppState {
    backend: Mutex<Backend>,
    cache_dir: PathBuf,
    ref_wav: String,
    takes: u32,
    /// Pecah teks per kalimat sebelum sintesis (CHUNK=0 mematikan).
    chunk: bool,
    /// Jumlah take per kalimat saat teks dipecah (CHUNK_TAKES).
    chunk_takes: u32,
    /// Di bawah jumlah kata ini, teks tidak dipecah (CHUNK_MIN_WORDS).
    chunk_min_words: usize,
    /// Target loudness LUFS; 0 = jangan dinormalkan (LOUDNESS).
    loudness: f32,
    /// Batas puncak dBFS setelah normalisasi (PEAK_DB).
    peak_db: f32,
    lang: &'static dyn lang::Lang,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn hash_key(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex::encode(&h.finalize()[..12])
}

fn wav_response(bytes: Vec<u8>, cached: bool, s: Option<&Synth>) -> Response {
    let mut b = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-cache", if cached { "hit" } else { "miss" });
    if let Some(s) = s {
        if s.dur > 0.0 {
            b = b.header("x-duration", format!("{:.2}", s.dur));
        }
        if let Some(w) = s.wer {
            b = b.header("x-wer", format!("{w:.3}"));
        }
        if let Some(t) = s.take {
            b = b.header("x-take", t.to_string());
        }
        if let Some(c) = s.chunks {
            b = b.header("x-chunks", c.to_string());
        }
        if let Some(t) = s.takes {
            b = b.header("x-takes", t.to_string());
        }
    }
    b.body(axum::body::Body::from(bytes)).unwrap()
}

#[derive(Deserialize)]
struct TtsReq {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    takes: Option<u32>,
}

async fn normalize_handler(
    State(st): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let text = q.get("text").cloned().unwrap_or_default();
    Json(serde_json::json!({
        "input": text,
        "normalized": st.lang.normalize(&text),
        "prepared": st.lang.prepare(&text),
        "lang": st.lang.code(),
    }))
    .into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let backend = st.backend.lock().map(|b| b.describe()).unwrap_or_default();
    Json(serde_json::json!({
        "ok": true,
        "engine": "laras",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": backend,
        "lang": st.lang.code(),
        "default_voice": st.ref_wav,
        "takes": st.takes,
        "chunk": st.chunk,
        "chunk_takes": st.chunk_takes,
        "chunk_min_words": st.chunk_min_words,
        "loudness": st.loudness,
        "peak_db": st.peak_db,
        "cache_dir": st.cache_dir.to_string_lossy(),
    }))
    .into_response()
}

/// Sambung beberapa wav jadi satu, dengan jeda pendek antar kalimat.
#[cfg(feature = "onnx")]
fn gabung_wav(bagian: &[PathBuf], out: &std::path::Path) -> Result<(), String> {
    use laras::onnx::wav;
    let jeda = vec![0.0f32; (0.15 * 24_000.0) as usize];
    let mut semua: Vec<f32> = Vec::new();
    for (i, p) in bagian.iter().enumerate() {
        let s = wav::read_mono(p).map_err(|e| format!("baca potongan: {e:#}"))?;
        if i > 0 {
            semua.extend_from_slice(&jeda);
        }
        semua.extend_from_slice(&s);
    }
    wav::write_mono(out, &semua).map_err(|e| format!("tulis gabungan: {e:#}"))
}

async fn tts(State(st): State<Arc<AppState>>, Json(req): Json<TtsReq>) -> Response {
    let text = st.lang.prepare(&req.text);
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, "teks kosong").into_response();
    }
    let ref_wav = req.voice.unwrap_or_else(|| st.ref_wav.clone());
    let takes_diminta = req.takes.unwrap_or(st.takes).clamp(1, 8);

    // Teks pendek tidak dipecah: satu lintasan dengan best-of-N + seleksi ASR
    // masih murah, dan pemecahan justru mematikan seleksi.
    #[cfg(feature = "onnx")]
    let kata = text.split_whitespace().count();
    #[cfg(feature = "onnx")]
    let potongan: Vec<String> = if st.chunk && kata > st.chunk_min_words {
        lang::potong_kalimat(&text)
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "onnx"))]
    let potongan: Vec<String> = Vec::new();

    // Teks yang dipecah memakai jatah take sendiri: seleksi ASR per kalimat
    // membuat paragraf tiga kalimat jadi ~75 s, dan tiap kalimat pendek sudah
    // cukup bagus dengan satu take.
    let takes = if potongan.len() > 1 {
        st.chunk_takes.clamp(1, 8)
    } else {
        takes_diminta
    };
    let key = hash_key(&format!("{text}|{ref_wav}|{takes}"));
    let cached_path = st.cache_dir.join(format!("{key}.wav"));

    if let Ok(bytes) = std::fs::read(&cached_path) {
        return wav_response(bytes, true, None);
    }

    // Sintesis memakan CPU penuh: jalankan di thread blocking, bukan di runtime.
    let st2 = Arc::clone(&st);
    let out = cached_path.clone();
    let text2 = text.clone();
    let ref2 = ref_wav.clone();
    let key2 = key.clone();
    let res = tokio::task::spawn_blocking(move || {
        let mut backend = st2
            .backend
            .lock()
            .map_err(|_| "backend poison".to_string())?;

        // Keluaran mentah sekitar −28 LUFS dengan puncak −10 dBFS. Naikkan ke
        // target loudness lalu batasi puncaknya, sekali di akhir supaya hasil
        // gabungan antar kalimat tetap rata.
        let rapikan = |s: Synth| -> Result<Synth, String> {
            #[cfg(feature = "onnx")]
            {
                if st2.loudness == 0.0 {
                    return Ok(s);
                }
                let mut x = laras::onnx::wav::read_mono(&out)
                    .map_err(|e| format!("baca wav untuk normalisasi: {e:#}"))?;
                let (gain, awal, akhir) =
                    laras::level::normalize(&mut x, 24_000.0, st2.loudness, st2.peak_db);
                laras::onnx::wav::write_mono(&out, &x)
                    .map_err(|e| format!("tulis wav ternormalisasi: {e:#}"))?;
                eprintln!("[level] {awal:.1} LUFS -> puncak {akhir:.1} dBFS (gain {gain:+.1} dB)");
            }
            Ok(s)
        };

        if potongan.len() <= 1 {
            return backend
                .synth(&key2, &text2, &ref2, takes, &out)
                .and_then(rapikan);
        }

        // Banyak kalimat: sintesis satu per satu lalu disambung. Model kecil
        // sering tidak berhenti di titik pada paragraf panjang dan mengulang
        // kata di ujung; memecahnya menghilangkan itu.
        #[cfg(feature = "onnx")]
        {
            let mut bagian: Vec<PathBuf> = Vec::with_capacity(potongan.len());
            let mut total = 0.0;
            for (i, k) in potongan.iter().enumerate() {
                let tmp = st2.cache_dir.join(format!("{key2}.p{i}.wav"));
                match backend.synth(&format!("{key2}-{i}"), k, &ref2, takes, &tmp) {
                    Ok(s) => {
                        total += s.dur;
                        bagian.push(tmp);
                    }
                    Err(e) => {
                        for p in &bagian {
                            let _ = std::fs::remove_file(p);
                        }
                        return Err(e);
                    }
                }
            }
            let hasil = gabung_wav(&bagian, &out);
            for p in &bagian {
                let _ = std::fs::remove_file(p);
            }
            hasil?;
            rapikan(Synth {
                dur: total,
                wer: None,
                take: None,
                chunks: Some(potongan.len()),
                takes: Some(takes),
            })
        }
        // Tanpa fitur `onnx` tidak ada cara menyambung wav: satu lintasan saja.
        #[cfg(not(feature = "onnx"))]
        {
            unreachable!("potongan kosong tanpa fitur onnx")
        }
    })
    .await;

    let synth = match res {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("tugas sintesis gagal: {e}"),
            )
                .into_response()
        }
    };

    match std::fs::read(&cached_path) {
        Ok(bytes) => wav_response(bytes, false, Some(&synth)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("wav tidak terbaca: {e}"),
        )
            .into_response(),
    }
}

#[tokio::main]
async fn main() {
    let lang_code = env_or("LARAS_LANG", "id");
    let Some(bahasa) = lang::get(&lang_code) else {
        eprintln!(
            "bahasa \"{lang_code}\" belum didukung. Tersedia: {}",
            lang::codes().join(", ")
        );
        std::process::exit(2);
    };

    let port: u16 = env_or("PORT", "8080").parse().unwrap_or(8080);
    let cache_dir = PathBuf::from(env_or("CACHE_DIR", "data/cache"));
    std::fs::create_dir_all(&cache_dir).expect("cache dir");
    let ref_wav = env_or("REF_WAV", "data/ref/suara.wav");
    let takes: u32 = env_or("TAKES", "3").parse().unwrap_or(3);
    let chunk = env_or("CHUNK", "1") != "0";
    let chunk_takes: u32 = env_or("CHUNK_TAKES", "1").parse().unwrap_or(1);
    let chunk_min_words: usize = env_or("CHUNK_MIN_WORDS", "20").parse().unwrap_or(20);
    let loudness: f32 = env_or("LOUDNESS", "-18").parse().unwrap_or(-16.0);
    let peak_db: f32 = env_or("PEAK_DB", "-1.5").parse().unwrap_or(-1.5);
    // 0 = pakai semua inti yang tersedia.
    let threads: usize = env_or("THREADS", "0").parse::<usize>().unwrap_or(0);

    #[cfg(feature = "onnx")]
    let backend = {
        use laras::onnx::Params;
        let models = PathBuf::from(env_or("MODELS_DIR", "data/models"));
        let mut params = Params::default();
        if let Ok(v) = std::env::var("TEMP") {
            if let Ok(v) = v.parse() {
                params.temperature = v;
            }
        }
        if let Ok(v) = std::env::var("EOS") {
            if let Ok(v) = v.parse() {
                params.eos_threshold = v;
            }
        }
        if let Ok(v) = std::env::var("SEED") {
            if let Ok(v) = v.parse() {
                params.seed = v;
            }
        }
        let threads = if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            threads
        };
        eprintln!(
            "memuat model native dari {} ({} thread, temp {})",
            models.display(),
            threads,
            params.temperature
        );

        // Pemilih take berbasis ASR: aktif hanya kalau ASR_MODEL diisi.
        #[cfg(feature = "asr")]
        let asr = match std::env::var("ASR_MODEL") {
            Ok(p) if !p.is_empty() => {
                let path = PathBuf::from(&p);
                eprintln!("memuat model ASR dari {}", path.display());
                let a = asr::Asr::new(&path, threads).expect("gagal memuat model ASR");
                Some(a)
            }
            _ => None,
        };

        Backend::Native(
            native::Native::new(
                models,
                threads,
                params,
                &lang_code,
                #[cfg(feature = "asr")]
                asr,
            )
            .expect("gagal memuat model ONNX"),
        )
    };

    #[cfg(not(feature = "onnx"))]
    let backend = {
        let _ = threads;
        let python = env_or("WORKER_PY", "python3");
        let script = env_or("WORKER_SCRIPT", "worker.py");
        eprintln!("menjalankan worker: {python} {script}");
        Backend::Worker(worker::Worker::spawn(
            &python,
            &script,
            &[
                ("WORKER_OUT", env_or("WORKER_OUT", "data/out")),
                ("USE_ASR", env_or("USE_ASR", "1")),
                ("EOS", env_or("EOS", "-3.0")),
                ("HF_HOME", env_or("HF_HOME", "data/hf")),
            ],
        ))
    };

    let state = Arc::new(AppState {
        backend: Mutex::new(backend),
        cache_dir,
        chunk,
        chunk_takes,
        chunk_min_words,
        loudness,
        peak_db,
        ref_wav,
        takes,
        lang: bahasa,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/normalize", get(normalize_handler))
        .route("/tts", post(tts))
        .with_state(state);

    let bind = env_or("BIND", "127.0.0.1");
    let addr = format!("{bind}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!(
        "laras {} (bahasa {}) listening on http://{addr}",
        env!("CARGO_PKG_VERSION"),
        bahasa.code()
    );
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_stabil_dan_pendek() {
        let a = hash_key("halo|ref.wav|3");
        assert_eq!(a, hash_key("halo|ref.wav|3"));
        assert_ne!(a, hash_key("halo|ref.wav|1"));
        assert_eq!(a.len(), 24);
    }
}
