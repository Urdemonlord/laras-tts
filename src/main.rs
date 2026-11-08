//! tts-gateway — HTTP gateway TTS Bahasa Indonesia.
//!
//! Tanggung jawab (semua di Rust, tanpa Python):
//!   * normalisasi angka: "Rp 12.500" -> "dua belas ribu lima ratus rupiah"
//!   * cache hasil berdasarkan hash teks + voice + takes
//!   * serialisasi akses ke worker (1 proses Python, model tetap di RAM)
//!
//! Worker (Python, `worker.py`) melakukan: chunking per kalimat, generasi,
//! best-of-N dipilih lewat WER Whisper, tulis WAV 24 kHz mono.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------- normalizer

const SATUAN: [&str; 12] = [
    "", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh",
    "sebelas",
];

// `is_multiple_of()` baru stabil di Rust 1.87; sengaja pakai `%` agar MSRV tetap 1.75.
#[allow(clippy::manual_is_multiple_of)]
fn ratus(n: u64) -> String {
    if n < 12 {
        return SATUAN[n as usize].to_string();
    }
    if n < 20 {
        return format!("{} belas", ratus(n - 10));
    }
    if n < 100 {
        let s = format!("{} puluh", ratus(n / 10));
        return if n % 10 != 0 {
            format!("{} {}", s, ratus(n % 10))
        } else {
            s
        };
    }
    let s = if n < 200 {
        "seratus".to_string()
    } else {
        format!("{} ratus", ratus(n / 100))
    };
    if n % 100 != 0 {
        format!("{} {}", s, ratus(n % 100))
    } else {
        s
    }
}

fn terbilang(n: u64) -> String {
    if n == 0 {
        return "nol".to_string();
    }
    let mut n = n;
    let mut parts: Vec<String> = Vec::new();
    let scales: [(u64, &str); 5] = [
        (1_000_000_000_000, "triliun"),
        (1_000_000_000, "miliar"),
        (1_000_000, "juta"),
        (1_000, "ribu"),
        (1, ""),
    ];
    for (nilai, nama) in scales {
        if n >= nilai {
            let q = n / nilai;
            n %= nilai;
            if nilai == 1000 && q == 1 {
                parts.push("seribu".to_string());
            } else if nama.is_empty() {
                parts.push(ratus(q));
            } else {
                parts.push(format!("{} {}", ratus(q), nama));
            }
        }
    }
    parts.join(" ")
}

fn re_rp() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\bRp\.?\s*([0-9](?:[0-9.,]*[0-9])?)").unwrap())
}

fn re_persen() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"([0-9](?:[0-9.,]*[0-9])?)\s*%").unwrap())
}

fn re_angka() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[0-9](?:[0-9.,]*[0-9])?").unwrap())
}

fn re_ribuan() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[0-9]{1,3}(\.[0-9]{3})+$").unwrap())
}

fn re_spasi() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}

/// "12.500" -> "dua belas ribu lima ratus", "32,5" -> "tiga puluh dua koma lima"
fn angka(tok: &str) -> String {
    let tok = tok.trim().trim_end_matches('.');
    if let Some((bulat, pecah)) = tok.split_once(',') {
        let b: String = bulat.replace('.', "");
        let b = if b.is_empty() { "0".to_string() } else { b };
        let mut out = format!("{} koma", terbilang(b.parse().unwrap_or(0)));
        for c in pecah.chars().filter(|c| c.is_ascii_digit()) {
            out.push(' ');
            out.push_str(&terbilang(c.to_digit(10).unwrap() as u64));
        }
        return out;
    }
    let clean = if re_ribuan().is_match(tok) {
        tok.replace('.', "")
    } else {
        tok.to_string()
    };
    if clean.parse::<u64>().is_err() {
        // bukan bilangan utuh (mis. "09.30", "4.2.1") -> baca digit per digit
        let mut out: Vec<String> = Vec::new();
        for c in tok.chars() {
            match c {
                '0'..='9' => out.push(terbilang(c.to_digit(10).unwrap() as u64)),
                '.' => out.push("titik".to_string()),
                ',' => out.push("koma".to_string()),
                _ => {}
            }
        }
        return out.join(" ");
    }
    terbilang(clean.parse().unwrap_or(0))
}

/// Normalisasi teks sebelum masuk model: semua digit jadi kata.
pub fn normalize(text: &str) -> String {
    let s = re_rp()
        .replace_all(text, |c: &regex::Captures| {
            format!("{} rupiah", angka(&c[1]))
        })
        .to_string();
    let s = re_persen()
        .replace_all(&s, |c: &regex::Captures| format!("{} persen", angka(&c[1])))
        .to_string();
    let s = re_angka()
        .replace_all(&s, |c: &regex::Captures| angka(&c[0]))
        .to_string();
    re_spasi().replace_all(s.trim(), " ").to_string()
}

// ------------------------------------------------------------------- worker

struct Worker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct AppState {
    worker: Mutex<Worker>,
    cache_dir: PathBuf,
    ref_wav: String,
    takes: u32,
}

const DEFAULT_POCKET_CFG: &str = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5";

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn hash_key(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex::encode(&h.finalize()[..12])
}

fn wav_response(bytes: Vec<u8>, cached: bool, dur: Option<f64>, wer: Option<f64>) -> Response {
    let mut b = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-cache", if cached { "hit" } else { "miss" });
    if let Some(d) = dur {
        b = b.header("x-duration", format!("{d:.2}"));
    }
    if let Some(w) = wer {
        b = b.header("x-wer", format!("{w:.3}"));
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
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let text = q.get("text").cloned().unwrap_or_default();
    Json(serde_json::json!({"input": text, "normalized": normalize(&text)})).into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let w = st.worker.lock().unwrap();
    Json(serde_json::json!({
        "ok": true,
        "pid": w._child.id(),
        "default_voice": st.ref_wav,
        "takes": st.takes,
        "cache_dir": st.cache_dir.to_string_lossy(),
    }))
    .into_response()
}

async fn tts(State(st): State<Arc<AppState>>, Json(req): Json<TtsReq>) -> Response {
    let text = normalize(&req.text);
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, "teks kosong").into_response();
    }
    let ref_wav = req.voice.unwrap_or_else(|| st.ref_wav.clone());
    let takes = req.takes.unwrap_or(st.takes).clamp(1, 5);
    let key = hash_key(&format!("{text}|{ref_wav}|{takes}"));
    let cached_path = st.cache_dir.join(format!("{key}.wav"));

    if let Ok(bytes) = std::fs::read(&cached_path) {
        return wav_response(bytes, true, None, None);
    }

    let mut w = st.worker.lock().unwrap();
    let payload = serde_json::json!({
        "id": key,
        "text": text,
        "ref": ref_wav,
        "takes": takes,
    });
    let mut line = payload.to_string();
    line.push('\n');
    if let Err(e) = w
        .stdin
        .write_all(line.as_bytes())
        .and_then(|_| w.stdin.flush())
    {
        return (
            StatusCode::BAD_GATEWAY,
            format!("worker tidak bisa ditulis: {e}"),
        )
            .into_response();
    }

    let mut resp = String::new();
    match w.stdout.read_line(&mut resp) {
        Ok(0) => return (StatusCode::BAD_GATEWAY, "worker mati").into_response(),
        Ok(_) => {}
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, format!("baca worker gagal: {e}")).into_response()
        }
    }

    let v: serde_json::Value = match serde_json::from_str(resp.trim()) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("respons worker bukan JSON: {e}: {resp}"),
            )
                .into_response()
        }
    };
    if v.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        let err = v.get("err").and_then(|x| x.as_str()).unwrap_or("unknown");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker error: {err}"),
        )
            .into_response();
    }
    let path = v.get("path").and_then(|x| x.as_str()).unwrap_or_default();
    let dur = v.get("dur").and_then(|x| x.as_f64());
    let wer = v.get("wer").and_then(|x| x.as_f64());
    match std::fs::read(path) {
        Ok(bytes) => {
            let _ = std::fs::copy(path, &cached_path);
            wav_response(bytes, false, dur, wer)
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("wav tidak terbaca: {e}"),
        )
            .into_response(),
    }
}

#[tokio::main]
async fn main() {
    let port: u16 = env_or("PORT", "8080").parse().unwrap_or(8080);
    let cache_dir = PathBuf::from(env_or("CACHE_DIR", "data/cache"));
    std::fs::create_dir_all(&cache_dir).expect("cache dir");
    let ref_wav = env_or("REF_WAV", "data/ref/suara.wav");
    let takes: u32 = env_or("TAKES", "3").parse().unwrap_or(3);

    let python = env_or("WORKER_PY", "python3");
    let script = env_or("WORKER_SCRIPT", "worker.py");
    eprintln!("menjalankan worker: {python} {script}");

    let mut child = Command::new(&python)
        .arg(&script)
        .env("POCKET_CFG", env_or("POCKET_CFG", DEFAULT_POCKET_CFG))
        .env("WORKER_OUT", env_or("WORKER_OUT", "data/out"))
        .env("USE_ASR", env_or("USE_ASR", "1"))
        .env("EOS", env_or("EOS", "-3.0"))
        .env("HF_HOME", env_or("HF_HOME", "data/hf"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("gagal spawn worker (cek WORKER_PY / WORKER_SCRIPT)");

    let stdin = child.stdin.take().expect("stdin worker");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout worker"));

    // Baris pertama worker adalah handshake {"ready":true,...}; harus dihabiskan
    // di sini, kalau tidak akan terbaca sebagai respons request pertama.
    let mut hello = String::new();
    match stdout.read_line(&mut hello) {
        Ok(n) if n > 0 => eprintln!("worker siap: {}", hello.trim()),
        _ => eprintln!("worker tidak mengirim handshake (lanjut saja)"),
    }

    let state = Arc::new(AppState {
        worker: Mutex::new(Worker {
            _child: child,
            stdin,
            stdout,
        }),
        cache_dir,
        ref_wav,
        takes,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/normalize", get(normalize_handler))
        .route("/tts", post(tts))
        .with_state(state);

    let bind = env_or("BIND", "127.0.0.1");
    let addr = format!("{bind}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!("tts-gateway listening on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terbilang_dasar() {
        assert_eq!(terbilang(0), "nol");
        assert_eq!(terbilang(11), "sebelas");
        assert_eq!(terbilang(15), "lima belas");
        assert_eq!(terbilang(100), "seratus");
        assert_eq!(terbilang(1000), "seribu");
        assert_eq!(terbilang(12500), "dua belas ribu lima ratus");
        assert_eq!(terbilang(2026), "dua ribu dua puluh enam");
        assert_eq!(terbilang(1_250_000), "satu juta dua ratus lima puluh ribu");
    }

    #[test]
    fn normalisasi_kalimat() {
        assert_eq!(
            normalize("Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."),
            "Harga saham naik lima belas persen menjadi dua belas ribu lima ratus rupiah per lembar pada dua ribu dua puluh enam."
        );
        assert_eq!(
            normalize("Suhu 32,5 derajat"),
            "Suhu tiga puluh dua koma lima derajat"
        );
        assert_eq!(normalize("Biaya Rp 5.000.000"), "Biaya lima juta rupiah");
        assert_eq!(
            normalize("tahun 1998"),
            "tahun seribu sembilan ratus sembilan puluh delapan"
        );
        assert_eq!(normalize("tanpa angka"), "tanpa angka");
        assert_eq!(normalize("   spasi   berlebih  "), "spasi berlebih");
        assert_eq!(
            normalize("pukul 09.30"),
            "pukul nol sembilan titik tiga nol"
        );
        assert_eq!(normalize("versi 4.2.1"), "versi empat titik dua titik satu");
    }
}
