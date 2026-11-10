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

// Modul yang sama dipakai oleh pustaka `laras` (lihat `src/lib.rs`), supaya
// gateway HTTP dan aplikasi desktop berbagi satu jalur sintesis.
use laras::lang;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[cfg(feature = "onnx")]
use laras::native;

#[cfg(feature = "asr")]
use laras::asr;

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

// Varian `Uji` (hanya saat tes) jauh lebih kecil dari `Native`, jadi clippy
// memperingatkan selisih ukuran yang tidak pernah muncul di build rilis.
#[cfg_attr(test, allow(clippy::large_enum_variant))]
enum Backend {
    #[cfg(feature = "onnx")]
    Native(native::Native),
    #[cfg(not(feature = "onnx"))]
    Worker(worker::Worker),
    /// Hanya dipakai tes: rute HTTP bisa diuji tanpa memuat model 438 MB.
    #[cfg(test)]
    Uji,
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
            #[cfg(test)]
            Backend::Uji => {
                let _ = (key, text, ref_wav, takes, out);
                Err("backend uji: tidak ada model".to_string())
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
            #[cfg(test)]
            Backend::Uji => serde_json::json!({ "kind": "uji" }),
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
    /// Slot sintesis paralel (MAX_CONCURRENT); sisanya menunggu di antrean.
    antrean: Arc<Semaphore>,
    /// Berapa lama boleh menunggu slot sebelum dijawab 429 (QUEUE_WAIT).
    tunggu_antrean: Duration,
    lang: &'static dyn lang::Lang,
}

// ---------------------------------------------------------------- penjaga
//
// Pengaman runtime: siapa yang boleh memanggil, seberapa sering, berapa lama,
// dan berapa banyak yang boleh menunggu sekaligus. Semuanya opsional supaya
// pemakaian di localhost tetap tanpa gesekan, tapi tersedia begitu port-nya
// dibuka ke jaringan.

/// Batas laju per klien dengan jendela tetap 60 detik.
struct Laju {
    batas: u32,
    jendela: Duration,
    isi: Mutex<HashMap<String, (Instant, u32)>>,
}

impl Laju {
    fn baru(batas: u32) -> Self {
        Self {
            batas,
            jendela: Duration::from_secs(60),
            isi: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` = boleh lewat; `Err(detik)` = tolak dan sebut sisa waktunya.
    fn izinkan(&self, kunci: &str) -> Result<(), u64> {
        let mut isi = self.isi.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        // Jangan biarkan peta tumbuh tanpa batas kalau kliennya banyak.
        if isi.len() > 4096 {
            isi.retain(|_, (t, _)| now.duration_since(*t) < self.jendela);
        }
        let e = isi.entry(kunci.to_string()).or_insert((now, 0));
        let lewat = now.duration_since(e.0);
        if lewat >= self.jendela {
            *e = (now, 1);
            return Ok(());
        }
        if e.1 >= self.batas {
            return Err(self.jendela.saturating_sub(lewat).as_secs().max(1));
        }
        e.1 += 1;
        Ok(())
    }
}

#[derive(Clone)]
struct Penjaga {
    token: Option<String>,
    laju: Option<Arc<Laju>>,
    batas_waktu: Duration,
    batas_body: usize,
}

impl Penjaga {
    fn dari_env(bind: &str) -> Self {
        let token = match std::env::var("LARAS_TOKEN") {
            Ok(t) if !t.trim().is_empty() => Some(t.trim().to_string()),
            _ => None,
        };
        let laju = match env_angka("RATE_LIMIT", 0u32) {
            0 => None,
            n => Some(Arc::new(Laju::baru(n))),
        };
        let batas_waktu = Duration::from_secs(env_angka("REQUEST_TIMEOUT", 300u64).max(1));
        let batas_body = env_angka("MAX_BODY_KB", 64usize).max(1) * 1024;

        let lokal = matches!(bind, "127.0.0.1" | "::1" | "localhost");
        if token.is_none() && !lokal {
            eprintln!(
                "[peringatan] BIND={bind} tanpa LARAS_TOKEN: siapa pun yang bisa \
                 menjangkau port ini boleh memakai CPU-nya. Set LARAS_TOKEN untuk mengunci."
            );
        }
        Self {
            token,
            laju,
            batas_waktu,
            batas_body,
        }
    }
}

/// Bandingkan dua string tanpa membocorkan isi lewat waktu eksekusi.
fn sama_konstan(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut beda = 0u8;
    for i in 0..a.len() {
        beda |= a[i] ^ b[i];
    }
    beda == 0
}

/// Klien mana yang sedang memanggil: hop pertama `X-Forwarded-For` kalau ada
/// (di belakang reverse proxy semua koneksi datang dari 127.0.0.1), kalau tidak
/// alamat socket-nya.
fn kunci_klien(req: &Request) -> String {
    if let Some(ip) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return ip.to_string();
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "tak-dikenal".to_string())
}

async fn cek_token(State(p): State<Penjaga>, req: Request, next: Next) -> Response {
    let Some(token) = p.token.as_deref() else {
        return next.run(req).await;
    };
    // Healthcheck container dan halaman depan harus tetap terbuka: halaman
    // statis tidak memakai CPU, dan tanpa ini pengunjung cuma melihat 401.
    // Sintesis tetap butuh token.
    if matches!(req.uri().path(), "/" | "/health") {
        return next.run(req).await;
    }
    let dikirim = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if sama_konstan(dikirim, token) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "butuh header Authorization: Bearer <token>\n",
        )
            .into_response()
    }
}

async fn cek_laju(State(p): State<Penjaga>, req: Request, next: Next) -> Response {
    let Some(laju) = p.laju.as_ref() else {
        return next.run(req).await;
    };
    if matches!(req.uri().path(), "/" | "/health") {
        return next.run(req).await;
    }
    match laju.izinkan(&kunci_klien(&req)) {
        Ok(()) => next.run(req).await,
        Err(detik) => (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, detik.to_string())],
            "terlalu banyak permintaan\n",
        )
            .into_response(),
    }
}

/// Batas waktu hanya mencakup sampai header respons siap; body yang mengalir
/// (`/tts/stream`) tidak dipotong di tengah jalan.
async fn batas_waktu(State(p): State<Penjaga>, req: Request, next: Next) -> Response {
    match tokio::time::timeout(p.batas_waktu, next.run(req)).await {
        Ok(r) => r,
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "permintaan melebihi batas waktu\n",
        )
            .into_response(),
    }
}

/// Alasan permintaan tidak mendapat slot. Dibuat kecil dan terpisah dari
/// `Response` supaya `Result`-nya tidak membengkak di setiap titik panggil.
#[derive(Debug)]
struct AntreanPenuh {
    status: StatusCode,
    pesan: &'static str,
    retry_after: Option<u64>,
}

impl AntreanPenuh {
    fn jadi_response(self) -> Response {
        let mut res = (self.status, self.pesan).into_response();
        if let Some(detik) = self.retry_after {
            if let Ok(v) = detik.to_string().parse() {
                res.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        res
    }
}

/// Ambil satu slot sintesis. Kalau antrean penuh sampai `tunggu` habis,
/// jawab 429 supaya klien tahu harus mencoba lagi — bukan menggantung.
async fn ambil_izin(
    antrean: &Arc<Semaphore>,
    tunggu: Duration,
) -> Result<OwnedSemaphorePermit, AntreanPenuh> {
    match tokio::time::timeout(tunggu, Arc::clone(antrean).acquire_owned()).await {
        Ok(Ok(izin)) => Ok(izin),
        Ok(Err(_)) => Err(AntreanPenuh {
            status: StatusCode::SERVICE_UNAVAILABLE,
            pesan: "antrean ditutup\n",
            retry_after: None,
        }),
        Err(_) => Err(AntreanPenuh {
            status: StatusCode::TOO_MANY_REQUESTS,
            pesan: "antrean sintesis penuh, coba lagi\n",
            retry_after: Some(5),
        }),
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_angka<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
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

/// Susunan rute + pengaman. Dipakai `main` dan tes, jadi tes HTTP benar-benar
/// menguji urutan layer yang dipakai di produksi.
fn bangun_router(state: Arc<AppState>, penjaga: Penjaga) -> Router {
    // Token paling luar: permintaan tanpa token tidak menghabiskan jatah laju.
    // `health` dikecualikan di dalam middleware supaya healthcheck tetap jalan.
    Router::new()
        .route("/", get(halaman_demo))
        .route("/health", get(health))
        .route("/normalize", get(normalize_handler))
        .route("/tts", post(tts))
        .route("/tts/stream", post(tts_stream))
        .with_state(state)
        .layer(DefaultBodyLimit::max(penjaga.batas_body))
        .layer(middleware::from_fn_with_state(penjaga.clone(), batas_waktu))
        .layer(middleware::from_fn_with_state(penjaga.clone(), cek_laju))
        .layer(middleware::from_fn_with_state(penjaga, cek_token))
}

/// Halaman depan: satu kotak teks dan tombol putar. Berkasnya di `demo/`,
/// ikut dikompilasi ke binary supaya image tidak perlu menyajikan direktori.
async fn halaman_demo() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../demo/index.html"),
    )
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

    // Slot diambil setelah cache dicek: permintaan yang sudah ada hasilnya
    // tidak perlu ikut mengantre di belakang sintesis yang berat.
    let _izin = match ambil_izin(&st.antrean, st.tunggu_antrean).await {
        Ok(izin) => izin,
        Err(tolak) => return tolak.jadi_response(),
    };

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

/// Header WAV 44 byte untuk aliran.
///
/// Ukuran total belum diketahui saat header dikirim, jadi `RIFF` dan `data`
/// diisi `0xFFFFFFFF`. Pemutar berbasis ffmpeg/mpv berhenti saat koneksi
/// ditutup; browser butuh `?fmt=pcm` karena `<audio>` menolak WAV tanpa
/// panjang pasti.
fn wav_header_stream() -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&u32::MAX.to_le_bytes()); // ukuran berkas: tak diketahui
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes()); // panjang blok fmt
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&24_000u32.to_le_bytes());
    h.extend_from_slice(&48_000u32.to_le_bytes()); // byte/detik
    h.extend_from_slice(&2u16.to_le_bytes()); // byte per frame
    h.extend_from_slice(&16u16.to_le_bytes()); // bit per sampel
    h.extend_from_slice(b"data");
    h.extend_from_slice(&u32::MAX.to_le_bytes()); // ukuran data: tak diketahui
    h
}

/// Sampel float → PCM 16-bit little-endian.
///
/// Tanpa fitur `onnx` backend Python sudah mengirim PCM 16-bit, jadi fungsi
/// ini hanya dipakai jalur native.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

#[derive(Deserialize)]
struct StreamQuery {
    /// `pcm` = PCM mentah 24 kHz mono 16-bit (untuk browser/JS),
    /// selain itu WAV berukuran tak diketahui.
    #[serde(default)]
    fmt: Option<String>,
}

/// `POST /tts/stream` — audio mengalir per kalimat, bukan menunggu utuh.
///
/// Paragraf 4 kalimat yang butuh ±11 s dengan `/tts` mulai berbunyi pada
/// kalimat pertama (±1–2 s). Tiap potongan dinormalkan levelnya sendiri,
/// karena normalisasi gabungan baru bisa dihitung setelah semua potongan ada.
async fn tts_stream(
    State(st): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<StreamQuery>,
    Json(req): Json<TtsReq>,
) -> Response {
    use axum::body::Bytes;
    use tokio_stream::wrappers::ReceiverStream;

    let pcm_saja = q.fmt.as_deref() == Some("pcm");
    let text = st.lang.prepare(&req.text);
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, "teks kosong").into_response();
    }
    let ref_wav = req.voice.unwrap_or_else(|| st.ref_wav.clone());
    let takes_diminta = req.takes.unwrap_or(st.takes).clamp(1, 8);

    #[cfg(feature = "onnx")]
    let potongan: Vec<String> = {
        let kata = text.split_whitespace().count();
        if st.chunk && kata > st.chunk_min_words {
            lang::potong_kalimat(&text)
        } else {
            vec![text.clone()]
        }
    };
    #[cfg(not(feature = "onnx"))]
    let potongan: Vec<String> = vec![text.clone()];

    let takes = if potongan.len() > 1 {
        st.chunk_takes.clamp(1, 8)
    } else {
        takes_diminta
    };
    let n = potongan.len();
    let key = hash_key(&format!("{text}|{ref_wav}|{takes}"));
    let cached = st.cache_dir.join(format!("{key}.wav"));

    let tipe = if pcm_saja {
        "audio/L16;rate=24000;channels=1"
    } else {
        "audio/wav"
    };

    // Hasil utuh sudah ada: tidak ada yang perlu dialirkan.
    if let Ok(bytes) = std::fs::read(&cached) {
        let body = if pcm_saja && bytes.len() > 44 && &bytes[..4] == b"RIFF" {
            bytes[44..].to_vec()
        } else {
            bytes
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, tipe)
            .header(header::CACHE_CONTROL, "no-store")
            .header("x-cache", "hit")
            .header("x-stream", "0")
            .header("x-chunks", n.to_string())
            .body(axum::body::Body::from(body))
            .unwrap();
    }

    // Satu slot untuk seluruh aliran: potongan dikirim berurutan dan modelnya
    // memang cuma satu, jadi menahan slot sampai selesai itu jujur.
    let _izin = match ambil_izin(&st.antrean, st.tunggu_antrean).await {
        Ok(izin) => izin,
        Err(tolak) => return tolak.jadi_response(),
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    let st2 = Arc::clone(&st);
    let key2 = key.clone();
    let ref2 = ref_wav.clone();
    let teks = potongan.clone();
    let n_chunks = n;

    tokio::task::spawn_blocking(move || {
        let kirim = |tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
                     b: Vec<u8>|
         -> bool { tx.blocking_send(Ok(Bytes::from(b))).is_ok() };

        if !pcm_saja && !kirim(&tx, wav_header_stream()) {
            return;
        }

        let mut backend = match st2.backend.lock() {
            Ok(b) => b,
            Err(_) => {
                let _ = tx.blocking_send(Err(std::io::Error::other("backend poison")));
                return;
            }
        };

        let jeda = vec![0u8; 3600 * 2]; // 0,15 s hening antar kalimat
        for (i, k) in teks.iter().enumerate() {
            let tmp = st2.cache_dir.join(format!("{key2}.s{i}.wav"));
            let hasil = backend.synth(&format!("{key2}-{i}"), k, &ref2, takes, &tmp);
            let bytes = match hasil {
                Ok(_) => {
                    #[cfg(feature = "onnx")]
                    {
                        let mut x = match laras::onnx::wav::read_mono(&tmp) {
                            Ok(x) => x,
                            Err(e) => {
                                let _ = std::fs::remove_file(&tmp);
                                let _ = tx.blocking_send(Err(std::io::Error::other(format!(
                                    "baca potongan: {e:#}"
                                ))));
                                return;
                            }
                        };
                        if st2.loudness != 0.0 {
                            laras::level::normalize(&mut x, 24_000.0, st2.loudness, st2.peak_db);
                        }
                        pcm16(&x)
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        // Backend Python sudah menulis wav 24 kHz 16-bit.
                        match std::fs::read(&tmp) {
                            Ok(b) if b.len() > 44 => b[44..].to_vec(),
                            Ok(b) => b,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(std::io::Error::other(format!(
                                    "baca potongan: {e}"
                                ))));
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    let _ = tx.blocking_send(Err(std::io::Error::other(e)));
                    return;
                }
            };
            let _ = std::fs::remove_file(&tmp);
            if i > 0 && !kirim(&tx, jeda.clone()) {
                return;
            }
            if !kirim(&tx, bytes) {
                return; // klien pergi
            }
            eprintln!("[stream] potongan {}/{} terkirim", i + 1, n_chunks);
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, tipe)
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-cache", "miss")
        .header("x-stream", "1")
        .header("x-chunks", n.to_string())
        .header("x-takes", takes.to_string())
        .body(axum::body::Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
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
        if let Ok(v) = std::env::var("FLOW_STEPS") {
            if let Ok(v) = v.parse() {
                params.flow_steps = v;
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

    // Dibaca sebelum server siap supaya peringatan soal BIND/token muncul
    // di log startup, bukan saat permintaan pertama ditolak.
    let bind = env_or("BIND", "127.0.0.1");
    let penjaga = Penjaga::dari_env(&bind);

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
        antrean: Arc::new(Semaphore::new(env_angka("MAX_CONCURRENT", 1usize).max(1))),
        tunggu_antrean: Duration::from_secs(env_angka("QUEUE_WAIT", 120u64)),
        lang: bahasa,
    });

    // Cek referensi lebih awal: lebih baik tahu sekarang daripada baru gagal
    // saat permintaan pertama masuk. Tidak fatal, karena tiap permintaan boleh
    // membawa `voice` sendiri.
    if !std::path::Path::new(&state.ref_wav).exists() {
        eprintln!(
            "peringatan: referensi suara {} tidak ada.\n  \
             ambil contoh CC0: scripts/unduh-model.sh\n  \
             atau set REF_WAV ke rekaman sendiri (WAV/MP3, 10–15 detik).",
            state.ref_wav
        );
    }

    let app = bangun_router(state, penjaga.clone());

    let addr = format!("{bind}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!(
        "laras {} (bahasa {}) listening on http://{addr} (token: {})",
        env!("CARGO_PKG_VERSION"),
        bahasa.code(),
        if penjaga.token.is_some() {
            "wajib"
        } else {
            "tidak dipakai"
        }
    );
    // `ConnectInfo` dipakai pembatas laju untuk tahu klien mana yang memanggil.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_stream_44_byte_dan_sah() {
        let h = wav_header_stream();
        assert_eq!(h.len(), 44);
        assert_eq!(&h[..4], b"RIFF");
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([h[20], h[21]]), 1, "PCM");
        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 1, "mono");
        assert_eq!(u32::from_le_bytes([h[24], h[25], h[26], h[27]]), 24_000);
        assert_eq!(u16::from_le_bytes([h[34], h[35]]), 16, "16-bit");
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes([h[40], h[41], h[42], h[43]]), u32::MAX);
    }

    #[test]
    fn pcm16_dua_byte_per_sampel_dan_dibatasi() {
        assert_eq!(pcm16(&[0.0, 1.0, -1.0, 2.0, -2.0]).len(), 10);
        let b = pcm16(&[1.0]);
        assert_eq!(i16::from_le_bytes([b[0], b[1]]), 32767);
        let b = pcm16(&[-2.0]);
        assert_eq!(i16::from_le_bytes([b[0], b[1]]), -32767);
    }

    #[test]
    fn cache_key_stabil_dan_pendek() {
        let a = hash_key("halo|ref.wav|3");
        assert_eq!(a, hash_key("halo|ref.wav|3"));
        assert_ne!(a, hash_key("halo|ref.wav|1"));
        assert_eq!(a.len(), 24);
    }

    // ------------------------------------------------------------ tes HTTP
    //
    // Rute diuji lewat `Service::oneshot`: tidak ada port yang dibuka, tapi
    // permintaannya melewati axum sungguhan — layer, ekstraktor, dan handler
    // yang sama dengan produksi.

    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn state_uji() -> Arc<AppState> {
        Arc::new(AppState {
            backend: Mutex::new(Backend::Uji),
            cache_dir: std::env::temp_dir().join("laras-uji-cache"),
            ref_wav: "ref.wav".to_string(),
            takes: 1,
            chunk: false,
            chunk_takes: 1,
            chunk_min_words: 20,
            loudness: 0.0,
            peak_db: -1.5,
            antrean: Arc::new(Semaphore::new(1)),
            tunggu_antrean: Duration::from_millis(30),
            lang: lang::get("id").expect("profil bahasa id"),
        })
    }

    fn penjaga_uji(token: Option<&str>, laju: Option<u32>, waktu_ms: u64) -> Penjaga {
        Penjaga {
            token: token.map(str::to_string),
            laju: laju.map(|n| Arc::new(Laju::baru(n))),
            batas_waktu: Duration::from_millis(waktu_ms),
            batas_body: 1024,
        }
    }

    async fn minta(app: Router, req: Request) -> (StatusCode, axum::http::HeaderMap, String) {
        let res = app.oneshot(req).await.expect("layer");
        let status = res.status();
        let headers = res.headers().clone();
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        (
            status,
            headers,
            String::from_utf8_lossy(&bytes).into_owned(),
        )
    }

    fn req_json(uri: &str, body: &str, token: Option<&str>) -> Request {
        let mut b = Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(t) = token {
            b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        b.body(axum::body::Body::from(body.to_string()))
            .expect("request")
    }

    fn req_get(uri: &str, xff: Option<&str>) -> Request {
        let mut b = Request::builder().uri(uri);
        if let Some(ip) = xff {
            b = b.header("x-forwarded-for", ip);
        }
        b.body(axum::body::Body::empty()).expect("request")
    }

    #[tokio::test]
    async fn health_tetap_jalan_tanpa_token() {
        let app = bangun_router(state_uji(), penjaga_uji(Some("rahasia"), None, 5_000));
        let (status, _, body) = minta(app, req_get("/health", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("\"ok\":true"), "{body}");
        assert!(body.contains("\"kind\":\"uji\""), "{body}");
    }

    #[tokio::test]
    async fn halaman_depan_terbuka_tanpa_token() {
        let app = bangun_router(state_uji(), penjaga_uji(Some("rahasia"), Some(1), 5_000));
        let (status, headers, body) = minta(app.clone(), req_get("/", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            headers
                .get(header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap()),
            Some("text/html; charset=utf-8")
        );
        assert!(body.contains("Laras"), "halaman harus punya judul");
        assert!(body.contains("/tts"), "halaman harus menyebut endpoint-nya");

        // Halaman depan juga tidak menghabiskan jatah laju: dengan jatah
        // 1/menit, permintaan berikutnya masih boleh lewat.
        let app = bangun_router(state_uji(), penjaga_uji(None, Some(1), 5_000));
        assert_eq!(
            minta(app.clone(), req_get("/", None)).await.0,
            StatusCode::OK
        );
        assert_eq!(
            minta(app, req_get("/normalize?text=a", None)).await.0,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn tts_tanpa_token_ditolak_401() {
        let app = bangun_router(state_uji(), penjaga_uji(Some("rahasia"), None, 5_000));
        let (status, headers, _) = minta(app, req_json("/tts", r#"{"text":"halo"}"#, None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            headers
                .get(header::WWW_AUTHENTICATE)
                .map(|v| v.to_str().unwrap()),
            Some("Bearer")
        );
    }

    #[tokio::test]
    async fn token_salah_ditolak_token_benar_diteruskan() {
        let app = bangun_router(state_uji(), penjaga_uji(Some("rahasia"), None, 5_000));
        let (status, _, _) = minta(
            app.clone(),
            req_json("/tts", r#"{"text":"halo"}"#, Some("salah")),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        // Lolos penjaga, lalu gagal di backend uji — yang penting bukan 401.
        let (status, _, body) =
            minta(app, req_json("/tts", r#"{"text":"halo"}"#, Some("rahasia"))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
        assert!(body.contains("backend uji"), "{body}");
    }

    #[tokio::test]
    async fn teks_kosong_ditolak_400() {
        let app = bangun_router(state_uji(), penjaga_uji(None, None, 5_000));
        let (status, _, body) = minta(app, req_json("/tts", r#"{"text":"   "}"#, None)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("kosong"), "{body}");
    }

    #[tokio::test]
    async fn body_json_rusak_ditolak_klien() {
        let app = bangun_router(state_uji(), penjaga_uji(None, None, 5_000));
        let (status, _, _) = minta(app, req_json("/tts", r#"{"text":"#, None)).await;
        assert!(status.is_client_error(), "{status}");
    }

    #[tokio::test]
    async fn normalize_lewat_rute_asli() {
        let app = bangun_router(state_uji(), penjaga_uji(None, None, 5_000));
        let (status, _, body) = minta(app, req_get("/normalize?text=Halo%2C%20dunia", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("\"normalized\""), "{body}");
        assert!(body.contains("\"lang\":\"id\""), "{body}");
    }

    #[tokio::test]
    async fn laju_dibatasi_per_klien_dan_ada_retry_after() {
        let app = bangun_router(state_uji(), penjaga_uji(None, Some(1), 5_000));
        assert_eq!(
            minta(app.clone(), req_get("/normalize?text=a", Some("10.0.0.1")))
                .await
                .0,
            StatusCode::OK
        );
        let (status, headers, _) =
            minta(app.clone(), req_get("/normalize?text=b", Some("10.0.0.1"))).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            headers.get(header::RETRY_AFTER).is_some(),
            "butuh Retry-After"
        );
        // Klien lain punya jatahnya sendiri.
        assert_eq!(
            minta(app, req_get("/normalize?text=c", Some("10.0.0.2")))
                .await
                .0,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn body_melebihi_batas_ditolak_413() {
        let app = bangun_router(state_uji(), penjaga_uji(None, None, 5_000));
        let besar = format!(r#"{{"text":"{}"}}"#, "a".repeat(2_000));
        let (status, _, _) = minta(app, req_json("/tts", &besar, None)).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn batas_waktu_memotong_permintaan_lambat() {
        let app = Router::new()
            .route(
                "/lambat",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    "selesai"
                }),
            )
            .layer(middleware::from_fn_with_state(
                penjaga_uji(None, None, 50),
                batas_waktu,
            ));
        let (status, _, _) = minta(app, req_get("/lambat", None)).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    }

    #[tokio::test]
    async fn antrean_penuh_dijawab_429_lalu_pulih() {
        let antrean = Arc::new(Semaphore::new(1));
        let pegang = ambil_izin(&antrean, Duration::from_millis(5))
            .await
            .expect("slot pertama harus dapat");
        let tolak = ambil_izin(&antrean, Duration::from_millis(20)).await;
        let tolak = tolak.expect_err("slot kedua harus ditolak");
        assert_eq!(tolak.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(tolak.retry_after, Some(5));
        assert_eq!(
            tolak.jadi_response().headers().get(header::RETRY_AFTER),
            Some(&"5".parse().unwrap())
        );
        drop(pegang);
        assert!(ambil_izin(&antrean, Duration::from_millis(20))
            .await
            .is_ok());
    }

    #[test]
    fn token_dibandingkan_tanpa_bocor_panjang() {
        assert!(sama_konstan("rahasia", "rahasia"));
        assert!(!sama_konstan("rahasia", "rahasib"));
        assert!(!sama_konstan("rahasia", "rahasiaa"));
        assert!(!sama_konstan("", "rahasia"));
    }
}
