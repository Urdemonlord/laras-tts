#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Laras desktop — muka tipis di atas pustaka `laras`.
//!
//! Tidak ada server HTTP dan tidak ada port: mesin ONNX dimuat di dalam proses
//! ini, lalu setiap permintaan sintesis memanggil `laras::native` yang sama
//! persis dengan yang dipakai gateway HTTP. Karena itu hasilnya identik dengan
//! `POST /tts`, hanya tanpa jaringan.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use laras::native::Native;
use laras::onnx::Params;

/// Satu take: di aplikasi desktop orang menunggu di depan layar, jadi waktu
/// tunggu lebih penting daripada variasi. Memilih di antara beberapa take butuh
/// penilai ASR (fitur `asr`), yang tidak dibundel di sini.
const TAKES: usize = 1;

const VERSI: &str = env!("CARGO_PKG_VERSION");

// ------------------------------------------------------------------ keadaan

struct Inti {
    /// `None` selama model belum dimuat; diisi sekali, dipakai berulang.
    mesin: Mutex<Option<Native>>,
    models: Mutex<PathBuf>,
    referensi: Mutex<PathBuf>,
    memuat: AtomicBool,
}

impl Inti {
    fn baru() -> Self {
        let models = cari_models();
        let referensi = cari_referensi();
        Self {
            mesin: Mutex::new(None),
            models: Mutex::new(models),
            referensi: Mutex::new(referensi),
            memuat: AtomicBool::new(false),
        }
    }
}

/// Urutan pencarian model: env, lalu folder data pengguna, lalu folder
/// `data/` di samping binary/bundle, lalu folder repo (memudahkan
/// `cargo run` saat mengembangkan).
///
/// Kandidat dari lokasi binary: paket (zip/tarball/.deb/.app) menaruh
/// `data/` di samping binary atau di samping bundle `.app`. Naik sampai
/// tiga tingkat supaya `.app/Contents/MacOS/laras-desktop` menemukan
/// `data/` di samping `Laras.app`.
fn calon_dari_exe(rel: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut d = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(PathBuf::from));
    for _ in 0..4 {
        let Some(p) = d else { break };
        out.push(p.join("data").join(rel));
        d = p.parent().map(PathBuf::from);
    }
    out
}

fn cari_models() -> PathBuf {
    let mut calon: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("MODELS_DIR") {
        calon.push(PathBuf::from(v));
    }
    if let Some(d) = dirs_data() {
        calon.push(d.join("models"));
    }
    calon.extend(calon_dari_exe("models"));
    if let Ok(d) = std::env::current_dir() {
        calon.push(d.join("data/models"));
        calon.push(d.join("../data/models"));
    }
    calon
        .iter()
        .find(|p| p.join("config.json").exists() || p.is_dir())
        .cloned()
        .unwrap_or_else(|| PathBuf::from("data/models"))
}

fn cari_referensi() -> PathBuf {
    let mut calon: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("REF_WAV") {
        calon.push(PathBuf::from(v));
    }
    if let Some(d) = dirs_data() {
        calon.push(d.join("ref/suara.mp3"));
    }
    calon.extend(calon_dari_exe("ref/suara.mp3"));
    if let Ok(d) = std::env::current_dir() {
        calon.push(d.join("data/ref/suara.mp3"));
        calon.push(d.join("../data/ref/suara.mp3"));
    }
    calon
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .unwrap_or_else(|| PathBuf::from("data/ref/suara.mp3"))
}

/// `~/.local/share/laras` di Linux (mengikuti XDG_DATA_HOME).
fn dirs_data() -> Option<PathBuf> {
    if let Ok(x) = std::env::var("XDG_DATA_HOME") {
        return Some(PathBuf::from(x).join("laras"));
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".local/share/laras"))
}

// -------------------------------------------------------------------- data

#[derive(Serialize, Clone)]
struct Status {
    versi: String,
    siap: bool,
    memuat: bool,
    models: String,
    models_ada: bool,
    referensi: String,
    referensi_ada: bool,
    inti_cpu: usize,
}

#[derive(Serialize)]
struct Hasil {
    berkas: String,
    durasi: f64,
    detik: f32,
    take: usize,
    /// Skor diagnostik dari mesin. Ini **bukan** WER: tanpa fitur `asr` nilainya
    /// jarak medoid di ruang latent, bukan ukuran salah-dengar. Tidak ditampilkan
    /// ke pengguna, hanya berguna saat menelusuri masalah.
    skor: f32,
    byte: u64,
}

fn status_dari(inti: &Inti) -> Status {
    let models = inti.models.lock().map(|p| p.clone()).unwrap_or_default();
    let referensi = inti.referensi.lock().map(|p| p.clone()).unwrap_or_default();
    let siap = inti.mesin.lock().map(|m| m.is_some()).unwrap_or(false);
    Status {
        versi: VERSI.to_string(),
        siap,
        memuat: inti.memuat.load(Ordering::Relaxed),
        models_ada: models.join("config.json").is_file() || models.is_dir(),
        models: models.display().to_string(),
        referensi_ada: referensi.is_file(),
        referensi: referensi.display().to_string(),
        inti_cpu: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

// ---------------------------------------------------------------- perintah

#[tauri::command]
fn status(inti: State<'_, Arc<Inti>>) -> Status {
    status_dari(&inti)
}

/// Muat model ke memori. Berat (ratusan MB, beberapa detik), jadi dijalankan
/// di thread terpisah supaya jendela tetap responsif.
#[tauri::command]
async fn siapkan(
    inti: State<'_, Arc<Inti>>,
    models: Option<String>,
    referensi: Option<String>,
) -> Result<Status, String> {
    let inti = inti.inner().clone();
    // Salinan kedua untuk tugas latar: `inti` yang asli tetap dipakai untuk
    // menyusun Status setelah model selesai dimuat.
    let untuk_muat = inti.clone();

    if let Some(m) = models.as_ref().filter(|s| !s.trim().is_empty()) {
        *inti.models.lock().map_err(|_| "kunci model rusak")? = PathBuf::from(m);
    }
    if let Some(r) = referensi.as_ref().filter(|s| !s.trim().is_empty()) {
        *inti.referensi.lock().map_err(|_| "kunci referensi rusak")? = PathBuf::from(r);
    }

    if inti.memuat.swap(true, Ordering::SeqCst) {
        return Err("model sedang dimuat".to_string());
    }

    let hasil = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let inti = untuk_muat;
        let models = inti.models.lock().map_err(|_| "kunci model rusak")?.clone();
        let referensi = inti
            .referensi
            .lock()
            .map_err(|_| "kunci referensi rusak")?
            .clone();

        if !models.is_dir() {
            return Err(format!(
                "folder model tidak ditemukan: {}\nJalankan ./scripts/unduh-model.sh dari repo, \
                 atau pilih folder lain.",
                models.display()
            ));
        }
        if !referensi.is_file() {
            return Err(format!(
                "berkas referensi tidak ditemukan: {}\nPilih rekaman 10–15 detik (WAV/MP3/m4a).",
                referensi.display()
            ));
        }

        let inti_cpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let mesin = Native::new(models, inti_cpu, Params::default(), "id")
            .map_err(|e| format!("gagal memuat model: {e:#}"))?;

        // Suara referensi diuji sekarang, bukan saat pengguna menekan tombol:
        // ffmpeg yang gagal lebih baik ketahuan di sini.
        let referensi_str = referensi.to_string_lossy().to_string();
        let mut mesin = mesin;
        mesin
            .use_voice(&referensi_str)
            .map_err(|e| format!("gagal membaca referensi: {e:#}"))?;

        *inti.mesin.lock().map_err(|_| "kunci mesin rusak")? = Some(mesin);
        Ok(())
    })
    .await;

    inti.memuat.store(false, Ordering::SeqCst);

    match hasil {
        Ok(Ok(())) => Ok(status_dari(&inti)),
        Ok(Err(pesan)) => Err(pesan),
        Err(e) => Err(format!("tugas latar gagal: {e}")),
    }
}

#[tauri::command]
async fn sintesis(
    app: AppHandle,
    inti: State<'_, Arc<Inti>>,
    teks: String,
) -> Result<Hasil, String> {
    let teks = teks.trim().to_string();
    if teks.is_empty() {
        return Err("teks masih kosong".to_string());
    }
    if teks.chars().count() > 5_000 {
        return Err("teks terlalu panjang (maksimal 5.000 karakter)".to_string());
    }

    let inti = inti.inner().clone();
    let cache = app
        .path()
        .app_cache_dir()
        .map_err(|e| format!("folder cache tidak tersedia: {e}"))?;
    std::fs::create_dir_all(&cache).map_err(|e| format!("gagal membuat cache: {e}"))?;

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let keluar = cache.join(format!("laras-{stamp}.wav"));

    tauri::async_runtime::spawn_blocking(move || -> Result<Hasil, String> {
        let referensi = inti
            .referensi
            .lock()
            .map_err(|_| "kunci referensi rusak")?
            .to_string_lossy()
            .to_string();

        let mut kunci = inti.mesin.lock().map_err(|_| "kunci mesin rusak")?;
        let mesin = kunci
            .as_mut()
            .ok_or_else(|| "mesin belum dimuat — tekan “Muat mesin” dulu".to_string())?;

        let mulai = Instant::now();
        let (durasi, take, skor) = mesin
            .synth_to(&teks, &referensi, TAKES, &keluar)
            .map_err(|e| format!("sintesis gagal: {e:#}"))?;
        let detik = mulai.elapsed().as_secs_f32();
        let byte = std::fs::metadata(&keluar).map(|m| m.len()).unwrap_or(0);

        Ok(Hasil {
            berkas: keluar.to_string_lossy().to_string(),
            durasi,
            detik,
            take,
            skor,
            byte,
        })
    })
    .await
    .map_err(|e| format!("tugas latar gagal: {e}"))?
}

/// Dialog pemilih berkas. Dijalankan sinkron: dialog harus hidup di thread
/// yang sama dengan loop peristiwa, dan perintah sinkron Tauri tidak memblokir
/// antarmuka.
#[tauri::command]
fn pilih_suara(app: AppHandle) -> Option<String> {
    let (kirim, terima) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .set_title("Pilih rekaman suara referensi (10–15 detik)")
        .add_filter("Audio", &["wav", "mp3", "m4a", "flac", "ogg", "opus"])
        .pick_file(move |jalur| {
            let _ = kirim.send(jalur);
        });
    terima
        .recv()
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn pilih_folder(app: AppHandle) -> Option<String> {
    let (kirim, terima) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .set_title("Pilih folder model (berisi config.json)")
        .pick_folder(move |jalur| {
            let _ = kirim.send(jalur);
        });
    terima
        .recv()
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn simpan_wav(app: AppHandle, sumber: String) -> Result<Option<String>, String> {
    let asal = PathBuf::from(&sumber);
    if !asal.is_file() {
        return Err("berkas hasil tidak ditemukan".to_string());
    }
    let (kirim, terima) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .set_title("Simpan audio")
        .set_file_name("laras.wav")
        .add_filter("WAV", &["wav"])
        .save_file(move |jalur| {
            let _ = kirim.send(jalur);
        });

    let tujuan = terima
        .recv()
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok());
    match tujuan {
        Some(t) => {
            std::fs::copy(&asal, &t).map_err(|e| format!("gagal menyimpan: {e}"))?;
            Ok(Some(t.display().to_string()))
        }
        None => Ok(None),
    }
}

#[tauri::command]
fn buka_folder(jalur: String) {
    let target = Path::new(&jalur);
    let dir = if target.is_dir() {
        target.to_path_buf()
    } else {
        target.parent().map(|p| p.to_path_buf()).unwrap_or_default()
    };
    // Tanpa crate tambahan: xdg-open ada di semua desktop Linux.
    let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
}

// ------------------------------------------------------------- mode kepala

/// Mode tanpa jendela: `laras-desktop --sintesis "teks" --keluar out.wav`.
///
/// Dipakai untuk menguji jalur sintesis di binary ini tanpa X server (mis. di
/// VPS tanpa layar) dan untuk skrip. Mengembalikan kode keluar bila mode ini
/// yang diminta, `None` bila aplikasi harus jalan normal.
fn mode_kepala() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    if !args.iter().any(|a| a == "--sintesis") {
        return None;
    }
    let ambil = |nama: &str| -> Option<String> {
        args.iter()
            .position(|a| a == nama)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let teks = match ambil("--sintesis") {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            eprintln!("pakai: laras-desktop --sintesis \"teks\" --keluar out.wav");
            return Some(2);
        }
    };
    let keluar = ambil("--keluar").unwrap_or_else(|| "laras.wav".to_string());
    let models = ambil("--models")
        .map(PathBuf::from)
        .unwrap_or_else(cari_models);
    let referensi = ambil("--suara")
        .map(PathBuf::from)
        .unwrap_or_else(cari_referensi);

    let inti_cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);

    eprintln!("model:     {}", models.display());
    eprintln!("referensi: {}", referensi.display());
    eprintln!("inti cpu:  {inti_cpu}");

    let mulai = Instant::now();
    let mut mesin = match Native::new(models, inti_cpu, Params::default(), "id") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("galat memuat model: {e:#}");
            return Some(1);
        }
    };
    let muat = mulai.elapsed().as_secs_f32();
    eprintln!("model dimuat dalam {muat:.2} s");

    let t = Instant::now();
    match mesin.synth_to(
        &teks,
        &referensi.to_string_lossy(),
        TAKES,
        Path::new(&keluar),
    ) {
        Ok((durasi, take, skor)) => {
            eprintln!(
                "ok: {durasi:.2} s audio dalam {:.2} s (take {}, skor {skor:.4}) → {keluar}",
                t.elapsed().as_secs_f32(),
                take + 1
            );
            Some(0)
        }
        Err(e) => {
            eprintln!("galat sintesis: {e:#}");
            Some(1)
        }
    }
}

// -------------------------------------------------------------------- main

fn main() {
    if let Some(kode) = mode_kepala() {
        std::process::exit(kode);
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(Inti::baru()))
        .invoke_handler(tauri::generate_handler![
            status,
            siapkan,
            sintesis,
            pilih_suara,
            pilih_folder,
            simpan_wav,
            buka_folder
        ])
        .run(tauri::generate_context!())
        .expect("gagal menjalankan aplikasi");
}
