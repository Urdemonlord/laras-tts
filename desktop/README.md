# Laras desktop

Muka tipis di atas pustaka `laras`. Tidak ada server HTTP, tidak ada port, dan
tidak ada proses Python: mesin ONNX dimuat di dalam proses aplikasi, lalu setiap
permintaan sintesis memanggil `laras::native` — jalur yang sama persis dengan
`POST /tts` di gateway. Hasilnya identik, hanya tanpa jaringan.

```
desktop/
├── ui/                 # antarmuka (HTML + CSS + JS, tanpa bundler/npm)
│   ├── index.html
│   ├── app.js
│   └── gaya.css
└── src-tauri/          # aplikasi Rust (Tauri v2)
    ├── Cargo.toml
    ├── build.rs
    ├── tauri.conf.json
    ├── icons/
    └── src/main.rs     # perintah: status, siapkan, sintesis, pilih_*, simpan_wav
```

## Kenapa Tauri, bukan Electron

Tauri memakai webview yang sudah ada di sistem (WebKitGTK di Linux, WebView2 di
Windows, WKWebView di macOS) dan memanggil Rust langsung. Electron membawa
Chromium sendiri: ±150 MB sebelum satu baris kode aplikasi ditulis, dan mesin
Laras harus dijalankan sebagai proses terpisah di belakang port lokal.

Konsekuensinya: **aplikasi ini butuh webview sistem**. Di Linux artinya paket
`webkit2gtk-4.1` (GNOME/KDE modern sudah punya).

## Prasyarat

| Kebutuhan | Linux | Catatan |
|---|---|---|
| Rust ≥ 1.92 | `rustup` | |
| WebKitGTK 4.1 | `libwebkit2gtk-4.1-dev` | runtime: `libwebkit2gtk-4.1-0` |
| GTK 3 | `libgtk-3-dev` | |
| librsvg | `librsvg2-dev` | ikon bundel |
| `cmake` | `cmake` | dipakai `onnxruntime` saat build |
| `ffmpeg` | `ffmpeg` | konversi MP3/m4a → WAV |
| Model | `./scripts/unduh-model.sh` | ±160 MB, sekali saja |

Debian/Ubuntu:

```bash
sudo apt install -y libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev \
  libayatana-appindicator3-dev libsoup-3.0-dev cmake ffmpeg
```

## Bangun

```bash
cd desktop/src-tauri
cargo build --release      # binary: target/release/laras-desktop
```

Untuk paket `.deb` / AppImage (butuh CLI Tauri sekali pasang):

```bash
cargo install tauri-cli --version "^2" --locked
cd desktop/src-tauri && cargo tauri build
```

Rilis resmi dibuat otomatis oleh `.github/workflows/desktop-release.yml` saat tag `desktop-v*` di-push: Linux (.deb, AppImage, tarball), macOS (tarball arm64 + x86_64), Windows (zip).

## Model dan referensi suara

Aplikasi mencari, berurutan:

1. `MODELS_DIR` / `REF_WAV` dari lingkungan,
2. `~/.local/share/laras/models` dan `~/.local/share/laras/ref/suara.mp3`,
3. `data/models` dan `data/ref/suara.mp3` di samping binary (atau di samping bundle `.app` di macOS),
4. `data/models` dan `data/ref/suara.mp3` relatif ke direktori kerja.

Referensi suara dicari sebagai `suara.wav` dulu, baru `suara.mp3`, di tiap lokasi di atas; WAV dibaca langsung tanpa ffmpeg, jadi paket desktop menyertakan keduanya.

Kalau tidak ketemu, panel **Pengaturan** terbuka dan kamu bisa menunjuk folder
sendiri. Tidak ada unduhan tersembunyi di dalam aplikasi — model diunduh lewat
skrip yang bisa kamu baca (`scripts/unduh-model.sh`, memverifikasi sha256).

## Memeriksa tampilan tanpa membangun binary

Tata letak bisa diperiksa tanpa mengompilasi Tauri: buka `ui/index.html` di
peramban WebKit dengan `window.__TAURI__` tiruan yang mengembalikan bentuk data
perintah sungguhan. Yang terverifikasi dengan cara ini adalah HTML/CSS — bukan
IPC Tauri-nya.

Di mesin tanpa layar, WebDriver WebKit bisa menjepret halamannya langsung
(tangkapan diambil dari mesin render, bukan dari X — jadi tidak terkena masalah
komposisi GL):

```bash
sudo apt install -y webkit2gtk-driver
Xvfb :77 -screen 0 900x900x24 &
DISPLAY=:77 WebKitWebDriver --port=4444 &
# POST /session, POST /session/{id}/url, lalu GET /session/{id}/screenshot
```

Catatan dari pemeriksaan pertama: `.hasil { display: flex }` mengalahkan atribut
`hidden`, sehingga kartu hasil tampil sebelum ada audio. Karena itu ada aturan
`[hidden] { display: none !important; }` di `gaya.css`.

## Batasan yang sama dengan gateway

Aplikasi ini tidak memperbaiki model. Artikulasi gugus konsonan tetap lemah
karena kosakata tokenizer 4.000 kata, tempo tetap mengikuti rekaman referensi,
dan teks panjang tetap dipecah per kalimat. Yang berbeda hanya kemasannya.

Satu take per permintaan: di desktop orang menunggu di depan layar, jadi waktu
tunggu lebih penting daripada variasi. Memilih di antara beberapa take butuh
penilai ASR (fitur `asr`), yang tidak dibundel di sini.
