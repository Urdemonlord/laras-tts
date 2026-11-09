<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/laras-banner-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/laras-banner-light.png">
    <img src="assets/laras-banner-dark.png" alt="Laras — Open-source Voice Synthesis Engine" width="760">
  </picture>
</p>

# Laras

**Open-source voice synthesis engine.** Give your words a voice.

CPU-first · Self-hosted · Voice cloning · Indonesian-first

[![CI](https://github.com/Urdemonlord/laras-tts/actions/workflows/ci.yml/badge.svg)](https://github.com/Urdemonlord/laras-tts/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#lisensi)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

Laras mengubah teks Bahasa Indonesia menjadi suara yang menyerupai suara
referensi 10–15 detik — di CPU, tanpa GPU, di mesin yang kamu miliki sendiri.

Bukan model baru. Ini lapisan produksi di atas
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) + bobot
[pocket-tts-indonesian](https://huggingface.co/anak10thn/pocket-tts-indonesian)
yang menambal empat hal yang membuat model mentah belum layak dipakai:

| Masalah model mentah | Yang dilakukan Laras |
|---|---|
| Angka dibaca kacau (`15` → "satu lima", WER 0,71) | normalizer angka Indonesia di Rust (`Rp 12.500` → "dua belas ribu lima ratus rupiah") |
| Berhenti di ~16 detik lalu mengarang | chunking per kalimat + jeda 0,12 s |
| Referensi >15 detik merusak hasil (WER 0,75–1,0) | potong referensi otomatis, ambil dari sampel bersuara pertama |
| Output tidak stabil antar-generate | N take, pilih yang WER-nya terendah (Whisper di dalam proses) |

Hasil terukur pada **394 kalimat** (Tatoeba, temp 0,10, satu take):
**WER 0,19–0,22**, 42–48% kalimat tanpa salah kata, kemiripan suara
**0,80–0,89** (Resemblyzer), tanpa GPU. Angka 0,00–0,08 yang beredar di
versi awal README berasal dari 12 kalimat pilihan — jangan dipakai.

Pemilihan take berbasis ASR memang menolong, walau tidak gratis. Pada 60
kalimat berpasangan (3 take, temp 0,10):

| | WER | kalimat bersih |
|---|---|---|
| take pertama (tanpa seleksi) | 0,143 | 53% |
| **dipilih ASR** | **0,108** | **63%** |
| oracle (batas atas, 3 take) | 0,098 | – |

Selisih berpasangan **−0,035, CI95 [−0,070, −0,001]**; menang 18%, kalah 3%,
seri 78%. Artinya selektor mengambil hampir seluruh keuntungan yang tersedia
(0,108 vs batas 0,098).

Ketiga take ditranskrip dalam **satu lintasan encoder** (digabung dengan jeda
1,5 s lalu dipetakan balik per waktu), dan itu memilih take yang persis sama
dengan cara terpisah — hanya lebih cepat: ±14–19 s per permintaan di 4 vCPU
versus ±22 s. Mendengar 2 detik pertama saja juga cukup akurat (0,105 vs
0,108) tapi tidak menghemat waktu, karena biayanya di jendela encoder.
Ongkos ASR tetap nyata: pakai untuk batch, bukan pratonton.

## Posisi proyek

**Produk: Bahasa Indonesia dulu. Arsitektur: tidak dikunci ke satu bahasa.**

Yang membuat Laras berguna hari ini adalah Bahasa Indonesia: belum ada engine
kloning suara open source yang jalan di CPU dan mendukung bahasa ini dengan
baik. Tapi bagian yang spesifik-bahasa dipisah ke `src/lang/`, di belakang
trait `Lang` (normalisasi + perbaikan fonem). Menambah bahasa = menambah satu
berkas, bukan menulis ulang engine:

```rust
// src/lang/mod.rs
pub trait Lang: Sync + Send {
    fn code(&self) -> &'static str;
    fn normalize(&self, text: &str) -> String;
    fn phoneme_fix(&self, text: &str) -> String { text.to_string() }
}
```

Bahasa dipilih saat runtime: `LARAS_LANG=id` (default).

## Coba cepat

```bash
git clone https://github.com/Urdemonlord/laras-tts && cd laras-tts
mkdir -p data/ref
ffmpeg -i rekaman.m4a -ar 24000 -ac 1 data/ref/suara.wav   # 10-15 detik
docker compose up --build
```

> Bobot `kyutai/pocket-tts` di Hugging Face **gated**: kamu perlu login dan
> menyetujui lisensinya sekali sebelum unduhan pertama berhasil
> (`huggingface-cli login`).

```bash
curl -X POST localhost:8080/tts -H 'content-type: application/json' \
  -d '{"text":"Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."}' \
  -o hasil.wav -D -
# x-cache: miss · x-duration: 5.20 · x-wer: 0.050
```

Contoh lengkap: [`examples/curl.sh`](examples/curl.sh).

## Instalasi manual

Butuh Rust ≥ 1.75, Python ≥ 3.10, ±2,5 GB RAM (model 438 MB + Whisper-small
untuk skoring).

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cargo build --release
REF_WAV=data/ref/suara.wav ./target/release/laras
```

Tanpa `POCKET_CFG`, Laras memakai bobot Indonesia yang sudah dipin ke commit
tertentu, jadi hasilnya reproducible.

## API

### `POST /tts`

```json
{"text": "Harga saham naik 15 persen menjadi Rp 12.500.", "voice": "/path/lain.wav", "takes": 3}
```

`voice` dan `takes` opsional (`takes` 1–5, default 3). Respons `audio/wav`
24 kHz mono PCM16 dengan header:

| Header | Isi |
|---|---|
| `x-cache` | `hit` atau `miss` |
| `x-duration` | durasi audio (detik) |
| `x-wer` | WER take terpilih (0 = sempurna) |

Kunci cache = `sha256(teks siap-pakai | voice | takes)`; request identik
dilayani ~0,9 ms.

### `GET /health`

```json
{"ok":true,"engine":"laras","version":"0.1.0","lang":"id","takes":3,"default_voice":"data/ref/suara.wav"}
```

### `GET /normalize?text=...`

Debug normalizer tanpa memuat model:

```bash
curl -s --get localhost:8080/normalize --data-urlencode 'text=Diskon 25% jadi Rp 89.900'
# {"input":"...","normalized":"Diskon dua puluh lima persen jadi delapan puluh sembilan ribu sembilan ratus rupiah","lang":"id"}
```

## Benchmark

CPU-only **bukan** berarti ringan di semua CPU. Angka di bawah adalah hasil
ukur nyata, bukan estimasi:

| Mesin | vCPU | RAM | RTF | RAM puncak | Catatan |
|---|---|---|---|---|---|
| Xeon 8255C (sandbox) | 4 | 8 GB | **0,26** | 2,1 GB | ±4× realtime |
| Xeon 8255C (VPS kecil) | 2 | 3,7 GB | 0,60–1,09 | 1,6 GB | ±1–1,7× realtime, layak tapi mepet |

Waktu respons HTTP (4 vCPU, `takes=3`):

| Kasus | takes | respons | durasi audio | WER |
|---|---|---|---|---|
| Angka + Rp | 3 | 11,8 s | 5,20 s | 0,05 |
| Angka + Rp, request identik | – | **0,9 ms** | 5,20 s | – |
| 2 kalimat | 3 | 13,4 s | 8,84 s | 0,04 |
| 2 kalimat | 1 | 3,7 s | 8,60 s | 0,12 |
| Jam + desimal + kode ruang | 3 | 11,0 s | 5,96 s | 0,16 |

`takes=1` ±3,6× lebih cepat tapi WER naik 3×. Pakai 1 take untuk pratinjau,
3 take untuk hasil final. Menyalakan `ASR_MODEL` menambah ±13 s (3 take) di
4 vCPU — lihat bagian pemilih take di `docs/onnx-port.md`.

## Konfigurasi

| Variabel | Default | Arti |
|---|---|---|
| `BIND` | `127.0.0.1` | alamat bind (`0.0.0.0` di Docker) |
| `PORT` | `8080` | port HTTP |
| `LARAS_LANG` | `id` | profil bahasa (`src/lang/`) |
| `REF_WAV` | `data/ref/suara.wav` | referensi suara default |
| `TAKES` | `3` | jumlah take untuk best-of-N |
| `CHUNK` | `1` | pecah teks per kalimat sebelum sintesis; `0` = satu lintasan |
| `CHUNK_MIN_WORDS` | `20` | teks ≤ sekian kata tidak dipecah (tetap best-of-N + ASR) |
| `CHUNK_TAKES` | `1` | take per kalimat saat teks dipecah |
| `LOUDNESS` | `-18` | target loudness LUFS keluaran; `0` = jangan dinormalkan |
| `PEAK_DB` | `-1.5` | batas puncak dBFS setelah normalisasi |
| `MODELS_DIR` | `data/onnx` | direktori graf ONNX (backend native) |
| `ASR_MODEL` | _(kosong)_ | path `ggml-*.bin`; kalau diisi, take dipilih dengan WER ASR |
| `ASR_CTX` | `0` (penuh 1500) | jendela encoder whisper; 768 ±2,2× lebih cepat |
| `ASR_THREADS` | `1` | thread whisper; di 4 vCPU 1 thread justru tercepat |
| `USE_ASR` | `1` | `0` = matikan skoring WER di backend `worker.py` |
| `CACHE_DIR` | `data/cache` | cache WAV |
| `WORKER_PY` | `python3` | interpreter worker |
| `WORKER_SCRIPT` | `worker.py` | path worker |
| `POCKET_CFG` | bobot ID (pin commit) | konfigurasi model |
| `EOS` | `-3.0` | ambang EOS model |
| `REF_MAX_S` | `15` | batas potong referensi (detik) |
| `HF_HOME` | `data/hf` | cache bobot model |

## Level audio

Keluaran mentah mesin ini sekitar **−28 LUFS** dengan puncak −10,7 dBFS: harus
dikeraskan sendiri di pemutar. Gateway menormalkannya sendiri, tanpa ffmpeg:

1. Loudness diukur dengan **ITU-R BS.1770** (K-weighting + blok 400 ms + gerbang
   absolut −70 LUFS dan relatif −10 LU), dihitung di `src/level.rs`. Filter
   diturunkan ulang dari parameter aslinya (`K = tan(π f₀ / fs)`) sehingga
   berlaku di 24 kHz, bukan hanya 48 kHz; pada 48 kHz koefisiennya sama persis
   dengan tabel spesifikasi (ada tesnya).
2. Gain dinaikkan ke target `LOUDNESS`, lalu puncak dibatasi dengan limiter
   lookahead 5 ms dan pelepasan 50 ms ke `PEAK_DB`.
3. Karena pembatasan menurunkan loudness, langkah 2 diulang maksimal tiga kali
   sampai selisih ke target < 0,3 dB.

Hasil pada empat kalimat uji: **−18,0 sampai −18,2 LUFS**, puncak −1,5 dBFS,
gain +9,8 sampai +11,3 dB. Tanpa normalisasi, selisih antar kalimat mencapai
1 dB dan semuanya terlalu pelan.

Untuk memeriksa satu berkas: `cargo run --release --bin lufs -- hasil.wav -18 -1.5`.

## Arsitektur

```
klien ──HTTP──> laras (Rust/axum)
                  │ 1. profil bahasa: normalisasi + perbaikan fonem
                  │ 2. pecah per kalimat (CHUNK), cache sha256(teks|voice|takes)
                  │ 3. serialisasi ke satu worker
                  ▼
               worker.py (Python, model tetap di RAM)
                  │ 4. chunking per kalimat
                  │ 5. generate N take
                  │ 6. pilih WER terendah
                  ▼
               WAV 24 kHz mono
```

**Kenapa Rust + Python?** Inferensi masih butuh PyTorch/Pocket TTS yang hanya
tersedia di Python. Yang di Rust adalah bagian yang menentukan pengalaman API:
parsing, normalisasi, cache, orkestrasi. Kontrak HTTP-nya stabil, jadi engine
bisa ditukar tanpa mengubah klien — dan itu memang rencananya: ONNX Runtime di
Rust, lalu Python hilang dari dependency.

**Kenapa satu worker?** Model 438 MB + state suara; banyak worker di mesin
kecil bikin OOM. Antrean lebih baik daripada crash. Untuk throughput, jalankan
beberapa instance di belakang nginx.

## Batasan yang sudah terukur

- **Referensi >15 detik merusak output** (WER 0,75–1,0) — dipotong otomatis.
- **Teks panjang harus di-chunk**; tanpa itu model berhenti di ~16 detik.
- **Digit wajib dinormalisasi**; tanpa normalizer WER 0,71, dengan normalizer 0,00.
- Gugus konsonan `kl-` kadang hilang ("kloning" → "koning") — batas model 6 layer.
- Kecepatan bicara mengikuti tempo referensi.
- Belum ada streaming; audio dikirim setelah selesai.
- Normalizer baru Bahasa Indonesia (dan belum menangani semua format tanggal).

## Roadmap

| Prioritas | Pekerjaan |
|---|---|
| P0 | Publish repo + benchmark hardware yang jelas ✅ |
| P1 | ONNX Runtime di Rust — inferensi native ✅, penyambungan ke gateway ⏳ |
| P1 | Streaming chunked response (kirim per kalimat) |
| P1 | Kamus fonem (`kl-`, `-ngg-`) berbasis uji dengar + WER |
| P2 | HF Spaces demo (referensi suara sintetis, bukan suara pribadi) |
| P2 | Pool worker + `SO_REUSEPORT` |
| P2 | Resampler polifase (kualitas akustik lebih baik) |
| P3 | SDK Rust / Python / JavaScript |
| P3 | Watermark audio (AudioSeal) sebagai opsi |

## Etika

Laras bisa mengkloning suara siapa pun. **Baca [ETHICS.md](ETHICS.md).**
Ringkasnya: pakai suaramu sendiri atau suara yang kamu punya izinnya; jangan
meniru figur publik; jangan untuk penipuan. Repo ini tidak menyertakan berkas
suara siapa pun dan CI menolak commit berkas audio.

## Lisensi

Kode: **MIT OR Apache-2.0**.

Bobot model punya lisensinya sendiri dan **tidak** tercakup lisensi kode.
Rincian lengkap per lapisan ada di [NOTICE](NOTICE). Yang paling penting:

| Lapisan | Lisensi |
|---|---|
| Kode Laras | MIT OR Apache-2.0 |
| Kode dependency | MIT / Apache-2.0 / ISC / BSD-3 |
| Kode Pocket TTS (Kyutai) | MIT |
| **Bobot Pocket TTS** | **CC-BY-4.0** (gated di HF) |
| **Bobot Indonesia (anak10thn)** | **CC-BY-4.0** |
| Dataset & suara referensi | milik masing-masing pemilik |

CC-BY-4.0 mengizinkan penggunaan komersial, **tapi** mewajibkan atribusi,
penyertaan tautan lisensi, dan penanda perubahan. Kalau kamu memakai bobot
Indonesia di produk, cantumkan: *"Pocket TTS Indonesian oleh anak10thn,
CC-BY-4.0"* beserta tautan ke halaman modelnya.

## Kredit

- Kyutai — [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (kode MIT, bobot CC-BY-4.0)
- anak10thn — bobot Bahasa Indonesia (CC-BY-4.0)
- OpenAI — Whisper small (MIT), untuk skoring WER
- SYSTRAN — faster-whisper (MIT)
