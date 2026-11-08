# tts-gateway

[![CI](https://github.com/Urdemonlord/tts-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Urdemonlord/tts-gateway/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#lisensi)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

TTS Bahasa Indonesia yang **jalan di CPU**, **bisa kloning suara sendiri**, dan
**bisa di-self-host**. Satu endpoint HTTP, satu berkas WAV keluar.

Ini bukan model baru. Ini lapisan produksi di atas
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) + bobot
[pocket-tts-indonesian](https://huggingface.co/anak10thn/pocket-tts-indonesian)
yang menambal empat masalah yang bikin model mentah tidak layak dipakai:

| Masalah model mentah | Yang dilakukan gateway |
|---|---|
| Angka dibaca kacau (`15` → "satu lima", WER 0,71) | normalizer angka Indonesia di Rust (`Rp 12.500` → "dua belas ribu lima ratus rupiah") |
| Berhenti di ~16 detik lalu mengarang | chunking per kalimat + jeda 0,12 s |
| Referensi >15 detik merusak hasil (WER 0,75–1,0) | potong referensi otomatis, ambil dari sampel bersuara pertama |
| Output tidak stabil antar-generate | N take, pilih yang WER-nya terendah (Whisper-small) |

Hasil: WER 0,00–0,08 pada teks normal, kemiripan suara 0,80–0,89 (Resemblyzer),
tanpa GPU sama sekali.

## Coba cepat

```bash
git clone https://github.com/Urdemonlord/tts-gateway && cd tts-gateway
mkdir -p data/ref
# taruh rekaman 10-15 detik di sini (lihat ref/README.md)
ffmpeg -i rekaman.m4a -ar 24000 -ac 1 data/ref/suara.wav
docker compose up --build
```

```bash
curl -X POST localhost:8080/tts -H 'content-type: application/json' \
  -d '{"text":"Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."}' \
  -o hasil.wav -D -
# x-cache: miss · x-duration: 5.20 · x-wer: 0.050
```

Contoh lengkap: [`examples/curl.sh`](examples/curl.sh).

## Instalasi manual

Butuh Rust ≥ 1.75, Python ≥ 3.10, ±2,5 GB RAM (worker memuat model 438 MB +
Whisper-small untuk skoring).

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cargo build --release
REF_WAV=data/ref/suara.wav \
POCKET_CFG="hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5" \
WORKER_PY=venv/bin/python \
./target/release/tts-gateway
```

## API

### `POST /tts`

```json
{"text": "Harga saham naik 15 persen menjadi Rp 12.500.", "voice": "/path/lain.wav", "takes": 3}
```

`voice` dan `takes` opsional. Respons `audio/wav` 24 kHz mono PCM16 dengan header:

| Header | Isi |
|---|---|
| `x-cache` | `hit` atau `miss` |
| `x-duration` | durasi audio (detik) |
| `x-wer` | WER take terpilih (0 = sempurna) |

Kunci cache = `sha256(teks ternormalisasi | voice | takes)`, jadi request identik
dilayani ~0,9 ms.

### `GET /health`

```json
{"ok": true, "pid": 1234, "default_voice": "data/ref/suara.wav", "takes": 3, "worker": {"ready": true}}
```

### `GET /normalize?text=...`

Debug normalizer tanpa memuat model:

```bash
curl -s --get localhost:8080/normalize \
  --data-urlencode 'text=Rapat pukul 09.30, naik 7,5 persen'
# {"input":"...","normalized":"Rapat pukul nol sembilan titik tiga nol, naik tujuh koma lima persen"}
```

## Hasil ukur

4 vCPU, tanpa GPU, `takes=3` kecuali disebut lain:

| Kasus | takes | waktu respons | durasi audio | WER |
|---|---|---|---|---|
| Angka + Rp | 3 | 11,8 s | 5,20 s | 0,05 |
| Angka + Rp, request identik | – | **0,9 ms** | 5,20 s | – |
| 2 kalimat | 3 | 13,4 s | 8,84 s | 0,04 |
| 2 kalimat | 1 | 3,7 s | 8,60 s | 0,12 |
| Jam + desimal + kode ruang | 3 | 11,0 s | 5,96 s | 0,16 |

`takes=1` ±3,6× lebih cepat tapi WER naik 3×. Kompromi yang wajar untuk
pratinjau; pakai 3 take untuk hasil final.

## Konfigurasi

| Variabel | Default | Arti |
|---|---|---|
| `BIND` | `127.0.0.1` | alamat bind (`0.0.0.0` di Docker) |
| `PORT` | `8080` | port HTTP |
| `REF_WAV` | `data/ref/suara.wav` | referensi suara default |
| `TAKES` | `3` | jumlah take untuk best-of-N |
| `USE_ASR` | `1` | `0` = matikan skoring WER (1 take, lebih cepat) |
| `CACHE_DIR` | `data/cache` | cache WAV |
| `WORKER_PY` | `python3` | interpreter worker |
| `WORKER_SCRIPT` | `worker.py` | path worker |
| `EOS` | `-3.0` | ambang EOS model |
| `REF_MAX_S` | `15` | batas potong referensi (detik) |
| `HF_HOME` | `data/hf` | cache bobot model |

## Arsitektur

```
klien ──HTTP──> tts-gateway (Rust/axum)
                  │ 1. normalisasi angka
                  │ 2. cache sha256(teks|voice|takes)
                  │ 3. serialisasi ke satu worker
                  ▼
               worker.py (Python, model tetap di RAM)
                  │ 4. chunking per kalimat
                  │ 5. generate N take
                  │ 6. pilih WER terendah
                  ▼
               WAV 24 kHz mono
```

**Kenapa Rust + Python, bukan satu bahasa?** Inferensi masih butuh
PyTorch/Pocket TTS yang hanya tersedia di Python. Yang dipindah ke Rust adalah
bagian yang menentukan pengalaman API: parsing, normalisasi, cache, dan
orchestrasi. Kontrak HTTP-nya stabil, jadi engine bisa ditukar nanti tanpa
mengubah klien. Port penuh ke Rust (candle/`ort` + ONNX INT8 yang sudah ada di
HF) adalah roadmap berikutnya — itu juga akan menurunkan RAM ke ±1 GB.

**Kenapa satu worker?** Model 438 MB + state suara; menjalankan banyak worker
di mesin kecil bikin OOM. Serialisasi antrean lebih baik daripada crash.
Untuk throughput, jalankan beberapa instance di belakang nginx.

## Batasan yang sudah terukur

- **Referensi >15 detik merusak output** (WER 0,75–1,0) — worker memotong otomatis.
- **Teks panjang harus di-chunk**; tanpa itu model berhenti di ~16 detik.
- **Digit wajib dinormalisasi**; tanpa normalizer WER 0,71, dengan normalizer 0,00.
- Gugus konsonan `kl-` kadang hilang ("kloning" → "koning") — batas model 6 layer.
- Kecepatan bicara mengikuti tempo referensi.
- Belum ada streaming; respons dikirim setelah audio selesai.
- Normalizer baru mendukung Bahasa Indonesia.

## Roadmap

- [ ] Port inferensi ke Rust murni (`ort` + ONNX INT8) → hapus dependensi Python
- [ ] Streaming chunked response (mulai kirim setelah kalimat pertama)
- [ ] Kamus koreksi fonem (`kl-`, `-ngg-`, dll.)
- [ ] Pool worker + `SO_REUSEPORT`
- [ ] Watermark audio (AudioSeal) sebagai opsi
- [ ] Dukungan bahasa lain lewat modul normalizer terpisah

## Etika

Proyek ini bisa mengkloning suara siapa pun. **Baca [ETHICS.md](ETHICS.md).**
Ringkasnya: pakai suaramu sendiri atau suara yang kamu punya izinnya; jangan
meniru figur publik, jangan untuk penipuan. Repo ini tidak menyertakan berkas
suara siapa pun dan CI menolak commit berkas audio.

## Lisensi

Kode: **MIT OR Apache-2.0** (pilih salah satu).

Bobot model **tidak** termasuk dalam lisensi ini:

- [pocket-tts](https://github.com/kyutai-labs/pocket-tts) — MIT (Kyutai)
- [pocket-tts-indonesian](https://huggingface.co/anak10thn/pocket-tts-indonesian) — **CC-BY-4.0**

Kalau kamu memakai bobot Indonesia di produk, kamu **wajib** mencantumkan
atribusi ke `anak10thn` beserta tautannya. Detail lengkap ada di [NOTICE](NOTICE).

## Kredit

- Kyutai — [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- anak10thn — bobot Bahasa Indonesia (dilatih dari dataset LEMAS)
- OpenAI — Whisper small (untuk skoring WER)
- SYSTRAN — faster-whisper
