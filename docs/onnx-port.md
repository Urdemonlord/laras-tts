# Port ONNX Runtime (native, CPU-only)

Dokumen ini mencatat hasil port inferensi Laras dari worker Python ke Rust +
ONNX Runtime. Tujuannya menghapus worker Python (2,1 GB RAM) tanpa mengubah
kualitas suara.

## Kenapa ONNX

Bundle `anak10thn/pocket-tts-indonesian` sudah menyediakan lima graf siap pakai:

| Graf | Input | Output | Peran |
|---|---|---|---|
| `mimi_encoder.onnx` | `audio [1,1,T]` f32 | `[1,F,1024]` | prompt suara |
| `text_conditioner.onnx` | `token_ids [1,N]` i64 | `[1,N,1024]` | embedding teks |
| `flow_lm_main_int8.onnx` | `sequence`, `text_embeddings`, 18 `state_*` | `conditioning`, `eos_logit`, 18 `out_state_*` | AR backbone |
| `flow_lm_flow_int8.onnx` | `c [1024]`, `s [1,1]`, `t [1,1]`, `x [1,32]` | `[1,32]` | ODE 1 langkah |
| `mimi_decoder_int8.onnx` | `latent [1,1,32]`, 56 `state_*` | audio 24 kHz | vocoder |

## Urutan pipeline

```
wav ──mimi_encoder──► prompt [1,T,1024] ──(prepend bos_before_voice)──┐
                                                                     ▼
teks ──BPE──► ids ──text_conditioner──► [1,N,1024] ──► flow_lm_main pass 1 (suara)
                                                        flow_lm_main pass 2 (teks)
                                                                     │
                          ┌──────────────────────────────────────────┘
                          ▼
      loop: flow_lm_main(frame) ──► cond, eos
            sample noise ──► flow_lm_flow ──► latent
            latent ──► mimi_decoder (1 frame/panggilan) ──► audio
```

## Tiga jebakan yang wajib diingat

### 1. Decoder harus streaming satu frame per panggilan

Graf decoder diekspor untuk inferensi inkremental. Memberinya 35 frame sekaligus
menghasilkan audio rusak:

| Cara panggil | Korelasi vs PyTorch |
|---|---|
| batch 35 frame | 0,378 |
| streaming 1 frame | **0,996** |

### 2. `bos_before_voice` wajib di-prepend

Tanpa token BOS di depan prompt suara, model tidak pernah memutuskan berhenti:

| Kondisi | Frame sampai EOS |
|---|---|
| tanpa BOS | 259 frame (21 s) |
| dengan BOS | **42 frame (3,6 s)** |

### 3. Conditioning teks + suara harus dua pass berurutan

Pass pertama memasukkan prompt suara, pass kedua memasukkan embedding teks.
Menggabungkannya dalam satu tensor membuat model mengabaikan teks.

## Verifikasi numerik

Dibandingkan terhadap implementasi PyTorch acuan (`kyutai/pocket-tts`) dan
terhadap pipeline Python ONNX:

| Tahap | Metrik |
|---|---|
| prompt suara (encoder) | cos 1,000000 |
| embedding teks | identik bit-per-bit |
| conditioning frame-0 | identik bit-per-bit (input sama) |
| output flow | identik bit-per-bit (input sama) |
| decoder, latent sama | corr **0,99999958** |
| tokenizer vs SentencePiece | 16/16 kasus uji identik |

## Catatan penting: sensitivitas graf int8

Graf int8 sangat sensitif terhadap bit terakhir. Pada pengujian, selisih
**4,8 × 10⁻⁷** pada prompt suara (yang muncul karena `ort` memakai build ONNX
Runtime berbeda dari wheel pip `onnxruntime`) membesar menjadi **0,023** pada
conditioning, lalu membuat latent antar-frame berbeda.

Artinya: **paritas bit-per-bit lintas build ONNX Runtime tidak realistis.**
Yang bisa dan sudah dibuktikan adalah paritas fungsional:

* diberikan input tensor yang sama, setiap graf menghasilkan output identik;
* perbedaan yang tersisa berasal dari pemilihan kernel MLAS yang berbeda antar
  build, bukan dari logika port.

Konsekuensi praktis: untuk uji regresi, gunakan `--noise` (deret noise yang sama)
**dan** latent yang sama (`LARAS_LATENTS_IN`) agar tahap yang diuji terisolasi.

## Tokenizer

`tokenizer.model` adalah SentencePiece **BPE** (skor piece = negatif rank),
bukan unigram. Implementasi di `src/onnx/tokenizer.rs`:

1. NFKC + casefold
2. spasi dan `-` menjadi satu spasi, spasi berlebih dibuang
3. awalan `▁`, spasi menjadi `▁`
4. merge pasangan dengan skor tertinggi sampai tidak ada kandidat
5. deret simbol tak dikenal digabung menjadi satu `<unk>`

Protobuf SentencePiece dibaca langsung (parser varint minimal), tanpa dependensi
`protobuf` atau `sentencepiece`.

## Pemakaian

```bash
cargo build --release --features onnx --bin laras_onnx

./target/release/laras_onnx \
  --models ./models \
  --ref suara-referensi.wav \
  --text "Selamat pagi, nama saya Laras." \
  --out hasil.wav
```

Opsi lain: `--threads`, `--seed`, `--temp`, `--eos`, `--noise`, `--dump-tokens`.

## Pemilih take berbasis ASR (fitur `asr`)

`Engine::synthesize_best` menghasilkan N kandidat lalu memilih satu. Tanpa
pendengaran, pemilihan itu nyaris tidak berguna: pada 394 kalimat / 2.361 take,
medoid latent hanya 0,210 vs 0,220 untuk take pertama (CI95 −0,032…+0,012),
dan model linear 22 fitur justru kalah (0,235–0,239). Karena itu pemilihan
dipindah ke ASR sungguhan: whisper.cpp lewat crate `whisper-rs`, di dalam
proses yang sama — tetap satu binary, tetap CPU.

### Biaya: jendela encoder adalah segalanya

whisper.cpp selalu menyandikan jendela 30 detik, sepanjang apa pun audionya.
Untuk audio 2,5 detik itu 12× kerja sia-sia. Hasil ukur (`small` q5_1, audio
2,5 s, 4 vCPU):

| `audio_ctx` | thread | waktu | transkrip |
|---|---|---|---|
| 1500 (bawaan) | 4 | 11,7 s | benar |
| 1500 | 1 | 9,9 s | benar |
| 768 | 4 | 5,4 s | benar |
| **768** | **1** | **4,4 s** | **benar** |
| 512 | 1 | 2,8 s | rusak (`Pasar` → `Lepasat`) |
| 384 | 4 | 2,8 s | rusak |

Dua hal yang berlawanan dengan intuisi:

* **Thread lebih banyak = lebih lambat.** Overhead thread pool ggml melebihi
  manfaat paralelnya pada beban sekecil ini. Default `ASR_THREADS=1`.
* **Model lebih kecil belum tentu cukup.** `base` q5_1 4× lebih cepat
  (0,8–2,7 s) tetapi transkripnya kacau (`pasal tradisional di turun, kita
  bisa tiba-tiba di sini`) — tidak bisa dipakai sebagai juri.

Karena itu `ASR_CTX=768` + `ASR_THREADS=1` jadi titik seimbang: ±4,4 s per
take, akurasi transkrip utuh. Untuk `takes=3` artinya ±13 s tambahan per
permintaan — hanya layak untuk batch, bukan pratonton interaktif.

### Hasil

60 kalimat berpasangan (Tatoeba, temp 0,10, `takes=3`, evaluasi WER memakai
faster-whisper `small` int8 di luar proses):

| Pemilih | WER | bersih |
|---|---|---|
| take pertama | 0,143 | 53% |
| **ASR (`small`, ctx 768)** | **0,108** | **63%** |
| oracle (terbaik dari 3 take) | 0,098 | – |

Selisih berpasangan −0,035 dengan CI95 [−0,070, −0,001]; menang 18% / kalah
3% / seri 78%. Selektor mengambil hampir seluruh celah ke oracle.

Satu kejujuran metodologis: penilai dan pemilih sama-sama keluarga
whisper-small, jadi WER absolutnya optimistis. Yang bisa dipercaya adalah
selisih berpasangannya, karena kedua sisi diukur dengan alat yang sama pada
teks yang sama.

### Dua detik sudah cukup — tapi tidak menghemat waktu

Kesalahan artikulasi terdengar di awal kalimat, jadi mendengar seluruh take
tidak menambah informasi untuk memilih (`ASR_MAX_S=2`, 60 kalimat yang sama):

| Cara dengar | WER pilihan | bersih |
|---|---|---|
| seluruh take | 0,108 | 63% |
| 2 detik pertama | 0,105 | 65% |

Selisih −0,003, CI95 [−0,014, +0,004], 97% kalimat memilih take yang sama
nilainya. Tapi waktunya **tidak** turun: biaya whisper ditentukan jendela
encoder (`audio_ctx`), bukan panjang audio. Memotong audio hanya menghemat
token dekoder — beberapa persen. `ASR_MAX_S` tetap disediakan, tapi bukan
tuas kecepatan.

### Satu lintasan untuk semua take

Yang benar-benar menghemat: take digabung berurutan dengan jeda senyap
(`ASR_GAP`, bawaan 1,5 s), ditranskrip **sekali**, lalu segmen dipetakan
balik ke take berdasarkan waktu mulai. Pada 60 kalimat, take yang dipilih
**identik 100%** dengan cara transkrip terpisah (WER 0,108 keduanya), dengan
waktu per permintaan turun dari ±22 s ke ±14–19 s di 4 vCPU.

Jeda 0,6 s tidak cukup: whisper menggabungkan ketiga take jadi satu segmen.
Kalau pemetaan gagal (segmen tidak terpecah), kode jatuh kembali ke transkrip
terpisah — lambat tapi benar.

`ASR_CTX=512` diuji 27 kalimat: WER 0,117 vs 0,099 pada subset yang sama,
jadi lebih buruk dan hanya menghemat ±1,7 s. Bawaan tetap 768.

### Paragraf panjang dipecah per kalimat

Pada teks satu paragraf, model kadang tidak menemukan titik henti dan
mengulang kata di ujung ("...tidak pernah ter-ter-ter"). Perbaikannya bukan di
pemilih take, tapi di pemotongan teks: `CHUNK=1` (bawaan) memecah masukan di
`.`, `!`, `?`, atau baris baru, menyintesis tiap kalimat, lalu menyambungnya
dengan jeda 150 ms. Kalimat yang lebih dari 28 kata dipecah lagi di koma, dan
potongan di bawah 4 kata digabung ke kalimat sebelumnya.

Paragraf uji yang sama:

| Masukan | Transkrip whisper |
|---|---|
| satu lintasan, satu take | "...seperti biasa, ya percaya bahwa rezeki tidak perc**ator-ter-ter-ter**..." |
| satu lintasan, tiga take + ASR | "...seperti biasa ya percaya barasik **ITT**." |
| **dipecah per kalimat** | "...seperti biasa. Ya percaya bahwa rezeki **tidak pernah tertukar**." |

Biayanya: satu lintasan model per kalimat (bukan per karakter), jadi paragraf
3 kalimat menjadi 3 lintasan pendek — totalnya justru lebih cepat daripada satu
lintasan panjang yang terus memancarkan token sampai batas konteks.

### Aturan otomatis: pendek pakai seleksi, panjang pakai potong

Dua-duanya tidak bisa dipakai bersamaan dengan murah: seleksi ASR per kalimat
membuat paragraf tiga kalimat jadi ±75 s. Jadi pembagiannya otomatis:

| Panjang teks | Yang terjadi | Contoh waktu (4 vCPU) |
|---|---|---|
| ≤ 20 kata (`CHUNK_MIN_WORDS`) | satu lintasan, `TAKES` take, dipilih ASR | 6 kata: ±25 s (permintaan pertama, termasuk pemanasan whisper) |
| > 20 kata | dipecah per kalimat, `CHUNK_TAKES=1` take, tanpa seleksi | 25 kata / 3 kalimat: ±4 s |

Ambang 20 kata dipilih karena pengulangan di ujung baru muncul pada paragraf
(25 kata ke atas), sementara teks pendek lebih diuntungkan seleksi take.
Respons membawa header `x-chunks` dan `x-takes` supaya pemanggil tahu jalur
mana yang dipakai; `CHUNK=0` memaksa satu lintasan untuk semuanya.

### Pemakaian

```bash
ASR_MODEL=./models/ggml-small-q5_1.bin ASR_CTX=768 ASR_THREADS=1 \
  TAKES=3 ./target/release/laras
```

`/health` melaporkan `"asr": true` kalau pemilih aktif. Tanpa `ASR_MODEL`,
pemilihan jatuh ke medoid bawaan (cepat, tetapi lihat angka di atas).

## Yang belum dikerjakan

* Pemilih take masih satu pass: seluruh take ditranskrip penuh. Yang lebih
  murah adalah transkrip sebagian (mis. 2 detik pertama) atau model khusus
  Indonesia yang lebih kecil dari `small` tetapi lebih akurat dari `base`.
* Resampler masih interpolasi linear (mengikuti acuan Python). Resampler
  polifase akan lebih baik secara akustik, tetapi memutus paritas dengan acuan.
* Dukungan MP3/FLAC pada input referensi (saat ini WAV).
