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
