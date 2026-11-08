# Kontribusi

Terima kasih. Aturan singkatnya:

## Sebelum mengirim PR

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all
ruff check worker.py norm_id.py   # opsional, untuk perubahan Python
```

CI menjalankan hal yang sama. PR yang membuat CI merah tidak di-merge.

## Aturan yang tidak bisa ditawar

1. **Jangan pernah commit berkas suara** (`*.wav`, `*.mp3`, `*.m4a`, `*.ogg`).
   Ini berlaku untuk rekaman referensi maupun hasil sintesis. Pakai
   `ref/README.md` untuk menjelaskan, bukan berkasnya.
2. **Jangan menambah dependensi tanpa cek lisensi.** Yang diterima: MIT,
   Apache-2.0, BSD, ISC. Yang perlu diskusi dulu: GPL/AGPL (karena menyebar
   ke seluruh binary), lisensi non-komersial, atau lisensi tak jelas.
3. **Perubahan perilaku normalizer wajib disertai unit test** di
   `src/main.rs` (`mod tests`). Ini bagian yang paling mudah rusak diam-diam.
4. **Jangan menurunkan kualitas demi kecepatan tanpa angka.** Sertakan
   perbandingan WER/waktu sebelum-sesudah.

## Gaya commit

`<area>: <ringkasan>` — mis. `normalizer: tangani format 09.30`,
`worker: cache state per referensi`, `docs: tambah batasan 15 detik`.

## Menambah dukungan bahasa

Semua yang spesifik-bahasa ada di `src/lang/`, di belakang trait `Lang`.
Untuk bahasa baru:

1. Buat `src/lang/xx.rs`, implementasikan `Lang` (minimal `code()` dan
   `normalize()`; `phoneme_fix()` opsional).
2. Daftarkan di `src/lang/mod.rs` (`static ALL`).
3. Tambah test: angka dasar, ribuan, desimal, mata uang, persen, dan format
   yang mudah salah (jam, versi, tanggal).
4. Pilih saat runtime dengan `LARAS_LANG=xx`.

Model perlu mendukung bahasa tersebut; normalizer hanya menyiapkan teks.
