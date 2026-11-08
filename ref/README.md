# Folder referensi suara

Repo ini **tidak menyertakan berkas suara apa pun**. Kamu harus menyediakan
sendiri berkas referensi.

## Cara menyiapkan

```bash
mkdir -p ref
ffmpeg -i rekaman.m4a -ar 24000 -ac 1 ref/suara.wav
```

Syarat yang sudah diuji:

| Parameter | Nilai | Kenapa |
|---|---|---|
| Durasi | **10–15 detik** | >15 s membuat model berhenti dini dan mengarang (WER 0,75–1,0) |
| Kanal | mono | stereo tidak menambah informasi, hanya memperlambat |
| Sample rate | 24–48 kHz | worker akan menyesuaikan |
| Isi | kalimat mengalir, tanpa musik | musik/latar bising ikut "dikloning" |
| Awal berkas | langsung bicara | worker memotong dari sampel bersuara pertama |
| Tempo | santai | kecepatan bicara hasil mengikuti referensi |

## Etika

Gunakan suaramu sendiri, atau suara orang lain **dengan izin**. Lihat
[ETHICS.md](../ETHICS.md).
