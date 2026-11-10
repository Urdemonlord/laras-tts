---
title: Laras — TTS Bahasa Indonesia
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: TTS Bahasa Indonesia dengan kloning suara, CPU-first
---

# Laras

Engine TTS Bahasa Indonesia yang berjalan **tanpa GPU**. Ketik teks, tekan
tombol, dengar hasilnya.

Kode sumber: <https://github.com/Urdemonlord/laras-tts>

## Yang perlu kamu tahu sebelum menilai suaranya

Ini model kecil yang jujur soal batasnya:

- **Artikulasi gugus konsonan adalah batas model, bukan setelan decoding.**
  "kloning" bisa terdengar "koning", "kereta" jadi "kretah". Menambah langkah
  ODE, menaikkan suhu, atau mengeja ulang tidak menolong — akarnya kosakata
  tokenizer 4.000. Memperbaikinya butuh latih ulang dengan kosakata 8–16 ribu,
  dan itu butuh GPU.
- **Suara mengikuti tempo rekaman referensi.** Referensi di Space ini adalah
  sampel Common Voice Indonesia berlisensi **CC0** (domain publik) — bukan
  rekaman pribadi siapa pun.
- **Teks panjang dipecah per kalimat** supaya tidak berhenti di ~16 detik.
- **Batas laju 6 permintaan/menit** karena CPU Space dipakai bersama. Kalau
  kena `429`, tunggu sebentar — `Retry-After` menyebutkan berapa detik.

## API-nya sama dengan yang bisa kamu jalankan sendiri

```bash
curl -s -X POST https://<space-ini>/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Halo, apa kabar?"}' -o halo.wav
```

`POST /tts/stream` mengirim per kalimat (byte pertama jauh lebih cepat untuk
teks panjang), `GET /health` untuk status, `GET /normalize` untuk melihat hasil
normalisasi teks. Semua tersedia di image publik
`ghcr.io/urdemonlord/laras-tts`.

## Lisensi

Kode: MIT OR Apache-2.0. Bobot model punya lisensinya sendiri — lihat
[NOTICE](https://github.com/Urdemonlord/laras-tts/blob/main/NOTICE).

Repo ini tidak menyimpan rekaman suara siapa pun. Kalau kamu memakai Laras
untuk mengkloning suara, pakai suaramu sendiri atau suara yang kamu punya
izinnya — baca
[ETHICS.md](https://github.com/Urdemonlord/laras-tts/blob/main/ETHICS.md).
