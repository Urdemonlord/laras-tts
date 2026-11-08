# Kebijakan penggunaan & etika

Proyek ini adalah kloning suara. Teknologinya netral, penyalahgunaannya tidak.
Aturan di bawah adalah syarat penggunaan, bukan sekadar imbauan.

## Yang diizinkan

- Suaramu sendiri.
- Suara orang lain **dengan izin tertulis** dari pemiliknya (untuk proyek
  pribadi, konten, atau produk).
- Suara sintetis/karakter yang tidak mengklaim identitas orang nyata.

## Yang tidak diizinkan

- Meniru pejabat publik, politisi, atau figur publik untuk pernyataan yang
  tidak pernah mereka ucapkan.
- Penipuan: meniru suara anggota keluarga/rekan kerja untuk meminta uang,
  kata sandi, atau akses.
- Konten seksual non-konsensual, pelecehan, atau ujaran kebencian.
- Melepas rekaman suara orang tanpa izinnya sebagai berkas referensi.

## Praktik teknis di repo ini

- **Tidak ada berkas suara siapa pun yang di-commit.** `ref/*.wav` masuk
  `.gitignore`; repo hanya menyediakan `ref/README.md`.
- Contoh keluaran (`out/`, `cache/`) juga diabaikan.
- Tidak ada filter konten bawaan. Tanggung jawab ada pada pengguna dan
  operator layanan.

## Kalau kamu men-deploy sebagai layanan

- Minta pernyataan izin sebelum menerima berkas referensi.
- Simpan rekaman referensi terenkripsi dan hapus sesuai permintaan pemiliknya.
- Tandai keluaran sebagai sintetis. Minimal: metadata `comment` pada berkas
  audio, atau label di UI. Untuk kebutuhan lebih kuat, tambahkan watermark
  audio (mis. AudioSeal).
- Sediakan kanal pelaporan penyalahgunaan.

## Pelaporan

Penyalahgunaan yang melibatkan proyek ini dapat dilaporkan lewat issue
bertanda `abuse` atau surel ke pemelihara. Laporan yang valid akan
ditindaklanjuti dengan penghapusan contoh, dokumentasi peringatan, atau
pemblokiran kontributor.
