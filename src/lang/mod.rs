//! Profil bahasa.
//!
//! Satu bahasa = satu implementasi [`Lang`]. Semua bagian yang spesifik
//! bahasa (normalisasi angka, perbaikan fonem) hidup di modul ini, bukan di
//! inti gateway. Menambah bahasa baru berarti menambah satu berkas di sini,
//! tanpa menyentuh `main.rs`.
//!
//! Ini yang membuat Laras bisa tumbuh dari "TTS Bahasa Indonesia" menjadi
//! engine multi-bahasa tanpa menulis ulang arsitekturnya.

pub mod id;

/// Profil satu bahasa.
pub trait Lang: Sync + Send {
    /// Kode ISO-639-1, mis. `"id"`.
    fn code(&self) -> &'static str;

    /// Normalisasi teks sebelum masuk model: semua digit dan simbol yang
    /// tidak bisa dibaca model diubah menjadi kata.
    fn normalize(&self, text: &str) -> String;

    /// Perbaikan fonem setelah normalisasi. Default: tidak mengubah apa pun.
    /// Dipakai untuk menambal kata yang sering salah ucap di model kecil.
    fn phoneme_fix(&self, text: &str) -> String {
        text.to_string()
    }

    /// Pipeline lengkap: normalisasi lalu perbaikan fonem.
    fn prepare(&self, text: &str) -> String {
        self.phoneme_fix(&self.normalize(text))
    }
}

/// Batas kata per potongan sintesis.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
const MIN_KATA: usize = 4;
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
const MAX_KATA: usize = 28;

/// Pecah teks menjadi potongan yang aman disintesis sekali jalan.
///
/// Model berhenti dengan baik pada kalimat pendek; pada paragraf panjang ia
/// kadang tidak menemukan titik henti lalu mengulang kata di ujung
/// ("...tidak pernah ter-ter-ter"). Memecah per kalimat menghilangkan itu,
/// dan kalimat yang masih terlalu panjang dipecah lagi di koma.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub fn potong_kalimat(teks: &str) -> Vec<String> {
    let mut kasar: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in teks.chars() {
        if ch == '\n' {
            buf.push(' ');
            continue;
        }
        buf.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\u{2026}') {
            kasar.push(std::mem::take(&mut buf));
        }
    }
    kasar.push(buf);

    let mut out: Vec<String> = Vec::new();
    for k in kasar {
        let k = k.trim();
        if k.is_empty() {
            continue;
        }
        for bagian in pecah_panjang(k) {
            let pendek = kata(&bagian) < MIN_KATA;
            let muat = out
                .last()
                .map(|l: &String| kata(l) + kata(&bagian) <= MAX_KATA)
                .unwrap_or(false);
            if pendek && muat {
                let last = out.last_mut().expect("sudah dicek");
                last.push(' ');
                last.push_str(&bagian);
            } else {
                out.push(bagian);
            }
        }
    }
    out
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn kata(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Potong kalimat yang kepanjangan, utamakan di koma.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn pecah_panjang(s: &str) -> Vec<String> {
    if kata(s) <= MAX_KATA {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut n = 0;
    for w in s.split_whitespace() {
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(w);
        n += 1;
        let koma = w.ends_with(',') && n >= MIN_KATA;
        if n >= MAX_KATA || koma {
            out.push(std::mem::take(&mut buf));
            n = 0;
        }
    }
    if !buf.trim().is_empty() {
        out.push(buf);
    }
    out
}

static INDONESIAN: id::Indonesian = id::Indonesian;

/// Semua bahasa yang tersedia. Tambahkan di sini saat menambah modul.
static ALL: [&dyn Lang; 1] = [&INDONESIAN];

pub fn all() -> &'static [&'static dyn Lang] {
    &ALL
}

/// Cari profil berdasarkan kode. `None` kalau belum didukung.
pub fn get(code: &str) -> Option<&'static dyn Lang> {
    all().iter().copied().find(|l| l.code() == code)
}

/// Kode semua bahasa yang tersedia, untuk pesan error.
pub fn codes() -> Vec<&'static str> {
    all().iter().map(|l| l.code()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry() {
        assert!(get("id").is_some());
        assert!(get("xx").is_none());
        assert_eq!(codes(), vec!["id"]);
    }

    #[test]
    fn potong_per_kalimat() {
        let t = "Pagi itu hujan turun. Di ujung gang, pedagang membuka warung. Ia percaya rezeki tidak tertukar.";
        let p = potong_kalimat(t);
        assert_eq!(p.len(), 3, "{p:?}");
        assert!(p[0].ends_with('.'));
    }

    #[test]
    fn kalimat_pendek_tidak_dipotong() {
        assert_eq!(potong_kalimat("Selamat pagi.").len(), 1);
    }

    #[test]
    fn kalimat_kepanjangan_dipecah() {
        let t = "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh sebelas dua belas tiga belas empat belas lima belas enam belas tujuh belas delapan belas sembilan belas dua puluh dua puluh satu dua puluh dua dua puluh tiga dua puluh empat dua puluh lima dua puluh enam dua puluh tujuh dua puluh delapan dua puluh sembilan tiga puluh.";
        let p = potong_kalimat(t);
        assert!(p.len() >= 2, "{p:?}");
        assert!(p
            .iter()
            .all(|x| x.split_whitespace().count() <= MAX_KATA + MIN_KATA));
    }

    #[test]
    fn prepare_menjalankan_normalisasi() {
        let l = get("id").unwrap();
        assert_eq!(l.prepare("naik 15 persen"), "naik lima belas persen");
    }
}
