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
    fn prepare_menjalankan_normalisasi() {
        let l = get("id").unwrap();
        assert_eq!(l.prepare("naik 15 persen"), "naik lima belas persen");
    }
}
