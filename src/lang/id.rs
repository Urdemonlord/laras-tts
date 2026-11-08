//! Profil Bahasa Indonesia.
//!
//! Fokus: mengubah semua digit dan simbol mata uang/persen menjadi kata yang
//! bisa dibaca model. Tanpa ini, model membaca "15" sebagai "satu lima" dan
//! WER melonjak dari 0,00 ke 0,71 (terukur).

use std::sync::OnceLock;

use regex::Regex;

use super::Lang;

pub struct Indonesian;

impl Lang for Indonesian {
    fn code(&self) -> &'static str {
        "id"
    }

    fn normalize(&self, text: &str) -> String {
        normalize(text)
    }

    fn phoneme_fix(&self, text: &str) -> String {
        phoneme_fix(text)
    }
}

// ---------------------------------------------------------------- angka

const SATUAN: [&str; 12] = [
    "", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh",
    "sebelas",
];

// `is_multiple_of()` baru stabil di Rust 1.87; sengaja pakai `%` agar MSRV tetap 1.75.
#[allow(clippy::manual_is_multiple_of)]
fn ratus(n: u64) -> String {
    if n < 12 {
        return SATUAN[n as usize].to_string();
    }
    if n < 20 {
        return format!("{} belas", ratus(n - 10));
    }
    if n < 100 {
        let s = format!("{} puluh", ratus(n / 10));
        return if n % 10 != 0 {
            format!("{} {}", s, ratus(n % 10))
        } else {
            s
        };
    }
    let s = if n < 200 {
        "seratus".to_string()
    } else {
        format!("{} ratus", ratus(n / 100))
    };
    if n % 100 != 0 {
        format!("{} {}", s, ratus(n % 100))
    } else {
        s
    }
}

pub fn terbilang(n: u64) -> String {
    if n == 0 {
        return "nol".to_string();
    }
    let mut n = n;
    let mut parts: Vec<String> = Vec::new();
    let scales: [(u64, &str); 5] = [
        (1_000_000_000_000, "triliun"),
        (1_000_000_000, "miliar"),
        (1_000_000, "juta"),
        (1_000, "ribu"),
        (1, ""),
    ];
    for (nilai, nama) in scales {
        if n >= nilai {
            let q = n / nilai;
            n %= nilai;
            if nilai == 1000 && q == 1 {
                parts.push("seribu".to_string());
            } else if nama.is_empty() {
                parts.push(ratus(q));
            } else {
                parts.push(format!("{} {}", ratus(q), nama));
            }
        }
    }
    parts.join(" ")
}

fn re_rp() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\bRp\.?\s*([0-9](?:[0-9.,]*[0-9])?)").unwrap())
}

fn re_persen() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"([0-9](?:[0-9.,]*[0-9])?)\s*%").unwrap())
}

fn re_angka() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[0-9](?:[0-9.,]*[0-9])?").unwrap())
}

fn re_ribuan() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[0-9]{1,3}(\.[0-9]{3})+$").unwrap())
}

fn re_spasi() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}

/// "12.500" -> "dua belas ribu lima ratus", "32,5" -> "tiga puluh dua koma lima"
fn angka(tok: &str) -> String {
    let tok = tok.trim().trim_end_matches('.');
    if let Some((bulat, pecah)) = tok.split_once(',') {
        let b: String = bulat.replace('.', "");
        let b = if b.is_empty() { "0".to_string() } else { b };
        let mut out = format!("{} koma", terbilang(b.parse().unwrap_or(0)));
        for c in pecah.chars().filter(|c| c.is_ascii_digit()) {
            out.push(' ');
            out.push_str(&terbilang(c.to_digit(10).unwrap() as u64));
        }
        return out;
    }
    let clean = if re_ribuan().is_match(tok) {
        tok.replace('.', "")
    } else {
        tok.to_string()
    };
    if clean.parse::<u64>().is_err() {
        // bukan bilangan utuh (mis. "09.30", "4.2.1") -> baca digit per digit
        let mut out: Vec<String> = Vec::new();
        for c in tok.chars() {
            match c {
                '0'..='9' => out.push(terbilang(c.to_digit(10).unwrap() as u64)),
                '.' => out.push("titik".to_string()),
                ',' => out.push("koma".to_string()),
                _ => {}
            }
        }
        return out.join(" ");
    }
    terbilang(clean.parse().unwrap_or(0))
}

/// Normalisasi teks sebelum masuk model: semua digit jadi kata.
pub fn normalize(text: &str) -> String {
    let s = re_rp()
        .replace_all(text, |c: &regex::Captures| {
            format!("{} rupiah", angka(&c[1]))
        })
        .to_string();
    let s = re_persen()
        .replace_all(&s, |c: &regex::Captures| format!("{} persen", angka(&c[1])))
        .to_string();
    let s = re_angka()
        .replace_all(&s, |c: &regex::Captures| angka(&c[0]))
        .to_string();
    re_spasi().replace_all(s.trim(), " ").to_string()
}

// ------------------------------------------------------------- fonem

/// Kata yang sering salah ucap model 6-layer, dipetakan ke ejaan yang lebih
/// mudah diucapkan. Sengaja kosong dulu: setiap entri harus lewat uji dengar
/// dan pengukuran WER, bukan tebakan.
///
/// Contoh bentuk entri: `("kloning", "keloning")`.
const FONEM: &[(&str, &str)] = &[];

fn phoneme_fix(text: &str) -> String {
    if FONEM.is_empty() {
        return text.to_string();
    }
    let mut out = text.to_string();
    for (dari, ke) in FONEM {
        out = out.replace(dari, ke);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terbilang_dasar() {
        assert_eq!(terbilang(0), "nol");
        assert_eq!(terbilang(11), "sebelas");
        assert_eq!(terbilang(15), "lima belas");
        assert_eq!(terbilang(100), "seratus");
        assert_eq!(terbilang(1000), "seribu");
        assert_eq!(terbilang(12500), "dua belas ribu lima ratus");
        assert_eq!(terbilang(2026), "dua ribu dua puluh enam");
        assert_eq!(terbilang(1_250_000), "satu juta dua ratus lima puluh ribu");
    }

    #[test]
    fn normalisasi_kalimat() {
        assert_eq!(
            normalize("Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."),
            "Harga saham naik lima belas persen menjadi dua belas ribu lima ratus rupiah per lembar pada dua ribu dua puluh enam."
        );
        assert_eq!(
            normalize("Suhu 32,5 derajat"),
            "Suhu tiga puluh dua koma lima derajat"
        );
        assert_eq!(normalize("Biaya Rp 5.000.000"), "Biaya lima juta rupiah");
        assert_eq!(
            normalize("tahun 1998"),
            "tahun seribu sembilan ratus sembilan puluh delapan"
        );
        assert_eq!(normalize("tanpa angka"), "tanpa angka");
        assert_eq!(normalize("   spasi   berlebih  "), "spasi berlebih");
        assert_eq!(
            normalize("pukul 09.30"),
            "pukul nol sembilan titik tiga nol"
        );
        assert_eq!(normalize("versi 4.2.1"), "versi empat titik dua titik satu");
    }

    #[test]
    fn phoneme_kosong_tidak_mengubah() {
        assert_eq!(phoneme_fix("kloning suara"), "kloning suara");
    }
}
