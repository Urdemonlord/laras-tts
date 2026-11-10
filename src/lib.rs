//! Laras — pustaka inti.
//!
//! Modul `onnx` menyediakan mesin inferensi CPU-only berbasis ONNX Runtime.
//! Modul ini hanya tersedia bila crate dibangun dengan fitur `onnx`.
//!
//! Modul `native` adalah jalur tingkat tinggi yang dipakai bersama oleh gateway
//! HTTP (`src/main.rs`) dan aplikasi desktop (`desktop/`): muat model, pilih
//! voice, hasilkan WAV. Tidak ada logika sintesis yang diduplikasi di antara
//! keduanya.

pub mod lang;
pub mod level;

#[cfg(feature = "onnx")]
pub mod native;
#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "asr")]
pub mod asr;
