//! Laras — pustaka inti.
//!
//! Modul `onnx` menyediakan mesin inferensi CPU-only berbasis ONNX Runtime.
//! Modul ini hanya tersedia bila crate dibangun dengan fitur `onnx`.

pub mod level;

#[cfg(feature = "onnx")]
pub mod onnx;
