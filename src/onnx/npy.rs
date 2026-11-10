//! Pembaca `.npy` minimal (cukup untuk `bos_before_voice.npy`).
//!
//! Format: magic `\x93NUMPY`, versi, panjang header, lalu dict Python berisi
//! `descr`, `fortran_order`, dan `shape`.

use std::path::Path;

use anyhow::{anyhow, Result};

/// Array float32 hasil pembacaan.
pub struct NpyF32 {
    /// Dimensi array.
    pub shape: Vec<usize>,
    /// Data row-major.
    pub data: Vec<f32>,
}

/// Baca berkas `.npy` bertipe float32 little-endian.
pub fn read_f32(path: &Path) -> Result<NpyF32> {
    let buf = std::fs::read(path).map_err(|e| anyhow!("gagal membaca {}: {e}", path.display()))?;
    parse_f32(&buf)
}

/// Parse isi `.npy` dari memori.
pub fn parse_f32(buf: &[u8]) -> Result<NpyF32> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("bukan berkas .npy"));
    }
    let major = buf[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize)
    } else {
        let n = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        (n, 12usize)
    };

    let header_end = header_start
        .checked_add(header_len)
        .filter(|e| *e <= buf.len())
        .ok_or_else(|| anyhow!("header .npy terpotong"))?;
    let header = String::from_utf8_lossy(&buf[header_start..header_end]).into_owned();

    let descr = extract(&header, "'descr': '", "'")
        .ok_or_else(|| anyhow!("descr tidak ditemukan di header"))?;
    if descr != "<f4" && descr != "|f4" {
        return Err(anyhow!("dtype {descr} belum didukung (butuh float32)"));
    }

    let shape_raw = extract(&header, "'shape': (", ")")
        .ok_or_else(|| anyhow!("shape tidak ditemukan di header"))?;
    let mut shape = Vec::new();
    for part in shape_raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(
            p.parse::<usize>()
                .map_err(|_| anyhow!("dimensi tidak valid: {p}"))?,
        );
    }

    let count: usize = shape.iter().product();
    let body = &buf[header_end..];
    if body.len() < count * 4 {
        return Err(anyhow!(
            "data .npy kurang: butuh {} byte, ada {}",
            count * 4,
            body.len()
        ));
    }
    let data = body[..count * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    Ok(NpyF32 { shape, data })
}

fn extract(hay: &str, start: &str, end: &str) -> Option<String> {
    let i = hay.find(start)? + start.len();
    let rest = &hay[i..];
    let j = rest.find(end)?;
    Some(rest[..j].to_string())
}

#[cfg(test)]
mod tests {
    use super::parse_f32;

    #[test]
    fn bulatkan_header_v1() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }";
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY\x01\x00");
        buf.extend_from_slice(&(header.len() as u16).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&2.5f32.to_le_bytes());

        let a = parse_f32(&buf).expect("harus terbaca");
        assert_eq!(a.shape, vec![1, 2]);
        assert_eq!(a.data, vec![1.5, 2.5]);
    }
}
