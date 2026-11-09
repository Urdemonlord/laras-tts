//! Diagnostik kecepatan whisper.cpp pada audio pendek.
//!
//! Pemakaian:
//!   cargo run --release --features asr --bin asr_bench -- model.bin audio.wav
//!
//! Menjalankan beberapa set parameter pada audio yang sama dan mencetak
//! waktu tempuh + hasil transkrip, supaya jelas mana yang mahal.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use laras::onnx::wav;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("model .bin");
    let audio_path = args.next().expect("audio .wav");

    let mut pcm = wav::read_mono(Path::new(&audio_path))?;
    if pcm.len() > 24_000 * 4 {
        pcm.truncate(24_000 * 4);
    }
    let pcm = wav::resample(&pcm, 24_000, 16_000);
    println!(
        "audio {:.2} s ({} sampel 16 kHz)",
        pcm.len() as f32 / 16_000.0,
        pcm.len()
    );

    let t0 = Instant::now();
    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("muat model: {:.2} s", t0.elapsed().as_secs_f32());

    let mut state = ctx.create_state().map_err(|e| anyhow::anyhow!("{e}"))?;

    // (nama, thread, no_ts, satu_segmen, maks_token, temp_inc, audio_ctx)
    // Audio pendek (±2 s) butuh jauh lebih sedikit jendela encoder daripada
    // 1500 frame bawaan; sapuan ini mencari titik termurah yang masih benar.
    let skenario: Vec<(&str, i32, bool, bool, i32, f32, i32)> = vec![
        ("ctx 768, 1 thread", 1, true, true, 128, 0.0, 768),
        ("ctx 512, 1 thread", 1, true, true, 128, 0.0, 512),
        ("ctx 384, 1 thread", 1, true, true, 128, 0.0, 384),
        ("ctx 320, 1 thread", 1, true, true, 128, 0.0, 320),
        ("ctx 256, 1 thread", 1, true, true, 128, 0.0, 256),
        ("ctx 192, 1 thread", 1, true, true, 128, 0.0, 192),
        ("ctx 768, 2 thread", 2, true, true, 128, 0.0, 768),
        ("ctx 320, 2 thread", 2, true, true, 128, 0.0, 320),
    ];

    for (nama, threads, no_ts, satu, maks, tinc, actx) in skenario {
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(threads);
        p.set_language(Some("id"));
        p.set_translate(false);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        p.set_no_timestamps(no_ts);
        p.set_single_segment(satu);
        p.set_temperature(0.0);
        p.set_temperature_inc(tinc);
        if maks > 0 {
            p.set_max_tokens(maks);
        }
        if actx > 0 {
            p.set_audio_ctx(actx);
        }

        let t = Instant::now();
        state.full(p, &pcm).map_err(|e| anyhow::anyhow!("{e}"))?;
        let dur = t.elapsed().as_secs_f32();

        let mut teks = String::new();
        let n = state.full_n_segments().unwrap_or(0);
        for i in 0..n {
            if let Ok(s) = state.full_get_segment_text(i) {
                teks.push_str(&s);
            }
        }
        println!(
            "{:<34} {:>6.2} s  {} segmen  \"{}\"",
            nama,
            dur,
            n,
            teks.trim()
        );
    }
    Ok(())
}
