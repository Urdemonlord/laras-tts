//! CLI uji mesin ONNX Laras.
//!
//! ```bash
//! laras_onnx --models ./models --ref suara.wav --text "Selamat pagi." --out out.wav
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use laras::onnx::wav;
use laras::onnx::{Engine, Params, SAMPLE_RATE};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        usage();
        return Ok(());
    }

    let models =
        PathBuf::from(arg(&args, "--models").ok_or_else(|| anyhow!("--models wajib diisi"))?);
    let text = arg(&args, "--text").ok_or_else(|| anyhow!("--text wajib diisi"))?;
    let threads = arg(&args, "--threads")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let mut eng = Engine::new(&models, threads)?;

    // Mode debug tokenizer: cetak id tanpa menjalankan model.
    if args.iter().any(|a| a == "--dump-tokens") {
        let (prepared, _) = laras::onnx::prepare_text(&text);
        let ids = eng.tokenizer().encode(&prepared);
        println!("teks siap  : {prepared:?}");
        println!("jumlah     : {}", ids.len());
        println!("ids        : {ids:?}");
        println!("decode     : {:?}", eng.tokenizer().decode(&ids));
        return Ok(());
    }

    let ref_path = PathBuf::from(arg(&args, "--ref").ok_or_else(|| anyhow!("--ref wajib diisi"))?);
    let out_path =
        PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "laras-out.wav".to_string()));

    let mut params = Params::default();
    if let Some(v) = arg(&args, "--seed") {
        params.seed = v.parse().map_err(|_| anyhow!("--seed bukan angka"))?;
    }
    if let Some(v) = arg(&args, "--temp") {
        params.temperature = v.parse().map_err(|_| anyhow!("--temp bukan angka"))?;
    }
    if let Some(v) = arg(&args, "--eos") {
        params.eos_threshold = v.parse().map_err(|_| anyhow!("--eos bukan angka"))?;
    }
    if let Some(v) = arg(&args, "--flow-steps") {
        params.flow_steps = v.parse().map_err(|_| anyhow!("--flow-steps bukan angka"))?;
    }

    // Noise eksternal untuk uji kesetaraan numerik terhadap implementasi Python.
    let noise: Option<Vec<f32>> = match arg(&args, "--noise") {
        Some(p) => Some(read_f32(&PathBuf::from(p))?),
        None => None,
    };

    let t0 = Instant::now();
    let samples = wav::read_mono(&ref_path)?;
    let dur_ref = samples.len() as f32 / SAMPLE_RATE as f32;
    eng.set_voice(&samples)?;
    let t_voice = t0.elapsed().as_secs_f32();

    let takes = arg(&args, "--takes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    let dump_dir = arg(&args, "--dump-dir").map(PathBuf::from);

    let t0 = Instant::now();
    let best = eng.synthesize_best(&text, &params, takes, noise.as_deref())?;
    let t_synth = t0.elapsed().as_secs_f32();
    let idx = best.index;
    let audio = best.audio().to_vec();
    let diag = best.diag().clone();
    let all: Vec<_> = best.diags().cloned().collect();

    // Simpan semua kandidat bila diminta (untuk riset/uji dengar).
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir)?;
        for (k, (_, a, lat)) in best.cands.iter().enumerate() {
            wav::write_mono(&dir.join(format!("take{k}.wav")), a)?;
            let mut raw = Vec::with_capacity(lat.len() * 4);
            for v in lat {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join(format!("take{k}.f32")), raw)?;
        }
    }

    wav::write_mono(&out_path, &audio)?;

    let dur = audio.len() as f32 / SAMPLE_RATE as f32;
    let peak = audio.iter().fold(0f32, |m, v| m.max(v.abs()));
    let words = text.split_whitespace().count();
    println!(
        "referensi  : {} ({dur_ref:.2}s, {} frame)",
        ref_path.display(),
        eng.voice_frames()
    );
    if takes > 1 {
        println!(
            "kandidat   : {takes} take, dipilih #{idx} (skor {:+.2})",
            diag.score(words)
        );
        println!("  take  frame  eos   conf    dur   rms    skor");
        for (k, d) in all.iter().enumerate() {
            let mark = if k == idx { "*" } else { " " };
            println!(
                "{mark} {k:4}  {:5}  {:4}  {:+6.2}  {:5.2}s  {:.3}  {:+6.2}",
                d.frames,
                d.eos_frame
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                d.eos_conf,
                d.dur,
                d.rms,
                d.score(words)
            );
        }
    }
    println!(
        "suara      : {dur:.2}s, peak {peak:.2}, {} sampel",
        audio.len()
    );
    println!(
        "waktu      : voice {t_voice:.2}s | sintesis {t_synth:.2}s | RTF {:.3}",
        t_synth / dur.max(0.01)
    );
    println!("keluaran   : {}", out_path.display());
    Ok(())
}

fn arg(args: &[String], key: &str) -> Option<String> {
    let i = args.iter().position(|a| a == key)?;
    args.get(i + 1).cloned()
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let raw = std::fs::read(path)?;
    if raw.len() % 4 != 0 {
        return Err(anyhow!("berkas noise harus kelipatan 4 byte"));
    }
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn usage() {
    println!(
        "laras_onnx — mesin TTS Laras (ONNX Runtime, CPU-only)\n\n\
         PEMAKAIAN:\n  \
         laras_onnx --models DIR --ref suara.wav --text \"...\" --out out.wav\n\n\
         OPSI:\n  \
         --models DIR     direktori berisi 5 graf ONNX + tokenizer.model\n  \
         --ref FILE       WAV suara referensi (mono/stereo, laju apa pun)\n  \
         --text STRING    teks yang akan diucapkan\n  \
         --out FILE       berkas WAV keluaran (default laras-out.wav)\n  \
         --threads N      jumlah thread intra-op (default: jumlah CPU)\n  \
         --seed N         seed RNG (default 1234)\n  \
         --temp F         temperatur sampling (default 0.3)\n  \
         --eos F          ambang logit EOS (default -4.0)\n  \
         --takes N        jumlah kandidat; dipilih otomatis tanpa ASR\n  \
         --dump-dir DIR   simpan semua kandidat sebagai take0.wav, take1.wav, ...\n  \
         --noise FILE     deret noise f32 mentah untuk uji determinisme\n  \
         --dump-tokens    cetak token saja, tanpa sintesis\n"
    );
}
