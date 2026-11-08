#!/usr/bin/env python3
"""Worker TTS untuk tts-gateway.

Protokol: satu JSON per baris di stdin, satu JSON per baris di stdout.
  -> {"id": "...", "text": "sudah dinormalisasi", "ref": "path.wav", "takes": 3}
  <- {"ok": true, "id": "...", "path": "/tmp/x.wav", "dur": 3.72, "wer": 0.0, "asr": "..."}
  <- {"ok": false, "id": "...", "err": "..."}

Model dimuat sekali (tetap di RAM), state suara di-cache per path referensi.
Referensi otomatis dipotong ke 15 detik — >15 s membuat model berhenti dini.
"""
import hashlib
import json
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf
from pocket_tts import TTSModel

from norm_id import normalize

CFG = os.environ.get(
    "POCKET_CFG",
    "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5",
)
OUT = os.environ.get("WORKER_OUT", "data/out")
USE_ASR = os.environ.get("USE_ASR", "1") == "1"
EOS = float(os.environ.get("EOS", "-3.0"))
REF_MAX_S = float(os.environ.get("REF_MAX_S", "15"))
SR_OUT = 24000
GAP_S = 0.12

os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


model = TTSModel.load_model(config=CFG, eos_threshold=EOS)
log(f"pocket-tts siap (eos={EOS})")

asr = None
if USE_ASR:
    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
    log("whisper-small siap untuk skoring best-of-N")

_states = {}
_ref_cache = {}


def prep_ref(path):
    """Potong referensi ke REF_MAX_S detik mulai dari sampel bersuara pertama."""
    if path in _ref_cache:
        return _ref_cache[path]
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    win = max(1, int(0.02 * sr))
    env = np.abs(a[: len(a) // win * win]).reshape(-1, win).max(axis=1)
    idx = np.where(env > 0.02)[0]
    start = int(idx[0] * win) if len(idx) else 0
    seg = a[start: start + int(REF_MAX_S * sr)]
    tag = hashlib.md5(path.encode()).hexdigest()[:8]
    tmp = os.path.join(OUT, f"ref_{tag}.wav")
    sf.write(tmp, seg, sr)
    _ref_cache[path] = tmp
    log(f"referensi {path} -> {tmp} ({len(seg) / sr:.1f}s)")
    return tmp


def state_for(path):
    if path not in _states:
        _states[path] = model.get_state_for_audio_prompt(prep_ref(path))
    return _states[path]


def chunk(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def norm_words(s):
    keep = "abcdefghijklmnopqrstuvwxyz "
    return " ".join("".join(c if c in keep else " " for c in s.lower()).split())


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def generate(text, ref, takes):
    st = state_for(ref)
    gap = np.zeros(int(GAP_S * SR_OUT), dtype=np.float32)
    target = norm_words(normalize(text))
    best = None
    for k in range(takes):
        parts = []
        for c in chunk(text):
            a = model.generate_audio(st, c)
            parts.append(a.detach().cpu().numpy().astype(np.float32).reshape(-1))
            parts.append(gap)
        audio = np.concatenate(parts[:-1]) if parts else np.zeros(1, dtype=np.float32)
        if asr is None:
            return {"audio": audio, "wer": None, "asr": None, "take": k}
        data = librosa.resample(audio, orig_sr=SR_OUT, target_sr=16000)
        segs, _ = asr.transcribe(data, language="id", beam_size=5)
        hyp = " ".join(s.text.strip() for s in segs)
        w = wer(target, norm_words(normalize(hyp)))
        log(f"  take{k}: wer={w:.2f} dur={len(audio) / SR_OUT:.2f}s")
        if best is None or w < best["wer"]:
            best = {"audio": audio, "wer": w, "asr": hyp, "take": k}
    return best


def handle(req):
    rid = req.get("id", "?")
    text = (req.get("text") or "").strip()
    ref = req.get("ref") or ""
    takes = int(req.get("takes", 3))
    if not text:
        return {"ok": False, "id": rid, "err": "teks kosong"}
    if not os.path.exists(ref):
        return {"ok": False, "id": rid, "err": f"referensi tidak ada: {ref}"}
    best = generate(text, ref, takes)
    path = os.path.join(OUT, f"{rid}.wav")
    a = best["audio"]
    import wave

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR_OUT)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return {
        "ok": True,
        "id": rid,
        "path": path,
        "dur": round(len(a) / SR_OUT, 2),
        "wer": None if best["wer"] is None else round(best["wer"], 3),
        "asr": best["asr"],
        "take": best["take"],
    }


def main():
    print(json.dumps({"ready": True, "asr": USE_ASR, "eos": EOS}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle(req)
        except Exception as e:  # noqa: BLE001
            resp = {"ok": False, "id": "?", "err": f"{type(e).__name__}: {e}"}
        print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
