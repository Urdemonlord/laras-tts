"""Normalizer angka Bahasa Indonesia untuk TTS (digit -> kata)."""
import re

SATUAN = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
          "sepuluh", "sebelas"]


def _ratus(n):
    if n < 12:
        return SATUAN[n]
    if n < 20:
        return _ratus(n - 10) + " belas"
    if n < 100:
        s = _ratus(n // 10) + " puluh"
        return s + (" " + _ratus(n % 10) if n % 10 else "")
    if n < 200:
        s = "seratus"
    else:
        s = _ratus(n // 100) + " ratus"
    return s + (" " + _ratus(n % 100) if n % 100 else "")


def terbilang(n):
    if n == 0:
        return "nol"
    parts = []
    for nilai, nama in ((10**12, "triliun"), (10**9, "miliar"), (10**6, "juta"),
                        (10**3, "ribu"), (1, "")):
        if n >= nilai:
            q, n = divmod(n, nilai)
            if nilai == 1000 and q == 1:
                parts.append("seribu")
            else:
                parts.append((_ratus(q) + " " + nama).strip())
    return " ".join(parts)


def _angka(tok):
    tok = tok.strip().rstrip(".")
    if "," in tok:
        bulat, pecah = tok.split(",", 1)
        bulat = bulat.replace(".", "") or "0"
        out = terbilang(int(bulat)) + " koma"
        return out + " " + " ".join(terbilang(int(d)) for d in pecah if d.isdigit())
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", tok):
        tok = tok.replace(".", "")
    if not tok.isdigit():
        out = []
        for c in tok:
            if c.isdigit():
                out.append(terbilang(int(c)))
            elif c == ".":
                out.append("titik")
            elif c == ",":
                out.append("koma")
        return " ".join(out)
    return terbilang(int(tok))


def normalize(text):
    text = re.sub(r"\bRp\.?\s*([\d.,]+)", lambda m: _angka(m.group(1)) + " rupiah", text)
    text = re.sub(r"([\d.,]+)\s*%", lambda m: _angka(m.group(1)) + " persen", text)
    text = re.sub(r"\d[\d.,]*", lambda m: _angka(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    for s in ["Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026.",
              "Suhu mencapai 32,5 derajat", "Tahun 1998 dan 1.250.000 orang", "Biaya Rp 5.000.000"]:
        print(s, "->", normalize(s))
