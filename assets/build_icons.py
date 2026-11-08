#!/usr/bin/env python3
"""Bangun set favicon + ikon PWA dari assets/laras-mark.png.

Butuh: pillow, numpy.

    uv run --with pillow --with numpy python assets/build_icons.py

Keluaran di assets/icons/:
    favicon-16x16.png, favicon-32x32.png   glyph bar (lihat catatan di bawah)
    favicon-48x48.png                     mark lengkap
    favicon.ico                           16 + 32 + 48 dalam satu berkas
    apple-touch-icon.png                  180x180, latar solid
    icon-192.png, icon-512.png            PWA "any"
    icon-maskable-512.png                 full-bleed, aman safe zone Android
    contact-sheet.png                     pratinjau semua ukuran

Kenapa 16/32 px pakai glyph bar, bukan mark lengkap:
mark Laras punya bar equalizer knockout + busur suara tipis. Pada 16 px busur
itu hilang dan knockout-nya menyatu jadi bercak gelap, jadi ikon terbaca
sebagai blob ungu. Tiga bar equalizer adalah bentuk paling sederhana dari mark
yang sama, dan bertahan sampai 16 px. Mulai 48 px mark lengkap dipakai lagi.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "icons"

BRAND_BG = (11, 6, 20)          # #0B0614, latar gelap ikon opaque
BRAND_TOP = (139, 92, 246)      # #8B5CF6
BRAND_BOT = (76, 29, 149)       # #4C1D95
MIN_COMPONENT = 300             # px; di bawah ini = artefak sumber


def load_mark() -> Image.Image:
    p = ROOT / "laras-mark.png"
    if not p.exists():
        raise SystemExit(f"{p} belum ada — jalankan assets/build.py dulu")
    return Image.open(p).convert("RGBA")


def drop_specks(img: Image.Image) -> Image.Image:
    """Buang komponen kecil terpisah (di sumber ada dash 32x5 dan bintik 2 px)."""
    arr = np.asarray(img).copy()
    mask = arr[:, :, 3] > 30
    h, w = mask.shape
    seen = np.zeros_like(mask)
    keep = np.zeros_like(mask)
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) >= MIN_COMPONENT:
                for y, x in pts:
                    keep[y, x] = True
    arr[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
    return Image.fromarray(arr, "RGBA")


def head_only(img: Image.Image) -> Image.Image:
    """Ambil komponen terbesar saja: siluet kepala, tanpa busur suara tipis."""
    arr = np.asarray(img).copy()
    mask = arr[:, :, 3] > 30
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) > len(best):
                best = pts
    keep = np.zeros_like(mask)
    for y, x in best:
        keep[y, x] = True
    arr[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
    return Image.fromarray(arr, "RGBA")


def ink_crop(img: Image.Image) -> Image.Image:
    """Potong ke bbox tinta supaya centering optis, bukan centering kanvas."""
    bbox = img.split()[-1].point(lambda v: 255 if v > 30 else 0).getbbox()
    return img.crop(bbox) if bbox else img


def place(img: Image.Image, size: int, fill: float, bg=None) -> Image.Image:
    """Skala `img` supaya sisi terpanjangnya = fill*size, lalu taruh di tengah."""
    scale = (size * fill) / max(img.size)
    m = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                   Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (bg + (255,)) if bg else (0, 0, 0, 0))
    canvas.paste(m, ((size - m.width) // 2, (size - m.height) // 2), m)
    return canvas


def bars(size: int) -> Image.Image:
    """Glyph tiga bar equalizer, digambar langsung di grid piksel."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bw = max(1, size // 8)
    gap = max(1, size // 8)
    total = 3 * bw + 2 * gap
    x0 = (size - total) // 2
    for i, frac in enumerate((0.55, 1.00, 0.72)):
        bh = max(2, round(size * 0.72 * frac))
        y0 = (size - bh) // 2
        d.rounded_rectangle([x0 + i * (bw + gap), y0,
                             x0 + i * (bw + gap) + bw - 1, y0 + bh - 1],
                            radius=max(0, bw // 2 - 1), fill=(255, 255, 255, 255))
    # gradien vertikal brand, di-clip ke bentuk bar
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / max(1, size - 1)
        gd.line([(0, y), (size, y)], fill=tuple(
            round(BRAND_TOP[c] + (BRAND_BOT[c] - BRAND_TOP[c]) * t) for c in range(3)) + (255,))
    grad.putalpha(img.split()[-1])
    return grad


def write_ico(path: Path, frames) -> None:
    """Tulis ICO multi-resolusi sendiri: Pillow hanya menyimpan 1 frame."""
    import io
    import struct

    blobs = []
    for img in frames:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        blobs.append((img.width, buf.getvalue()))

    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries, payload = b"", b""
    for side, data in blobs:
        entries += struct.pack("<BBBBHHII", side % 256, side % 256, 0, 0, 1, 32,
                               len(data), offset)
        offset += len(data)
        payload += data
    path.write_bytes(head + entries + payload)


def head_bars(img: Image.Image, size: int, fill: float = 0.92) -> Image.Image:
    """Kepala saja + tiga bar putih tebal. Busur tipis dibuang: di 48 px jadi noise."""
    src = ink_crop(head_only(img))
    scale = (size * fill) / max(src.size)
    m = src.resize((round(src.width * scale), round(src.height * scale)),
                   Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox, oy = (size - m.width) // 2, (size - m.height) // 2
    canvas.paste(m, (ox, oy), m)
    d = ImageDraw.Draw(canvas)
    bw = max(1, size // 12)
    gap = max(1, size // 12)
    x0 = ox + int(m.width * 0.30)
    for i, frac in enumerate((0.34, 0.52, 0.40)):
        bh = max(2, round(m.height * frac))
        yy = oy + (m.height - bh) // 2
        d.rectangle([x0 + i * (bw + gap), yy,
                     x0 + i * (bw + gap) + bw - 1, yy + bh - 1],
                    fill=(255, 255, 255, 255))
    return canvas


def squircle_preview(img: Image.Image, box: int) -> Image.Image:
    """Pratinjau potong membulat, meniru ikon Android/iOS."""
    m = img.resize((box, box), Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", (box * 4, box * 4), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, box * 4 - 1, box * 4 - 1], fill=255)
    m.putalpha(mask.resize((box, box), Image.Resampling.LANCZOS))
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mark = ink_crop(drop_specks(load_mark()))

    small = {}
    for s in (16, 32):
        small[s] = bars(s)
        small[s].save(OUT / f"favicon-{s}x{s}.png")
    mid = head_bars(mark, 48)
    mid.save(OUT / "favicon-48x48.png")
    write_ico(OUT / "favicon.ico", [small[16], small[32], mid])

    place(mark, 180, 0.72, BRAND_BG).convert("RGB").save(OUT / "apple-touch-icon.png")
    for s in (192, 512):
        place(mark, s, 0.80, BRAND_BG).convert("RGB").save(OUT / f"icon-{s}.png")
    place(mark, 512, 0.60, BRAND_BG).convert("RGB").save(OUT / "icon-maskable-512.png")

    # --- contact sheet
    tiles = [("16", small[16], 16), ("32", small[32], 32), ("48", mid, 48)]
    big = [("apple 180", Image.open(OUT / "apple-touch-icon.png"), 110),
           ("icon 192", Image.open(OUT / "icon-192.png"), 110),
           ("maskable", Image.open(OUT / "icon-maskable-512.png"), 110)]
    sheet = Image.new("RGB", (760, 510), (248, 248, 250))
    d = ImageDraw.Draw(sheet)
    x = 24
    for label, im, show in tiles:
        d.text((x, 18), f"{label}px", fill=(40, 40, 40))
        sheet.paste(im.resize((show, show), Image.Resampling.NEAREST), (x, 36),
                    im.resize((show, show), Image.Resampling.NEAREST))
        x += show + 28
    x = 24
    for label, im, show in big:
        d.text((x, 200), label, fill=(40, 40, 40))
        sq = squircle_preview(im.convert("RGBA"), show)
        sheet.paste(sq, (x, 218), sq)
        x += show + 28
    # maskable: uji safe zone (lingkaran 80%) di atas latar terang
    mz = Image.open(OUT / "icon-maskable-512.png").resize((110, 110), Image.Resampling.LANCZOS)
    test = Image.new("RGB", (110, 110), (255, 255, 255))
    test.paste(mz, (0, 0))
    td = ImageDraw.Draw(test)   # gambar di kanvas tile, bukan di sheet
    td.ellipse([11, 11, 98, 98], outline=(220, 30, 60), width=1)
    d.text((24 + 3 * 138, 200), "safe zone 80%", fill=(40, 40, 40))
    sheet.paste(test, (24 + 3 * 138, 218))

    # baris zoom 8x: bukti keterbacaan ukuran kecil
    z = 8
    d.text((24, 348), "zoom 8x", fill=(40, 40, 40))
    for i, (im, lbl) in enumerate(((small[16], "16"), (small[32], "32"))):
        zb = im.resize((16 * z, 16 * z), Image.Resampling.NEAREST)
        sheet.paste(zb, (24 + i * (16 * z + 20), 366), zb)
    sheet.save(OUT / "contact-sheet.png")

    for f in sorted(OUT.iterdir()):
        size = "ICO (16/32/48)" if f.suffix == ".ico" else str(Image.open(f).size)
        print(f"{f.name:26s} {size}")


if __name__ == "__main__":
    main()
