#!/usr/bin/env python3
"""Bangun aset logo Laras dari lockup sumber.

Sumber: satu berkas gambar (lockup horizontal: mark ungu + wordmark + tagline)
di latar hitam. Skrip ini TIDAK menggambar ulang tipografi — hanya memetakan
ulang luminansi ke warna target, jadi bentuk huruf asli tetap utuh.

Masalah yang diperbaiki:
  * wordmark di sumber ~#181818 di atas #000 (kontras 1,3:1) -> dinaikkan
  * padding vertikal berlebih -> dipotong ke bbox konten + margin

Keluaran (assets/):
  laras-banner-dark.png    1280 px, latar hitam, teks terang
  laras-banner-light.png   1280 px, latar putih, teks gelap
  laras-mark.png           mark saja, latar transparan, kanvas persegi
  laras-social.png         1280x640, untuk social preview GitHub

Pemakaian:
  python3 assets/build.py assets/source/lockup-original.jpg
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- warna target ---------------------------------------------------------
DARK = {
    "bg": (0, 0, 0),
    "wordmark": (242, 242, 242),
    "tagline": (168, 168, 168),
}
LIGHT = {
    "bg": (255, 255, 255),
    "wordmark": (17, 17, 17),
    "tagline": (105, 105, 105),
}

# --- ambang luminansi (diukur dari sumber) --------------------------------
BG_MAX = 10          # di bawah ini dianggap latar
WORDMARK_LO, WORDMARK_HI = 10, 24
TAGLINE_LO, TAGLINE_HI = 40, 84
MARK_LO, MARK_HI = 5, 18


def smooth(x, lo, hi):
    """Ramp linear 0..1, dijepit. Cukup untuk tepi antialias."""
    return np.clip((x - lo) / float(hi - lo), 0.0, 1.0)


def load(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    lum = a.max(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    return a, lum, sat


def split_regions(lum, sat):
    """Pisahkan area mark (berwarna) dari area teks (netral) secara horizontal."""
    colored_cols = (sat > 25).sum(axis=0) > 3
    xs = np.where(colored_cols)[0]
    mark_x1 = int(xs.max()) if len(xs) else 0
    # celah antara mark dan teks
    gap = int(mark_x1 + 0.03 * lum.shape[1])
    return mark_x1, gap


def bands(lum, sat, x_from):
    """Cari pita vertikal wordmark dan tagline di area teks."""
    text = (lum > BG_MAX) & ~(sat > 25)
    prof = text[:, x_from:].sum(axis=1)
    rows = np.where(prof > prof.max() * 0.02)[0]
    if len(rows) == 0:
        return (0, lum.shape[0]), (0, 0)
    # tagline = pita paling bawah yang terpisah dari wordmark
    top = rows.min()
    bottom = rows.max()
    # cari celah terbesar di paruh bawah
    mid = int(top + (bottom - top) * 0.75)
    seg = prof[mid:bottom + 1]
    if seg.max() == 0:
        return (top, bottom), (0, 0)
    # tagline dimulai pada baris pertama setelah celah kosong
    empty = np.where(seg <= 1)[0]
    if len(empty) == 0:
        return (top, bottom), (0, 0)
    tag_top = mid + int(empty[0])
    while tag_top < bottom and prof[tag_top] <= 1:
        tag_top += 1
    return (top, tag_top - 1), (tag_top, bottom)


def build_layers(path):
    a, lum, sat = load(path)
    mark_x1, gap = split_regions(lum, sat)
    word_band, tag_band = bands(lum, sat, gap)

    h, w = lum.shape
    yy = np.arange(h)[:, None]

    is_mark = np.zeros((h, w), bool)
    is_mark[:, :gap] = True
    is_word = (yy >= word_band[0]) & (yy <= word_band[1]) & ~is_mark
    is_tag = (yy >= tag_band[0]) & (yy <= tag_band[1]) & ~is_mark & (tag_band[1] > 0)

    # Halo JPEG di sekitar siluet itu netral dan gelap; kalau ikut terbawa,
    # hasilnya jadi outline hitam di atas latar putih. Buang.
    halo = (sat < 18) & (lum < 70)
    alpha_mark = smooth(lum, MARK_LO, MARK_HI) * is_mark
    alpha_mark[halo & is_mark] = 0.0
    alpha_word = smooth(lum, WORDMARK_LO, WORDMARK_HI) * is_word
    alpha_tag = smooth(lum, TAGLINE_LO, TAGLINE_HI) * is_tag

    return a, lum, sat, alpha_mark, alpha_word, alpha_tag, (word_band, tag_band, mark_x1)


def compose(a, alpha_mark, alpha_word, alpha_tag, theme):
    h, w = alpha_word.shape
    out = np.zeros((h, w, 3), np.float32)
    bg = np.array(theme["bg"], np.float32)
    out[:] = bg

    def over(color, alpha):
        c = np.array(color, np.float32)
        return out * (1 - alpha[..., None]) + c * alpha[..., None]

    # mark: warna asli dipertahankan, hanya latar yang dibuang
    out = out * (1 - alpha_mark[..., None]) + a * alpha_mark[..., None]
    out = over(theme["wordmark"], alpha_word)
    out = over(theme["tagline"], alpha_tag)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def content_bbox(alpha_mark, alpha_word, alpha_tag, thr=0.06):
    m = (alpha_mark + alpha_word + alpha_tag) > thr
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def crop_pad(img, bbox, pad_frac_x=0.05, pad_frac_y=0.10):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad_frac_x), int(h * pad_frac_y)
    box = (max(0, x0 - px), max(0, y0 - py),
           min(img.width, x1 + px), min(img.height, y1 + py))
    return img.crop(box)


def main():
    if len(sys.argv) < 2:
        sys.exit("pakai: python3 assets/build.py <lockup.jpg>")
    src = Path(sys.argv[1])
    outdir = Path(__file__).parent

    a, lum, sat, am, aw, at, info = build_layers(src)
    bbox = content_bbox(am, aw, at)

    dark = compose(a, am, aw, at, DARK)
    light = compose(a, am, aw, at, LIGHT)

    # banner: potong ke konten + margin, lalu skala ke lebar 1280
    banner_dark = crop_pad(dark, bbox, 0.045, 0.22)
    banner_light = crop_pad(light, bbox, 0.045, 0.22)
    scale = 1280 / banner_dark.width
    size = (1280, round(banner_dark.height * scale))
    banner_dark.resize(size, Image.Resampling.LANCZOS).save(outdir / "laras-banner-dark.png")
    banner_light.resize(size, Image.Resampling.LANCZOS).save(outdir / "laras-banner-light.png")

    # mark saja, latar transparan, kanvas persegi
    mark_x1 = info[2]
    mark_alpha = np.zeros_like(am)
    mark_alpha[:, :mark_x1 + int(0.02 * am.shape[1])] = am[:, :mark_x1 + int(0.02 * am.shape[1])]
    ys, xs = np.where(mark_alpha > 0.30)
    mb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    rgba = np.dstack([a, (np.clip(mark_alpha, 0, 1) * 255)])
    mark_img = Image.fromarray(rgba.astype(np.uint8), "RGBA").crop(mb)
    side = int(max(mark_img.size) * 1.18)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(
        mark_img,
        ((side - mark_img.width) // 2, (side - mark_img.height) // 2),
        mark_img,
    )
    canvas.resize((512, 512), Image.Resampling.LANCZOS).save(outdir / "laras-mark.png")

    # social preview 1280x640 (latar hitam, aman untuk tema terang & gelap)
    social = Image.new("RGB", (1280, 640), DARK["bg"])
    b = banner_dark
    sc = min((1280 * 0.88) / b.width, (640 * 0.80) / b.height)
    nb = b.resize((round(b.width * sc), round(b.height * sc)), Image.Resampling.LANCZOS)
    social.paste(nb, ((1280 - nb.width) // 2, (640 - nb.height) // 2))
    social.save(outdir / "laras-social.png")

    print("wordmark band", info[0], "tagline band", info[1], "mark_x1", info[2])
    for f in ["laras-banner-dark.png", "laras-banner-light.png", "laras-mark.png", "laras-social.png"]:
        p = outdir / f
        print(f"{f:26s} {Image.open(p).size}")


if __name__ == "__main__":
    main()
