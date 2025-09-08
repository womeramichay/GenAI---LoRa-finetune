# scripts/preprocess.py
import argparse, pathlib, json
from typing import Optional
from PIL import Image, ImageOps, ImageFilter

def center_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return im.crop((left, top, left + m, top + m))

def to_white_bg(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg.convert("RGB")
    else:
        im = im.convert("RGB")
    return im

def maybe_threshold(im: Image.Image, thresh: Optional[int]) -> Image.Image:
    if thresh is None:
        return im
    g = ImageOps.grayscale(im)
    g = g.filter(ImageFilter.MedianFilter(3))
    bw = g.point(lambda p: 255 if p >= thresh else 0, mode="1").convert("L")
    return Image.merge("RGB", (bw, bw, bw))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threshold", type=int, default=None, help="0-255 (e.g., 200 for crisp stencil)")
    ap.add_argument("--write_captions", action="store_true", help="create .txt per image from filename")
    args = ap.parse_args()

    in_dir  = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "captions").mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    count = 0
    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        im = Image.open(p)
        im = center_square(im)
        im = to_white_bg(im)
        if im.size != (args.size, args.size):
            im = im.resize((args.size, args.size), Image.LANCZOS)
        im = maybe_threshold(im, args.threshold)

        out_img = (out_dir / "images" / (p.stem + ".png"))
        im.save(out_img)

        if args.write_captions:
            cap = (p.stem.replace("_", " ").replace("-", " ")
                   + ", minimal line-art tattoo, stencil, no shading")
            (out_dir / "captions" / (p.stem + ".txt")).write_text(cap, encoding="utf-8")
        count += 1

    meta = {"count": count, "size": args.size, "threshold": args.threshold}
    (out_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Processed {count} images → {out_dir}")

if __name__ == "__main__":
    main()
