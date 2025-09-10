# scripts/preprocess.py
# Unified preprocessing with two variants: "vanilla" and "blip"
import argparse, pathlib, json, re
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

# ---- caption helpers ----
SUBJECT_WORDS_RE = re.compile(r"[A-Za-z0-9]+")

def clamp_words(text: str, max_words: int) -> str:
    words = SUBJECT_WORDS_RE.findall(text.lower())
    return " ".join(words[:max_words]) if words else "design"

def subject_from_filename(stem: str, max_words: int) -> str:
    name = stem.replace("_", " ").replace("-", " ")
    return clamp_words(name, max_words)

def build_caption(subject: str, style: str, suffix: str, max_subject_words: int) -> str:
    subject = clamp_words(subject, max_subject_words)
    style = style.strip()
    if style:
        return f"{subject}, {style}, {suffix}"
    else:
        return f"{subject}, {suffix}"

def load_blip(device: str = "cuda"):
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    d = device if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(d)
    return proc, model, d

def blip_subject(proc, model, device, pil_image: Image.Image, max_new_tokens=25) -> str:
    inputs = proc(images=pil_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, help="e.g. tattoo_v3")
    ap.add_argument("--variant", choices=["vanilla","blip"], required=True)
    ap.add_argument("--input_dir", required=True, help="data/raw/<dataset_name>")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--max_subject_words", type=int, default=2)
    ap.add_argument("--style", default="", help="optional style tokens (<=2 words), e.g. 'clean line' or 'graffiti'")
    ap.add_argument("--suffix", default="minimal line-art tattoo, stencil, high contrast, no shading")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_root = pathlib.Path("data/processed") / args.dataset_name / args.variant
    img_out = out_root / "images"
    cap_out = out_root / "captions"
    img_out.mkdir(parents=True, exist_ok=True)
    cap_out.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in exts]

    # Load BLIP once if needed
    blip_bundle = None
    if args.variant == "blip":
        blip_bundle = load_blip()

    count = 0
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = center_square(im)
        im = to_white_bg(im)
        if im.size != (args.size, args.size):
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            im = im.resize((args.size, args.size), resample)
        im = maybe_threshold(im, args.threshold)

        if args.variant == "vanilla":
            subject = subject_from_filename(p.stem, args.max_subject_words)
        else:
            proc, model, device = blip_bundle
            raw = blip_subject(proc, model, device, im)
            subject = clamp_words(raw, args.max_subject_words)

        caption = build_caption(subject, args.style, args.suffix, args.max_subject_words)

        im.save(img_out / f"{p.stem}.png")
        (cap_out / f"{p.stem}.txt").write_text(caption, encoding="utf-8")
        count += 1

    meta = {
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "count": count,
        "size": args.size,
        "threshold": args.threshold,
        "style": args.style,
        "suffix": args.suffix,
        "max_subject_words": args.max_subject_words,
    }
    (out_root / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Processed {count} images → {out_root}")

if __name__ == "__main__":
    main()
