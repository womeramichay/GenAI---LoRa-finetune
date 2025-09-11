# scripts/enrich_captions.py
import argparse, pathlib, re, json
from typing import Optional, List
from PIL import Image
from tqdm import tqdm

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

GENERIC_PATTERNS = [
    r"^\s*an? image of\s*",
    r"^\s*an? illustration of\s*",
    r"^\s*an? drawing of\s*",
    r"^\s*an? photo of\s*",
    r"^\s*an? picture of\s*",
    r"\b(clean line[s]?,?\s*)?\b(minimal(ist)? line[- ]?art,?\s*)?",
]

def clean_caption(text: str) -> str:
    t = text.strip()
    # remove boilerplate openers
    for pat in GENERIC_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    # collapse spaces & trailing punctuation
    t = re.sub(r"\s+", " ", t).strip(" ,.;:-")
    # lowercase first letter to match our dataset style
    if t and t[0].isalpha():
        t = t[0].lower() + t[1:]
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Folder with *.png images")
    ap.add_argument("--out_dir", required=True, help="Where to save captions/*.txt")
    ap.add_argument("--min_words", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=30)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--append_style", default=", clean line tattoo, high contrast, stencil",
                    help="Style suffix to ensure downstream conditioning has style words.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    args = ap.parse_args()

    img_dir = pathlib.Path(args.img_dir)
    out_dir = pathlib.Path(args.out_dir)
    cap_dir = out_dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    # load BLIP large
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16 if args.device == "cuda" else None)
    model.to(args.device)
    model.eval()

    imgs: List[pathlib.Path] = sorted(img_dir.glob("*.png"))
    assert len(imgs) > 0, f"No images in {img_dir}"

    # save meta
    meta = {
        "source_images": str(img_dir),
        "model": args.model,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "append_style": args.append_style,
        "count": len(imgs),
    }
    (out_dir / "enrich_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def cap_one_batch(batch_imgs: List[pathlib.Path]):
        images = [Image.open(p).convert("RGB") for p in batch_imgs]
        inputs = processor(images=images, return_tensors="pt").to(args.device, dtype=torch.float16 if args.device=="cuda" else None)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                min_length=args.min_words,            # tokens (approx words, but helps)
                max_length=args.max_words + 10,       # small buffer
                no_repeat_ngram_size=2,
            )
        texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        return [clean_caption(t) for t in texts]

    # iterate batches
    for i in tqdm(range(0, len(imgs), args.batch_size), total=(len(imgs)+args.batch_size-1)//args.batch_size):
        batch = imgs[i:i+args.batch_size]
        caps = cap_one_batch(batch)

        for p, c in zip(batch, caps):
            # enforce min words at the word level
            words = [w for w in re.split(r"\s+", c) if w]
            if len(words) < args.min_words:
                # try to extend slightly with style tokens
                c = f"{c}, detail, outline, subject emphasis"
            # final suffix (style kept at the tail)
            if args.append_style:
                c = f"{c}{args.append_style}"
            (cap_dir / f"{p.stem}.txt").write_text(c.strip(), encoding="utf-8")

    print(f"Saved {len(imgs)} captions → {cap_dir}")

if __name__ == "__main__":
    main()
