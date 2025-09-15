# scripts/auto_caption_blip.py
import argparse, os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch

# BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
# BLIP-1
from transformers import BlipProcessor, BlipForConditionalGeneration


def chunked(it, n):
    it = list(it)
    for i in range(0, len(it), n):
        yield it[i:i+n]


def load_model(model_name: str, device: str):
    """
    If model_name contains 'blip2', load BLIP-2; otherwise load BLIP-1.
    Returns (processor, model, is_blip2).
    """
    is_blip2 = ("blip2" in model_name.lower())
    if is_blip2:
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    model = model.to(device)
    model.eval()
    return processor, model, is_blip2


@torch.no_grad()
def caption_batch(processor, model, is_blip2, images, device, max_new_tokens=32):
    if is_blip2:
        inputs = processor(images=images, return_tensors="pt").to(device, dtype=model.dtype)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        texts = processor.batch_decode(out, skip_special_tokens=True)
    else:
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        texts = processor.batch_decode(out, skip_special_tokens=True)
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Folder with preprocessed PNGs")
    ap.add_argument("--out_dir", required=True, help="Where to write one .txt per image")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large",
                    help="BLIP-1 (default): Salesforce/blip-image-captioning-large "
                         "or BLIP-2: e.g. Salesforce/blip2-opt-2.7b")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Gather images
    imgs = sorted([p for p in img_dir.glob("*.png")])
    if not imgs:
        raise SystemExit(f"No PNGs found in {img_dir}")

    print(f"Loading model: {args.model}")
    processor, model, is_blip2 = load_model(args.model, device)

    # Process in batches
    for batch in tqdm(list(chunked(imgs, args.batch_size)), desc="Captioning"):
        images = [Image.open(p).convert("RGB") for p in batch]
        texts = caption_batch(processor, model, is_blip2, images, device, args.max_new_tokens)

        for p, t in zip(batch, texts):
            (out_dir / f"{p.stem}.txt").write_text(t + "\n", encoding="utf-8")

    print(f"Saved captions to: {out_dir}")


if __name__ == "__main__":
    main()
