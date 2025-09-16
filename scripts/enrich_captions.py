import argparse, os
from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm

# Use BLIP v1 (works well, avoids BLIP-2 tokenizer issues)
from transformers import BlipProcessor, BlipForConditionalGeneration

def dedup_words(text: str) -> str:
    # simple pass to reduce repeated tokens like "tattoo tattoo"
    seen = set()
    out: List[str] = []
    for w in text.split():
        w_norm = w.lower().strip(",.;:!?")
        if w_norm in seen and len(w_norm) > 3:
            continue
        seen.add(w_norm)
        out.append(w)
    return " ".join(out)

def trim_to_max_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",      required=True, help="Folder with preprocessed PNGs")
    ap.add_argument("--out_dir",      required=True, help="Where to write enriched .txt captions")
    ap.add_argument("--append_style", default=", clean line tattoo, high contrast, stencil, no shading")
    ap.add_argument("--max_words",    type=int, default=64)
    ap.add_argument("--batch_size",   type=int, default=8)
    ap.add_argument("--model",        default="Salesforce/blip-image-captioning-large",
                    help="BLIP captioning model (v1).")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    processor = BlipProcessor.from_pretrained(args.model)
    model     = BlipForConditionalGeneration.from_pretrained(args.model).to(device)

    # Collect images
    imgs = sorted([p for p in img_dir.glob("*.png")])
    if not imgs:
        raise RuntimeError(f"No PNG images found in {img_dir}")

    batch = []
    for p in tqdm(imgs, desc="Caption+enrich"):
        batch.append(p)

        if len(batch) == args.batch_size:
            caption_and_write(batch, processor, model, device, out_dir,
                              append_style=args.append_style, max_words=args.max_words)
            batch = []

    if batch:
        caption_and_write(batch, processor, model, device, out_dir,
                          append_style=args.append_style, max_words=args.max_words)

    print(f"Saved enriched captions to: {out_dir}")

@torch.inference_mode()
def caption_and_write(
    paths: List[Path],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: str,
    out_dir: Path,
    append_style: str,
    max_words: int,
):
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)

    # Use beam search for a bit more descriptive captions
    generated_ids = model.generate(
        **inputs,
        max_length=48,
        num_beams=5,
        length_penalty=1.0,
        repetition_penalty=1.2,
    )
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for p, base in zip(paths, captions):
        base = base.strip()
        base = dedup_words(base)
        # append style if not present
        styled = (base + " " + append_style).strip()
        styled = styled.replace("  ", " ").strip(" ,.;")
        styled = trim_to_max_words(styled, max_words)

        (out_dir / f"{p.stem}.txt").write_text(styled, encoding="utf-8")

if __name__ == "__main__":
    main()
