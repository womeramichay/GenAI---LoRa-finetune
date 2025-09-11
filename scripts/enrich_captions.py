# scripts/enrich_captions.py
import argparse, pathlib, re, json
from typing import List
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def generate_blip(model, processor, batch_imgs: List[pathlib.Path], device, max_new_tokens=40):
    images = [Image.open(p).convert("RGB") for p in batch_imgs]
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, repetition_penalty=1.2)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]

def clean_line(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)                   # data/processed/<dataset>/images
    ap.add_argument("--out_dir", required=True)                   # data/processed/<dataset>/blip_plus
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    ap.add_argument("--append_style", default=", clean line tattoo, high contrast, stencil, no shading")
    ap.add_argument("--max_words", type=int, default=24)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map={"": 0} if dev.type == "cuda" else None)
    model.to(dev)

    img_dir = pathlib.Path(args.img_dir)
    out_root = pathlib.Path(args.out_dir)
    cap_dir = out_root / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.png"))
    count = 0
    for i in tqdm(range(0, len(imgs), args.batch_size), total=(len(imgs)+args.batch_size-1)//args.batch_size):
        batch = imgs[i:i+args.batch_size]
        caps = generate_blip(model, processor, batch, dev)
        for p, c in zip(batch, caps):
            words = [w for w in re.split(r"[^a-zA-Z0-9]+", clean_line(c)) if w]
            words = words[:args.max_words]
            # add style tokens
            enriched = " ".join(words) + args.append_style
            (cap_dir / f"{p.stem}.txt").write_text(enriched.strip(), encoding="utf-8")
            count += 1

    (out_root / "enrich_meta.json").write_text(
        json.dumps({"count": count, "model": args.model, "append_style": args.append_style}, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {count} BLIP+ captions → {cap_dir}")

if __name__ == "__main__":
    main()
