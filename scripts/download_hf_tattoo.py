# scripts/download_hf_tattoo.py
import os, argparse
from datasets import load_dataset
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. Drozdik/tattoo_v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--outdir", default="data/raw/hf_tattoo_v3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split)
    saved = 0
    for i, rec in enumerate(ds):
        img = rec.get("image") or rec.get("img")
        if img is None:
            continue
        im = img.convert("RGB")
        im.save(os.path.join(args.outdir, f"{i:06d}.png"))
        saved += 1
    print(f"saved {saved} images to {args.outdir}")

if __name__ == "__main__":
    main()
