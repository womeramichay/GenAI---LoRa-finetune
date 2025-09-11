# scripts/preprocess.py
import argparse, pathlib, json
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True)           # e.g. tattoo_v3
    ap.add_argument("--variant", default="vanilla")            # kept for backward compat; not used when images_only
    ap.add_argument("--input_dir", required=True)              # e.g. data/raw/tattoo_v3
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out_root", default="data/processed")    # base processed dir
    ap.add_argument("--images_only", action="store_true")      # NEW: only write images/
    args = ap.parse_args()

    in_root = pathlib.Path(args.input_dir)
    out_root = pathlib.Path(args.out_root) / args.dataset_name
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in sorted(in_root.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (args.size, args.size):
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            im = im.resize((args.size, args.size), resample)
        (img_dir / f"{count:06d}.png").parent.mkdir(parents=True, exist_ok=True)
        im.save(img_dir / f"{count:06d}.png")
        count += 1

    # meta
    (out_root / "preprocess_meta.json").write_text(
        json.dumps({"dataset_name": args.dataset_name, "size": args.size, "count": count}, indent=2),
        encoding="utf-8"
    )

    print(f"Processed {count} images → {img_dir}")
    # We intentionally do not write captions here anymore.
    # Caption scripts will populate data/processed/<dataset>/<variant>/captions

if __name__ == "__main__":
    main()
