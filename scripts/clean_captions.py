# scripts/clean_captions.py
import argparse, pathlib, re, json

def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def default_caption() -> str:
    return "minimal line-art tattoo, stencil, high contrast, no shading"

def caption_from_name(p: pathlib.Path) -> str:
    # optional heuristic from filename tokens
    base = p.stem.lower().replace("_", " ").replace("-", " ")
    base = re.sub(r"\d+", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or "tattoo design"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)                 # data/processed/<dataset>/images
    ap.add_argument("--out_dir", required=True)                 # data/processed/<dataset>/vanilla
    ap.add_argument("--append_style", default=", clean line tattoo, high contrast, stencil, no shading")
    ap.add_argument("--max_words", type=int, default=8)
    args = ap.parse_args()

    img_dir = pathlib.Path(args.img_dir)
    out_root = pathlib.Path(args.out_dir)
    cap_dir = out_root / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in sorted(img_dir.glob("*.png")):
        words = caption_from_name(p).split()
        words = words[:args.max_words] if words else default_caption().split()
        cap = clean_text(" ".join(words)) + args.append_style
        (cap_dir / f"{p.stem}.txt").write_text(cap.strip(), encoding="utf-8")
        count += 1

    (out_root / "vanilla_meta.json").write_text(
        json.dumps({"count": count, "append_style": args.append_style}, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {count} vanilla captions → {cap_dir}")

if __name__ == "__main__":
    main()
