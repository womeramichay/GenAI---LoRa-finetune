import argparse, pathlib, re

STOPWORDS = {"a", "the", "an", "of", "and", "in", "black", "white", "line", "art", "tattoo", "stencil"}
STYLE_TRAILER = ", clean line, minimal line-art tattoo, stencil, high contrast, no shading"

def extract_subject(txt: str) -> str:
    # Keep 2–5 most meaningful tokens from BLIP, drop style/color fillers.
    words = re.findall(r"[a-z0-9]+", txt.lower())
    words = [w for w in words if w not in STOPWORDS]
    # collapse repeated words
    dedup = []
    for w in words:
        if not dedup or dedup[-1] != w:
            dedup.append(w)
    # choose up to 4
    core = " ".join(dedup[:4]).strip()
    return core or "tattoo motif"

def subject_from_filename(p: pathlib.Path) -> str:
    # fallback from file name, e.g. "owl_sketch_03.png" -> "owl sketch"
    base = p.stem.lower()
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\d+", "", base).strip()
    base = re.sub(r"\s+", " ", base)
    if not base:
        return "tattoo motif"
    # take first 2–3 words
    return " ".join(base.split()[:3])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_dir", required=True)
    ap.add_argument("--images_dir", required=True)  # to fallback on filenames if caption is empty
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cap_dir = pathlib.Path(args.captions_dir)
    img_dir = pathlib.Path(args.images_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img in sorted(img_dir.glob("*.png")):
        cap_path = cap_dir / f"{img.stem}.txt"
        raw = cap_path.read_text(encoding="utf-8").strip() if cap_path.exists() else ""
        subj = extract_subject(raw) if raw else subject_from_filename(img)
        # enforce template
        clean = f"{subj}{STYLE_TRAILER}"
        (out_dir / f"{img.stem}.txt").write_text(clean, encoding="utf-8")

    print(f"Cleaned captions written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
