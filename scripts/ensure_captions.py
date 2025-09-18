# scripts/ensure_captions.py
import argparse, shutil, sys, tempfile
from pathlib import Path
import subprocess

def p(*a, **k): print(*a, **k, flush=True)

def run_live(cmd_args):
    p(">>>", " ".join(f'"{a}"' if (" " in str(a)) else str(a) for a in cmd_args))
    proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    return proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blip","blip_plus","vanilla"], required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--cap_dir", required=True)
    # BLIP args
    ap.add_argument("--blip_model", default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    # BLIP+ style
    ap.add_argument("--append_style", default=", clean line tattoo, high contrast, stencil, no shading")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    cap_dir = Path(args.cap_dir)
    cap_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.glob("*.png")])
    if not images:
        raise SystemExit(f"No PNG images in {img_dir}")

    existing = {p.stem for p in cap_dir.glob("*.txt")}
    missing  = [p for p in images if p.stem not in existing]

    if args.mode == "vanilla":
        # Nothing to do here; vanilla uses .txt beside images or main handles autofill.
        p("[ensure] vanilla mode: nothing to do.")
        return 0

    if not missing:
        p("[ensure] captions are up-to-date; nothing to generate.")
        return 0

    p(f"[ensure] {len(missing)} captions missing → generating with {args.mode}...")

    # Build a small temp folder containing only missing images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tmp_imgs = tmp / "imgs"
        tmp_imgs.mkdir(parents=True, exist_ok=True)

        for src in missing:
            shutil.copy2(src, tmp_imgs / src.name)

        if args.mode == "blip":
            # run auto_caption_blip.py → writes .txt into cap_dir
            cmd = [
                sys.executable, "scripts/auto_caption_blip.py",
                "--img_dir", str(tmp_imgs),
                "--out_dir", str(cap_dir),
                "--batch_size", str(args.batch_size),
                "--max_new_tokens", str(args.max_new_tokens),
                "--model", args.blip_model,
            ]
        else:
            # blip_plus → run enrich_captions.py
            cmd = [
                sys.executable, "scripts/enrich_captions.py",
                "--img_dir", str(tmp_imgs),
                "--out_dir", str(cap_dir),
                "--append_style", args.append_style,
                "--batch_size", str(args.batch_size),
                "--model", args.blip_model,
            ]

        rc = run_live(cmd)
        if rc != 0:
            raise SystemExit(rc)

    # Final sanity
    now = len(list(cap_dir.glob("*.txt")))
    p(f"[ensure] Done. Now have {now} caption files at {cap_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
