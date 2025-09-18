#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main runner: pick a variant (vanilla | blip | blip_plus), ensure captions for that variant,
optionally preprocess, then train. Windows-safe printing of the exact commands it runs.

Variants:
  - vanilla   -> use .txt captions next to images (or --vanilla_autofill to generate basic ones)
  - blip      -> ensure BLIP captions in processed/<dataset>/captions_blip/
  - blip_plus -> ensure enriched BLIP captions in processed/<dataset>/captions_blip_plus/

Examples:
  python main.py --list_variants
  python main.py --variant vanilla
  python main.py --variant blip
  python main.py --variant blip_plus --fresh
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------------------
# Variant presets (safe for ~6–8 GB GPUs)
# ---------------------------
VARIANTS = {
    # "vanilla":     dict(rank=4, alpha=8, resolution=384, batch_size=1, grad_accum=16, max_steps=600, caption_mode="vanilla"),
    "blip":        dict(rank=4, alpha=8, resolution=384, batch_size=1, grad_accum=16, max_steps=1000, caption_mode="blip"),
    "blip_plus":   dict(rank=4, alpha=8, resolution=384, batch_size=1, grad_accum=16, max_steps=1000, caption_mode="blip_plus"),
    # you can add more here (e.g., res448, lowmem, etc.)
}

# ---------------------------
# Defaults (paths)
# ---------------------------
DATASET_NAME_DEFAULT = "tattoo_v3_subset2000"
RAW_ROOT_DEFAULT      = Path("data/raw")
PROCESSED_ROOT_BASE   = Path("data/processed")
RUNS_ROOT             = Path("runs/lora")

# ---------------------------
# Helpers
# ---------------------------
def p(*a, **k):
    print(*a, **k, flush=True)

def run_live(cmd_args, env=None):
    """Run a command with live stdout; return code is returned."""
    p("\n>>>", " ".join(f'"{a}"' if (" " in str(a)) else str(a) for a in cmd_args))
    proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    return proc.returncode

def count_in_dir(path: Path, exts):
    if not path.exists():
        return 0
    total = 0
    for ext in exts:
        total += len(list(path.glob(f"*{ext}")))
    return total

def choose_run_dir(base_dir: Path, want_name: str, fresh: bool) -> Path:
    """
    If fresh=False and directory exists, reuse (resume).
    If fresh=True and directory exists, create unique suffix: _v2, _v3, ...
    """
    rd = base_dir / want_name
    if not rd.exists():
        return rd
    if not fresh:
        return rd
    i = 2
    while True:
        rd2 = base_dir / f"{want_name}_v{i}"
        if not rd2.exists():
            return rd2
        i += 1

def save_run_meta(run_dir: Path, payload: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Variant runner with caption check and training.")
    # dataset & dirs
    ap.add_argument("--dataset", type=str, default=DATASET_NAME_DEFAULT, help="Dataset name (used in processed paths).")
    ap.add_argument("--img_dir", type=str, default=None, help="Override: directory with images.")
    ap.add_argument("--cap_dir", type=str, default=None, help="Override: directory with captions (.txt).")
    ap.add_argument("--raw_root", type=str, default=str(RAW_ROOT_DEFAULT), help="Raw data root (for optional preprocess).")
    ap.add_argument("--processed_root_base", type=str, default=str(PROCESSED_ROOT_BASE), help="Processed data root base.")
    ap.add_argument("--skip_preprocess", action="store_true", help="Skip preprocess step.")
    ap.add_argument("--preprocess_size", type=int, default=384, help="Resize used in scripts/preprocess.py when needed.")

    # captions (BLIP / BLIP+)
    ap.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP model for auto caption.")
    ap.add_argument("--blip_batch_size", type=int, default=2)
    ap.add_argument("--blip_max_new_tokens", type=int, default=32)
    ap.add_argument("--blip_plus_append", type=str, default=", clean line tattoo, high contrast, stencil, no shading")

    # vanilla fallback (only if you want it to auto-fill when .txt are missing)
    ap.add_argument("--vanilla_autofill", action="store_true",
                    help="If vanilla has missing captions, auto-create basic captions from filenames.")

    # training / variant
    ap.add_argument("--variant", type=str, default="vanilla", help="Variant key (see --list_variants).")
    ap.add_argument("--list_variants", action="store_true", help="List available variants and exit.")
    ap.add_argument("--fresh", action="store_true", help="Force new run directory (do not resume).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--eval_steps_infer", type=int, default=16)
    ap.add_argument("--eval_guidance", type=float, default=6.0)
    ap.add_argument("--eval_width", type=int, default=None, help="If not set, equals resolution.")
    ap.add_argument("--eval_height", type=int, default=None, help="If not set, equals resolution.")

    args = ap.parse_args()

    if args.list_variants:
        p("Available variants:")
        for k, v in VARIANTS.items():
            p(f"  {k}: {v}")
        sys.exit(0)

    if args.variant not in VARIANTS:
        raise SystemExit(f"Unknown variant '{args.variant}'. Use --list_variants to see options.")

    var = VARIANTS[args.variant]
    eval_w = args.eval_width or var["resolution"]
    eval_h = args.eval_height or var["resolution"]

    # Resolve dirs
    processed_root = Path(args.processed_root_base) / args.dataset
    default_img_dir = processed_root / "images"
    img_dir = Path(args.img_dir).resolve() if args.img_dir else default_img_dir.resolve()

    # Variant-specific caption location
    if args.cap_dir:
        cap_dir = Path(args.cap_dir).resolve()
    else:
        if var["caption_mode"] == "vanilla":
            cap_dir = img_dir  # .txt next to .png
        elif var["caption_mode"] == "blip":
            cap_dir = (processed_root / "captions_blip").resolve()
        else:  # blip_plus
            cap_dir = (processed_root / "captions_blip_plus").resolve()

    p("[main] Config")
    p(f"  dataset    = {args.dataset}")
    p(f"  img_dir    = {img_dir}")
    p(f"  cap_dir    = {cap_dir}")
    p(f"  variant    = {args.variant} -> {var}")

    # ---------------------------
    # Optional preprocess (only if counts differ)
    # ---------------------------
    if not args.skip_preprocess:
        raw_dir = Path(args.raw_root)
        raw_n  = count_in_dir(raw_dir, [".png", ".jpg", ".jpeg", ".webp"])
        proc_n = count_in_dir(img_dir, [".png", ".jpg", ".jpeg", ".webp"])
        p(f"[check] raw={raw_n} | processed={proc_n} @ {img_dir}")
        if raw_n != proc_n:
            pre_script = Path("scripts/preprocess.py")
            if pre_script.exists():
                p("[preprocess] counts differ -> running scripts/preprocess.py")
                cmd = [
                    sys.executable, str(pre_script),
                    "--dataset_name", args.dataset,
                    "--input_dir", str(raw_dir),
                    "--size", str(args.preprocess_size),
                    "--out_root", str(Path(args.processed_root_base)),
                    "--images_only",
                ]
                rc = run_live(cmd)
                if rc != 0:
                    raise RuntimeError("preprocess.py failed")
            else:
                p("[preprocess] scripts/preprocess.py not found; skipping.")
        else:
            p("[preprocess] up-to-date; skipping.")
    else:
        p("[preprocess] Skipped by flag.")

    # ---------------------------
    # Ensure captions for the chosen variant
    # ---------------------------
    ensure_script = Path("scripts/ensure_captions.py")
    if var["caption_mode"] in ("blip", "blip_plus"):
        if not ensure_script.exists():
            raise FileNotFoundError("scripts/ensure_captions.py not found.")
        cmd_ec = [
            sys.executable, str(ensure_script),
            "--mode", var["caption_mode"],
            "--img_dir", str(img_dir),
            "--cap_dir", str(cap_dir),
            "--blip_model", args.blip_model,
            "--batch_size", str(args.blip_batch_size),
            "--max_new_tokens", str(args.blip_max_new_tokens),
            "--append_style", args.blip_plus_append,
        ]
        rc = run_live(cmd_ec)
        if rc != 0:
            raise RuntimeError("ensure_captions.py failed")
    else:
        # vanilla: verify we have captions (or make basic ones if asked)
        pngs = sorted([p for p in img_dir.glob("*.png")])
        txts = sorted([p for p in img_dir.glob("*.txt")])
        if len(pngs) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        if len(txts) < len(pngs):
            if args.vanilla_autofill:
                p("[vanilla] Missing captions detected -> generating basic captions from filenames.")
                clean_script = Path("scripts/clean_captions.py")
                out_root = processed_root / "vanilla"
                out_root.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, str(clean_script),
                    "--img_dir", str(img_dir),
                    "--out_dir", str(out_root),
                    "--append_style", args.blip_plus_append,
                ]
                rc = run_live(cmd)
                if rc != 0:
                    raise RuntimeError("clean_captions.py failed")
                cap_dir = out_root / "captions"
            else:
                raise RuntimeError(
                    f"Found {len(pngs)} images but only {len(txts)} captions in {img_dir}. "
                    f"Add --vanilla_autofill to auto-create simple captions."
                )

    # ---------------------------
    # Build run dir + train
    # ---------------------------
    tag = f"{args.dataset}_{args.variant}_r{var['rank']}a{var['alpha']}_res{var['resolution']}"
    run_dir = choose_run_dir(RUNS_ROOT, tag, fresh=args.fresh)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_meta(run_dir, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "img_dir": str(img_dir),
        "cap_dir": str(cap_dir),
        "variant": args.variant,
        "variant_params": var,
        "caption_mode": var["caption_mode"],
    })

    train_script = Path("scripts/train_lora.py")
    if not train_script.exists():
        raise FileNotFoundError("scripts/train_lora.py not found.")

    tr = [
        sys.executable, str(train_script),
        "--data_images", str(img_dir),
        "--data_captions", str(cap_dir),
        "--output_dir", str(run_dir),
        "--rank", str(var["rank"]),
        "--alpha", str(var["alpha"]),
        "--resolution", str(var["resolution"]),
        "--max_steps", str(var["max_steps"]),
        "--batch_size", str(var["batch_size"]),
        "--grad_accum", str(var["grad_accum"]),
        "--val_split", str(0.05),
        "--save_every", str(args.save_every),
        "--eval_every", str(args.eval_every),
        "--eval_steps_infer", str(args.eval_steps_infer),
        "--eval_guidance", str(args.eval_guidance),
        "--eval_width", str(eval_w),
        "--eval_height", str(eval_h),
        "--seed", str(args.seed),
        "--eval_clip",
    ]
    # resume if weights exist
    if not args.fresh:
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            tr += ["--resume_from_dir", str(weights_dir)]

    p(f"\n[train] {args.variant} -> {run_dir}  (resume={'False' if args.fresh else 'True'})")
    rc = run_live(tr)
    if rc != 0:
        raise RuntimeError(f"Training failed with code {rc}")
    p("\n[done] Training completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        p(f"\n[ERROR] {e}")
        sys.exit(1)
