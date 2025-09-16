#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-button runner for the Tattoo LoRA project.

- Idempotent: only does missing steps (preprocess → captions → train → compare).
- Streams child process output so you can see tqdm and prints in PyCharm.
- Storage friendly defaults (few checkpoints, lightweight eval images).

Edit CONFIG below, then run this file.
"""

from __future__ import annotations
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# =========================
# EDIT ME (your defaults)
# =========================
USE_CONFIG = True  # True = use CONFIG; False = use CLI (not needed for PyCharm)

@dataclass
class Config:
    # Data
    raw_dir: str = "data/raw"                    # folder with your source PNGs
    dataset_name: str = "tattoo_v3_subset2000"   # under data/processed/<dataset_name>
    size: int = 512                              # preprocess square size

    # Which variants to train
    variants: List[str] = None                   # ["vanilla", "blip", "blip_plus"]

    # LoRA / training hyperparams
    rank: int = 8
    alpha: int = 8
    max_steps: int = 600
    batch_size: int = 1
    grad_accum: int = 16
    save_every: int = 1000       # large => keeps only best/final in practice
    eval_every: int = 100

    # Eval image settings (VRAM-friendly)
    eval_steps: int = 16
    eval_guidance: float = 6.0
    eval_w: int = 448
    eval_h: int = 448
    eval_clip: bool = True       # set False if you want to avoid CLIP download

    # Optional: train text encoder LoRA too
    train_text_encoder: bool = False

    # Optional side-by-side compare at the end
    do_compare: bool = False#True
    prompt: str = "owl, clean line, minimal line-art tattoo, stencil, high contrast, no shading"
    compare_steps: int = 18
    compare_guidance: float = 6.5
    compare_w: int = 448
    compare_h: int = 448

    # Quick skips
    skip_preprocess: bool = False
    skip_caption: bool = False   # skips BLIP/BLIP+ generation (not recommended on first run)
    skip_train: bool = False

DEFAULT_VARIANTS = ["vanilla"]#, "blip", "blip_plus"]
CONFIG = Config(variants=DEFAULT_VARIANTS.copy())

# =========================
# Internals
# =========================
REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"

def run(cmd, cwd=REPO) -> None:
    """Run a command and stream its output live, robust to Unicode on Windows."""
    import os, io, sys, subprocess
    print("\n>>>", " ".join(str(c) for c in cmd))

    # Force UTF-8 from the child; if not, we still 'replace' undecodable bytes.
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    # Decode as UTF-8; if the child prints something else, don't crash—replace the char.
    with io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            print(line, end="")  # tqdm-friendly
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with code {rc}")


def count_files(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern))) if d and d.exists() else 0

def best_or_final_exists(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    return any(run_dir.glob("*best.safetensors")) or any(run_dir.glob("*final.safetensors"))

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def discover_variants(variants: Optional[List[str]]) -> List[str]:
    allowed = {"vanilla", "blip", "blip_plus"}
    out, seen = [], set()
    for v in (variants or DEFAULT_VARIANTS):
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"Unknown variant '{v}'. Allowed: {sorted(allowed)}")
        if v not in seen:
            out.append(v); seen.add(v)
    return out

# =========================
# Pipeline steps
# =========================
def maybe_preprocess(raw_dir: Path, processed_root: Path, size: int = 512) -> None:
    """
    If processed_root/images/*.png count differs from raw_dir/*.png, run scripts/preprocess.py.

    preprocess.py signature (current):
      --dataset_name <name> --input_dir <raw_images_dir> [--size 512] [--out_root data/processed] [--images_only]
    It writes into: <out_root>/<dataset_name>/images
    """
    dataset_name = processed_root.name
    raw_pngs = count_files(raw_dir, "*.png")
    proc_dir = processed_root / "images"
    proc_pngs = count_files(proc_dir, "*.png")

    print(f"[check] raw={raw_pngs} | processed={proc_pngs} @ {proc_dir}")
    if raw_pngs == 0:
        raise FileNotFoundError(f"No PNGs in raw dir: {raw_dir}")

    if proc_pngs != raw_pngs:
        print("[preprocess] counts differ -> running scripts/preprocess.py")
        ensure_dir(proc_dir)  # ensure parent exists
        cmd = [
            sys.executable, str(SCRIPTS / "preprocess.py"),
            "--dataset_name", dataset_name,
            "--input_dir", str(raw_dir),
            "--size", str(size),
            "--out_root", str(processed_root.parent),  # typically data/processed
            "--images_only",
        ]
        run(cmd)
    else:
        print("[preprocess] up-to-date; skipping.")

def maybe_caption_blip(processed_root: Path) -> Path:
    img_dir = processed_root / "images"
    out_dir = processed_root / "captions_blip"
    img_n = count_files(img_dir, "*.png")
    cap_n = count_files(out_dir, "*.txt")
    if cap_n != img_n:
        print("[caption] BLIP baseline -> auto_caption_blip.py")
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(SCRIPTS / "auto_caption_blip.py"),
            "--img_dir", str(img_dir),
            "--out_dir", str(out_dir),
        ]
        run(cmd)
    else:
        print("[caption] BLIP up-to-date.")
    return out_dir

def maybe_caption_blip_plus(processed_root: Path) -> Path:
    img_dir = processed_root / "images"
    out_dir = processed_root / "captions_blip_plus"
    img_n = count_files(img_dir, "*.png")
    cap_n = count_files(out_dir, "*.txt")
    if cap_n != img_n:
        print("[caption] BLIP+ enriched -> enrich_captions.py")
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(SCRIPTS / "enrich_captions.py"),
            "--img_dir", str(img_dir),
            "--out_dir", str(out_dir),
        ]
        run(cmd)
    else:
        print("[caption] BLIP+ up-to-date.")
    return out_dir

def train_variant(
    variant: str,
    processed_root: Path,
    runs_root: Path,
    cfg: Config,
) -> Path:
    img_dir = processed_root / "images"
    if variant == "vanilla":
        # pass a dir with NO .txt files so trainer uses built-in caption
        cap_dir = img_dir
    elif variant == "blip":
        cap_dir = maybe_caption_blip(processed_root)
    elif variant == "blip_plus":
        cap_dir = maybe_caption_blip_plus(processed_root)
    else:
        raise ValueError(variant)

    run_dir = runs_root / f"{processed_root.name}_{variant}_r{cfg.rank}a{cfg.alpha}"
    if best_or_final_exists(run_dir):
        print(f"[train] found weights in {run_dir} -> skipping.")
        return run_dir

    ensure_dir(run_dir)
    cmd = [
        sys.executable, str(SCRIPTS / "train_lora.py"),
        "--data_images", str(img_dir),
        "--data_captions", str(cap_dir),
        "--output_dir", str(run_dir),
        "--rank", str(cfg.rank), "--alpha", str(cfg.alpha),
        "--max_steps", str(cfg.max_steps),
        "--batch_size", str(cfg.batch_size),
        "--grad_accum", str(cfg.grad_accum),
        "--save_every", str(cfg.save_every),
        "--eval_every", str(cfg.eval_every),
        "--eval_steps_infer", str(cfg.eval_steps),
        "--eval_guidance", str(cfg.eval_guidance),
        "--eval_width", str(cfg.eval_w),
        "--eval_height", str(cfg.eval_h),
    ]
    if cfg.eval_clip:
        cmd.append("--eval_clip")
    if cfg.train_text_encoder:
        cmd.append("--train_text_encoder")

    print(f"[train] {variant} -> {run_dir}")
    run(cmd)
    return run_dir

def maybe_compare(prompt: str, variant_dirs: List[Path], out_dir: Path,
                  steps: int, guidance: float, width: int, height: int) -> None:
    if not variant_dirs:
        print("[compare] no variants; skipping.")
        return
    ensure_dir(out_dir)
    cmd = [
        sys.executable, str(SCRIPTS / "compare_lora_infer.py"),
        "--prompt", prompt,
        "--outdir", str(out_dir),
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--width", str(width),
        "--height", str(height),
    ]
    # compare script supports up to 3 slots; map in order
    slots = ["--vanilla_lora_dir", "--blip_lora_dir", "--blip_plus_lora_dir"]
    for i, d in enumerate(variant_dirs[:3]):
        cmd += [slots[i], str(d)]
    print(f"[compare] -> {out_dir}")
    run(cmd)

# =========================
# Entrypoint (PyCharm-friendly)
# =========================
def main():
    if USE_CONFIG:
        cfg = CONFIG
        print("[main] Using inline CONFIG (edit at top).")
    else:
        # If you ever want CLI parsing again, you can add it here.
        cfg = CONFIG
        print("[main] Using defaults (no CLI implemented in this mode).")

    raw_dir = Path(cfg.raw_dir)
    processed_root = Path("data/processed") / cfg.dataset_name
    runs_root = Path("runs/lora")
    samples_root = Path("runs/samples")

    # 1) Preprocess
    if not cfg.skip_preprocess:
        maybe_preprocess(raw_dir, processed_root, size=cfg.size)
    else:
        print("[skip] preprocess")

    # 2) Train each variant
    trained_dirs: List[Path] = []
    variants = discover_variants(cfg.variants)
    for v in variants:
        if cfg.skip_caption and v in ("blip", "blip_plus"):
            print(f"[skip] caption for {v}")
            # still need a captions path; fallback to images dir (behaves like vanilla)
        if not cfg.skip_train:
            rd = train_variant(v, processed_root, runs_root, cfg)
            trained_dirs.append(rd)
        else:
            print(f"[skip] training for {v}")

    # 3) Compare (optional)
    if cfg.do_compare:
        # If run in a fresh session with no new training, try to auto-collect existing dirs
        if not trained_dirs:
            trained_dirs = [
                d for d in runs_root.glob(f"{processed_root.name}_*_r{cfg.rank}a{cfg.alpha}") if d.is_dir()
            ]
        out = samples_root / f"compare_{cfg.dataset_name}_r{cfg.rank}a{cfg.alpha}"
        maybe_compare(
            prompt=cfg.prompt,
            variant_dirs=trained_dirs,
            out_dir=out,
            steps=cfg.compare_steps,
            guidance=cfg.compare_guidance,
            width=cfg.compare_w,
            height=cfg.compare_h,
        )

    print("\nAll done ✨")

if __name__ == "__main__":
    main()
