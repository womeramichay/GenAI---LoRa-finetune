# main.py
from __future__ import print_function

import sys, subprocess
from pathlib import Path

# =========================
# USER CONFIG
# =========================
ROOT = Path(__file__).resolve().parent

# Dataset tag and size
DATASET_NAME = "tattoo_v3"
IMAGE_SIZE   = 512

# Which variants to run (order matters)
# Choose any: "vanilla", "blip", "blip_plus"
VARIANTS = ["blip"]  # e.g. ["vanilla", "blip", "blip_plus"]

# Training hyper-params (shared)
RANK = 8
ALPHA = 8
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_STEPS  = 500
SAVE_EVERY = 50
EVAL_EVERY = 50
EARLY_STOP_PATIENCE = 4

# Optional resume per-variant (set to None to disable)
RESUME = {
    "vanilla": None,
    "blip": None,
    "blip_plus": None,
    # Example:
    # "vanilla": {
    #     "dir": r"C:\Users\Omer\tattoo-genai\runs\lora\tattoo_v3_vanilla_r8a8",
    #     "weight": "sd15_lora_r8_a8_step50.safetensors",
    #     "step": 50,
    # },
}

# =========================
# PROJECT PATHS
# =========================
PY          = sys.executable
SCRIPTS     = ROOT / "scripts"

RAW_DIR     = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed" / DATASET_NAME
IMAGES_DIR  = PROCESSED / "images"

# Caption dirs
CAP_VANILLA = IMAGES_DIR                # vanilla uses default caption inside train script
CAP_BLIP    = PROCESSED / "captions_blip"
CAP_BLIP_P  = PROCESSED / "captions_blip_plus"

# temp folder used by blip_plus (BLIP baseline first)
TMP_BLIP    = PROCESSED / "_tmp_blip"

# Output (weights) per variant
RUNS_DIR    = ROOT / "runs" / "lora"
OUT_VAN     = RUNS_DIR / f"{DATASET_NAME}_vanilla_r{RANK}a{ALPHA}"
OUT_BLIP    = RUNS_DIR / f"{DATASET_NAME}_blip_r{RANK}a{ALPHA}"
OUT_BLIP_P  = RUNS_DIR / f"{DATASET_NAME}_blip_plus_r{RANK}a{ALPHA}"

# =========================
# HELPERS
# =========================
def echo(x: str) -> None:
    print(x, flush=True)

def run(cmd, cwd: Path = None):
    """Run a subprocess and stream output; raise on error."""
    echo(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def count_pngs(p: Path) -> int:
    return sum(1 for _ in p.glob("*.png"))

# =========================
# PREPROCESS (images)
# =========================
def ensure_preprocessed_images() -> None:
    """If processed image count != raw count, (re)build processed images."""
    raw_count = count_pngs(RAW_DIR)
    proc_count = count_pngs(IMAGES_DIR)
    if raw_count == 0:
        raise RuntimeError(f"No PNGs found in RAW_DIR: {RAW_DIR}")
    if raw_count != proc_count:
        echo(f"[preprocess] Rebuilding – count mismatch (raw={raw_count}, processed={proc_count}).")
        (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
        run([
            PY, str(SCRIPTS / "preprocess.py"),
            "--dataset_name", DATASET_NAME,
            "--variant", "images",
            "--input_dir", str(RAW_DIR),
            "--size", str(IMAGE_SIZE),
            "--out_root", str(ROOT / "data" / "processed"),
            "--images_only",
        ], cwd=ROOT)
    else:
        echo(f"[preprocess] Skipped – processed image count matches raw: {proc_count}")

# =========================
# CAPTIONS (per-variant)
# =========================
def ensure_captions_vanilla() -> None:
    echo("[captions] Vanilla: using default caption inside the training script (no .txt files).")

def ensure_captions_blip() -> None:
    if CAP_BLIP.exists() and any(CAP_BLIP.glob("*.txt")):
        echo("[captions] BLIP: found existing captions — skipping.")
        return
    echo("[captions] BLIP: generating captions…")
    CAP_BLIP.mkdir(parents=True, exist_ok=True)
    # NOTE: auto_caption_blip.py expects --img_dir and --batch_size
    run([
        PY, str(SCRIPTS / "auto_caption_blip.py"),
        "--img_dir", str(IMAGES_DIR),
        "--out_dir", str(CAP_BLIP),
        "--batch_size", "8",
    ], cwd=ROOT)

def ensure_captions_blip_plus() -> None:
    if CAP_BLIP_P.exists() and any(CAP_BLIP_P.glob("*.txt")):
        echo("[captions] BLIP+: found existing captions — skipping.")
        return

    # 1) Baseline BLIP to TMP_BLIP (if not already there)
    if not (TMP_BLIP.exists() and any(TMP_BLIP.glob("*.txt"))):
        echo("[captions] BLIP+: baseline BLIP first…")
        TMP_BLIP.mkdir(parents=True, exist_ok=True)
        run([
            PY, str(SCRIPTS / "auto_caption_blip.py"),
            "--img_dir", str(IMAGES_DIR),
            "--out_dir", str(TMP_BLIP),
            "--batch_size", "8",
        ], cwd=ROOT)

    # 2) Enrich into CAP_BLIP_P
    echo("[captions] BLIP+: enriching captions…")
    CAP_BLIP_P.mkdir(parents=True, exist_ok=True)
    # enrich_captions.py typically expects --in_dir/--out_dir and optional style suffix
    run([
        PY, str(SCRIPTS / "enrich_captions.py"),
        "--in_dir", str(TMP_BLIP),
        "--out_dir", str(CAP_BLIP_P),
        "--style_suffix", ", clean line tattoo, high contrast, stencil, no shading",
    ], cwd=ROOT)

# =========================
# TRAIN (per-variant)
# =========================
def build_train_cmd(images_dir: Path, captions_dir: Path, out_dir: Path, resume_cfg: dict = None):
    cmd = [
        PY, str(SCRIPTS / "train_lora.py"),
        "--data_images", str(images_dir),
        "--data_captions", str(captions_dir),
        "--output_dir", str(out_dir),
        "--rank", str(RANK),
        "--alpha", str(ALPHA),
        "--batch_size", str(BATCH_SIZE),
        "--grad_accum", str(GRAD_ACCUM),
        "--max_steps", str(MAX_STEPS),
        "--save_every", str(SAVE_EVERY),
        "--eval_every", str(EVAL_EVERY),
        "--early_stop_patience", str(EARLY_STOP_PATIENCE),
    ]
    if resume_cfg:
        if resume_cfg.get("dir") and resume_cfg.get("weight"):
            cmd += [
                "--resume_from_dir", str(resume_cfg["dir"]),
                "--resume_weight_name", str(resume_cfg["weight"]),
            ]
        if resume_cfg.get("step") is not None:
            cmd += ["--resume_step", str(int(resume_cfg["step"]))]
    return cmd

def train_vanilla() -> None:
    out_dir = OUT_VAN
    out_dir.mkdir(parents=True, exist_ok=True)
    echo(f"[train] vanilla → {out_dir}")
    cmd = build_train_cmd(IMAGES_DIR, CAP_VANILLA, out_dir, RESUME.get("vanilla"))
    run(cmd, cwd=ROOT)

def train_blip() -> None:
    out_dir = OUT_BLIP
    out_dir.mkdir(parents=True, exist_ok=True)
    echo(f"[train] blip → {out_dir}")
    cmd = build_train_cmd(IMAGES_DIR, CAP_BLIP, out_dir, RESUME.get("blip"))
    run(cmd, cwd=ROOT)

def train_blip_plus() -> None:
    out_dir = OUT_BLIP_P
    out_dir.mkdir(parents=True, exist_ok=True)
    echo(f"[train] blip_plus → {out_dir}")
    cmd = build_train_cmd(IMAGES_DIR, CAP_BLIP_P, out_dir, RESUME.get("blip_plus"))
    run(cmd, cwd=ROOT)

# =========================
# PUBLIC RUNNERS (easy to call individually)
# =========================
def run_vanilla():
    ensure_preprocessed_images()
    ensure_captions_vanilla()
    train_vanilla()

def run_blip():
    ensure_preprocessed_images()
    ensure_captions_blip()
    train_blip()

def run_blip_plus():
    ensure_preprocessed_images()
    ensure_captions_blip_plus()
    train_blip_plus()

# =========================
# MAIN (multi-variant)
# =========================
def main():
    print("=== tattoo-genai: main runner ===")
    print("ROOT:", ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("Variants to run:", VARIANTS)

    # 1) ensure images once
    ensure_preprocessed_images()

    # 2) captions + train per variant
    for v in VARIANTS:
        if v == "vanilla":
            ensure_captions_vanilla()
            train_vanilla()
        elif v == "blip":
            ensure_captions_blip()
            train_blip()
        elif v == "blip_plus":
            ensure_captions_blip_plus()
            train_blip_plus()
        else:
            raise ValueError(f"Unknown variant in VARIANTS: {v}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
