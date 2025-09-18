# scripts/run_variants.py
import argparse, pathlib, subprocess, sys, shlex, os, json, time

"""
Orchestrator to run captioning -> training -> comparison -> plots
for one or more LoRA variants:
  - vanilla   : uses your existing "clean" captions (style suffix only or your edited files)
  - blip      : auto captions via BLIP
  - blip_plus : auto captions via BLIP + enrichment

It expects your existing scripts:
  scripts/preprocess.py
  scripts/auto_caption_blip.py
  scripts/enrich_captions.py
  scripts/clean_captions.py
  scripts/train_lora.py
  scripts/compare_lora_infer.py
  scripts/plot_training.py

Training uses your train_lora.py (with val split, eval CSV, early stop, save-every-N).
"""

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

# -------- helpers --------
def run(cmd, cwd=None):
    print(f"\n[run] {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# -------- pipeline steps --------
def step_preprocess(raw_dir, out_dir, size):
    # Optional: Use if you want to re-make square PNGs
    cmd = f"{sys.executable} {SCRIPTS/'preprocess.py'} --in_dir \"{raw_dir}\" --out_dir \"{out_dir}\" --size {size}"
    run(cmd)

def step_caption_blip(img_dir, cap_dir):
    cmd = f'{sys.executable} {SCRIPTS/"auto_caption_blip.py"} --img_dir "{img_dir}" --out_dir "{cap_dir}"'
    run(cmd)

def step_enrich_captions(img_dir, out_dir):
    cmd = f'{sys.executable} {SCRIPTS/"enrich_captions.py"} --img_dir "{img_dir}" --out_dir "{out_dir}"'
    run(cmd)

def step_clean_captions(img_dir, cap_root, style_suffix):
    cmd = f'{sys.executable} {SCRIPTS/"clean_captions.py"} --img_dir "{img_dir}" --out_dir "{cap_root}" --append_style "{style_suffix}"'
    run(cmd)


def step_train(img_dir, cap_dir, out_dir, steps, batch, grad_accum, save_every, eval_every, rank, alpha, seed):
    cmd = (
        f"{sys.executable} {SCRIPTS/'train_lora.py'} "
        f"--data_images \"{img_dir}\" --data_captions \"{cap_dir}\" "
        f"--output_dir \"{out_dir}\" "
        f"--max_steps {steps} --batch_size {batch} --grad_accum {grad_accum} "
        f"--save_every {save_every} --eval_every {eval_every} "
        f"--rank {rank} --alpha {alpha} --seed {seed}"
    )
    run(cmd)

def step_compare(prompt, vanilla_dir, vanilla_weight, blip_dir=None, blip_weight=None, blip_plus_dir=None, blip_plus_weight=None, outdir=None):
    cmd = f"{sys.executable} {SCRIPTS/'compare_lora_infer.py'} --prompt {shlex.quote(prompt)} --vanilla_lora_dir \"{vanilla_dir}\" --vanilla_lora_weight \"{vanilla_weight}\""
    if blip_dir and blip_weight:
        cmd += f" --blip_lora_dir \"{blip_dir}\" --blip_lora_weight \"{blip_weight}\""
    if blip_plus_dir and blip_plus_weight:
        cmd += f" --blip_plus_lora_dir \"{blip_plus_dir}\" --blip_plus_lora_weight \"{blip_plus_weight}\""
    if outdir:
        ensure_dir(pathlib.Path(outdir))
        cmd += f" --outdir \"{outdir}\""
    run(cmd)

def step_plot(csv_path):
    cmd = f"{sys.executable} {SCRIPTS/'plot_training.py'} --csv \"{csv_path}\""
    run(cmd)

# -------- orchestrate --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tattoo_v3")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--style_suffix", default=", clean line tattoo, high contrast, stencil, no shading")

    ap.add_argument("--variants", nargs="+", default=["vanilla", "blip", "blip_plus"], choices=["vanilla","blip","blip_plus"])
    ap.add_argument("--do_preprocess", action="store_true", help="Rebuild PNGs from raw")
    ap.add_argument("--do_caption", action="store_true", help="Recompute captions for requested variants")
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_compare", action="store_true")
    ap.add_argument("--do_plot", action="store_true")

    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--prompt", default="owl, minimal line-art tattoo, clean lines, high-contrast, no shading, stencil")
    args = ap.parse_args()

    # Paths
    data = REPO / "data" / "processed" / args.dataset
    raw_dir = REPO / "data" / "raw" / args.dataset
    vanilla_dir = data / "vanilla"
    blip_dir    = data / "blip"
    blipp_dir   = data / "blip_plus"

    out_base = REPO / "runs" / "lora"
    out_van  = out_base / f"{args.dataset}_vanilla_r{args.rank}a{args.alpha}"
    out_blip = out_base / f"{args.dataset}_blip_r{args.rank}a{args.alpha}"
    out_blpp = out_base / f"{args.dataset}_blip_plus_r{args.rank}a{args.alpha}"

    # 1) Preprocess
    if args.do_preprocess:
        prep_out = data / "processed_pngs"
        prep_out.mkdir(parents=True, exist_ok=True)
        step_preprocess(raw_dir, prep_out, args.image_size)

    # 2) Captioning per variant (only if requested)
    if args.do_caption:
        if "vanilla" in args.variants:
            imgd = vanilla_dir / "images"; capd = vanilla_dir / "captions"
            ensure_dir(imgd); ensure_dir(capd)
            # If you already have preprocessed images, just copy there or set vanilla_dir/images beforehand.
            # Here we only create clean captions with a style suffix:
            step_clean_captions(imgd, capd, args.style_suffix)

        if "blip" in args.variants:
            imgd = blip_dir / "images"; capd = blip_dir / "captions"
            ensure_dir(imgd); ensure_dir(capd)
            step_caption_blip(imgd, capd)

        if "blip_plus" in args.variants:
            imgd = blipp_dir / "images"; capd = blipp_dir / "captions"
            ensure_dir(imgd); ensure_dir(capd)
            tmp_cap = blipp_dir / "captions_blip"
            ensure_dir(tmp_cap)
            step_caption_blip(imgd, tmp_cap)            # BLIP
            step_enrich_captions(tmp_cap, capd)         # + enrichment

    # 3) Train per variant
    if args.do_train:
        if "vanilla" in args.variants:
            step_train(
                img_dir=str(vanilla_dir / "images"),
                cap_dir=str(vanilla_dir / "captions"),
                out_dir=str(out_van),
                steps=args.max_steps, batch=args.batch_size, grad_accum=args.grad_accum,
                save_every=args.save_every, eval_every=args.eval_every,
                rank=args.rank, alpha=args.alpha, seed=args.seed
            )
        if "blip" in args.variants:
            step_train(
                img_dir=str(blip_dir / "images"),
                cap_dir=str(blip_dir / "captions"),
                out_dir=str(out_blip),
                steps=args.max_steps, batch=args.batch_size, grad_accum=args.grad_accum,
                save_every=args.save_every, eval_every=args.eval_every,
                rank=args.rank, alpha=args.alpha, seed=args.seed
            )
        if "blip_plus" in args.variants:
            step_train(
                img_dir=str(blipp_dir / "images"),
                cap_dir=str(blipp_dir / "captions"),
                out_dir=str(out_blpp),
                steps=args.max_steps, batch=args.batch_size, grad_accum=args.grad_accum,
                save_every=args.save_every, eval_every=args.eval_every,
                rank=args.rank, alpha=args.alpha, seed=args.seed
            )

    # 4) Compare images
    if args.do_compare:
        v_weight = "sd15_lora_r{}_a{}_final.safetensors".format(args.rank, args.alpha)
        b_weight = "sd15_lora_r{}_a{}_final.safetensors".format(args.rank, args.alpha)
        p_weight = "sd15_lora_r{}_a{}_final.safetensors".format(args.rank, args.alpha)

        outdir = REPO / "runs" / "samples" / f"compare_{int(time.time())}"
        va = str(out_van); vb = str(out_blip) if "blip" in args.variants else None
        vp = str(out_blpp) if "blip_plus" in args.variants else None
        step_compare(
            prompt=args.prompt,
            vanilla_dir=va, vanilla_weight=v_weight,
            blip_dir=vb, blip_weight=b_weight if vb else None,
            blip_plus_dir=vp, blip_plus_weight=p_weight if vp else None,
            outdir=str(outdir)
        )

    # 5) Plot latest CSVs for each variant (simple heuristic: pick the most recent CSV in runs/logs/)
    if args.do_plot:
        logs_dir = REPO / "runs" / "logs"
        if logs_dir.exists():
            for variant_name, out_dir in [
                ("vanilla", out_van),
                ("blip", out_blip),
                ("blip_plus", out_blpp),
            ]:
                if variant_name in args.variants:
                    # Find a CSV that contains the out_dir name in filename
                    cands = sorted([p for p in logs_dir.glob("*.csv") if out_dir.name in p.name], key=lambda p: p.stat().st_mtime)
                    if cands:
                        step_plot(str(cands[-1]))
        else:
            print("[plot] No logs found under runs/logs")

    print("\n[done] Orchestration complete.\n")


if __name__ == "__main__":
    main()
