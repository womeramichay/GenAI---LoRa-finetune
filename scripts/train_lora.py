#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning for SD1.5 UNet using PEFT backend (diffusers ≥0.29).

Highlights:
- AMP + grad checkpointing (VRAM friendly for ~6GB GPUs)
- Grad accumulation defaults to 32 to reduce loss jitter
- Saves best checkpoint by val loss (weights/best.safetensors)
- Optional early stopping (--early_stop_patience)
- Logs to runs/logs/<run_name>/metrics.csv and auto-saves loss plot

Usage (vanilla captions next to images):
  python scripts/train_lora.py ^
    --data_images  "data\\processed\\tattoo_v3_subset2000\\images" ^
    --data_captions "data\\processed\\tattoo_v3_subset2000\\images" ^
    --output_dir   "runs\\lora\\tattoo_v3_subset2000_vanilla_r4a8_res384_s500"

Usage (BLIP+ captions):
  python scripts/train_lora.py ^
    --data_images  "data\\processed\\tattoo_v3_subset2000\\images" ^
    --data_captions "data\\processed\\tattoo_v3_subset2000\\captions_blip_plus" ^
    --output_dir   "runs\\lora\\tattoo_v3_subset2000_blip_plus_r4a8_res384_s500"
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer

try:
    from peft import LoraConfig
except Exception:
    raise SystemExit(
        "PEFT is required. Please run:  pip install peft==0.10.0\n"
        "Then re-run this script."
    )

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ------------------
# Dataset
# ------------------
class TattooDataset(Dataset):
    def __init__(self, image_dir: str, captions_dir: str, resolution: int, vanilla_caption: str):
        self.image_dir = Path(image_dir)
        self.captions_dir = Path(captions_dir)
        self.res = resolution
        self.fallback = vanilla_caption

        self.items = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        img = img.crop((left, top, left + m, top + m))
        if m != self.res:
            img = img.resize((self.res, self.res), Image.BICUBIC)
        return img

    def _find_caption(self, img_path: Path) -> str:
        cap = (self.captions_dir / img_path.name).with_suffix(".txt")
        if cap.exists():
            try:
                return cap.read_text(encoding="utf-8").strip()
            except Exception:
                return self.fallback
        return self.fallback

    def __getitem__(self, idx):
        p = self.items[idx]
        img = self._load_image(p)
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, 0..1
        arr = arr.transpose(2, 0, 1)                     # CHW
        arr = (arr * 2.0) - 1.0                          # [-1,1]
        tensor = torch.from_numpy(arr)
        caption = self._find_caption(p)
        return tensor, caption


def collate_fn(batch: List[Tuple[torch.Tensor, str]]):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    caps = [b[1] for b in batch]
    return imgs, caps


# ------------------
# Save / Eval / Plot
# ------------------
def save_lora_safetensors(unet, out_dir: Path, filename: str, meta: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_attn"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    unet.save_attn_procs(str(tmp_dir))
    src = tmp_dir / "pytorch_lora_weights.safetensors"
    dst = out_dir / filename
    if dst.exists():
        dst.unlink()
    src.replace(dst)

    for p in tmp_dir.glob("*"):
        p.unlink()
    tmp_dir.rmdir()

    if meta is not None:
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return str(dst)


@torch.no_grad()
def evaluate_noise_mse(unet, vae, tokenizer, text_encoder, noise_sched, dl, device):
    unet.eval()
    vae.eval()
    text_encoder.eval()
    total = 0.0
    count = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16
    for imgs, caps in dl:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=amp_dtype):
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bsz = lat.shape[0]
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=device)
            eps = torch.randn_like(lat)
            noisy = noise_sched.add_noise(lat, eps, t)

            tokens = tokenizer(
                caps, padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt"
            ).to(device)
            txt = text_encoder(**tokens).last_hidden_state

            pred = unet(noisy, t, encoder_hidden_states=txt).sample
            loss = F.mse_loss(pred.float(), eps.float(), reduction="mean")

        total += loss.item() * bsz
        count += bsz
    return total / max(count, 1)


def plot_loss_csv(csv_path: Path, out_png: Path, title: str):
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        if "train_loss" in df.columns:
            plt.plot(df["step"], df["train_loss"], label="train")
        if "val_loss" in df.columns:
            plt.plot(df["step"], df["val_loss"], label="val")
        plt.xlabel("step"); plt.ylabel("loss (noise MSE)")
        plt.title(title)
        plt.legend()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[plot] failed to save plot: {e}")


# ------------------
# Train
# ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_images", required=True)
    ap.add_argument("--data_captions", required=True)
    ap.add_argument("--output_dir", default="runs/lora/run")
    ap.add_argument("--pretrained", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--train_text_encoder", action="store_true")
    ap.add_argument("--max_steps", type=int, default=500)            # <= your request
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)            # <= your request
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)     # enable clipping by default
    ap.add_argument("--lr_unet", type=float, default=8e-5)           # slightly lower LR helps stability
    ap.add_argument("--lr_text", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", default="runs/logs")
    ap.add_argument("--resume_from_dir", default="")
    ap.add_argument("--eval_steps_infer", type=int, default=16)
    ap.add_argument("--eval_guidance", type=float, default=6.0)
    ap.add_argument("--eval_width", type=int, default=384)
    ap.add_argument("--eval_height", type=int, default=384)
    ap.add_argument("--eval_clip", action="store_true")
    ap.add_argument("--vanilla_caption", type=str, default="minimal line-art tattoo, stencil, high-contrast")
    ap.add_argument("--early_stop_patience", type=int, default=0, help="Stop after this many evals without improvement (0=off)")
    ap.add_argument("--plot_at_end", action="store_true", help="Write loss plot to runs/logs/<run>/loss.png at the end")
    ap.add_argument("--debug_samples", type=int, default=0, help="Print this many random sample captions at start")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    out_dir = Path(args.output_dir)
    weights_dir = out_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    log_dir_base = Path(args.log_dir)
    run_log_dir = log_dir_base / out_dir.name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = run_log_dir / "metrics.csv"
    if not log_csv.exists():
        log_csv.write_text("step,train_loss,val_loss,clip_score\n", encoding="utf-8")

    # Data
    ds = TattooDataset(args.data_images, args.data_captions, args.resolution, args.vanilla_caption)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    print(f"[data] images: {len(ds)} | train: {n_train} | val: {n_val}")
    print(f"[data] captions_dir: {Path(args.data_captions).resolve()}")
    if args.debug_samples > 0:
        idxs = random.sample(range(len(ds)), k=min(args.debug_samples, len(ds)))
        print("[data] sample captions:")
        for i in idxs:
            img_path = ds.items[i]
            cap = ds._find_caption(img_path)
            print(f"  - {img_path.name}: {cap[:120]}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                          collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=min(4, args.batch_size), shuffle=False, num_workers=0, pin_memory=True,
                        collate_fn=collate_fn)

    # Model: load pipeline to grab parts
    print("Loading SD1.5…")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.float16 if use_cuda else torch.float32, safety_checker=None
    )
    pipe.to(device)

    vae = pipe.vae
    tokenizer: CLIPTokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # Scheduler for training objective
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)

    # VRAM helpers
    unet.enable_gradient_checkpointing()
    torch.backends.cuda.matmul.allow_tf32 = True

    # PEFT LoRA adapter on UNet
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_cfg)  # adapter name "default"

    # Freeze base UNet, train only LoRA; force LoRA params to float32 so GradScaler is happy
    for p in unet.parameters():
        p.requires_grad_(False)
    for n, p in unet.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
            if p.dtype != torch.float32:
                p.data = p.data.float()

    # Text encoder optional
    if args.train_text_encoder:
        text_encoder.train()
        for p in text_encoder.parameters():
            p.requires_grad_(True)
    else:
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)

    # Optimizer (only LoRA + maybe text encoder)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": args.lr_unet, "weight_decay": args.weight_decay}]
    if args.train_text_encoder:
        groups.append({"params": text_encoder.parameters(), "lr": args.lr_text, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)

    # Resume: load previously-saved LoRA weights if provided
    if args.resume_from_dir:
        try:
            unet.load_attn_procs(args.resume_from_dir, weight_name="autosave_last.safetensors")
            print(f"[resume] Loaded LoRA from {args.resume_from_dir}/autosave_last.safetensors")
        except Exception as e:
            try:
                unet.load_attn_procs(args.resume_from_dir)
                print(f"[resume] Loaded LoRA from {args.resume_from_dir} (default name)")
            except Exception as e2:
                print(f"[resume] Failed to load from {args.resume_from_dir}: {e} / {e2}")

    # Trackers for best + early stop
    best_val = float("inf")
    bad_evals = 0
    stop_training = False
    best_path = None

    # Train loop
    global_step = 0
    accum = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    def do_eval_and_save(step, train_loss_val):
        nonlocal best_val, bad_evals, stop_training, best_path
        val_loss = evaluate_noise_mse(unet, vae, tokenizer, text_encoder, noise_sched, dl_val, device)
        clip_score = float("nan")

        # log
        with log_csv.open("a", encoding="utf-8") as f:
            f.write(f"{step},{train_loss_val:.6f},{val_loss:.6f},{clip_score}\n")

        # always autosave "last"
        save_lora_safetensors(unet, weights_dir, "autosave_last.safetensors", meta={
            "step": step, "rank": args.rank, "alpha": args.alpha, "resolution": args.resolution,
            "seed": args.seed, "pretrained": args.pretrained,
        })

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            bad_evals = 0
            best_path = save_lora_safetensors(unet, weights_dir, "best.safetensors", meta={
                "step": step, "rank": args.rank, "alpha": args.alpha, "resolution": args.resolution,
                "seed": args.seed, "pretrained": args.pretrained, "val_loss": val_loss,
            })
            print(f"[eval] step {step} | train_loss={train_loss_val:.4f} | val_loss={val_loss:.4f} -> BEST ✓")
        else:
            bad_evals += 1
            print(f"[eval] step {step} | train_loss={train_loss_val:.4f} | val_loss={val_loss:.4f} -> no improve ({bad_evals})")

        # early stop if enabled
        if args.early_stop_patience > 0 and bad_evals >= args.early_stop_patience:
            print(f"[early-stop] No improvement for {bad_evals} evals. Stopping.")
            stop_training = True

    unet.train()

    while global_step < args.max_steps and not stop_training:
        for imgs, caps in dl_train:
            global_step += 1
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=amp_dtype):
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                bsz = imgs.shape[0]
                t = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=device)
                eps = torch.randn_like(latents)
                noisy = noise_sched.add_noise(latents, eps, t)

                tokens = tokenizer(
                    caps, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt"
                ).to(device)
                txt = text_encoder(**tokens).last_hidden_state

            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=amp_dtype):
                pred = unet(noisy, t, encoder_hidden_states=txt).sample
                loss = F.mse_loss(pred.float(), eps.float(), reduction="mean")

            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            accum += 1

            if accum % args.grad_accum == 0:
                if args.clip_grad_norm > 0:
                    # Only unscale if we actually plan to clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            if global_step % 10 == 0:
                print(f"step {global_step} | loss {loss.item() * args.grad_accum:.4f}")

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                do_eval_and_save(global_step, loss.item() * args.grad_accum)
                if stop_training:
                    break

            if global_step >= args.max_steps:
                break
        # break outer loop on early stop
        if stop_training:
            break

    final_path = save_lora_safetensors(unet, weights_dir, "final.safetensors", meta={
        "step": global_step,
        "rank": args.rank,
        "alpha": args.alpha,
        "resolution": args.resolution,
        "seed": args.seed,
        "pretrained": args.pretrained,
    })
    print(f"[done] Finished {global_step} steps. Saved final LoRA -> {final_path}")
    if best_path:
        print(f"[info] Best checkpoint by val_loss: {best_path}")
    else:
        print("[info] No best checkpoint recorded (maybe no evals?)")

    # Save args for traceability
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Auto-plot if requested
    if args.plot_at_end:
        plot_loss_csv(
            log_csv,
            run_log_dir / "loss.png",
            title=out_dir.name
        )
        print(f"[plot] Saved loss curve -> {run_log_dir / 'loss.png'}")


if __name__ == "__main__":
    main()
