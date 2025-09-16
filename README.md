# Tattoo LoRA — Tattoo line-art generator (Stable Diffusion 1.5 + LoRA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HF-diffusers-ffd21e.svg)](https://github.com/huggingface/diffusers)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What & Why (Model + Method)

**Model used:** `runwayml/stable-diffusion-v1-5` (SD-1.5) fine-tuned with **LoRA** (Low-Rank Adaptation) on tattoo line-art images.

**Why SD-1.5?**
- Mature, well-documented base model with strong community tooling (Diffusers, PEFT/LoRA).
- Lightweight enough for **single-GPU training** (≈6–8 GB VRAM) while still producing clean, high-contrast line art.
- Reproducible checkpoints and predictable behavior for portfolio projects.

**Why LoRA (instead of full finetune / DreamBooth)?**
- **Parameter-efficient:** we only train a few million adapter weights (rank *r* ≈ 8), not the whole UNet.
- **Fits on consumer GPUs:** works with batch_size=1 + gradient accumulation.
- **Composable & portable:** LoRA weights are small `.safetensors` you can load on top of SD-1.5 anywhere.

---

## Method (Training & Evaluation)

### Training objective (how we train)
- We keep the SD-1.5 **UNet** and **VAE/Text Encoder** frozen (optionally LoRA-train the text encoder).
- Inject LoRA adapters into the UNet’s **attention projections**: `to_q`, `to_k`, `to_v`, `to_out.0`.
- Supervision: **denoising MSE** (predict the noise ε) with the SD-1.5 DDPM scheduler.
- Optimizer: **AdamW** with cosine schedule + warmup, gradient accumulation, optional grad-norm clipping.
- Data: line-art PNGs resized to 512×512; captions vary by variant (see below).

### Captioning variants (what we compare)
| Variant   | Caption source                                                              | When to use                           |
|-----------|-----------------------------------------------------------------------------|---------------------------------------|
| `vanilla` | Fixed style string inside trainer: *“minimal line-art tattoo, stencil, high-contrast”* | Baseline style adherence              |
| `blip`    | Auto captions from **BLIP** (`Salesforce/blip-image-captioning-large`)      | Natural language descriptions         |
| `blip_plus` | BLIP + style suffix: *“clean line tattoo, high contrast, stencil, no shading”* (dedup + trim) | Stronger style conditioning |

### Evaluation (how we measure)
1. **Validation denoising MSE** on a held-out set (proxy for over/under-fitting). Logged to CSV as `val_loss`.
2. **Qualitative eval images** every *N* steps from fixed prompts (same seeds → apples-to-apples across runs).
3. **Optional CLIP score** (`openai/clip-vit-base-patch32`): average text-image similarity for the fixed prompts. Logged as `clip_score`.

> Logs are in `runs/logs/*.csv` with columns: `step, train_loss, val_loss, clip_score`.  
> Eval images are saved under `runs/samples/<run_name>/eval_stepXXXX/`.

---

## Highlights

- **End-to-end workflow**: preprocessing → BLIP/BLIP+ captioning → LoRA training → evaluation → inference.
- **Three variants** to compare style conditioning approaches.
- **Reproducible**: one-button `main.py` orchestrates the pipeline, and a Streamlit dashboard visualizes results.
- **Storage & VRAM aware**: lightweight eval images, large `save_every` (keeps best/final), CPU offload at inference.

---

## Repository layout
tattoo-genai/
├─ data/
│ ├─ raw/ # source PNGs
│ └─ processed/<dataset>/
│ ├─ images/ # preprocessed images (512x512)
│ ├─ captions_blip/ # BLIP captions
│ └─ captions_blip_plus/ # BLIP+ captions
├─ runs/
│ ├─ lora/ # LoRA weights per run (best/final)
│ ├─ logs/ # CSV logs
│ └─ samples/ # eval/ad-hoc images
├─ scripts/
│ ├─ preprocess.py # resize/crop images
│ ├─ auto_caption_blip.py # BLIP baseline captions
│ ├─ enrich_captions.py # BLIP+ enrichment (suffix + dedup)
│ ├─ train_lora.py # LoRA trainer (MSE + eval images + CLIP)
│ ├─ compare_lora_infer.py # side-by-side inference for runs
│ └─ plot_training.py # plot CSV logs
├─ app/
│ └─ dashboard.py # Streamlit dashboard (auto-discovers runs)
└─ main.py # one-button pipeline driver (idempotent)
