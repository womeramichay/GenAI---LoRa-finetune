**What I built**

I fine-tuned Stable Diffusion 1.5 to generate clean, high-contrast tattoo line art.
Instead of retraining the whole model, I trained small adapter layers using LoRA.
These adapters are lightweight files you can load on top of the base model.

Why this base model:

Stable Diffusion 1.5 is well understood, has great of community tools, and runs on a single consumer GPU. It gives predictable results and reproducible checkpoints.

**Why LoRA:**
* Small adapters instead of full fine-tuning save time and resorces.
* Efficient: I train a few million adapter weights instead of the entire network.
* Fits on everyday hardware: I used my personal PC and GPU, which let me learn from many images while keeping memory low and runtime reasonable, The LoRA weights are small and easy to store.
  
**Data & preprocessing**

* Dataset: Tattoo images from [Kaggle](https://www.kaggle.com/datasets/faiqueali/tattoos).
* Cleaning: Convert to RGB, center-crop to a square, and resize, I trained at 384x384 (I started with 512x512 buti change from time limitation). This reduces memory use and increases throughput on a single GPU, which speeds up iteration.
* Split: I keep 5% of the images aside for validation so I can track generalization during training.

**Captions:**

I guide the model with captions and tried three sources:
1) vanilla – a short, generic tattoo prompt used for every image (simple fallback).
2) BLIP – automatic per-image captions from a pretrained BLIP model.
3) BLIP+ – the same BLIP captions but enriched with style hints like “clean line tattoo, stencil, high contrast.”

**Why BLIP:**
It gives me reliable captions out of the box, runs fast on my setup, and integrates cleanly with the LoRA training pipeline so I get consistent supervision without manual labeling. BLIP+ simply nudges the captions toward the look I want (line-art, high-contrast).

**How I trained:**

* I keep the base model parts fixed (UNet, VAE, text encoder) and insert LoRA adapters inside the UNet attention blocks (to_q, to_k, to_v, to_out.0).
* The learning target is standard for diffusion models: predict the added noise during denoising (MSE loss).
* I use AdamW, warmup where helpful, optional gradient-norm clipping, and mixed precision for speed and lower memory.

**Training settings**
Effective larger batches without extra VRAM: I keep the per-step batch small and use gradient accumulation to simulate a larger batch. This gives more stable updates while staying within memory limits.

**Training steps:**
I ran ~500 steps (quick checks) and ~1000 steps (longer runs) to compare curves. Short runs help me iterate, longer runs push quality once settings look good.

**Model selection**
After each evaluation I save a checkpoint, and I also save a final one at the end. I can pick the checkpoint with the lowest validation loss as the best generalizing weights.

**Results and takeaways**

To compare caption strategies I fixed the random seed and trained three variants with identical hyper-parameters: vanilla, BLIP, and BLIP+. The training/validation noise-MSE curves are nearly overlapping, and I did not observe a consistent, statistically meaningful gap between the variants under these settings. In short: on this dataset and budget (≈2000 images, 5% val, 512–1024 steps, LoRA r=8,a=8 @ 384²), caption choice did not materially change the denoising loss.

Why this can happen:
Noise-MSE measures the diffusion objective, not image quality; different captions can yield similar MSE.
With LoRA capacity (r=8) and short training horizon, the model may be bottlenecked by steps/capacity rather than caption richness.
Line-art structure carries strong signal generic prompts already align well, so BLIP/BLIP+ adds limited incremental supervision here.

**Next Step**
Train longer (2k–4k steps) or increase LoRA capacity (e.g., r=16) and/or train the text encoder.
Raise resolution to 512x512 and keep effective batch large (e.g., bs=4, grad-accum=16).
Evaluate generation quality directly: fixed prompt grids across checkpoints (use the compare script), CLIP-score on generated sets, or human side-by-side ratings.
Try caption dropout / mix “vanilla” + BLIP to reduce over-conditioning; or cleaner captions (deduplicate, lowercase, remove stopwords).




