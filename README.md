What I built-
I fine-tuned the “Stable Diffusion 1.5” image generator to produce clean, high-contrast tattoo line art. Instead of retraining the whole model I trained small adapter layers (LoRA method). These adapters are lightweight files which you can load on top of the base model.

Why this base model:
Stable Diffusion 1.5 is well understood, has plenty of community tools, and runs on a single consumer GPU. It gives predictable results and reproducible checkpoints—perfect for a portfolio project.

Why LoRA: (small adapters) instead of full fine-tuning

* It is efficient: only train a few million adapter weights instade the entire network.

* It fits on everyday hardware: I useded my personal PC and GPU, in that why i could learn from many images and in the same time keep memory low still and have a resonable runtime.

How I trained-
First i used images of tattos [kaggle](https://www.kaggle.com/datasets/faiqueali/tattoos)
I keep the base model parts fixed and add LoRA adapters inside the attention blocks.

The learning target is simple and standard for diffusion models: predict the noise that was added to an image while denoising.

I use AdamW as the optimizer, warmup where helpful, optional gradient-norm clipping, and mixed precision for speed and lower memory.

I used captions guide the learning. I tried three caption sources:

1) vanilla: a short, generic tattoo prompt used for every image (fallback, no .txt files needed).

2) BLIP: automatic captions generated from each image using blip model.

3) BLIP_PLUS: the same BLIP captions but enriched with tattoo-style hints (e.g., “clean line tattoo, stencil, high contrast”).

I chose to use blip model 

During this project i started with train
What I changed to learn faster and from more images

Image size: we trained at 384×384 (not 512×512). This uses less memory and lets us see more mini-batches per hour, which usually improves results faster on a single GPU.

“Bigger” batches without extra memory: we keep the per-step batch small (so it fits in memory) and accumulate gradients over several mini-batches before taking an optimizer step. This gives the effect of a larger batch—more stable updates—without needing more VRAM.

Training steps: we ran both ~500 steps (quick runs) and ~1000 steps (longer runs) to compare curves. Short runs help iterate on settings; longer runs push quality once settings look good.

Validation split: we keep 5% of images aside for validation and measure the same loss there every N steps. This tells us when the model is actually improving and lets us pick a good checkpoint.

Picking the weights: after each evaluation we save a checkpoint; at the end we also save the final one. You can select the checkpoint with the lowest validation loss (often the best generalization).

Outcome

All three variants trained cleanly on a single Windows machine.

Loss curves and CSV logs are saved per run so you can compare vanilla vs. blip vs. blip_plus.

In practice, blip_plus captions often give nicer prompts for generation, while blip and vanilla are good baselines for speed and sanity checks.
