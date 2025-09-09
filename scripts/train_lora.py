# scripts/train_lora.py
import os, math, argparse, pathlib, json, random
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from accelerate import Accelerator
from tqdm import tqdm

class ImageTextFolder(Dataset):
    def __init__(self, img_dir: str, cap_dir: Optional[str], size: int = 512):
        p = pathlib.Path(img_dir)
        self.imgs = sorted([x for x in p.glob("*.png")])
        self.cap_dir = pathlib.Path(cap_dir) if cap_dir else None
        self.size = size
        self.tf = T.Compose([T.ToTensor(), T.Normalize([0.5],[0.5])])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        p = self.imgs[i]
        im = Image.open(p).convert("RGB")
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size), Image.LANCZOS)
        x = self.tf(im)
        cap = "minimal line-art tattoo, stencil, high-contrast"
        if self.cap_dir:
            c = self.cap_dir / (p.stem + ".txt")
            if c.exists(): cap = c.read_text(encoding="utf-8").strip()
        return {"pixel_values": x, "caption": cap}

def save_lora(pipe, out_dir, tag, cfg):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.safetensors"
    LoraLoaderMixin.save_lora_weights(pipe.unet, path)
    print(f"[save] {path}")
    (path.parent.parent / "configs").mkdir(parents=True, exist_ok=True)
    (path.parent.parent / "configs" / f"{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_images", required=True)
    ap.add_argument("--data_captions", default="data/processed/captions")
    ap.add_argument("--output_dir", default="lora/weights")
    ap.add_argument("--pretrained", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--train_text_encoder", action="store_true")
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=125)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_unet", type=float, default=1e-4)
    ap.add_argument("--lr_text", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.grad_accum)
    dev = acc.device; is_main = acc.is_main_process

    if is_main: print("Loading SD1.5…")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to(dev)
    pipe.enable_attention_slicing()
    pipe.unet.enable_gradient_checkpointing()
    if args.train_text_encoder:
        pipe.text_encoder.gradient_checkpointing_enable()

    for name, m in pipe.unet.attn_processors.items():
        pipe.unet.set_attn_processor(LoRAAttnProcessor2_0(hidden_size=m.hidden_size, rank=args.rank))

    params = []
    for n, p in pipe.unet.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True); params.append(p)
        else:
            p.requires_grad_(False)
    if args.train_text_encoder:
        for n, p in pipe.text_encoder.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True); params.append(p)
            else:
                p.requires_grad_(False)

    opt = torch.optim.AdamW(params, lr=args.lr_unet, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    noise_sched = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    ds = ImageTextFolder(args.data_images, args.data_captions, size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    num_updates = args.max_steps
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=num_updates)

    pipe.unet, opt, dl, sched = acc.prepare(pipe.unet, opt, dl, sched)
    if args.train_text_encoder: pipe.text_encoder = acc.prepare_model(pipe.text_encoder)

    step = 0
    tag_base = f"sd15_lora_r{args.rank}_a{args.alpha}"
    pbar = tqdm(total=args.max_steps, disable=not is_main)
    while step < args.max_steps:
        for batch in dl:
            with acc.accumulate(pipe.unet):
                toks = pipe.tokenizer(list(batch["caption"]), padding="max_length",
                                      max_length=pipe.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt")
                with torch.no_grad():
                    enc = pipe.text_encoder(toks.input_ids.to(dev))[0]

                px = batch["pixel_values"].to(dev, dtype=torch.float16)
                with torch.no_grad():
                    lat = pipe.vae.encode(px).latent_dist.sample() * 0.18215

                t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (lat.shape[0],), device=dev, dtype=torch.long)
                eps = torch.randn_like(lat)
                lat_noisy = noise_sched.add_noise(lat, eps, t)

                pred = pipe.unet(lat_noisy, t, encoder_hidden_states=enc).sample
                loss = torch.nn.functional.mse_loss(pred.float(), eps.float())

                acc.backward(loss)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            if acc.sync_gradients:
                step += 1
                if is_main: pbar.set_description(f"step {step} | loss {loss.item():.4f}"); pbar.update(1)
                if is_main and (step % args.save_every == 0 or step == args.max_steps):
                    save_lora(pipe, args.output_dir, f"{tag_base}_step{step}", vars(args))
            if step >= args.max_steps: break
    if is_main:
        save_lora(pipe, args.output_dir, f"{tag_base}_final", vars(args))
        print("Done.")

if __name__ == "__main__":
    main()
