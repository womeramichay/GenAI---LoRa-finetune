# scripts/train_lora.py
import argparse, pathlib, json, random, csv, time
from typing import Optional, List
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

# LoRA via PEFT backend (diffusers >=0.29)
from peft import LoraConfig

# ------------------------ Dataset ------------------------ #
class ImageTextFolder(Dataset):
    def __init__(self, img_dir: Optional[str], cap_dir: Optional[str], size: int = 512,
                 paths: Optional[List[Path]] = None):
        if paths is not None:
            self.imgs: List[Path] = sorted(paths)
        else:
            p = Path(img_dir)
            self.imgs: List[Path] = sorted([x for x in p.glob("*.png")])
        self.cap_dir = Path(cap_dir) if cap_dir else None
        self.size = size
        self.tf = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        im = Image.open(p).convert("RGB")
        if im.size != (self.size, self.size):
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            im = im.resize((self.size, self.size), resample)
        x = self.tf(im)

        cap = "minimal line-art tattoo, stencil, high-contrast"
        if self.cap_dir:
            c = self.cap_dir / (p.stem + ".txt")
            if c.exists():
                cap = c.read_text(encoding="utf-8").strip()

        return {"pixel_values": x, "caption": cap}

# ------------------------ Utils ------------------------ #
def save_lora(pipe, out_dir, tag, cfg, accelerator=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.safetensors"

    unet_for_save = pipe.unet
    if accelerator is not None:
        try:
            unet_for_save = accelerator.unwrap_model(unet_for_save)
        except Exception:
            pass

    pipe.save_lora_weights(
        save_directory=out,
        unet_lora_layers=unet_for_save,
        weight_name=f"{tag}.safetensors",
    )
    print(f"[save] {path}")

    cfg_dir = out.parent / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def evaluate(pipe, dl, dev, noise_sched, max_batches: int = 64, seed: int = 1234) -> float:
    """Validation: average denoising MSE over up to `max_batches`."""
    def randn_like_compat(x, g=None):
        try:
            return torch.randn_like(x, generator=g)
        except TypeError:
            return torch.randn(x.shape, device=x.device, dtype=x.dtype)

    pipe.unet.eval()
    losses = []
    g = torch.Generator(device=dev).manual_seed(seed)

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= max_batches:
                break

            toks = pipe.tokenizer(
                list(batch["caption"]),
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            enc = pipe.text_encoder(toks.input_ids.to(dev))[0]

            px = batch["pixel_values"].to(dev, dtype=torch.float16)
            lat = pipe.vae.encode(px).latent_dist.sample(generator=g) * 0.18215

            try:
                t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (lat.shape[0],), device=dev, dtype=torch.long, generator=g)
            except TypeError:
                t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (lat.shape[0],), device=dev, dtype=torch.long)

            eps = randn_like_compat(lat, g)
            lat_noisy = noise_sched.add_noise(lat, eps, t)

            pred = pipe.unet(lat_noisy, t, encoder_hidden_states=enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), eps.float())
            losses.append(loss.item())

    pipe.unet.train()
    return float(sum(losses) / max(1, len(losses)))

# ------------------------ Main ------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_images", required=True)
    ap.add_argument("--data_captions", required=True)
    ap.add_argument("--output_dir", default="runs/lora/exp")
    ap.add_argument("--pretrained", default="runwayml/stable-diffusion-v1-5")

    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)

    ap.add_argument("--train_text_encoder", action="store_true")
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--val_split", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)

    ap.add_argument("--lr_unet", type=float, default=1e-4)
    ap.add_argument("--lr_text", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=100)

    ap.add_argument("--early_stop_patience", type=int, default=6)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", default="runs/logs")

    # resume support
    ap.add_argument("--resume_from_dir", default=None, help="Folder with saved *.safetensors")
    ap.add_argument("--resume_weight_name", default=None, help="LoRA weight file to resume from")
    ap.add_argument("--resume_step", type=int, default=0, help="Global step to start from")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.grad_accum)
    dev = acc.device
    is_main = acc.is_main_process

    if is_main:
        print("Loading SD1.5…")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to(dev)
    pipe.enable_attention_slicing()
    pipe.unet.enable_gradient_checkpointing()
    if args.train_text_encoder:
        pipe.text_encoder.gradient_checkpointing_enable()

    # ----- Add LoRA adapter ONCE -----
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    adapter_name = "tattoo"
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
        target_modules=target_modules
    )
    pipe.unet.add_adapter(lora_cfg, adapter_name=adapter_name)
    pipe.unet.set_adapter(adapter_name)

    # ----- Resume: ONLY load weights into existing adapter -----
    if args.resume_from_dir:
        try:
            pipe.load_lora_weights(
                args.resume_from_dir,
                weight_name=(args.resume_weight_name if args.resume_weight_name else None),
                adapter_name=adapter_name,
            )
            print(f"[resume] loaded via load_lora_weights: {args.resume_weight_name or '(default)'}")
        except Exception as e:
            print(f"[resume] load_lora_weights failed ({e}); trying state_dict fallback…")
            from safetensors.torch import load_file
            wpath = Path(args.resume_from_dir) / (args.resume_weight_name or "")
            state = load_file(str(wpath), device="cpu")
            for k in list(state.keys()):
                if state[k].dtype == torch.float16:
                    state[k] = state[k].float()
            missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
            print(f"[resume] loaded state dict: missing={len(missing)} unexpected={len(unexpected)}")

    # trainable parameters = LoRA only (fp32 for AMP stability)
    params = []
    for n, p in pipe.unet.named_parameters():
        req = ("lora_" in n.lower())
        p.requires_grad_(req)
        if req:
            p.data = p.data.float()
            params.append(p)

    if args.train_text_encoder:
        for n, p in pipe.text_encoder.named_parameters():
            req = ("lora_" in n.lower())
            p.requires_grad_(req)
            if req:
                p.data = p.data.float()
                params.append(p)

    opt = torch.optim.AdamW(params, lr=args.lr_unet, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay, eps=1e-8)
    noise_sched = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    # ----- Train/Val split -----
    img_dir = Path(args.data_images)
    all_imgs = sorted(img_dir.glob("*.png"))
    n_total = len(all_imgs); assert n_total > 0, f"No images found in {img_dir}"
    n_val = max(1, int(n_total * args.val_split))
    random.Random(args.seed).shuffle(all_imgs)
    val_imgs = all_imgs[:n_val]
    train_imgs = all_imgs[n_val:]

    ds_train = ImageTextFolder(None, args.data_captions, size=args.resolution, paths=train_imgs)
    ds_val   = ImageTextFolder(None, args.data_captions, size=args.resolution, paths=val_imgs)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # LR schedule for remaining steps
    remaining_steps = max(1, args.max_steps - max(0, args.resume_step))
    warmup_left = max(0, args.warmup_steps - max(0, args.resume_step))
    sched = get_cosine_schedule_with_warmup(opt,
                                            num_warmup_steps=warmup_left,
                                            num_training_steps=remaining_steps)

    # prepare with Accelerate
    pipe.unet, opt, dl_train, sched = acc.prepare(pipe.unet, opt, dl_train, sched)
    if args.train_text_encoder:
        pipe.text_encoder = acc.prepare_model(pipe.text_encoder)

    # ----- Logging -----
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exp_tag = f"sd15_lora_r{args.rank}_a{args.alpha}"
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"{out_dir.name}_{int(time.time())}.csv"
    if is_main:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    # ----- Train loop with eval + early stop -----
    global_step = max(0, args.resume_step)
    best_val, bad_epochs = float("inf"), 0
    pbar = tqdm(total=args.max_steps, initial=global_step, disable=not is_main)

    while global_step < args.max_steps:
        for batch in dl_train:
            with acc.accumulate(pipe.unet):
                toks = pipe.tokenizer(
                    list(batch["caption"]),
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
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

                # Clip only on real optimizer steps (when gradients are syncing), and guard AMP quirk
                if acc.sync_gradients and args.clip_grad_norm and args.clip_grad_norm > 0:
                    try:
                        acc.clip_grad_norm_(params, args.clip_grad_norm)
                    except RuntimeError as e:
                        if "unscale_() has already been called" in str(e):
                            # benign with some torch/accelerate combos; skip this step’s clipping
                            pass
                        else:
                            raise

                opt.step();
                sched.step();
                opt.zero_grad(set_to_none=True)

            if acc.sync_gradients:
                global_step += 1
                if is_main:
                    pbar.set_description(f"step {global_step} | loss {loss.item():.4f}")
                    pbar.update(1)

                if is_main and (global_step % args.save_every == 0 or global_step == args.max_steps):
                    save_lora(pipe, args.output_dir, f"{exp_tag}_step{global_step}", vars(args), acc)

                if is_main and (global_step % args.eval_every == 0 or global_step == args.max_steps):
                    val_loss = evaluate(pipe, dl_val, dev, noise_sched)
                    with open(log_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([global_step, float(loss.item()), float(val_loss)])
                    print(f"[eval] step {global_step} | train {loss.item():.4f} | val {val_loss:.4f}")

                    if val_loss + args.early_stop_min_delta < best_val:
                        best_val = val_loss
                        bad_epochs = 0
                        save_lora(pipe, args.output_dir, f"{exp_tag}_best", vars(args), acc)
                    else:
                        bad_epochs += 1
                        if bad_epochs >= args.early_stop_patience:
                            print(f"[early-stop] no improvement for {bad_epochs} evals; stopping at step {global_step}.")
                            global_step = args.max_steps
                            break

            if global_step >= args.max_steps:
                break

    if is_main:
        save_lora(pipe, args.output_dir, f"{exp_tag}_final", vars(args), acc)
        print("Done.")

if __name__ == "__main__":
    main()
