# scripts/train_lora.py
import argparse, pathlib, json, random, csv, time, os, contextlib
from typing import Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

# LoRA via PEFT (recommended with diffusers >= 0.29)
from peft import LoraConfig
from safetensors.torch import load_file as safe_load

# ------------------------ Dataset ------------------------ #
class ImageTextFolder(Dataset):
    """
    Loads PNGs + matching captions from folders, OR takes an explicit list of image paths.
    """
    def __init__(
        self,
        img_dir: Optional[str],
        cap_dir: Optional[str],
        size: int = 512,
        paths: Optional[List[pathlib.Path]] = None,
    ):
        if paths is not None:
            self.imgs: List[pathlib.Path] = sorted(paths)
        else:
            p = pathlib.Path(img_dir)
            self.imgs: List[pathlib.Path] = sorted([x for x in p.glob("*.png")])
        self.cap_dir = pathlib.Path(cap_dir) if cap_dir else None
        self.size = size
        self.tf = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.imgs)

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

# ------------------------ Quiet helpers ------------------------ #
class TeeToFile(contextlib.AbstractContextManager):
    """Redirect stdout/stderr to a file (still prints final few lines when desired)."""
    def __init__(self, path: pathlib.Path, append: bool = True):
        self.path = path
        self.append = append
        self._stdout, self._stderr = None, None
        self._fh = None

    def __enter__(self):
        mode = "a" if self.append else "w"
        self._fh = open(self.path, mode, encoding="utf-8")
        self._stdout, self._stderr = os.dup(1), os.dup(2)
        os.dup2(self._fh.fileno(), 1)
        os.dup2(self._fh.fileno(), 2)
        return self

    def __exit__(self, *exc):
        try:
            os.dup2(self._stdout, 1)
            os.dup2(self._stderr, 2)
        finally:
            if self._fh:
                self._fh.close()

def set_quiet_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    # If you’re fully cached and want zero hub chatter, you can uncomment:
    # os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ------------------------ Save/Load utils ------------------------ #
def save_lora(pipe, out_dir, tag, cfg, accelerator=None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.safetensors"

    # unwrap UNet if wrapped by accelerate
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

    # store run config
    cfg_dir = out.parent / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def save_state(out_dir: pathlib.Path, tag: str, step: int, best_val: float, bad_epochs: int, opt, sched):
    state = {
        "step": step,
        "best_val": best_val,
        "bad_epochs": bad_epochs,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, out_dir / f"{tag}_state.pt")

def load_state(path: pathlib.Path, opt, sched):
    ckpt = torch.load(path, map_location="cpu")
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    if ckpt.get("rng") is not None:
        torch.set_rng_state(ckpt["rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    step = int(ckpt.get("step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    print(f"[resume] restored state from {path} @ step {step}")
    return step, best_val, bad_epochs

def evaluate(pipe, dl, dev, noise_sched, max_batches: int = 64, seed: int = 1234) -> float:
    """
    Validation: same denoising MSE, averaged over up to `max_batches`.
    Fixed seed reduces variance between evals.
    """
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

            t = torch.randint(
                0, noise_sched.config.num_train_timesteps,
                (lat.shape[0],), device=dev, dtype=torch.long, generator=g
            )
            eps = torch.randn_like(lat, generator=g)
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

    ap.add_argument("--train_text_encoder", action="store_true")  # off by default
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

    # Resume options
    ap.add_argument("--resume_from_dir", default=None)          # load LoRA weights into adapter before training
    ap.add_argument("--resume_weight_name", default=None)       # e.g., sd15_lora_r8_a8_step150.safetensors
    ap.add_argument("--resume_state_path", default=None)        # full state (opt/sched/step) .pt file

    # QUIET MODE
    ap.add_argument("--quiet", action="store_true", help="Mute bars/logs; write to file instead of spamming console")
    ap.add_argument("--log_file", default=None, help="Optional path to capture stdout/stderr when --quiet")

    args = ap.parse_args()

    # Quiet setup (avoid spamming ChatGPT / terminal)
    if args.quiet:
        set_quiet_env()

    # Repro
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.grad_accum)
    dev = acc.device
    is_main = acc.is_main_process

    # ----- Load pipeline -----
    def run():
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

        # ----- Add LoRA to UNet attention projections -----
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian", target_modules=target_modules
        )
        adapter_name = "tattoo"
        pipe.unet.add_adapter(lora_cfg, adapter_name=adapter_name)
        pipe.unet.set_adapter(adapter_name)

        # --- Optional: resume LoRA weights into adapter (weights-only) ---
        if args.resume_from_dir:
            wpath = pathlib.Path(args.resume_from_dir) / (
                args.resume_weight_name or "sd15_lora_r8_a8_final.safetensors"
            )
            state = safe_load(wpath, device="cpu")
            for k in list(state.keys()):
                if state[k].dtype == torch.float16:
                    state[k] = state[k].float()  # AMP stability
            missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
            print(f"[resume] loaded {wpath.name}; missing={len(missing)} unexpected={len(unexpected)}")

        # Trainable params = LoRA only (keep them in fp32 so AMP can unscale)
        for n, p in pipe.unet.named_parameters():
            requires = ("lora_" in n.lower())
            p.requires_grad_(requires)
            if requires:
                p.data = p.data.float()
        params = [p for n, p in pipe.unet.named_parameters() if "lora_" in n.lower()]

        if args.train_text_encoder:
            for n, p in pipe.text_encoder.named_parameters():
                requires = ("lora_" in n.lower())
                p.requires_grad_(requires)
                if requires:
                    p.data = p.data.float()
            params += [p for n, p in pipe.text_encoder.named_parameters() if "lora_" in n.lower()]

        opt = torch.optim.AdamW(
            params, lr=args.lr_unet, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8
        )
        noise_sched = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

        # ----- Train/Val split -----
        img_dir = pathlib.Path(args.data_images)
        all_imgs = sorted(img_dir.glob("*.png"))
        n_total = len(all_imgs)
        assert n_total > 0, f"No images found in {img_dir}"
        n_val = max(1, int(n_total * args.val_split))
        random.Random(args.seed).shuffle(all_imgs)
        val_imgs = all_imgs[:n_val]
        train_imgs = all_imgs[n_val:]

        ds_train = ImageTextFolder(None, args.data_captions, size=args.resolution, paths=train_imgs)
        ds_val   = ImageTextFolder(None, args.data_captions, size=args.resolution, paths=val_imgs)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
        )

        # prepare with Accelerate (train loader only)
        pipe.unet, opt, dl_train, sched = acc.prepare(pipe.unet, opt, dl_train, sched)
        if args.train_text_encoder:
            pipe.text_encoder = acc.prepare_model(pipe.text_encoder)

        # ----- Logging / output dirs -----
        out_dir = pathlib.Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_tag = f"sd15_lora_r{args.rank}_a{args.alpha}"
        log_dir = pathlib.Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        log_csv = log_dir / f"{out_dir.name}_{int(time.time())}.csv"
        if is_main:
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "train_loss", "val_loss"])

        # ----- Optional: Resume full training state -----
        step, best_val, bad_epochs = 0, float("inf"), 0
        if args.resume_state_path and pathlib.Path(args.resume_state_path).exists():
            s_step, s_best, s_bad = load_state(pathlib.Path(args.resume_state_path), opt, sched)
            step, best_val, bad_epochs = s_step, s_best, s_bad

        # Progress bar
        pbar = tqdm(
            total=args.max_steps,
            initial=step,
            disable=(not is_main) or args.quiet,
            leave=False,
            dynamic_ncols=False,
        )

        # ----- Train loop with eval + early stopping -----
        while step < args.max_steps:
            for batch in dl_train:
                with acc.accumulate(pipe.unet):
                    # tokenize
                    toks = pipe.tokenizer(
                        list(batch["caption"]),
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        enc = pipe.text_encoder(toks.input_ids.to(dev))[0]

                    # latents + noise
                    px = batch["pixel_values"].to(dev, dtype=torch.float16)
                    with torch.no_grad():
                        lat = pipe.vae.encode(px).latent_dist.sample() * 0.18215

                    t = torch.randint(
                        0, noise_sched.config.num_train_timesteps,
                        (lat.shape[0],), device=dev, dtype=torch.long
                    )
                    eps = torch.randn_like(lat)
                    lat_noisy = noise_sched.add_noise(lat, eps, t)

                    # predict noise, MSE
                    pred = pipe.unet(lat_noisy, t, encoder_hidden_states=enc).sample
                    loss = torch.nn.functional.mse_loss(pred.float(), eps.float())

                    acc.backward(loss)
                    if args.clip_grad_norm and args.clip_grad_norm > 0:
                        acc.clip_grad_norm_(params, args.clip_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

                # One logical step finished?
                if acc.sync_gradients:
                    step += 1
                    if is_main and not args.quiet:
                        pbar.set_description(f"step {step} | loss {loss.item():.4f}")
                        pbar.update(1)

                    # periodic SAVE (weights + state)
                    if is_main and (step % args.save_every == 0 or step == args.max_steps):
                        save_lora(pipe, args.output_dir, f"{exp_tag}_step{step}", vars(args), acc)
                        save_state(out_dir, f"{exp_tag}_step{step}", step, best_val, bad_epochs, opt, sched)

                    # periodic EVAL
                    if is_main and (step % args.eval_every == 0 or step == args.max_steps):
                        val_loss = evaluate(pipe, dl_val, dev, noise_sched)
                        with open(log_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([step, float(loss.item()), float(val_loss)])
                        if not args.quiet:
                            print(f"[eval] step {step} | train {loss.item():.4f} | val {val_loss:.4f}")

                        # early stopping on best val
                        if val_loss + args.early_stop_min_delta < best_val:
                            best_val = val_loss
                            bad_epochs = 0
                            save_lora(pipe, args.output_dir, f"{exp_tag}_best", vars(args), acc)
                            save_state(out_dir, f"{exp_tag}_best", step, best_val, bad_epochs, opt, sched)
                        else:
                            bad_epochs += 1
                            if bad_epochs >= args.early_stop_patience:
                                if not args.quiet:
                                    print(f"[early-stop] no improvement for {bad_epochs} evals; stopping at step {step}.")
                                step = args.max_steps
                                break

                if step >= args.max_steps:
                    break

        if is_main:
            save_lora(pipe, args.output_dir, f"{exp_tag}_final", vars(args), acc)
            save_state(out_dir, f"{exp_tag}_final", step, best_val, bad_epochs, opt, sched)
            if not args.quiet:
                print("Done.")

    # Actually run (optionally capturing all output)
    if args.quiet:
        log_dir = pathlib.Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        log_path = pathlib.Path(args.log_file) if args.log_file else (log_dir / "train_quiet.log")
        with TeeToFile(log_path):
            set_quiet_env()
            main_out = run()
    else:
        run()

if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    main()
