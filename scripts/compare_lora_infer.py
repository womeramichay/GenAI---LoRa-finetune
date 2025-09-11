# scripts/compare_lora_infer.py
import argparse, pathlib, math, re, os, random
from typing import Optional, List, Tuple
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def pick_weight(dirpath: pathlib.Path) -> pathlib.Path:
    # prefer *_best.safetensors, else *_final.safetensors, else any *.safetensors
    best = sorted(dirpath.glob("*_best.safetensors"))
    if best: return best[-1]
    fin = sorted(dirpath.glob("*_final.safetensors"))
    if fin: return fin[-1]
    allw = sorted(dirpath.glob("*.safetensors"))
    if not allw:
        raise FileNotFoundError(f"No .safetensors found in {dirpath}")
    return allw[-1]

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def load_base_pipe(pretrained: str, device: torch.device):
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe

def try_load_lora(pipe, dirpath: pathlib.Path, weight_name: str) -> str:
    """Load LoRA saved by diffusers.save_lora_weights (our train script).
       Returns how it was loaded (string tag)."""
    try:
        pipe.load_lora_weights(dirpath, weight_name=weight_name)
        return "diffusers"
    except Exception:
        # Some older diffusers may need PEFT fallback
        try:
            pipe.load_lora_weights(dirpath, weight_name=weight_name, use_safetensors=True)
            return "peft-fallback"
        except Exception as e:
            raise RuntimeError(f"Failed loading LoRA from {dirpath}/{weight_name}: {e}")

def render_grid(imgs: List[Image.Image], rows: int, cols: int, cell_pad: int = 4) -> Image.Image:
    assert len(imgs) > 0
    w, h = imgs[0].size
    W = cols * w + (cols - 1) * cell_pad
    H = rows * h + (rows - 1) * cell_pad
    grid = Image.new("RGB", (W, H), (30, 30, 30))
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(imgs): break
            x = c * (w + cell_pad)
            y = r * (h + cell_pad)
            grid.paste(imgs[k], (x, y))
            k += 1
    return grid

def sample_set(pipe, prompt: str, seeds: List[int], steps: int, guidance: float, width: int, height: int):
    outs = []
    for s in seeds:
        g = torch.Generator(device=pipe.device).manual_seed(s)
        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                   height=height, width=width, generator=g).images[0]
        outs.append(img)
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--pretrained", default="runwayml/stable-diffusion-v1-5")

    # any combination of the following three sets is allowed
    ap.add_argument("--vanilla_lora_dir", default=None)
    ap.add_argument("--vanilla_lora_weight", default=None)

    ap.add_argument("--blip_lora_dir", default=None)
    ap.add_argument("--blip_lora_weight", default=None)

    ap.add_argument("--blip_plus_lora_dir", default=None)
    ap.add_argument("--blip_plus_lora_weight", default=None)

    ap.add_argument("--count", type=int, default=9)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)

    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = pathlib.Path(args.outdir); ensure_dir(outdir)

    # fixed seeds so all variants get identical noise
    random.seed(args.seed)
    seeds = [random.randint(0, 2**31 - 1) for _ in range(args.count)]

    # For each provided LoRA, load base pipe fresh (isolated), attach LoRA, sample, save grid.
    def run_one(tag: str, lora_dir: Optional[str], lora_name: Optional[str]) -> Optional[Image.Image]:
        if not lora_dir: return None
        ldir = pathlib.Path(lora_dir)
        weight = pathlib.Path(lora_name) if lora_name else pick_weight(ldir)
        pipe = load_base_pipe(args.pretrained, device)
        how = try_load_lora(pipe, ldir, weight.name)
        print(f"[{tag}] loaded via: {how}")
        imgs = sample_set(pipe, args.prompt, seeds, args.steps, args.guidance, args.width, args.height)
        grid = render_grid(imgs, args.rows, args.cols)
        grid_path = outdir / f"{slug(tag)}_{slug(args.prompt)}.png"
        grid.save(grid_path)
        print(f"[{tag}] saved grid → {grid_path}")
        return grid

    grids = []
    names = []

    g_v = run_one("vanilla", args.vanilla_lora_dir, args.vanilla_lora_weight)
    if g_v: grids.append(g_v); names.append("vanilla")

    g_b = run_one("blip", args.blip_lora_dir, args.blip_lora_weight)
    if g_b: grids.append(g_b); names.append("blip")

    g_bp = run_one("blip_plus", args.blip_plus_lora_dir, args.blip_plus_lora_weight)
    if g_bp: grids.append(g_bp); names.append("blip_plus")

    # Combined side-by-side panel (if >=2 provided)
    if len(grids) >= 2:
        w, h = grids[0].size
        pad = 10
        combo = Image.new("RGB", (len(grids) * w + (len(grids) - 1) * pad, h), (20, 20, 20))
        x = 0
        for g in grids:
            combo.paste(g, (x, 0))
            x += w + pad
        combo_path = outdir / f"compare_{slug('_'.join(names))}_{slug(args.prompt)}.png"
        combo.save(combo_path)
        print(f"[combined] saved → {combo_path}")

if __name__ == "__main__":
    main()
