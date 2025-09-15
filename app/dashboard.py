# app/dashboard.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob
import subprocess, sys, os

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
SAMPLES = RUNS / "samples"
LORA = RUNS / "lora"

st.set_page_config(page_title="Tattoo LoRA Dashboard", layout="wide")

st.title("🖊️ Tattoo LoRA — Training Dashboard")

# Discover variants/runs
run_dirs = sorted(p for p in LORA.glob("*") if p.is_dir())
variants = [p.name for p in run_dirs]
if not variants:
    st.warning("No runs found under runs/lora/* yet. Train via `python main.py` first.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
picked = st.sidebar.multiselect("Variants to show", variants, default=variants)
prompt = st.sidebar.text_input("Quick inference prompt", "owl, clean line, minimal line-art tattoo, stencil, high contrast, no shading")
go_infer = st.sidebar.button("▶️ Run quick inference for picked variants")

# Loss plots (per-variant)
st.subheader("📉 Loss — per variant")
cols = st.columns(min(3, len(picked)) or 1)
for i, name in enumerate(picked):
    with cols[i % len(cols)]:
        out_dir = LORA / name
        # find matching CSV (logs are named with run-folder + timestamp)
        pattern = str(LOGS / f"{out_dir.name}_*.csv")
        csvs = sorted(glob.glob(pattern))
        if not csvs:
            st.info(f"No log CSV for `{name}` yet.")
            continue
        df = pd.read_csv(csvs[-1])
        fig, ax = plt.subplots()
        ax.plot(df["step"], df["train_loss"], label="train")
        ax.plot(df["step"], df["val_loss"], label="val")
        ax.set_title(name)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
        st.pyplot(fig)

# Overlay plot (all vs all)
st.subheader("📊 Loss — all vs all (validation)")
if picked:
    fig, ax = plt.subplots()
    for name in picked:
        out_dir = LORA / name
        csvs = sorted(glob.glob(str(LOGS / f"{out_dir.name}_*.csv")))
        if not csvs: continue
        df = pd.read_csv(csvs[-1])
        ax.plot(df["step"], df["val_loss"], label=name)
    ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend()
    st.pyplot(fig)

# Samples gallery
st.subheader("🖼️ Sample images")
gcols = st.columns(min(3, len(picked)) or 1)
for i, name in enumerate(picked):
    with gcols[i % len(gcols)]:
        st.markdown(f"**{name}**")
        samp_dir = SAMPLES / name
        if not samp_dir.exists():
            st.info("No samples found yet.")
            continue
        # show up to 6 PNGs
        imgs = sorted(samp_dir.glob("*.png"))[:6]
        for p in imgs:
            st.image(str(p), use_container_width=True, caption=p.name)

# Optional quick inference (calls compare script the same way main.py does)
if go_infer:
    PY = sys.executable
    for name in picked:
        out_dir = LORA / name
        weight = out_dir / f"sd15_lora_r8_a8_final.safetensors"
        if not weight.exists():
            alt = out_dir / f"sd15_lora_r8_a8_best.safetensors"
            if alt.exists(): weight = alt
        out_samples = SAMPLES / f"{name}_ad_hoc"
        out_samples.mkdir(parents=True, exist_ok=True)
        cmd = [
            PY, str(ROOT / "scripts" / "compare_lora_infer.py"),
            "--prompt", prompt,
            "--vanilla_lora_dir", str(out_dir),
            "--vanilla_lora_weight", weight.name,
            "--outdir", str(out_samples)
        ]
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
            st.success(f"Inference done for {name}. See {out_samples}")
        except Exception as e:
            st.error(f"{name}: {e}")

st.caption("Tip: Launch with `streamlit run app/dashboard.py` from your conda env.")
