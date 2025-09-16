#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
SAMPLES = RUNS / "samples"
LORA = RUNS / "lora"

st.set_page_config(page_title="Tattoo LoRA Dashboard", layout="wide")
st.title("🖊️ Tattoo LoRA — Training Dashboard (auto-discovered models)")


# ---------- discovery helpers ----------
def _weight_priority(files: List[Path]) -> Optional[Path]:
    """Pick weight by priority: *best* -> *final* -> last modified *.safetensors."""
    if not files:
        return None
    best = [p for p in files if "best" in p.stem.lower()]
    if best:
        return sorted(best)[-1]
    final = [p for p in files if "final" in p.stem.lower()]
    if final:
        return sorted(final)[-1]
    return max(files, key=lambda p: p.stat().st_mtime)  # latest by mtime


def _latest_log(run_name: str) -> Optional[Path]:
    if not LOGS.exists():
        return None
    csvs = sorted(glob.glob(str(LOGS / f"{run_name}_*.csv")))
    return Path(csvs[-1]) if csvs else None


def _read_log_tail(csv_path: Path) -> Dict[str, Optional[float]]:
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return {"step": None, "val_loss": None, "clip_score": None}
        last = df.iloc[-1]
        return {
            "step": float(last.get("step", float("nan"))),
            "val_loss": float(last.get("val_loss", float("nan"))),
            "clip_score": float(last.get("clip_score", float("nan"))) if "clip_score" in df.columns else None,
        }
    except Exception:
        return {"step": None, "val_loss": None, "clip_score": None}


def _discover_models() -> pd.DataFrame:
    rows = []
    if not LORA.exists():
        return pd.DataFrame(columns=[
            "run_name", "path", "best?", "final?", "picked_weight", "last_step", "last_val_loss", "last_clip_score"
        ])

    for run_dir in sorted([p for p in LORA.glob("*") if p.is_dir()]):
        weights = sorted(run_dir.glob("*.safetensors"))
        if not weights:
            # skip empty run folders (pre-created but not trained)
            continue

        picked = _weight_priority(weights)
        has_best = any("best" in p.stem.lower() for p in weights)
        has_final = any("final" in p.stem.lower() for p in weights)

        log = _latest_log(run_dir.name)
        tail = _read_log_tail(log) if log else {"step": None, "val_loss": None, "clip_score": None}

        rows.append({
            "run_name": run_dir.name,
            "path": str(run_dir.resolve()),
            "best?": "✅" if has_best else "—",
            "final?": "✅" if has_final else "—",
            "picked_weight": picked.name if picked else "—",
            "last_step": tail["step"],
            "last_val_loss": tail["val_loss"],
            "last_clip_score": tail["clip_score"],
        })

    cols = ["run_name", "path", "best?", "final?", "picked_weight", "last_step", "last_val_loss", "last_clip_score"]
    return pd.DataFrame(rows, columns=cols)


def _plot_loss(csv_path: Path, title: str = ""):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    if {"step", "train_loss", "val_loss"}.issubset(df.columns):
        ax.plot(df["step"], df["train_loss"], label="train", linestyle="--", alpha=0.7)
        ax.plot(df["step"], df["val_loss"], label="val", linewidth=2)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(title or csv_path.stem); ax.legend(loc="upper left")
        if "clip_score" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df["step"], df["clip_score"], label="clip", alpha=0.45)
            ax2.set_ylabel("CLIP score")
    else:
        ax.text(0.1, 0.5, f"{csv_path.name}\nmissing expected columns", fontsize=12)
        ax.axis("off")
    st.pyplot(fig)


def _run_compare(prompt: str, dirs: List[Path], outdir: Path, steps=20, guidance=6.5, width=448, height=448):
    """Call scripts/compare_lora_infer.py and surface real errors."""
    py = sys.executable
    script = str(ROOT / "scripts" / "compare_lora_infer.py")
    cmd = [
        py, script, "--prompt", prompt, "--outdir", str(outdir),
        "--steps", str(steps), "--guidance", str(guidance),
        "--width", str(width), "--height", str(height),
    ]
    slots = ["--vanilla_lora_dir", "--blip_lora_dir", "--blip_plus_lora_dir"]
    for i, d in enumerate(dirs[:3]):
        if d is not None:
            cmd += [slots[i], str(d)]
    outdir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        st.error(
            f"compare_lora_infer failed (code {res.returncode})\n\n"
            f"STDERR:\n{res.stderr}\n\nSTDOUT:\n{res.stdout}"
        )
    return res.returncode


# ---------- discovery UI ----------
st.sidebar.subheader("Discovery")
if st.sidebar.button("🔄 Refresh"):
    st.experimental_rerun()

search_q = st.sidebar.text_input("Filter runs (substring)", "")

catalog_df = _discover_models()
if search_q.strip():
    q = search_q.strip().lower()
    catalog_df = catalog_df[catalog_df["run_name"].str.lower().str.contains(q) | catalog_df["path"].str.lower().str.contains(q)]

if catalog_df.empty:
    st.warning("No trained models found (needs at least one *.safetensors under runs/lora/<run>).")
    st.stop()

st.subheader("📦 Discovered models")
st.dataframe(
    catalog_df[["run_name", "best?", "final?", "picked_weight", "last_step", "last_val_loss", "last_clip_score"]],
    hide_index=True,
    use_container_width=True,
)

# Build a list of valid run directories (guaranteed to have weights)
RUN_INDEX = {row.run_name: Path(row.path) for row in catalog_df.itertuples(index=False)}

# ---------- shared generation params ----------
st.sidebar.header("Generation params")
steps = st.sidebar.slider("Steps", 10, 40, 20)
guidance = st.sidebar.slider("Guidance", 4.0, 10.0, 6.5, 0.5)
width = st.sidebar.selectbox("Width", [384, 448, 512], index=1)
height = st.sidebar.selectbox("Height", [384, 448, 512], index=1)
prompt_default = "owl, clean line, minimal line-art tattoo, stencil, high contrast, no shading"

tab_single, tab_compare = st.tabs(["🎯 Single Model", "🧪 Compare Models"])

# ========================= Single Model =========================
with tab_single:
    left, right = st.columns([1, 2], gap="large")

    with left:
        pick = st.selectbox("Pick a model", list(RUN_INDEX.keys()), index=0)
        prompt_single = st.text_input("Prompt", prompt_default)
        gen_one = st.button("▶️ Generate with selected model")

        st.subheader("📉 Training metrics")
        log = _latest_log(pick)
        if log:
            _plot_loss(log, title=pick)
        else:
            st.info("No CSV log found for this model yet.")

    with right:
        st.subheader("ℹ️ Model path")
        st.code(str(RUN_INDEX[pick].resolve()), language="text")
        if gen_one:
            outdir = SAMPLES / f"{pick}_ad_hoc"
            code = _run_compare(prompt_single, [RUN_INDEX[pick]], outdir, steps, guidance, width, height)
            if code == 0:
                st.success(f"Generated into: {outdir}")
                imgs = sorted(outdir.glob("*.png"))[-3:]
                for p in imgs:
                    st.image(str(p), use_container_width=True, caption=p.name)

# ========================= Compare Models =========================
with tab_compare:
    st.markdown("Pick **2–3 models**. One prompt will be used for all.")
    default_picks = list(RUN_INDEX.keys())[:2]
    picks = st.multiselect("Models to compare", list(RUN_INDEX.keys()), default=default_picks)
    prompt_cmp = st.text_input("Compare prompt", prompt_default, key="cmp_prompt")
    gen_cmp = st.button("▶️ Generate side-by-side for picked models")

    st.subheader("📊 Validation loss overlay (and CLIP if available)")
    if picks:
        fig, ax = plt.subplots()
        for name in picks:
            log = _latest_log(name)
            if not log:
                continue
            df = pd.read_csv(log)
            if {"step", "val_loss"}.issubset(df.columns):
                ax.plot(df["step"], df["val_loss"], label=name)
        ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend()
        st.pyplot(fig)

    if gen_cmp and len(picks) >= 2:
        outdir = SAMPLES / f"compare_{'_'.join(picks)}"
        code = _run_compare(prompt_cmp, [RUN_INDEX[n] for n in picks], outdir, steps, guidance, width, height)
        if code == 0:
            st.success(f"Generated into: {outdir}")
            imgs = sorted(outdir.glob("*.png"))
            for img in imgs:
                st.image(str(img), use_container_width=True, caption=img.name)

st.caption("Models are auto-discovered from folders under runs/lora/* that contain .safetensors. "
           "Weights are auto-picked by priority: *best* → *final* → latest file.")
