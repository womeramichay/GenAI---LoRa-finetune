# app/dashboard.py
import streamlit as st
import subprocess, pathlib, sys, os, time, shutil
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root
SCRIPTS = ROOT / "scripts"
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
LORA_DIR = RUNS / "lora"
SAMPLES_DIR = RUNS / "samples"

# ---------- Helpers ----------
def ensure_dirs():
    LOGS.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def tail_file(path: pathlib.Path, n: int = 80) -> str:
    if not path.exists():
        return "[no log yet]"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except Exception as e:
        return f"[tail error: {e}]"

def list_csvs():
    return sorted(LOGS.glob("*.csv"))

def list_lora_runs():
    return [p for p in sorted(LORA_DIR.glob("*")) if p.is_dir()]

def list_lora_weights(run_dir: pathlib.Path):
    if not run_dir or not run_dir.exists():
        return []
    return sorted(run_dir.glob("*.safetensors"))

def pick_latest_weight(run_dir: pathlib.Path):
    if not run_dir or not run_dir.exists():
        return None
    cands = list_lora_weights(run_dir)
    if not cands:
        return None
    finals = [p for p in cands if "final" in p.name.lower()] or [p for p in cands if "best" in p.name.lower()]
    return (finals[-1] if finals else cands[-1])

def human_relpath(p: pathlib.Path):
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def format_lora_dir(p: pathlib.Path):
    return p.name

def log_line(msg: str):
    st.write(f"**{msg}**")

# ---------- UI ----------
st.set_page_config(page_title="Tattoo LoRA Dashboard", layout="wide")
ensure_dirs()

st.title("🖋️ Tattoo LoRA Dashboard")
st.caption(human_relpath(ROOT))

tab_train, tab_logs, tab_infer, tab_cleanup, tab_export = st.tabs(
    ["Train", "Logs/Curves", "Inference", "Cleanup", "Export README"]
)

# =========================================================
#                       TRAIN
# =========================================================
with tab_train:
    st.header("Train a LoRA")

    # Dataset + params
    colA, colB = st.columns([2, 1])
    with colA:
        dataset = st.selectbox("Dataset variant", ["vanilla", "blip", "blip_clean"], index=0)
        size = st.number_input("Resolution", 256, 768, 512, step=64)
        rank = st.number_input("LoRA rank (r)", 2, 128, 8, step=2)
        alpha = st.number_input("LoRA alpha", 2, 256, 8, step=2)
        max_steps = st.number_input("Max steps", 50, 10000, 250, step=50)
        save_every = st.number_input("Save every (steps)", 10, 1000, 50, step=10)
        eval_every = st.number_input("Eval every (steps)", 10, 1000, 50, step=10)
    with colB:
        batch_size = st.number_input("Batch size", 1, 8, 1, step=1)
        grad_accum = st.number_input("Grad accumulation", 1, 64, 8, step=1)
        clip_gn = st.number_input("Clip grad norm", 0.0, 10.0, 1.0, step=0.1)
        seed = st.number_input("Seed", 0, 999999, 42, step=1)
        quiet = st.checkbox("Quiet logging", value=True)

    # Folders
    base_dir = ROOT / f"data/processed/tattoo_v3/{dataset}"
    images_dir = base_dir / "images"
    captions_dir = base_dir / "captions"
    out_dir = LORA_DIR / f"tattoo_v3_{dataset}_r{rank}a{alpha}"
    log_file = LOGS / f"train_{dataset}_r{rank}a{alpha}_{int(time.time())}.log"

    st.markdown(f"""
- **Images**: `{human_relpath(images_dir)}`
- **Captions**: `{human_relpath(captions_dir)}`
- **Output (weights)**: `{human_relpath(out_dir)}`
- **Log**: `{human_relpath(log_file)}`
    """)

    # Resume options
    st.subheader("Resume from existing checkpoint (optional)")
    existing_runs = [None] + list_lora_runs()
    resume_run = st.selectbox("Choose a run to resume", existing_runs, format_func=lambda p: format_lora_dir(p) if p else "None")
    resume_weight = None
    if resume_run:
        weights = list_lora_weights(resume_run)
        resume_weight = st.selectbox("Choose a weight file", [None] + weights, format_func=lambda p: p.name if p else "None")

    # Build command
    cmd = [
        sys.executable, str(SCRIPTS / "train_lora.py"),
        "--data_images", str(images_dir),
        "--data_captions", str(captions_dir),
        "--output_dir", str(out_dir),
        "--resolution", str(size),
        "--rank", str(rank),
        "--alpha", str(alpha),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--max_steps", str(max_steps),
        "--save_every", str(save_every),
        "--eval_every", str(eval_every),
        "--clip_grad_norm", str(clip_gn),
        "--seed", str(seed),
        "--log_file", str(log_file),
    ]
    if quiet:
        cmd.append("--quiet")
    if resume_run and resume_weight:
        cmd += ["--resume_from_dir", str(resume_run), "--resume_weight_name", resume_weight.name]

    st.code(" ".join(cmd), language="bash")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🚀 Start Training"):
            ensure_dirs()
            # Start subprocess; let it run in background
            st.session_state["train_proc"] = subprocess.Popen(cmd, cwd=str(ROOT))
            st.success(f"Started training PID={st.session_state['train_proc'].pid}")

    with c2:
        if "train_proc" in st.session_state:
            p = st.session_state["train_proc"]
            st.info(f"Training PID: {p.pid} (running={p.poll() is None})")
            if st.button("Refresh Log Tail"):
                st.text(tail_file(log_file, n=100))

# =========================================================
#                      LOGS / CURVES
# =========================================================
with tab_logs:
    st.header("Training Logs & Curves")
    csvs = list_csvs()
    if not csvs:
        st.info("No CSV logs found yet. Train first.")
    else:
        sel = st.selectbox("Select CSV", csvs, index=len(csvs)-1, format_func=lambda p: p.name)
        if sel and sel.exists():
            # show tail
            if st.button("Show last 80 lines"):
                st.text(tail_file(sel, n=80))
            # plot
            try:
                df = pd.read_csv(sel)
                if not {"step", "train_loss", "val_loss"}.issubset(df.columns):
                    st.warning(f"{sel.name} doesn't contain expected columns.")
                else:
                    fig, ax = plt.subplots()
                    ax.plot(df["step"], df["train_loss"], label="train")
                    ax.plot(df["step"], df["val_loss"], label="val")
                    ax.set_xlabel("step"); ax.set_ylabel("loss")
                    ax.set_title(sel.name); ax.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not read/plot: {e}")

# =========================================================
#                        INFERENCE
# =========================================================
with tab_infer:
    st.header("Compare Inference")
    prompt = st.text_input("Prompt", "owl, clean line, minimal line-art tattoo, stencil, high contrast, no shading")
    steps = st.number_input("Steps", 5, 100, 25, step=5)
    guidance = st.number_input("CFG Guidance", 1.0, 20.0, 7.0, step=0.5)
    seed = st.number_input("Seed", 0, 1_000_000, 42, step=1)
    outdir = SAMPLES_DIR / "ui_compare"
    outdir.mkdir(parents=True, exist_ok=True)

    runs = [None] + list_lora_runs()
    vanilla = st.selectbox("LoRA A (optional)", runs, index=0, format_func=lambda p: format_lora_dir(p) if p else "None")
    blip = st.selectbox("LoRA B (optional)", runs, index=0, format_func=lambda p: format_lora_dir(p) if p else "None")

    def choose_weight(d: pathlib.Path):
        w = pick_latest_weight(d)
        if not w:
            st.warning(f"No weights found in {human_relpath(d)}")
        return w

    if st.button("🎨 Run Compare"):
        cmd = [
            sys.executable, str(SCRIPTS / "compare_lora_infer.py"),
            "--prompt", prompt,
            "--outdir", str(outdir),
            "--steps", str(steps),
            "--guidance", str(guidance),
            "--seed", str(seed),
            "--quiet", "--log_file", str(LOGS / "infer_ui.log"),
        ]
        if vanilla:
            vw = choose_weight(vanilla)
            if vw:
                cmd += ["--vanilla_lora_dir", str(vanilla), "--vanilla_lora_weight", vw.name]
        if blip:
            bw = choose_weight(blip)
            if bw:
                cmd += ["--blip_lora_dir", str(blip), "--blip_lora_weight", bw.name]

        st.code(" ".join(cmd), language="bash")
        subprocess.run(cmd, cwd=str(ROOT))
        st.success(f"Saved to: {human_relpath(outdir)}")

        cols = st.columns(3)
        for i, name in enumerate(["base.png", "vanilla_lora.png", "blip_lora.png"]):
            p = outdir / name
            if p.exists():
                cols[i].image(str(p), caption=name, use_column_width=True)
            else:
                cols[i].write(f"{name} not generated.")

# =========================================================
#                        CLEANUP
# =========================================================
with tab_cleanup:
    st.header("Cleanup: free disk space safely")
    st.caption("Tip: keep FINAL/BEST weights, remove intermediate *step*.safetensors and old samples.")

    # List candidates
    intermed = []
    for run in list_lora_runs():
        for w in run.glob("*.safetensors"):
            if "step" in w.name.lower():  # intermediate
                intermed.append(w)

    sample_pngs = sorted(SAMPLES_DIR.rglob("*.png"))

    st.subheader("Intermediate LoRA weights (*step*.safetensors)")
    sel_intermed = st.multiselect(
        "Select intermediate weights to delete",
        intermed,
        format_func=lambda p: human_relpath(p)
    )

    st.subheader("Sample images (runs/samples/**.png)")
    sel_samples = st.multiselect(
        "Select sample images to delete",
        sample_pngs,
        format_func=lambda p: human_relpath(p)
    )

    if st.button("🗑️ Delete Selected"):
        deleted = 0
        for p in sel_intermed + sel_samples:
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                st.warning(f"Could not delete {human_relpath(p)}: {e}")
        st.success(f"Deleted {deleted} files.")

# =========================================================
#                     EXPORT README
# =========================================================
with tab_export:
    st.header("Export README Snapshot")
    st.caption("Creates README_snapshot.md summarizing best checkpoints, curves, and sample images.")

    # Pick a run to summarize
    run = st.selectbox("Pick a run", list_lora_runs(), format_func=format_lora_dir)
    if run:
        best = None
        weights = list_lora_weights(run)
        finals = [w for w in weights if "final" in w.name.lower()]
        bests  = [w for w in weights if "best" in w.name.lower()]
        best = (bests[-1] if bests else (finals[-1] if finals else (weights[-1] if weights else None)))
        st.markdown(f"- **Selected run**: `{human_relpath(run)}`")
        st.markdown(f"- **Best weight**: `{best.name if best else 'N/A'}`")

    # Pick log CSV (optional)
    csvs = list_csvs()
    csv_sel = st.selectbox("Pick training CSV (optional)", [None] + csvs, index=0, format_func=lambda p: p.name if p else "None")

    # Pick 0–3 sample images to embed
    all_samples = sorted(SAMPLES_DIR.glob("*/*.png")) + sorted(SAMPLES_DIR.glob("*.png"))
    sample_sel = st.multiselect("Pick sample images to include (up to 6)", all_samples[:200], max_selections=6,
                                format_func=lambda p: human_relpath(p))

    if st.button("📝 Create README_snapshot.md"):
        out_md = ROOT / "README_snapshot.md"
        lines = []
        lines.append(f"# Tattoo LoRA Snapshot\n")
        lines.append(f"- Export time: {time.ctime()}")
        if run:
            lines.append(f"- Run: `{human_relpath(run)}`")
        if best:
            lines.append(f"- Best weight: `{best.name}`")
        lines.append("")
        if csv_sel and csv_sel.exists():
            lines.append(f"## Training Curves\n")
            lines.append(f"CSV: `{human_relpath(csv_sel)}`")
            try:
                df = pd.read_csv(csv_sel)
                if {"step", "train_loss", "val_loss"}.issubset(df.columns):
                    # Save a quick plot alongside snapshot
                    fig_path = ROOT / "runs" / "logs" / f"{csv_sel.stem}_plot.png"
                    fig, ax = plt.subplots()
                    ax.plot(df["step"], df["train_loss"], label="train")
                    ax.plot(df["step"], df["val_loss"], label="val")
                    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
                    ax.set_title(csv_sel.name)
                    fig.savefig(fig_path, bbox_inches="tight")
                    plt.close(fig)
                    lines.append(f"![Training Curves]({human_relpath(fig_path)})")
                else:
                    lines.append("_CSV missing columns for plotting._")
            except Exception as e:
                lines.append(f"_Could not plot CSV: {e}_")

        if sample_sel:
            lines.append("\n## Samples\n")
            for p in sample_sel:
                lines.append(f"**{p.name}**")
                lines.append(f"\n![]({human_relpath(p)})\n")

        (out_md).write_text("\n".join(lines), encoding="utf-8")
        st.success(f"Wrote {human_relpath(out_md)}")
