# scripts/plot_training.py
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One or more CSV log files.")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels (same length as --csv).")
    ap.add_argument("--out", default=None, help="Output PNG path; if omitted, uses first CSV name with _plot.png")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.csv):
        raise ValueError("labels length must match number of csv files")

    plt.figure(figsize=(8,5))
    for i, path in enumerate(args.csv):
        df = pd.read_csv(path)
        label = args.labels[i] if args.labels else pathlib.Path(path).stem
        if "step" not in df.columns or "train_loss" not in df.columns or "val_loss" not in df.columns:
            print(f"WARNING: {path} missing expected columns [step, train_loss, val_loss]; skipping.")
            continue
        plt.plot(df["step"], df["train_loss"], linestyle="--", alpha=0.7, label=f"{label} (train)")
        plt.plot(df["step"], df["val_loss"],  linewidth=2.0,              label=f"{label} (val)")

    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("LoRA training & validation loss")
    plt.legend()
    plt.grid(True, alpha=0.25)

    out = args.out
    if not out:
        first = pathlib.Path(args.csv[0])
        out = str(first.with_name(first.stem + "_plot.png"))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot → {out}")

if __name__ == "__main__":
    main()
