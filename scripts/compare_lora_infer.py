# scripts/plot_training.py
import argparse, pandas as pd, matplotlib.pyplot as plt, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out = pathlib.Path(args.out) if args.out else csv_path.with_suffix(".png")

    plt.figure()
    plt.plot(df["step"], df["train_loss"], label="train")
    plt.plot(df["step"], df["val_loss"], label="val")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("LoRA Training")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved plot to {out.resolve()}")

if __name__ == "__main__":
    main()
