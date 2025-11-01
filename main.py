"""Entry point to train the TCN model and save artifacts.

Usage:
    python main.py --epochs 100 --batch-size 64 --outdir ./outputs

This script creates a timestamped subdirectory under --outdir and runs Train.train.train.
It saves a small JSON summary after training with paths and basic metrics.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from Train.train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train TCN model and save artifacts")
    p.add_argument("--epochs", type=int, default=1000, help="Maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--outdir", type=str, default="./outputs", help="Base output directory")
    p.add_argument("--no-save", action="store_true", help="Do not save artifacts (for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training. Artifacts will be saved to: {run_dir}")

    try:
        result = train(epochs=args.epochs, batch_size=args.batch_size, save_dir=str(run_dir))
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"Training failed with exception: {e}")
        raise

    # Save a small summary JSON with returned metrics and locations
    summary = {
        "timestamp": stamp,
        "out_dir": str(run_dir),
        "training_time_s": result.get("training_time_s"),
        "metrics": result.get("metrics"),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Run finished. Summary written to:", summary_path)
    print("Artifacts saved under:", run_dir)


if __name__ == "__main__":
    main()
