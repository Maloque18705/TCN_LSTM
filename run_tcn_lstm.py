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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Train TCN model and save artifacts")
    p.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--outdir", type=str, default="./outputs/tcn_lstm", help="Base output directory")
    p.add_argument("--no-save", action="store_true", help="Do not save artifacts (for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"run_TCN_LSTM_{stamp}"
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

    # If training history is available, plot Loss (MSE) and MAE curves and save to outputs
    history = result.get("history") if result is not None else None
    if history and isinstance(history, dict):
        try:
            # Plot MSE Loss
            if 'loss' in history:
                plt.figure(figsize=(10, 5))
                plt.plot(history.get('loss', []), label='Train Loss')
                plt.plot(history.get('val_loss', []), label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (MSE)')
                plt.title('Training and Validation MSE Loss - TCN-LSTM')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(run_dir / 'loss_mse.png')
                plt.close()

            # Plot MAE
            # metric key may be 'mean_absolute_error' or 'mae'
            mae_key = None
            for k in ('mean_absolute_error', 'mae'):
                if k in history:
                    mae_key = k
                    break

            if mae_key is not None:
                plt.figure(figsize=(10, 5))
                plt.plot(history.get(mae_key, []), label='Train MAE')
                plt.plot(history.get(f'val_{mae_key}', []), label='Validation MAE')
                plt.xlabel('Epochs')
                plt.ylabel('MAE')
                plt.title('Training and Validation MAE - TCN-LSTM')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(run_dir / 'mae.png')
                plt.close()
        except Exception as e:
            print(f"Warning: failed to plot history: {e}")

    print("Run finished. Summary written to:", summary_path)
    print("Artifacts saved under:", run_dir)


if __name__ == "__main__":
    main()
